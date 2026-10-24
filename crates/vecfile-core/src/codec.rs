//! Byte codecs for vectors, bitmaps, compressed blobs and sealed segments.
//!
//! Arrays are stored as raw little-endian bytes so a segment part loads with
//! a single copy. Everything read back is validated: a corrupted file yields
//! [`StorageError::Corrupt`](crate::StorageError::Corrupt), never a panic.

use redb::{ReadableTable, Table};
use roaring::RoaringBitmap;
use vecfile_index::Metric;
use vecfile_index::hnsw::GraphParts;
use vecfile_index::segment::{SealedSegment, SegmentParts};

use crate::records::{SegmentHeader, decode, encode};
use crate::tables::SegmentPart;
use crate::{Result, corrupt};

/// zstd level for document content: fast, with most of the ratio.
const ZSTD_LEVEL: i32 = 3;

pub fn metric_to_u8(m: Metric) -> u8 {
    match m {
        Metric::Cosine => 0,
        Metric::Dot => 1,
        Metric::L2 => 2,
    }
}

pub fn metric_from_u8(b: u8) -> Result<Metric> {
    match b {
        0 => Ok(Metric::Cosine),
        1 => Ok(Metric::Dot),
        2 => Ok(Metric::L2),
        _ => Err(corrupt(format!("unknown metric {b}"))),
    }
}

pub fn vector_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

/// Copies a stored vector into `out`; fails if the length is wrong.
pub fn read_vector(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let dst: &mut [u8] = bytemuck::cast_slice_mut(out);
    if bytes.len() != dst.len() {
        return Err(corrupt(format!(
            "vector has {} bytes, expected {}",
            bytes.len(),
            dst.len()
        )));
    }
    dst.copy_from_slice(bytes);
    Ok(())
}

pub fn bitmap_bytes(b: &RoaringBitmap) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.serialized_size());
    b.serialize_into(&mut out)
        .expect("writing to a Vec cannot fail");
    out
}

pub fn read_bitmap(bytes: &[u8]) -> Result<RoaringBitmap> {
    RoaringBitmap::deserialize_from(bytes).map_err(|e| corrupt(format!("bitmap: {e}")))
}

pub fn compress(content: &[u8]) -> Vec<u8> {
    zstd::bulk::compress(content, ZSTD_LEVEL)
        .expect("zstd compression of an in-memory buffer cannot fail")
}

/// Decompresses a blob whose original length is known, refusing anything
/// that would expand beyond it.
pub fn decompress(blob: &[u8], len: u64) -> Result<Vec<u8>> {
    let len = usize::try_from(len).map_err(|_| corrupt("content length overflows usize"))?;
    let out = zstd::bulk::decompress(blob, len).map_err(|e| corrupt(format!("zstd: {e}")))?;
    if out.len() != len {
        return Err(corrupt(format!(
            "content is {} bytes, expected {len}",
            out.len()
        )));
    }
    Ok(out)
}

fn pod_vec<T: bytemuck::Pod>(bytes: &[u8], what: &str) -> Result<Vec<T>> {
    if !bytes.len().is_multiple_of(size_of::<T>()) {
        return Err(corrupt(format!(
            "{what}: {} bytes is not a whole number of elements",
            bytes.len()
        )));
    }
    let mut out = vec![T::zeroed(); bytes.len() / size_of::<T>()];
    bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(bytes);
    Ok(out)
}

type SegmentsTable<'t> = Table<'t, (u32, u64), &'static [u8]>;
type SegmentDataTable<'t> = Table<'t, (u32, u64, u8), &'static [u8]>;

/// Writes a sealed segment's header and parts.
pub fn write_segment(
    headers: &mut SegmentsTable<'_>,
    data: &mut SegmentDataTable<'_>,
    coll: u32,
    id: u64,
    seg: &SealedSegment,
) -> Result<()> {
    let p = seg.to_parts();
    let header = SegmentHeader {
        dim: p.dim as u32,
        metric: metric_to_u8(p.metric),
        len: p.ids.len() as u32,
        m: p.graph.m,
        entry: p.graph.entry,
        max_level: p.graph.max_level,
        quant_scale: p.quant_scale,
    };
    headers.insert((coll, id), encode(&header).as_slice())?;
    let parts: [(SegmentPart, &[u8]); 9] = [
        (SegmentPart::Levels, &p.graph.levels),
        (SegmentPart::Level0, bytemuck::cast_slice(&p.graph.level0)),
        (
            SegmentPart::UpperOffsets,
            bytemuck::cast_slice(&p.graph.upper_off),
        ),
        (SegmentPart::Upper, bytemuck::cast_slice(&p.graph.upper)),
        (
            SegmentPart::QuantOffset,
            bytemuck::cast_slice(&p.quant_offset),
        ),
        (SegmentPart::Codes, bytemuck::cast_slice(&p.codes)),
        (SegmentPart::Corrections, bytemuck::cast_slice(&p.corr)),
        (SegmentPart::Ids, bytemuck::cast_slice(&p.ids)),
        (SegmentPart::Vectors, bytemuck::cast_slice(&p.vectors)),
    ];
    for (part, bytes) in parts {
        data.insert((coll, id, part as u8), bytes)?;
    }
    Ok(())
}

/// Removes a sealed segment's header and parts.
pub fn delete_segment(
    headers: &mut SegmentsTable<'_>,
    data: &mut SegmentDataTable<'_>,
    coll: u32,
    id: u64,
) -> Result<()> {
    headers.remove((coll, id))?;
    for part in SegmentPart::ALL {
        data.remove((coll, id, part as u8))?;
    }
    Ok(())
}

/// Reads one array of a segment, copied once straight out of the stored value.
fn read_part<T: bytemuck::Pod>(
    data: &impl ReadableTable<(u32, u64, u8), &'static [u8]>,
    coll: u32,
    id: u64,
    part: SegmentPart,
) -> Result<Vec<T>> {
    let guard = data
        .get((coll, id, part as u8))?
        .ok_or_else(|| corrupt(format!("segment {id} is missing part {part:?}")))?;
    pod_vec(guard.value(), &format!("segment {id} part {part:?}"))
}

/// Reads and validates a sealed segment.
pub fn read_segment(
    headers: &impl ReadableTable<(u32, u64), &'static [u8]>,
    data: &impl ReadableTable<(u32, u64, u8), &'static [u8]>,
    coll: u32,
    id: u64,
) -> Result<SealedSegment> {
    let header_bytes = headers
        .get((coll, id))?
        .ok_or_else(|| corrupt(format!("segment {id} has no header")))?;
    let h: SegmentHeader = decode(header_bytes.value(), "segment header")?;
    let parts = SegmentParts {
        dim: h.dim as usize,
        metric: metric_from_u8(h.metric)?,
        graph: GraphParts {
            m: h.m,
            entry: h.entry,
            max_level: h.max_level,
            levels: read_part(data, coll, id, SegmentPart::Levels)?,
            level0: read_part(data, coll, id, SegmentPart::Level0)?,
            upper_off: read_part(data, coll, id, SegmentPart::UpperOffsets)?,
            upper: read_part(data, coll, id, SegmentPart::Upper)?,
        },
        quant_offset: read_part(data, coll, id, SegmentPart::QuantOffset)?,
        quant_scale: h.quant_scale,
        codes: read_part(data, coll, id, SegmentPart::Codes)?,
        corr: read_part(data, coll, id, SegmentPart::Corrections)?,
        vectors: read_part(data, coll, id, SegmentPart::Vectors)?,
        ids: read_part(data, coll, id, SegmentPart::Ids)?,
    };
    if parts.ids.len() != h.len as usize {
        return Err(corrupt(format!(
            "segment {id}: header says {} rows, ids say {}",
            h.len,
            parts.ids.len()
        )));
    }
    SealedSegment::from_parts(parts).map_err(|e| corrupt(format!("segment {id}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{SEGMENT_DATA, SEGMENTS};
    use redb::{Database, ReadableDatabase};
    use vecfile_index::hnsw::HnswParams;
    use vecfile_index::rng::SplitMix64;

    fn segment(n: usize, dim: usize) -> SealedSegment {
        let mut r = SplitMix64::new(3);
        let data: Vec<f32> = (0..n * dim).map(|_| r.next_f64_open() as f32).collect();
        let ids: Vec<u32> = (10..10 + n as u32).collect();
        SealedSegment::build(&ids, &data, dim, Metric::Cosine, &HnswParams::default())
    }

    #[test]
    fn segments_round_trip_through_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::create(dir.path().join("t.vdb")).unwrap();
        let seg = segment(500, 12);
        let tx = db.begin_write().unwrap();
        {
            let mut h = tx.open_table(SEGMENTS).unwrap();
            let mut d = tx.open_table(SEGMENT_DATA).unwrap();
            write_segment(&mut h, &mut d, 1, 7, &seg).unwrap();
        }
        tx.commit().unwrap();

        let rx = db.begin_read().unwrap();
        let h = rx.open_table(SEGMENTS).unwrap();
        let d = rx.open_table(SEGMENT_DATA).unwrap();
        assert_eq!(read_segment(&h, &d, 1, 7).unwrap(), seg);
        assert!(read_segment(&h, &d, 1, 8).is_err());
    }

    #[test]
    fn corrupted_parts_are_errors_not_panics() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::create(dir.path().join("t.vdb")).unwrap();
        let seg = segment(200, 4);
        let tx = db.begin_write().unwrap();
        {
            let mut h = tx.open_table(SEGMENTS).unwrap();
            let mut d = tx.open_table(SEGMENT_DATA).unwrap();
            write_segment(&mut h, &mut d, 1, 1, &seg).unwrap();
        }
        tx.commit().unwrap();

        let mut r = SplitMix64::new(11);
        for round in 0..90 {
            let part = SegmentPart::ALL[round % SegmentPart::ALL.len()] as u8;
            let tx = db.begin_write().unwrap();
            {
                let mut d = tx.open_table(SEGMENT_DATA).unwrap();
                let mut bytes = d.get((1, 1, part)).unwrap().unwrap().value().to_vec();
                match r.next_u64() % 3 {
                    0 if !bytes.is_empty() => {
                        let i = (r.next_u64() as usize) % bytes.len();
                        bytes[i] ^= 1 << (r.next_u64() % 8);
                    }
                    1 => {
                        bytes.truncate(bytes.len().saturating_sub(1 + (r.next_u64() as usize % 8)))
                    }
                    _ => bytes.extend_from_slice(&[0xAB; 4]),
                }
                d.insert((1, 1, part), bytes.as_slice()).unwrap();
            }
            tx.commit().unwrap();
            let rx = db.begin_read().unwrap();
            let h = rx.open_table(SEGMENTS).unwrap();
            let d = rx.open_table(SEGMENT_DATA).unwrap();
            // Either a valid segment (a flipped bit in a code or offset) or an error.
            let _ = read_segment(&h, &d, 1, 1);
            drop((h, d, rx));
            // Restore the original for the next round.
            let tx = db.begin_write().unwrap();
            {
                let mut h = tx.open_table(SEGMENTS).unwrap();
                let mut d = tx.open_table(SEGMENT_DATA).unwrap();
                write_segment(&mut h, &mut d, 1, 1, &seg).unwrap();
            }
            tx.commit().unwrap();
        }
    }

    #[test]
    fn blobs_and_bitmaps() {
        let text = "hello hello hello hello".repeat(100);
        let blob = compress(text.as_bytes());
        assert!(blob.len() < text.len());
        assert_eq!(
            decompress(&blob, text.len() as u64).unwrap(),
            text.as_bytes()
        );
        assert!(decompress(&blob, 10).is_err());
        assert!(decompress(b"not zstd", 10).is_err());

        let b: RoaringBitmap = [1u32, 5, 100_000].into_iter().collect();
        assert_eq!(read_bitmap(&bitmap_bytes(&b)).unwrap(), b);
        assert!(read_bitmap(&[1, 2, 3]).is_err());

        let v = [1.5f32, -2.0, 3.25];
        let mut out = [0f32; 3];
        read_vector(vector_bytes(&v), &mut out).unwrap();
        assert_eq!(out, v);
        assert!(read_vector(&[0; 5], &mut out).is_err());
    }
}
