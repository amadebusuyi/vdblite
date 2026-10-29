//! Sealed segments in regions of the file, and the space regions use.
//!
//! A segment is written as one region: its arrays one after another, each
//! aligned for direct use, the largest (the full vectors: f16 for int8
//! segments, f32 otherwise) last and page-aligned. A [`SegmentRecord`] in the `segments` table says where each
//! array sits. Loading a segment maps the region and borrows the arrays in
//! place ([`vdblite_index::Buf`]); nothing is copied.
//!
//! The record carries an xxh3 of every array except the full vectors, which
//! loading checks. The full vectors are left out on purpose: checking them
//! would read every row at open and make the whole region resident, while
//! the graph walk never touches most of them. A damaged one can at worst
//! give a wrong distance in a re-rank, never a panic.

use std::sync::Arc;

use half::f16;
use half::slice::HalfFloatSliceExt;
use redb::{ReadableTable, Table};
use vdblite_index::buf::{Backing, Buf};
use vdblite_index::hnsw::GraphParts;
use vdblite_index::segment::{FullVectors, SealedSegment, SegmentArrays, SegmentParts};
use xxhash_rust::xxh3::Xxh3;

use crate::codec::{metric_from_u8, metric_to_u8, read_vector};
use crate::container::{Container, PAGE};
use crate::records::{BulkRecord, SegmentHeader, SegmentRecord, decode};
use crate::tables::SegmentPart;
use crate::{Result, StorageError, corrupt};

/// The arrays of a segment region, in storage order: the full vectors,
/// the largest, last.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Levels = 0,
    Level0 = 1,
    UpperOffsets = 2,
    Upper = 3,
    Codes = 4,
    Corrections = 5,
    Ids = 6,
    LabelOffsets = 7,
    LabelBytes = 8,
    /// A binary quantizer's centroid and rotation (f32).
    BinaryState = 9,
    Full = 10,
}

const PARTS: usize = 11;
/// Records written before binary codes list 10 arrays, the full vectors
/// last.
const PARTS_BEFORE_BINARY: usize = 10;
/// Alignment of every array but the last.
const ALIGN: u64 = 64;

pub fn io(e: std::io::Error) -> StorageError {
    StorageError::Io(e)
}

/// The arrays of `a` as bytes, in [`Part`] order. The codes are int8
/// codes or 1-bit codes, whichever the segment has.
fn part_bytes<'a>(a: &SegmentArrays<'a>) -> [&'a [u8]; PARTS] {
    let codes = if a.bits.is_empty() {
        bytemuck::cast_slice(a.codes)
    } else {
        bytemuck::cast_slice(a.bits)
    };
    [
        a.levels,
        bytemuck::cast_slice(a.level0),
        bytemuck::cast_slice(a.upper_off),
        bytemuck::cast_slice(a.upper),
        codes,
        bytemuck::cast_slice(a.corr),
        bytemuck::cast_slice(a.ids),
        bytemuck::cast_slice(a.label_offsets),
        a.label_bytes,
        bytemuck::cast_slice(a.binary_state),
        a.full.bytes(),
    ]
}

/// Bytes per element of each array, as records count them. Codes and the
/// full vectors are counted in bytes: their kind follows from the binary
/// state and from rows × dimension.
const ELEM_SIZE: [u64; PARTS] = [1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 1];

/// Where each array goes in a region, and the region's length.
fn layout(bytes: &[&[u8]; PARTS]) -> ([u64; PARTS], u64) {
    let mut offs = [0u64; PARTS];
    let mut at = 0u64;
    for (i, b) in bytes.iter().enumerate() {
        let align = if i == Part::Full as usize {
            PAGE
        } else {
            ALIGN
        };
        at = at.div_ceil(align) * align;
        offs[i] = at;
        at += b.len() as u64;
    }
    (offs, at.max(1))
}

/// The length of the region `seg` needs.
pub fn region_len(seg: &SealedSegment) -> u64 {
    layout(&part_bytes(&seg.arrays())).1
}

/// xxh3 of every array but the last (the full vectors).
fn checksum(bytes: &[&[u8]]) -> u64 {
    let mut h = Xxh3::new();
    for b in &bytes[..bytes.len() - 1] {
        h.update(&(b.len() as u64).to_le_bytes());
        h.update(b);
    }
    h.digest()
}

/// Writes `seg` into the region at physical `offset` (of at least
/// [`region_len`] bytes) and returns its record. The caller syncs the file
/// before committing the record.
pub fn write_segment(c: &Container, seg: &SealedSegment, offset: u64) -> Result<SegmentRecord> {
    let a = seg.arrays();
    let bytes = part_bytes(&a);
    let (offs, len) = layout(&bytes);
    for (b, &o) in bytes.iter().zip(&offs) {
        c.write_at(offset + o, b).map_err(io)?;
    }
    Ok(SegmentRecord {
        dim: a.dim as u32,
        metric: metric_to_u8(a.metric),
        len: a.ids.len() as u32,
        m: a.m,
        entry: a.entry,
        max_level: a.max_level,
        quant_scale: a.quant_scale,
        quant_offset: a.quant_offset.to_vec(),
        region: (offset, len),
        parts: offs
            .iter()
            .zip(&bytes)
            .zip(ELEM_SIZE)
            .map(|((&o, b), size)| (o, b.len() as u64 / size))
            .collect(),
        checksum: checksum(&bytes),
    })
}

/// Maps a segment's region and loads it, borrowing every array in place.
/// The mapping leases `epoch` for as long as the segment lives.
pub fn map_segment(c: &Arc<Container>, rec: &SegmentRecord, epoch: u64) -> Result<SealedSegment> {
    let count = rec.parts.len();
    if count != PARTS && count != PARTS_BEFORE_BINARY {
        return Err(corrupt(format!(
            "segment record lists {count} arrays, expected {PARTS}"
        )));
    }
    let (offset, len) = rec.region;
    let file_len = c.physical_len().map_err(io)?;
    if offset.checked_add(len).is_none_or(|end| end > file_len) {
        return Err(corrupt(format!(
            "segment region {offset}+{len} lies past the end of the file"
        )));
    }
    let region: Arc<dyn Backing> = Arc::new(c.map(offset, len, epoch).map_err(io)?);
    fn buf<T: bytemuck::Pod>(region: &Arc<dyn Backing>, (o, n): (u64, u64)) -> Result<Buf<T>> {
        let (o, n) = (usize::try_from(o), usize::try_from(n));
        match (o, n) {
            (Ok(o), Ok(n)) => Buf::shared(region.clone(), o, n).map_err(corrupt),
            _ => Err(corrupt("segment array offset overflows")),
        }
    }
    let at = |part: Part| match (count, part) {
        (PARTS, _) => rec.parts[part as usize],
        (_, Part::BinaryState) => (0, 0),
        (_, Part::Full) => rec.parts[count - 1],
        (_, _) => rec.parts[part as usize],
    };
    let full = {
        let (o, bytes) = at(Part::Full);
        let values = rec.len as u64 * rec.dim as u64;
        if bytes == 2 * values {
            FullVectors::F16(buf(&region, (o, values))?)
        } else if bytes == 4 * values {
            FullVectors::F32(buf(&region, (o, values))?)
        } else {
            return Err(corrupt(format!(
                "segment region at {offset}: {bytes} bytes of vectors for {} rows",
                rec.len
            )));
        }
    };
    let binary_state: Buf<f32> = buf(&region, at(Part::BinaryState))?;
    let (codes, bits) = {
        let (o, bytes) = at(Part::Codes);
        if binary_state.is_empty() {
            (buf::<i8>(&region, (o, bytes))?, Buf::empty())
        } else if bytes.is_multiple_of(8) {
            (Buf::empty(), buf::<u64>(&region, (o, bytes / 8))?)
        } else {
            return Err(corrupt(format!(
                "segment region at {offset}: 1-bit codes are not whole words"
            )));
        }
    };
    let parts = SegmentParts {
        dim: rec.dim as usize,
        metric: metric_from_u8(rec.metric)?,
        graph: GraphParts {
            m: rec.m,
            entry: rec.entry,
            max_level: rec.max_level,
            levels: buf(&region, at(Part::Levels))?,
            level0: buf(&region, at(Part::Level0))?,
            upper_off: buf(&region, at(Part::UpperOffsets))?,
            upper: buf(&region, at(Part::Upper))?,
        },
        quant_offset: rec.quant_offset.clone(),
        quant_scale: rec.quant_scale,
        codes,
        bits,
        binary_state: binary_state.to_vec(),
        corr: buf(&region, at(Part::Corrections))?,
        full,
        ids: buf(&region, at(Part::Ids))?,
        label_offsets: buf(&region, at(Part::LabelOffsets))?,
        label_bytes: buf(&region, at(Part::LabelBytes))?,
    };
    let codes: &[u8] = if parts.bits.is_empty() {
        bytemuck::cast_slice(&parts.codes)
    } else {
        bytemuck::cast_slice(&parts.bits)
    };
    let mut bytes: Vec<&[u8]> = vec![
        &parts.graph.levels,
        bytemuck::cast_slice(&parts.graph.level0),
        bytemuck::cast_slice(&parts.graph.upper_off),
        bytemuck::cast_slice(&parts.graph.upper),
        codes,
        bytemuck::cast_slice(&parts.corr),
        bytemuck::cast_slice(&parts.ids),
        bytemuck::cast_slice(&parts.label_offsets),
        &parts.label_bytes,
    ];
    if count == PARTS {
        bytes.push(bytemuck::cast_slice(&binary_state));
    }
    bytes.push(parts.full.bytes());
    if checksum(&bytes) != rec.checksum {
        return Err(corrupt(format!(
            "segment region at {offset} fails its checksum"
        )));
    }
    if parts.ids.len() != rec.len as usize {
        return Err(corrupt(format!(
            "segment record says {} rows, its ids say {}",
            rec.len,
            parts.ids.len()
        )));
    }
    SealedSegment::from_parts(parts).map_err(|e| corrupt(format!("segment at {offset}: {e}")))
}

// ---------------------------------------------------------------- bulk loads

/// Bytes a bulk checkpoint of `count` vectors of `dim` needs, and where its
/// vectors start.
fn bulk_layout(count: usize, dim: usize, half: bool) -> (u64, u64) {
    let width = if half { 2 } else { 4 };
    let vectors_at = ((4 * count) as u64).div_ceil(PAGE) * PAGE;
    (vectors_at + (width * count * dim) as u64, vectors_at)
}

/// The region length a bulk checkpoint needs.
pub fn bulk_region_len(count: usize, dim: usize, half: bool) -> u64 {
    bulk_layout(count, dim, half).0.max(1)
}

/// Writes a bulk checkpoint (ids and their prepared vectors) at `offset`.
/// With `half`, the vectors are stored in half precision: enough for a
/// collection whose segments keep f16 vectors anyway.
pub fn write_bulk(
    c: &Container,
    ids: &[u32],
    vectors: &[f32],
    half: bool,
    offset: u64,
) -> Result<BulkRecord> {
    let dim = if ids.is_empty() {
        0
    } else {
        vectors.len() / ids.len()
    };
    let (len, vectors_at) = bulk_layout(ids.len(), dim, half);
    c.write_at(offset, bytemuck::cast_slice(ids)).map_err(io)?;
    if half {
        let mut narrow = vec![f16::ZERO; vectors.len()];
        narrow.convert_from_f32_slice(vectors);
        c.write_at(offset + vectors_at, bytemuck::cast_slice(&narrow))
    } else {
        c.write_at(offset + vectors_at, bytemuck::cast_slice(vectors))
    }
    .map_err(io)?;
    Ok(BulkRecord {
        region: (offset, len.max(1)),
        count: ids.len() as u32,
        dim: dim as u32,
        half,
    })
}

/// Reads a bulk checkpoint back: its chunk ids and prepared vectors.
pub fn read_bulk(c: &Container, rec: &BulkRecord) -> Result<(Vec<u32>, Vec<f32>)> {
    let (count, dim) = (rec.count as usize, rec.dim as usize);
    let (len, vectors_at) = bulk_layout(count, dim, rec.half);
    if rec.region.1 < len {
        return Err(corrupt("bulk checkpoint region is too short"));
    }
    let mut ids = vec![0u32; count];
    c.read_at(rec.region.0, bytemuck::cast_slice_mut(&mut ids))
        .map_err(io)?;
    let mut vectors = vec![0f32; count * dim];
    let at = rec.region.0 + vectors_at;
    if rec.half {
        let mut narrow = vec![f16::ZERO; count * dim];
        c.read_at(at, bytemuck::cast_slice_mut(&mut narrow))
            .map_err(io)?;
        narrow.convert_to_f32_slice(&mut vectors);
    } else {
        c.read_at(at, bytemuck::cast_slice_mut(&mut vectors))
            .map_err(io)?;
    }
    Ok((ids, vectors))
}

// ---------------------------------------------------------------- free space

pub type FreeTable<'t> = Table<'t, u64, (u64, u64)>;

/// Space for a region of `len` bytes: freed space no process can still be
/// reading, or else new space at the end of the file.
pub fn allocate(free: &mut FreeTable<'_>, c: &Container, len: u64) -> Result<u64> {
    let need = len.div_ceil(PAGE) * PAGE;
    let mut found = None;
    for row in free.iter()? {
        let (k, v) = row?;
        let (size, freed_at) = v.value();
        if size >= need && c.reusable(freed_at).map_err(io)? {
            found = Some((k.value(), size, freed_at));
            break;
        }
    }
    let Some((offset, size, freed_at)) = found else {
        return c.allocate_end(need).map_err(io);
    };
    free.remove(offset)?;
    if size > need {
        free.insert(offset + need, (size - need, freed_at))?;
    }
    Ok(offset)
}

/// Records a region's space as free from `epoch` on, joined with free
/// space on either side, so a later, larger region can reuse it. A joined
/// block keeps the later of the epochs: it is reusable when all its parts
/// are.
pub fn release(free: &mut FreeTable<'_>, (offset, len): (u64, u64), epoch: u64) -> Result<()> {
    let (mut start, mut size, mut epoch) = (offset, len.div_ceil(PAGE) * PAGE, epoch);
    let before = free
        .range(..offset)?
        .next_back()
        .transpose()?
        .map(|(k, v)| (k.value(), v.value()));
    if let Some((at, (n, e))) = before
        && at + n == start
    {
        free.remove(at)?;
        (start, size, epoch) = (at, size + n, epoch.max(e));
    }
    let after = free.get(start + size)?.map(|v| v.value());
    if let Some((n, e)) = after {
        free.remove(start + size)?;
        (size, epoch) = (size + n, epoch.max(e));
    }
    free.insert(start, (size, epoch))?;
    Ok(())
}

// ---------------------------------------------------------------- version 2

/// Reads a sealed segment stored the version 2 way (`segment_data` values),
/// with its full vectors from the `vectors` table where they still exist
/// and decoded from its codes otherwise. For upgrading a file.
pub fn read_legacy_segment(
    headers: &impl ReadableTable<(u32, u64), &'static [u8]>,
    data: &impl ReadableTable<(u32, u64, u8), &'static [u8]>,
    vectors: &impl ReadableTable<(u32, u32), &'static [u8]>,
    coll: u32,
    id: u64,
) -> Result<SealedSegment> {
    let header_bytes = headers
        .get((coll, id))?
        .ok_or_else(|| corrupt(format!("segment {id} has no header")))?;
    let h: SegmentHeader = decode(header_bytes.value(), "segment header")?;
    let part = |p: SegmentPart| -> Result<Vec<u8>> {
        Ok(data
            .get((coll, id, p as u8))?
            .ok_or_else(|| corrupt(format!("segment {id} is missing part {p:?}")))?
            .value()
            .to_vec())
    };
    fn pod<T: bytemuck::Pod>(bytes: Vec<u8>, what: &str) -> Result<Vec<T>> {
        if !bytes.len().is_multiple_of(size_of::<T>()) {
            return Err(corrupt(format!("{what}: not a whole number of elements")));
        }
        let mut out = vec![T::zeroed(); bytes.len() / size_of::<T>()];
        bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(&bytes);
        Ok(out)
    }
    let dim = h.dim as usize;
    let metric = metric_from_u8(h.metric)?;
    let ids: Vec<u32> = pod(part(SegmentPart::Ids)?, "ids")?;
    let codes: Vec<i8> = pod(part(SegmentPart::Codes)?, "codes")?;
    let corr: Vec<f32> = pod(part(SegmentPart::Corrections)?, "corrections")?;
    let quant_offset: Vec<f32> = pod(part(SegmentPart::QuantOffset)?, "quantizer")?;
    let mut full: Vec<f32> = pod(part(SegmentPart::Vectors)?, "vectors")?;
    if full.is_empty() && !ids.is_empty() {
        // An int8 segment: full vectors lived in the `vectors` table.
        if quant_offset.len() != dim || codes.len() != ids.len() * dim {
            return Err(corrupt(format!("segment {id}: arrays disagree on length")));
        }
        let quant = vdblite_index::quant::ScalarQuantizer::from_parts(
            metric,
            quant_offset.clone(),
            h.quant_scale,
        );
        full = vec![0f32; ids.len() * dim];
        for (local, (&chunk, row)) in ids.iter().zip(full.chunks_exact_mut(dim)).enumerate() {
            match vectors.get((coll, chunk))? {
                Some(v) => {
                    read_vector(v.value(), row)?;
                    vdblite_index::segment::prepare(metric, row);
                }
                // Deleted since: its code still steers the graph.
                None => quant.decode(&codes[local * dim..(local + 1) * dim], row),
            }
        }
    }
    let parts = SegmentParts {
        dim,
        metric,
        graph: GraphParts {
            m: h.m,
            entry: h.entry,
            max_level: h.max_level,
            levels: part(SegmentPart::Levels)?.into(),
            level0: pod::<u32>(part(SegmentPart::Level0)?, "level0")?.into(),
            upper_off: pod::<u32>(part(SegmentPart::UpperOffsets)?, "upper offsets")?.into(),
            upper: pod::<u32>(part(SegmentPart::Upper)?, "upper")?.into(),
        },
        quant_offset,
        quant_scale: h.quant_scale,
        full: FullVectors::new(full, !codes.is_empty()),
        codes: codes.into(),
        bits: Buf::empty(),
        binary_state: Vec::new(),
        corr: corr.into(),
        ids: ids.into(),
        label_offsets: Buf::empty(),
        label_bytes: Buf::empty(),
    };
    if parts.ids.len() != h.len as usize {
        return Err(corrupt(format!("segment {id}: header and ids disagree")));
    }
    SealedSegment::from_parts(parts).map_err(|e| corrupt(format!("segment {id}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::FREE_REGIONS;
    use vdblite_index::Metric;
    use vdblite_index::hnsw::HnswParams;
    use vdblite_index::rng::SplitMix64;

    fn segment(n: usize, dim: usize, seed: u64) -> SealedSegment {
        let mut r = SplitMix64::new(seed);
        let mut data: Vec<f32> = (0..n * dim).map(|_| r.next_f64_open() as f32).collect();
        for row in data.chunks_exact_mut(dim) {
            vdblite_index::segment::prepare(Metric::Cosine, row);
        }
        let ids: Vec<u32> = (10..10 + n as u32).collect();
        SealedSegment::build(&ids, &data, dim, Metric::Cosine, &HnswParams::default())
    }

    #[test]
    fn segments_round_trip_through_regions() {
        let dir = tempfile::tempdir().unwrap();
        let c = Container::open(&dir.path().join("s.vdb"), true).unwrap();
        let seg = segment(700, 12, 3);
        let at = c.allocate_end(region_len(&seg)).unwrap();
        let rec = write_segment(&c, &seg, at).unwrap();
        c.sync().unwrap();
        let back = map_segment(&c, &rec, 1).unwrap();
        assert!(back.is_shared());
        assert_eq!(back, seg);
        assert_eq!(back.vector(5), seg.vector(5));
        assert_eq!(back.label(5), None);

        let labelled = seg.with_labels(|g| Some(format!("key-{g}")));
        let at = c.allocate_end(region_len(&labelled)).unwrap();
        let rec = write_segment(&c, &labelled, at).unwrap();
        let back = map_segment(&c, &rec, 1).unwrap();
        assert_eq!(back, labelled);
        assert_eq!(back.label(5), Some(&b"key-15"[..]));

        let binary = labelled.requantize(vdblite_index::segment::Quantization::Binary);
        let at = c.allocate_end(region_len(&binary)).unwrap();
        let rec = write_segment(&c, &binary, at).unwrap();
        let back = map_segment(&c, &rec, 1).unwrap();
        assert_eq!(back, binary);
        assert!(back.is_shared());
    }

    #[test]
    fn damaged_regions_and_records_are_errors_not_panics() {
        let dir = tempfile::tempdir().unwrap();
        let c = Container::open(&dir.path().join("d.vdb"), true).unwrap();
        let seg = segment(300, 8, 4);
        let at = c.allocate_end(region_len(&seg)).unwrap();
        let rec = write_segment(&c, &seg, at).unwrap();
        let mut r = SplitMix64::new(9);
        // Flip a bit in any array but the full vectors: the checksum catches
        // it. (Padding between arrays is never read.)
        let checked: Vec<(u64, u64)> = rec.parts[..Part::Full as usize]
            .iter()
            .zip(ELEM_SIZE)
            .filter(|((_, n), _)| *n > 0)
            .map(|(&(o, n), size)| (o, n * size))
            .collect();
        for _ in 0..40 {
            let (o, len) = checked[(r.next_u64() as usize) % checked.len()];
            let pos = at + o + r.next_u64() % len;
            let mut b = [0u8];
            c.read_at(pos, &mut b).unwrap();
            c.write_at(pos, &[b[0] ^ (1 << (r.next_u64() % 8))])
                .unwrap();
            assert!(map_segment(&c, &rec, 1).is_err());
            c.write_at(pos, &b).unwrap();
        }
        assert!(map_segment(&c, &rec, 1).is_ok());
        // Records pointing nowhere sensible.
        for bad in [
            SegmentRecord {
                region: (at, u64::MAX),
                ..rec.clone()
            },
            SegmentRecord {
                parts: rec.parts[..3].to_vec(),
                ..rec.clone()
            },
            SegmentRecord {
                parts: rec.parts.iter().map(|&(o, n)| (o + 1, n)).collect(),
                ..rec.clone()
            },
            SegmentRecord {
                len: rec.len + 1,
                ..rec.clone()
            },
        ] {
            assert!(map_segment(&c, &bad, 1).is_err());
        }
    }

    #[test]
    fn freed_space_is_reused_once_unleased() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.vdb");
        let c = Container::open(&path, true).unwrap();
        let db = redb::Database::builder()
            .create_with_backend(c.redb_backend())
            .unwrap();
        let tx = db.begin_write().unwrap();
        {
            let mut free = tx.open_table(FREE_REGIONS).unwrap();
            let a = allocate(&mut free, &c, 3 * PAGE).unwrap();
            let lease = c.lease(1).unwrap();
            release(&mut free, (a, 3 * PAGE), 2).unwrap();
            // A lease on epoch 1 may still read it: new space instead.
            let b = allocate(&mut free, &c, PAGE).unwrap();
            assert_ne!(b, a);
            drop(lease);
            // Now it's reused, and the rest stays free.
            assert_eq!(allocate(&mut free, &c, PAGE).unwrap(), a);
            assert_eq!(free.get(a + PAGE).unwrap().unwrap().value(), (2 * PAGE, 2));
            // Neighbours join: freeing the page between two free blocks
            // leaves one block, at the later epoch.
            release(&mut free, (a, PAGE), 5).unwrap();
            assert_eq!(free.get(a).unwrap().unwrap().value(), (3 * PAGE, 5));
            release(&mut free, (b, PAGE), 1).unwrap();
            let blocks: Vec<_> = free.iter().unwrap().map(|r| r.unwrap().1.value()).collect();
            assert_eq!(blocks.len(), if b == a + 3 * PAGE { 1 } else { 2 });
        }
        tx.commit().unwrap();
    }
}
