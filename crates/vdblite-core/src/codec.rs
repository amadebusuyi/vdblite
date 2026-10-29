//! Byte codecs for vectors, bitmaps and compressed blobs. (Sealed segments
//! live in regions; see [`crate::segments`].)
//!
//! Everything read back is validated: a corrupted file yields
//! [`StorageError::Corrupt`](crate::StorageError::Corrupt), never a panic.

use roaring::RoaringBitmap;
use vdblite_index::Metric;

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

#[cfg(test)]
mod tests {
    use super::*;

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
