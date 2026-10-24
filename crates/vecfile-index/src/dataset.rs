//! Loaders for the TEXMEX benchmark formats (`.fvecs`, `.ivecs`) and SIFT1M,
//! and a generator of synthetic embedding-shaped vectors.
//!
//! Each record is a little-endian `i32` dimension followed by that many
//! 4-byte little-endian values. Every record in a file has the same dimension.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use crate::distance::normalize;
use crate::rng::SplitMix64;

/// A row-major matrix of vectors that all share one dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct Vectors<T> {
    pub dim: usize,
    pub data: Vec<T>,
}

impl<T> Vectors<T> {
    /// Number of vectors.
    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector `i`.
    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    pub fn rows(&self) -> std::slice::ChunksExact<'_, T> {
        self.data.chunks_exact(self.dim.max(1))
    }
}

/// Reads an `.fvecs` file, keeping at most `limit` vectors when given.
pub fn read_fvecs(path: impl AsRef<Path>, limit: Option<usize>) -> io::Result<Vectors<f32>> {
    read_vecs(path.as_ref(), limit, f32::from_le_bytes)
}

/// Reads an `.ivecs` file, keeping at most `limit` vectors when given.
pub fn read_ivecs(path: impl AsRef<Path>, limit: Option<usize>) -> io::Result<Vectors<i32>> {
    read_vecs(path.as_ref(), limit, i32::from_le_bytes)
}

fn read_vecs<T>(
    path: &Path,
    limit: Option<usize>,
    decode: fn([u8; 4]) -> T,
) -> io::Result<Vectors<T>> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut r = BufReader::with_capacity(1 << 20, file);

    let mut word = [0u8; 4];
    match r.read_exact(&mut word) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof && file_len == 0 => {
            return Ok(Vectors {
                dim: 0,
                data: Vec::new(),
            });
        }
        Err(e) => return Err(e),
    }
    let dim = i32::from_le_bytes(word);
    if dim <= 0 {
        return Err(invalid(path, format!("dimension {dim} is not positive")));
    }
    let dim = dim as usize;
    let record = 4 * (dim as u64 + 1);
    if file_len % record != 0 {
        return Err(invalid(
            path,
            format!("{file_len} bytes is not a whole number of {record}-byte records"),
        ));
    }
    let total = (file_len / record) as usize;
    let n = limit.map_or(total, |l| l.min(total));

    let mut data = Vec::with_capacity(n * dim);
    let mut buf = vec![0u8; dim * 4];
    for i in 0..n {
        if i > 0 {
            r.read_exact(&mut word)?;
            let d = i32::from_le_bytes(word);
            if d as i64 != dim as i64 {
                return Err(invalid(
                    path,
                    format!("record {i} has dimension {d}, expected {dim}"),
                ));
            }
        }
        r.read_exact(&mut buf)?;
        data.extend(buf.as_chunks::<4>().0.iter().map(|&c| decode(c)));
    }
    Ok(Vectors { dim, data })
}

fn invalid(path: &Path, msg: String) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}: {msg}", path.display()),
    )
}

/// The SIFT1M benchmark: 1M base vectors and 10K queries of 128 dimensions,
/// with the exact 100 nearest neighbours (L2) of each query.
///
/// Fetch it with `scripts/fetch-sift1m.sh`.
#[derive(Clone, Debug)]
pub struct Sift1M {
    pub base: Vectors<f32>,
    pub queries: Vectors<f32>,
    /// Row `i` lists the ids of query `i`'s nearest base vectors, closest first.
    pub groundtruth: Vec<Vec<u32>>,
}

impl Sift1M {
    /// Loads the dataset from `dir` (the extracted `sift/` folder), keeping the
    /// first `query_limit` queries when given.
    pub fn load(dir: impl AsRef<Path>, query_limit: Option<usize>) -> io::Result<Self> {
        let dir = dir.as_ref();
        let base = read_fvecs(dir.join("sift_base.fvecs"), None)?;
        let queries = read_fvecs(dir.join("sift_query.fvecs"), query_limit)?;
        let gt = read_ivecs(dir.join("sift_groundtruth.ivecs"), query_limit)?;
        if base.dim != queries.dim {
            return Err(invalid(
                dir,
                format!("base dim {} != query dim {}", base.dim, queries.dim),
            ));
        }
        if gt.len() != queries.len() {
            return Err(invalid(
                dir,
                format!(
                    "{} ground-truth rows for {} queries",
                    gt.len(),
                    queries.len()
                ),
            ));
        }
        let groundtruth = gt
            .rows()
            .map(|row| row.iter().map(|&id| id as u32).collect())
            .collect();
        Ok(Sift1M {
            base,
            queries,
            groundtruth,
        })
    }
}

/// Synthetic vectors shaped like text embeddings: clustered points in a
/// 48-dim latent space, projected up to `dim`, plus small isotropic noise,
/// then normalized.
///
/// Pure Gaussian noise would be far harder than real embeddings (its
/// intrinsic dimension is `dim`); this keeps the intrinsic dimension low, as
/// real models do. Memory and latency carry over to real data; recall should
/// be measured on a real embedding set too.
pub struct SyntheticEmbeddings {
    dim: usize,
    projection: Vec<f32>, // dim x LATENT
    centers: Vec<f32>,    // CLUSTERS x LATENT
}

impl SyntheticEmbeddings {
    const LATENT: usize = 48;
    const CLUSTERS: usize = 2000;

    pub fn new(dim: usize, seed: u64) -> Self {
        let mut r = SplitMix64::new(seed);
        let scale = 1.0 / (Self::LATENT as f32).sqrt();
        SyntheticEmbeddings {
            dim,
            projection: (0..dim * Self::LATENT)
                .map(|_| gaussian(&mut r) * scale)
                .collect(),
            centers: (0..Self::CLUSTERS * Self::LATENT)
                .map(|_| gaussian(&mut r))
                .collect(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Writes vector number `i` (the same for the same seed) into `out`.
    pub fn vector(&self, i: u64, out: &mut [f32]) {
        const LATENT: usize = SyntheticEmbeddings::LATENT;
        assert_eq!(out.len(), self.dim, "out must have dim elements");
        let mut r = SplitMix64::new(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xABCD);
        let c = (r.next_u64() % Self::CLUSTERS as u64) as usize;
        let z: Vec<f32> = (0..LATENT)
            .map(|j| self.centers[c * LATENT + j] + 0.6 * gaussian(&mut r))
            .collect();
        for (d, o) in out.iter_mut().enumerate() {
            let row = &self.projection[d * LATENT..(d + 1) * LATENT];
            *o = row.iter().zip(&z).map(|(a, b)| a * b).sum::<f32>() + 0.08 * gaussian(&mut r);
        }
        normalize(out);
    }
}

fn gaussian(r: &mut SplitMix64) -> f32 {
    let (u1, u2) = (r.next_f64_open(), r.next_f64_open());
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_records(path: &Path, records: &[(i32, Vec<[u8; 4]>)]) {
        let mut f = File::create(path).unwrap();
        for (dim, vals) in records {
            f.write_all(&dim.to_le_bytes()).unwrap();
            for v in vals {
                f.write_all(v).unwrap();
            }
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("vecfile-dataset-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn round_trips_fvecs() {
        let p = tmp("a.fvecs");
        let rows = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let recs: Vec<_> = rows
            .iter()
            .map(|r| (3, r.iter().map(|x| x.to_le_bytes()).collect()))
            .collect();
        write_records(&p, &recs);

        let v = read_fvecs(&p, None).unwrap();
        assert_eq!(v.dim, 3);
        assert_eq!(v.len(), 2);
        assert_eq!(v.row(1), &[4.0, 5.0, 6.0]);
        assert_eq!(read_fvecs(&p, Some(1)).unwrap().len(), 1);
    }

    #[test]
    fn rejects_truncated_file() {
        let p = tmp("trunc.ivecs");
        let mut bytes = Vec::new();
        bytes.extend(2i32.to_le_bytes());
        bytes.extend(7i32.to_le_bytes());
        std::fs::write(&p, bytes).unwrap();
        let err = read_ivecs(&p, None).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_mixed_dimensions() {
        let p = tmp("mixed.ivecs");
        // Two records of equal byte size but different declared dims cannot occur,
        // so make the second header lie about its dim.
        write_records(
            &p,
            &[
                (2, vec![1i32.to_le_bytes(), 2i32.to_le_bytes()]),
                (3, vec![1i32.to_le_bytes(), 2i32.to_le_bytes()]),
            ],
        );
        let err = read_ivecs(&p, None).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn empty_file_is_empty() {
        let p = tmp("empty.fvecs");
        std::fs::write(&p, []).unwrap();
        assert!(read_fvecs(&p, None).unwrap().is_empty());
    }
}
