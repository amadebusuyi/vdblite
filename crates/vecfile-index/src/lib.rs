//! Vector search for vecfile: SIMD distance kernels, exact (brute-force)
//! search, HNSW graphs, int8 quantization, segments, and the evaluation
//! tools used to measure recall.

pub mod brute_force;
pub mod dataset;
pub mod distance;
pub mod hnsw;
pub mod index;
pub mod quant;
pub mod recall;
pub mod rng;
pub mod segment;
pub mod topk;
pub mod visited;

pub use distance::Metric;
pub use index::{IndexConfig, VectorIndex};
pub use topk::{Neighbor, TopK};
