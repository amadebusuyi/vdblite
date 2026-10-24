//! Exact nearest-neighbour search by scanning every vector.
//!
//! This is the ground truth every approximate index is measured against, and
//! the search used for small or heavily filtered segments.
//!
//! Vectors are stored row-major in one flat slice: vector `i` is
//! `data[i * dim..(i + 1) * dim]` and its id is `i`.

use rayon::prelude::*;

use crate::distance::{Kernels, Metric, kernels};
use crate::topk::{Neighbor, TopK};

/// Queries handled together by one thread, so each block of base vectors
/// loaded into cache is reused this many times.
const QUERY_TILE: usize = 32;

/// Target size of one block of base vectors, sized to sit in L2 cache.
const BASE_TILE_BYTES: usize = 128 * 1024;

/// The `k` nearest vectors in `data` to `query`, closest first.
///
/// # Panics
/// If `dim` is zero, `data.len()` is not a multiple of `dim`, `query.len()`
/// differs from `dim`, or there are more than `u32::MAX` vectors.
pub fn search(data: &[f32], dim: usize, metric: Metric, query: &[f32], k: usize) -> Vec<Neighbor> {
    check_shape(data, dim);
    assert_eq!(query.len(), dim, "query length must equal dim");
    let kern = kernels();
    let mut top = TopK::new(k);
    for (id, v) in data.chunks_exact(dim).enumerate() {
        top.push(id as u32, metric.distance(kern, query, v));
    }
    top.into_sorted_vec()
}

/// [`search`] for many queries at once (row-major, like `data`), in parallel.
///
/// Queries are processed in tiles against cache-sized blocks of `data`, so
/// memory traffic drops by roughly the tile size compared with running each
/// query on its own.
pub fn search_batch(
    data: &[f32],
    dim: usize,
    metric: Metric,
    queries: &[f32],
    k: usize,
) -> Vec<Vec<Neighbor>> {
    check_shape(data, dim);
    assert_eq!(
        queries.len() % dim,
        0,
        "queries length must be a multiple of dim"
    );
    let kern = kernels();
    let base_tile = (BASE_TILE_BYTES / (dim * size_of::<f32>())).max(1);
    // Shrink tiles for small batches so every thread gets work.
    let per_thread = (queries.len() / dim).div_ceil(rayon::current_num_threads());
    let query_tile = per_thread.clamp(1, QUERY_TILE);

    queries
        .par_chunks(query_tile * dim)
        .flat_map_iter(|tile| search_tile(kern, data, dim, metric, tile, k, base_tile))
        .collect()
}

fn search_tile(
    kern: &Kernels,
    data: &[f32],
    dim: usize,
    metric: Metric,
    tile: &[f32],
    k: usize,
    base_tile: usize,
) -> Vec<Vec<Neighbor>> {
    let mut tops: Vec<TopK> = (0..tile.len() / dim).map(|_| TopK::new(k)).collect();
    for (block_idx, block) in data.chunks(base_tile * dim).enumerate() {
        let first_id = block_idx * base_tile;
        for (q, top) in tile.chunks_exact(dim).zip(&mut tops) {
            for (offset, v) in block.chunks_exact(dim).enumerate() {
                top.push((first_id + offset) as u32, metric.distance(kern, q, v));
            }
        }
    }
    tops.into_iter().map(TopK::into_sorted_vec).collect()
}

fn check_shape(data: &[f32], dim: usize) {
    assert!(dim > 0, "dim must be positive");
    assert_eq!(data.len() % dim, 0, "data length must be a multiple of dim");
    assert!(
        data.len() / dim <= u32::MAX as usize,
        "too many vectors for u32 ids"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Sort every vector by distance; the slowest possible correct answer.
    fn naive(data: &[f32], dim: usize, metric: Metric, q: &[f32], k: usize) -> Vec<Neighbor> {
        let kern = Kernels::SCALAR;
        let mut all: Vec<Neighbor> = data
            .chunks_exact(dim)
            .enumerate()
            .map(|(i, v)| Neighbor::new(i as u32, metric.distance(&kern, q, v)))
            .collect();
        all.sort();
        all.truncate(k);
        all
    }

    fn ids(ns: &[Neighbor]) -> Vec<u32> {
        ns.iter().map(|n| n.id).collect()
    }

    #[test]
    fn finds_exact_match_first() {
        let data = [0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 1.1, 0.9];
        let hits = search(&data, 2, Metric::L2, &[1.0, 1.0], 2);
        assert_eq!(ids(&hits), [1, 3]);
        assert_eq!(hits[0].dist, 0.0);
    }

    #[test]
    fn k_larger_than_data_returns_everything() {
        let data = [1.0, 2.0, 3.0];
        assert_eq!(search(&data, 1, Metric::Dot, &[1.0], 10).len(), 3);
    }

    #[test]
    #[should_panic(expected = "query length")]
    fn rejects_wrong_query_length() {
        search(&[1.0, 2.0], 2, Metric::L2, &[1.0], 1);
    }

    fn arb_case() -> impl Strategy<Value = (usize, Vec<f32>, Vec<f32>, usize)> {
        (1usize..40, 0usize..300, 1usize..70, 0usize..20).prop_flat_map(|(dim, n, nq, k)| {
            (
                Just(dim),
                prop::collection::vec(-10.0f32..10.0, n * dim),
                prop::collection::vec(-10.0f32..10.0, nq * dim),
                Just(k),
            )
        })
    }

    proptest! {
        #[test]
        fn search_matches_naive((dim, data, queries, k) in arb_case()) {
            for metric in [Metric::L2, Metric::Dot] {
                for q in queries.chunks_exact(dim) {
                    let got = search(&data, dim, metric, q, k);
                    let want = naive(&data, dim, metric, q, k);
                    // SIMD and scalar sums may differ in the last bits, which can
                    // swap near-ties; compare distances with a tolerance instead of ids.
                    prop_assert_eq!(got.len(), want.len());
                    for (g, w) in got.iter().zip(&want) {
                        prop_assert!((g.dist - w.dist).abs() <= 1e-3 * w.dist.abs().max(1.0),
                            "got {:?} want {:?}", g, w);
                    }
                }
            }
        }

        #[test]
        fn batch_matches_single((dim, data, queries, k) in arb_case()) {
            let batch = search_batch(&data, dim, Metric::L2, &queries, k);
            prop_assert_eq!(batch.len(), queries.len() / dim);
            for (q, got) in queries.chunks_exact(dim).zip(&batch) {
                prop_assert_eq!(got, &search(&data, dim, Metric::L2, q, k));
            }
        }
    }
}
