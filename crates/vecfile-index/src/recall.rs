//! Recall: the fraction of the true nearest neighbours a search returned.

use crate::topk::Neighbor;

/// Mean recall@k over all queries.
///
/// For each query, counts how many of `truth[..k]` appear in `found[..k]` and
/// divides by `k`. A result list shorter than `k` counts its missing entries
/// as misses.
///
/// # Panics
/// If `k` is zero, the two lists have different lengths, or a truth row has
/// fewer than `k` entries.
pub fn recall_at_k<F, T>(found: &[F], truth: &[T], k: usize) -> f64
where
    F: AsRef<[u32]>,
    T: AsRef<[u32]>,
{
    assert!(k > 0, "k must be positive");
    assert_eq!(found.len(), truth.len(), "one result list per query");
    if found.is_empty() {
        return 1.0;
    }
    let hits: usize = found
        .iter()
        .zip(truth)
        .map(|(f, t)| {
            let t = t.as_ref();
            assert!(t.len() >= k, "truth row has {} entries, need {k}", t.len());
            let t = &t[..k];
            let f = f.as_ref();
            f[..f.len().min(k)]
                .iter()
                .filter(|id| t.contains(id))
                .count()
        })
        .sum();
    hits as f64 / (found.len() * k) as f64
}

/// Mean recall@k where any result at least as close as the k-th true
/// neighbour counts as a hit, so ties at equal distance are not penalised.
///
/// `kth_dist[q]` is the distance from query `q` to its k-th true neighbour.
/// A result counts when its distance is within `kth_dist * (1 + 1e-3)`, the
/// same tolerance ann-benchmarks uses. Datasets with duplicate vectors
/// (SIFT1M has some) need this: ids at a tied distance are interchangeable.
///
/// # Panics
/// If `k` is zero or the two lists have different lengths.
pub fn recall_at_k_with_ties(found: &[Vec<Neighbor>], kth_dist: &[f32], k: usize) -> f64 {
    assert!(k > 0, "k must be positive");
    assert_eq!(found.len(), kth_dist.len(), "one result list per query");
    if found.is_empty() {
        return 1.0;
    }
    let hits: usize = found
        .iter()
        .zip(kth_dist)
        .map(|(f, &limit)| {
            let limit = limit + 1e-3 * limit.abs();
            f.iter().take(k).filter(|n| n.dist <= limit).count()
        })
        .sum();
    hits as f64 / (found.len() * k) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_count_as_hits() {
        // Truth's 2nd neighbour is at distance 4; we returned a different id at 4.
        let found = vec![vec![Neighbor::new(0, 1.0), Neighbor::new(9, 4.0)]];
        assert_eq!(recall_at_k_with_ties(&found, &[4.0], 2), 1.0);
        let worse = vec![vec![Neighbor::new(0, 1.0), Neighbor::new(9, 5.0)]];
        assert_eq!(recall_at_k_with_ties(&worse, &[4.0], 2), 0.5);
    }

    #[test]
    fn perfect_and_partial() {
        let truth = [vec![1, 2, 3], vec![4, 5, 6]];
        assert_eq!(recall_at_k(&truth, &truth, 3), 1.0);
        let found = [vec![3, 1, 9], vec![7, 8, 9]];
        assert_eq!(recall_at_k(&found, &truth, 3), 2.0 / 6.0);
    }

    #[test]
    fn order_within_k_does_not_matter_and_short_lists_miss() {
        let truth = [vec![1, 2, 3, 4]];
        assert_eq!(recall_at_k(&[vec![2, 1]], &truth, 2), 1.0);
        assert_eq!(recall_at_k(&[vec![1]], &truth, 2), 0.5);
    }
}
