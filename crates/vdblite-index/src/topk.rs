//! Bounded top-k selection.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A search result: a vector's position and its distance to the query.
///
/// Ordered by distance, then by id, so results are deterministic when
/// distances tie. Distances compare with `f32::total_cmp`.
#[derive(Clone, Copy, Debug)]
pub struct Neighbor {
    pub id: u32,
    pub dist: f32,
}

impl Neighbor {
    pub fn new(id: u32, dist: f32) -> Self {
        Neighbor { id, dist }
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Neighbor {}

/// Keeps the `k` smallest neighbours pushed into it.
///
/// Internally a max-heap whose root is the current worst kept result, so each
/// push is O(1) when rejected and O(log k) when accepted.
#[derive(Clone, Debug)]
pub struct TopK {
    k: usize,
    heap: BinaryHeap<Neighbor>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            heap: BinaryHeap::with_capacity(k),
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Offers a candidate; it is kept if it beats the current worst result.
    #[inline]
    pub fn push(&mut self, id: u32, dist: f32) {
        let n = Neighbor::new(id, dist);
        if self.heap.len() < self.k {
            self.heap.push(n);
        } else if let Some(mut worst) = self.heap.peek_mut()
            && n < *worst
        {
            *worst = n;
        }
    }

    /// The distance a candidate must beat to be kept, or `+inf` while not yet full.
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::INFINITY
        } else {
            self.heap.peek().map_or(f32::NEG_INFINITY, |n| n.dist)
        }
    }

    /// Empties the collector so it can be reused without reallocating.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// The kept neighbours, closest first.
    pub fn into_sorted_vec(self) -> Vec<Neighbor> {
        self.heap.into_sorted_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_smallest_in_order() {
        let mut t = TopK::new(3);
        for (id, d) in [(0, 5.0), (1, 1.0), (2, 4.0), (3, 0.5), (4, 9.0), (5, 2.0)] {
            t.push(id, d);
        }
        let ids: Vec<u32> = t.into_sorted_vec().iter().map(|n| n.id).collect();
        assert_eq!(ids, [3, 1, 5]);
    }

    #[test]
    fn ties_break_on_id() {
        let mut t = TopK::new(2);
        for id in [7, 3, 5, 1] {
            t.push(id, 1.0);
        }
        let ids: Vec<u32> = t.into_sorted_vec().iter().map(|n| n.id).collect();
        assert_eq!(ids, [1, 3]);
    }

    #[test]
    fn zero_k_keeps_nothing() {
        let mut t = TopK::new(0);
        t.push(0, 1.0);
        assert!(t.is_empty());
    }

    #[test]
    fn threshold_tracks_worst() {
        let mut t = TopK::new(2);
        assert_eq!(t.threshold(), f32::INFINITY);
        t.push(0, 3.0);
        t.push(1, 1.0);
        assert_eq!(t.threshold(), 3.0);
        t.push(2, 2.0);
        assert_eq!(t.threshold(), 2.0);
    }
}
