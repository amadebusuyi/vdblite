//! A visited set that clears in O(1).
//!
//! Each slot holds the epoch in which it was last visited; starting a new
//! search bumps the epoch instead of zeroing the array. The array is only
//! wiped when the 16-bit epoch wraps, once every 65,535 searches.

#[derive(Clone, Debug, Default)]
pub struct Visited {
    marks: Vec<u16>,
    epoch: u16,
}

impl Visited {
    pub fn new(n: usize) -> Self {
        Visited {
            marks: vec![0; n],
            epoch: 0,
        }
    }

    /// Starts a new search over `n` nodes, growing the array if needed.
    pub fn reset(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    /// Marks `i` visited; returns true if it was not visited before.
    #[inline]
    pub fn insert(&mut self, i: u32) -> bool {
        let slot = &mut self.marks[i as usize];
        if *slot == self.epoch {
            false
        } else {
            *slot = self.epoch;
            true
        }
    }

    #[inline]
    pub fn contains(&self, i: u32) -> bool {
        self.marks[i as usize] == self.epoch
    }

    pub fn capacity(&self) -> usize {
        self.marks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_forgets_and_wraps() {
        let mut v = Visited::new(4);
        v.reset(4);
        assert!(v.insert(2));
        assert!(!v.insert(2));
        for _ in 0..70_000 {
            v.reset(4);
            assert!(!v.contains(2));
            assert!(v.insert(2));
        }
    }
}
