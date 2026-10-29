//! Arrays that are either owned or borrowed from shared, immutable memory.
//!
//! A segment built in this process owns its arrays. One loaded from a file
//! borrows them from a memory-mapped region of that file, so every process
//! that opens the file shares one copy and nothing is copied at open. Both
//! read the same way: a [`Buf<T>`] derefs to `&[T]`.

use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use bytemuck::Pod;

/// Immutable bytes that stay valid, at a fixed address, for as long as the
/// value lives: a memory-mapped file region, or a heap buffer.
pub trait Backing: Send + Sync {
    fn bytes(&self) -> &[u8];
}

impl Backing for Vec<u8> {
    fn bytes(&self) -> &[u8] {
        self
    }
}

/// An array of `T`, owned or borrowed from a [`Backing`].
pub struct Buf<T: Pod> {
    inner: Inner<T>,
}

enum Inner<T: Pod> {
    Owned(Vec<T>),
    Shared {
        /// Keeps the memory behind `ptr` alive and unchanged.
        _owner: Arc<dyn Backing>,
        ptr: *const T,
        len: usize,
        _t: PhantomData<T>,
    },
}

// SAFETY: an owned `Buf` is a `Vec<T>` of plain data. A shared one points
// into memory its `Arc<dyn Backing>` keeps alive and immutable, and
// `Backing` is `Send + Sync`; the pointer is only ever read through.
unsafe impl<T: Pod> Send for Buf<T> {}
// SAFETY: as for `Send`: nothing mutates the memory behind a shared `Buf`.
unsafe impl<T: Pod> Sync for Buf<T> {}

impl<T: Pod> Buf<T> {
    pub fn owned(v: Vec<T>) -> Self {
        Buf {
            inner: Inner::Owned(v),
        }
    }

    /// Borrows `count` elements starting at byte `offset` of `owner`.
    /// Fails if the range is out of bounds or misaligned for `T`.
    pub fn shared(owner: Arc<dyn Backing>, offset: usize, count: usize) -> Result<Self, String> {
        let bytes = owner.bytes();
        let size = count
            .checked_mul(size_of::<T>())
            .ok_or_else(|| "array length overflows".to_string())?;
        let end = offset
            .checked_add(size)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| {
                format!(
                    "array of {size} bytes at {offset} overruns a {}-byte region",
                    bytes.len()
                )
            })?;
        let slice: &[T] = bytemuck::try_cast_slice(&bytes[offset..end])
            .map_err(|e| format!("array at {offset} is not aligned: {e}"))?;
        let ptr = slice.as_ptr();
        Ok(Buf {
            inner: Inner::Shared {
                _owner: owner,
                ptr,
                len: count,
                _t: PhantomData,
            },
        })
    }

    pub fn empty() -> Self {
        Self::owned(Vec::new())
    }

    /// Whether the elements live in shared memory rather than this
    /// process's heap.
    pub fn is_shared(&self) -> bool {
        matches!(self.inner, Inner::Shared { .. })
    }

    /// The elements as an owned vector (a copy, if shared).
    pub fn into_vec(self) -> Vec<T> {
        match self.inner {
            Inner::Owned(v) => v,
            Inner::Shared { .. } => self.to_vec(),
        }
    }
}

impl<T: Pod> Deref for Buf<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        match &self.inner {
            Inner::Owned(v) => v,
            // SAFETY: `ptr` and `len` were checked against the backing's
            // bytes (bounds and alignment) when this `Buf` was made, and the
            // `Arc` held beside them keeps those bytes alive and unchanged.
            Inner::Shared { ptr, len, .. } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }
}

impl<'a, T: Pod> IntoIterator for &'a Buf<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Pod> From<Vec<T>> for Buf<T> {
    fn from(v: Vec<T>) -> Self {
        Self::owned(v)
    }
}

impl<T: Pod> Default for Buf<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T: Pod> Clone for Buf<T> {
    fn clone(&self) -> Self {
        match &self.inner {
            Inner::Owned(v) => Self::owned(v.clone()),
            Inner::Shared {
                _owner, ptr, len, ..
            } => Buf {
                inner: Inner::Shared {
                    _owner: _owner.clone(),
                    ptr: *ptr,
                    len: *len,
                    _t: PhantomData,
                },
            },
        }
    }
}

impl<T: Pod + PartialEq> PartialEq for Buf<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Pod + fmt::Debug> fmt::Debug for Buf<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_shared() { "shared" } else { "owned" };
        write!(f, "Buf<{kind}, {} elements>", self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes backed by `u32`s, so they are 4-aligned by construction.
    struct Words(Vec<u32>);

    impl Backing for Words {
        fn bytes(&self) -> &[u8] {
            bytemuck::cast_slice(&self.0)
        }
    }

    #[test]
    fn shared_buffers_read_like_owned_ones() {
        let owner: Arc<dyn Backing> = Arc::new(Words((0..16).collect()));
        let b = Buf::<u32>::shared(owner.clone(), 8, 4).unwrap();
        assert_eq!(&*b, &[2, 3, 4, 5]);
        assert!(b.is_shared());
        assert_eq!(b.clone(), Buf::owned(vec![2, 3, 4, 5]));
        assert_eq!(b.into_vec(), vec![2, 3, 4, 5]);
        // Out of bounds, misaligned, or overflowing: errors, not panics.
        assert!(Buf::<u32>::shared(owner.clone(), 60, 2).is_err());
        assert!(Buf::<u32>::shared(owner.clone(), 2, 1).is_err());
        assert!(Buf::<u32>::shared(owner, 0, usize::MAX / 2).is_err());
    }
}
