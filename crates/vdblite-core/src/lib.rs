//! Single-file storage for vdblite.
//!
//! Everything a database holds lives in one file managed by
//! [redb](https://github.com/cberner/redb): an ACID, copy-on-write B-tree
//! store with checksummed pages, a single writer and MVCC readers. This is
//! Option B of the design (borrow the pager); see `docs/design/phase-5.md`.
//!
//! This crate defines the table layout ([`tables`]), the persisted record
//! types ([`records`]), metadata values and their sortable index encoding
//! ([`meta`]), the codecs for vectors, bitmaps and sealed segments
//! ([`codec`]), and the change log reading processes follow ([`changes`]).
//! Transactions and the public API live in the `vdblite` crate.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "vdblite stores arrays in little-endian byte order and supports little-endian targets only"
);

pub mod changes;
pub mod codec;
pub mod container;
pub mod meta;
pub mod records;
pub mod segments;
pub mod tables;

use std::fmt;

/// Format version written into every new file's catalog.
///
/// Version 2 added the `changes` table. Version 3 keeps sealed segments, and
/// the full vectors of their chunks, in mapped regions instead of
/// `segment_data` and `vectors` values. Older files are upgraded in place
/// when a writer opens them.
pub const FORMAT_VERSION: u32 = 3;

/// Errors from the storage layer.
#[derive(Debug)]
pub enum StorageError {
    /// The underlying store failed (I/O, lock, transaction).
    Store(redb::Error),
    /// Bytes in the file did not decode into a valid record.
    Corrupt(String),
    /// Reading or writing the file itself failed.
    Io(std::io::Error),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::Store(e) => write!(f, "storage error: {e}"),
            StorageError::Corrupt(m) => write!(f, "corrupt data: {m}"),
            StorageError::Io(e) => write!(f, "file error: {e}"),
        }
    }
}

impl std::error::Error for StorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StorageError::Store(e) => Some(e),
            StorageError::Corrupt(_) => None,
            StorageError::Io(e) => Some(e),
        }
    }
}

impl<E: Into<redb::Error>> From<E> for StorageError {
    fn from(e: E) -> Self {
        StorageError::Store(e.into())
    }
}

pub type Result<T, E = StorageError> = std::result::Result<T, E>;

pub(crate) fn corrupt(msg: impl Into<String>) -> StorageError {
    StorageError::Corrupt(msg.into())
}
