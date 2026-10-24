//! Single-file storage for vecfile.
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
//! Transactions and the public API live in the `vecfile` crate.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "vecfile stores arrays in little-endian byte order and supports little-endian targets only"
);

pub mod changes;
pub mod codec;
pub mod meta;
pub mod records;
pub mod tables;

use std::fmt;

/// Format version written into every new file's catalog.
///
/// Version 2 added the `changes` table. A version 1 file is upgraded in
/// place when a writer opens it.
pub const FORMAT_VERSION: u32 = 2;

/// Errors from the storage layer.
#[derive(Debug)]
pub enum StorageError {
    /// The underlying store failed (I/O, lock, transaction).
    Store(redb::Error),
    /// Bytes in the file did not decode into a valid record.
    Corrupt(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::Store(e) => write!(f, "storage error: {e}"),
            StorageError::Corrupt(m) => write!(f, "corrupt data: {m}"),
        }
    }
}

impl std::error::Error for StorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StorageError::Store(e) => Some(e),
            StorageError::Corrupt(_) => None,
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
