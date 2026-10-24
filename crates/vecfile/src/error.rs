use std::fmt;

use vecfile_core::StorageError;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Everything that can go wrong in vecfile.
#[derive(Debug)]
pub enum Error {
    /// The file or a transaction failed, or stored bytes were invalid.
    Storage(StorageError),
    /// A vector had the wrong number of dimensions.
    Dimension { expected: usize, got: usize },
    /// A collection with this name already exists.
    CollectionExists(String),
    /// No collection with this name exists.
    NoSuchCollection(String),
    /// A value passed in was invalid (field name, vector count, non-finite value…).
    InvalidArgument(String),
    /// The embedder failed.
    Embedder(BoxError),
    /// Vectors planned for a document no longer cover the chunks it needs,
    /// because it changed after planning. Plan again and retry.
    StaleEmbeddings { key: String },
    /// The file was written by an incompatible version of vecfile.
    Version { found: u32, supported: u32 },
    /// Sealing or compacting failed: reported by [`crate::Db::optimize`],
    /// by [`crate::Db::wait_for_maintenance`] for background work, and by a
    /// commit whose write buffer was over its limit. Nothing committed was
    /// lost; background maintenance retries after the next commit.
    Maintenance(Box<Error>),
    /// The database was opened read-only.
    ReadOnly,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Storage(e) => write!(f, "{e}"),
            Error::Dimension { expected, got } => {
                write!(f, "expected a {expected}-dimensional vector, got {got}")
            }
            Error::CollectionExists(n) => write!(f, "collection {n:?} already exists"),
            Error::NoSuchCollection(n) => write!(f, "no collection named {n:?}"),
            Error::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Error::Embedder(e) => write!(f, "embedder failed: {e}"),
            Error::StaleEmbeddings { key } => {
                write!(f, "embeddings planned for {key:?} are stale; plan again")
            }
            Error::Version { found, supported } => {
                write!(
                    f,
                    "file format version {found} is not supported (this build reads {supported})"
                )
            }
            Error::Maintenance(e) => write!(f, "index maintenance failed: {e}"),
            Error::ReadOnly => write!(f, "the database is open read-only"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Storage(e) => Some(e),
            Error::Embedder(e) => Some(e.as_ref()),
            Error::Maintenance(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<StorageError> for Error {
    fn from(e: StorageError) -> Self {
        Error::Storage(e)
    }
}

macro_rules! from_redb {
    ($($t:ty),*) => {$(
        impl From<$t> for Error {
            fn from(e: $t) -> Self { Error::Storage(StorageError::from(e)) }
        }
    )*};
}
from_redb!(
    redb::Error,
    redb::DatabaseError,
    redb::TransactionError,
    redb::TableError,
    redb::StorageError,
    redb::CommitError
);

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) fn invalid(msg: impl Into<String>) -> Error {
    Error::InvalidArgument(msg.into())
}

/// Runs a read against the file, turning a panic inside the storage engine
/// (which redb can raise on some corrupted pages) into an error.
///
/// This only works in builds that unwind; with `panic = "abort"` a panic
/// still ends the process.
pub(crate) fn guard<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| {
        Err(Error::Storage(StorageError::Corrupt(
            "the storage engine panicked reading the file; it is likely corrupted".into(),
        )))
    })
}
