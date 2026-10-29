//! The table layout of a `.vdb` file.
//!
//! Every per-collection table is keyed by the collection's numeric id first,
//! so a collection's rows are contiguous and one set of tables serves all
//! collections.
//!
//! | Table | Key | Value |
//! | --- | --- | --- |
//! | `catalog` | collection name | [`CollectionRecord`](crate::records::CollectionRecord) |
//! | `docs` | (coll, doc id) | [`DocRecord`](crate::records::DocRecord) |
//! | `keys` | (coll, user key) | doc id |
//! | `blobs` | (coll, doc id) | zstd-compressed content |
//! | `chunks` | (coll, chunk id) | [`ChunkRecord`](crate::records::ChunkRecord) |
//! | `vectors` | (coll, chunk id) | f32 vector, little-endian |
//! | `pending` | (coll, chunk id) | () — chunk is in the mutable segment |
//! | `dead` | (coll, chunk id) | sealed segment id holding its tombstone |
//! | `meta_index` | (coll, field ∅ value) | Roaring bitmap of chunk ids |
//! | `segments` | (coll, segment id) | [`SegmentRecord`](crate::records::SegmentRecord) |
//! | `segment_data` | (coll, segment id, part) | raw array bytes (version 2 files only) |
//! | `free_regions` | physical offset | (length, epoch freed at) |
//! | `reserved_regions` | physical offset | length of a region being written |
//! | `bulk` | (coll, seq) | [`BulkRecord`](crate::records::BulkRecord) of an unfinished bulk load |
//! | `postings` | (coll, term, chunk id) | term count and chunk length |
//! | `term_df` | (coll, term) | chunks containing the term |
//! | `chunk_terms` | (coll, chunk id) | the chunk's distinct terms |
//! | `lexical` | coll | [`LexicalStats`](crate::records::LexicalStats) |
//! | `changes` | (coll, seq) | [`Change`](crate::records::Change) — see [`crate::changes`] |

use redb::TableDefinition;

pub const CATALOG: TableDefinition<&str, &[u8]> = TableDefinition::new("catalog");
pub const DOCS: TableDefinition<(u32, u64), &[u8]> = TableDefinition::new("docs");
pub const KEYS: TableDefinition<(u32, &str), u64> = TableDefinition::new("keys");
pub const BLOBS: TableDefinition<(u32, u64), &[u8]> = TableDefinition::new("blobs");
pub const CHUNKS: TableDefinition<(u32, u32), &[u8]> = TableDefinition::new("chunks");
pub const VECTORS: TableDefinition<(u32, u32), &[u8]> = TableDefinition::new("vectors");
pub const PENDING: TableDefinition<(u32, u32), ()> = TableDefinition::new("pending");
pub const DEAD: TableDefinition<(u32, u32), u64> = TableDefinition::new("dead");
pub const META_INDEX: TableDefinition<(u32, &[u8]), &[u8]> = TableDefinition::new("meta_index");
pub const SEGMENTS: TableDefinition<(u32, u64), &[u8]> = TableDefinition::new("segments");
pub const SEGMENT_DATA: TableDefinition<(u32, u64, u8), &[u8]> =
    TableDefinition::new("segment_data");
/// Region space no record references any more, by physical offset, with
/// its length and the epoch it was freed at (see [`crate::container`]).
pub const FREE_REGIONS: TableDefinition<u64, (u64, u64)> = TableDefinition::new("free_regions");
/// Checkpoints of an unfinished bulk load: (coll, sequence) to a
/// [`BulkRecord`](crate::records::BulkRecord), a region of chunk ids and
/// their vectors. Finishing the load builds them into a segment and
/// deletes the rows; rows a crash left are finished at the next open.
pub const BULK: TableDefinition<(u32, u64), &[u8]> = TableDefinition::new("bulk");
/// Region space handed to a region being written outside the write lock,
/// by physical offset, with its length. Its record's commit removes the
/// row; rows left by a crash go back to `free_regions` at the next open.
pub const RESERVED_REGIONS: TableDefinition<u64, u64> = TableDefinition::new("reserved_regions");
/// Keyword index postings: one row per (term, chunk); the value packs the
/// term's count in the chunk (high 32 bits) and the chunk's length in terms.
pub const POSTINGS: TableDefinition<(u32, &str, u32), u64> = TableDefinition::new("postings");
/// Number of chunks containing each term (BM25's document frequency).
pub const TERM_DF: TableDefinition<(u32, &str), u32> = TableDefinition::new("term_df");
/// Each chunk's distinct terms, so deleting a chunk removes its postings.
pub const CHUNK_TERMS: TableDefinition<(u32, u32), &[u8]> = TableDefinition::new("chunk_terms");
/// Keyword-index statistics per collection; a row exists iff the collection
/// has a keyword index.
pub const LEXICAL: TableDefinition<u32, &[u8]> = TableDefinition::new("lexical");
/// Each collection's change log, so processes reading the file can follow
/// the writer's index (see [`crate::changes`]).
pub const CHANGES: TableDefinition<(u32, u64), &[u8]> = TableDefinition::new("changes");
/// Database-wide settings, such as the format version.
pub const SETTINGS: TableDefinition<&str, &[u8]> = TableDefinition::new("settings");

/// Parts of a sealed segment, stored as separate values so each is a plain
/// array that loads with one copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SegmentPart {
    Levels = 0,
    Level0 = 1,
    UpperOffsets = 2,
    Upper = 3,
    QuantOffset = 4,
    Codes = 5,
    Corrections = 6,
    Ids = 7,
    Vectors = 8,
}

impl SegmentPart {
    pub const ALL: [SegmentPart; 9] = [
        SegmentPart::Levels,
        SegmentPart::Level0,
        SegmentPart::UpperOffsets,
        SegmentPart::Upper,
        SegmentPart::QuantOffset,
        SegmentPart::Codes,
        SegmentPart::Corrections,
        SegmentPart::Ids,
        SegmentPart::Vectors,
    ];
}
