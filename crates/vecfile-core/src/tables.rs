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
//! | `segments` | (coll, segment id) | [`SegmentHeader`](crate::records::SegmentHeader) |
//! | `segment_data` | (coll, segment id, part) | raw array bytes |
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
