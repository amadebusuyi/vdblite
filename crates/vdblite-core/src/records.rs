//! Records persisted in the file, encoded with postcard.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::meta::MetaValue;
use crate::{Result, corrupt};

/// A collection's schema and counters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollectionRecord {
    pub id: u32,
    pub name: String,
    pub dim: u32,
    /// 0 cosine, 1 dot, 2 L2.
    pub metric: u8,
    /// 0 int8, 1 none (f32).
    pub quantization: u8,
    pub chunking: ChunkingRecord,
    pub hnsw: HnswRecord,
    pub seal_threshold: u32,
    pub next_doc: u64,
    pub next_chunk: u32,
    pub next_segment: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkingRecord {
    /// The whole document is one chunk.
    None,
    Structural {
        target_tokens: u32,
        max_tokens: u32,
        overlap_tokens: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswRecord {
    pub m: u32,
    pub ef_construction: u32,
    pub ef_search: u32,
    pub layered: bool,
}

/// A document: metadata, content summary and its chunk list.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DocRecord {
    pub key: String,
    pub meta: Vec<(String, MetaValue)>,
    /// 0 markdown, 1 code, 2 html, 3 text.
    pub format: u8,
    pub content_hash: u64,
    pub content_len: u64,
    pub version: u64,
    pub created_ms: u64,
    pub updated_ms: u64,
    /// Chunk ids in ordinal order.
    pub chunks: Vec<u32>,
    pub sections: Vec<SectionRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionRecord {
    pub level: u8,
    pub heading: (u64, u64),
    pub range: (u64, u64),
    pub parent: Option<u32>,
}

/// One chunk: where it sits in its parent document.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub doc: u64,
    pub ordinal: u32,
    pub start: u64,
    pub end: u64,
    pub overlap_start: u64,
    pub section: Option<u32>,
    pub tokens: u32,
    pub hash: u64,
}

/// Totals for a collection's keyword index (BM25 needs the average length).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalStats {
    /// Chunks in the keyword index.
    pub chunks: u64,
    /// Sum of their lengths, in weighted term units.
    pub total_len: u64,
    /// Layout of the postings this describes; an index with an older version
    /// is rebuilt rather than read.
    pub version: u32,
}

/// One entry of a collection's change log: an index change the writer
/// applied in memory, in commit order, so processes reading the file can
/// apply it to their own copy of the index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Change {
    /// Chunks removed, then chunks added to the write buffer (their vectors
    /// are in the `vectors` table).
    Write {
        deletes: Vec<u32>,
        inserts: Vec<u32>,
    },
    /// The write buffer's chunks listed in this segment were sealed into it.
    Seal { segment: u64 },
    /// Segments were replaced by one rebuilt from their live chunks (or by
    /// nothing, when none were left).
    Compact {
        removed: Vec<u64>,
        added: Option<u64>,
    },
}

/// A sealed segment: its scalar fields, and where its arrays sit in the
/// segment's region of the file (see [`crate::segments`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegmentRecord {
    pub dim: u32,
    pub metric: u8,
    pub len: u32,
    pub m: u32,
    pub entry: u32,
    pub max_level: u8,
    pub quant_scale: f32,
    pub quant_offset: Vec<f32>,
    /// Physical offset and length of the region.
    pub region: (u64, u64),
    /// Per array, in [`crate::segments::Part`] order: byte offset within
    /// the region and element count.
    pub parts: Vec<(u64, u64)>,
    /// xxh3 of every array except the full vectors.
    pub checksum: u64,
}

/// A checkpoint of an unfinished bulk load: `count` chunk ids (u32), then,
/// page-aligned, their prepared vectors (`dim` each, f16 if `half`, else
/// f32), in one region.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BulkRecord {
    pub region: (u64, u64),
    pub count: u32,
    pub dim: u32,
    pub half: bool,
}

/// The scalar fields of a sealed segment in a version 2 file, whose arrays
/// were stored as `segment_data` values. Read only to upgrade such a file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegmentHeader {
    pub dim: u32,
    pub metric: u8,
    pub len: u32,
    pub m: u32,
    pub entry: u32,
    pub max_level: u8,
    pub quant_scale: f32,
}

pub fn encode<T: Serialize>(value: &T) -> Vec<u8> {
    postcard::to_allocvec(value).expect("serializing to a Vec cannot fail")
}

pub fn decode<T: DeserializeOwned>(bytes: &[u8], what: &str) -> Result<T> {
    postcard::from_bytes(bytes).map_err(|e| corrupt(format!("{what}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip() {
        let d = DocRecord {
            key: "readme".into(),
            meta: vec![
                ("lang".into(), "en".into()),
                ("tags".into(), vec!["a", "b"].into()),
            ],
            format: 0,
            content_hash: 7,
            content_len: 100,
            version: 2,
            created_ms: 1,
            updated_ms: 2,
            chunks: vec![1, 2, 3],
            sections: vec![SectionRecord {
                level: 1,
                heading: (2, 7),
                range: (0, 100),
                parent: None,
            }],
        };
        assert_eq!(decode::<DocRecord>(&encode(&d), "doc").unwrap(), d);
        assert!(decode::<DocRecord>(&[0xff, 0xff], "doc").is_err());

        for c in [
            Change::Write {
                deletes: vec![4],
                inserts: vec![5, 6],
            },
            Change::Seal { segment: 3 },
            Change::Compact {
                removed: vec![1, 2],
                added: None,
            },
        ] {
            assert_eq!(decode::<Change>(&encode(&c), "change").unwrap(), c);
        }
    }
}
