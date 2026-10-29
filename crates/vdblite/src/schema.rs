use vdblite_chunk::ChunkConfig;
use vdblite_core::codec::{metric_from_u8, metric_to_u8};
use vdblite_core::records::{ChunkingRecord, CollectionRecord, HnswRecord};
use vdblite_index::hnsw::HnswParams;
use vdblite_index::segment::Quantization;
use vdblite_index::{IndexConfig, Metric};

use crate::error::{Result, invalid};

/// What a collection stores. Fixed when the collection is created.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schema {
    pub dim: usize,
    pub metric: Metric,
    pub quant: Quant,
    pub chunking: Chunking,
}

impl Schema {
    /// A schema with int8 quantization and default structural chunking.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Schema {
            dim,
            metric,
            quant: Quant::Int8,
            chunking: Chunking::default(),
        }
    }
}

/// In-memory representation of indexed vectors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Quant {
    /// int8 codes for the graph walk, f16 re-ranking.
    #[default]
    Int8,
    /// Full f32 vectors walk the graph. Use when int8 costs too much recall.
    None,
    /// 1-bit codes for the graph walk (an eighth of int8's size), f16
    /// re-ranking.
    Binary,
}

/// How documents are split into chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chunking {
    /// Each document is one chunk.
    None,
    /// Split on structure, then size. A single block is only split when it
    /// exceeds twice `target_tokens`.
    Structural {
        target_tokens: usize,
        overlap_tokens: usize,
    },
}

impl Default for Chunking {
    fn default() -> Self {
        Chunking::Structural {
            target_tokens: 400,
            overlap_tokens: 50,
        }
    }
}

impl Chunking {
    pub(crate) fn config(self) -> Option<ChunkConfig> {
        match self {
            Chunking::None => None,
            Chunking::Structural {
                target_tokens,
                overlap_tokens,
            } => Some(ChunkConfig {
                target_tokens,
                max_tokens: target_tokens * 2,
                overlap_tokens,
            }),
        }
    }
}

/// Merges grow segments up to this many times the seal threshold (about 1M
/// chunks by default). Fewer, larger graphs search with less work: a query
/// walks each segment's graph in full.
const SEGMENT_GROWTH: usize = 16;

/// Index tuning for a collection; [`IndexOptions::default`] suits most uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexOptions {
    pub hnsw: HnswParams,
    /// Chunks buffered in the mutable segment before it is sealed into a graph.
    pub seal_threshold: usize,
    /// Keep a BM25 keyword index for hybrid search (see `Search::text`).
    pub lexical: bool,
}

impl Default for IndexOptions {
    fn default() -> Self {
        IndexOptions {
            hnsw: HnswParams::default(),
            seal_threshold: 65_536,
            lexical: true,
        }
    }
}

pub(crate) fn validate(schema: &Schema, opts: &IndexOptions) -> Result<()> {
    if schema.dim == 0 || schema.dim > vdblite_index::distance::MAX_DIM {
        return Err(invalid(format!(
            "dim must be 1..={}",
            vdblite_index::distance::MAX_DIM
        )));
    }
    if let Chunking::Structural { target_tokens, .. } = schema.chunking
        && target_tokens == 0
    {
        return Err(invalid("target_tokens must be positive"));
    }
    if opts.hnsw.m < 2 || opts.hnsw.ef_construction == 0 || opts.hnsw.ef_search == 0 {
        return Err(invalid("hnsw m must be >= 2 and ef values positive"));
    }
    if opts.seal_threshold == 0 {
        return Err(invalid("seal_threshold must be positive"));
    }
    Ok(())
}

pub(crate) fn to_record(
    id: u32,
    name: &str,
    schema: &Schema,
    opts: &IndexOptions,
) -> CollectionRecord {
    CollectionRecord {
        id,
        name: name.to_owned(),
        dim: schema.dim as u32,
        metric: metric_to_u8(schema.metric),
        quantization: match schema.quant {
            Quant::Int8 => 0,
            Quant::None => 1,
            Quant::Binary => 2,
        },
        chunking: match schema.chunking {
            Chunking::None => ChunkingRecord::None,
            Chunking::Structural {
                target_tokens,
                overlap_tokens,
            } => ChunkingRecord::Structural {
                target_tokens: target_tokens as u32,
                max_tokens: (target_tokens * 2) as u32,
                overlap_tokens: overlap_tokens as u32,
            },
        },
        hnsw: HnswRecord {
            m: opts.hnsw.m as u32,
            ef_construction: opts.hnsw.ef_construction as u32,
            ef_search: opts.hnsw.ef_search as u32,
            layered: opts.hnsw.layered,
        },
        seal_threshold: opts.seal_threshold as u32,
        next_doc: 1,
        next_chunk: 0,
        next_segment: 1,
    }
}

pub(crate) fn from_record(r: &CollectionRecord) -> Result<(Schema, IndexOptions)> {
    let schema = Schema {
        dim: r.dim as usize,
        metric: metric_from_u8(r.metric)?,
        quant: match r.quantization {
            1 => Quant::None,
            2 => Quant::Binary,
            _ => Quant::Int8,
        },
        chunking: match r.chunking {
            ChunkingRecord::None => Chunking::None,
            ChunkingRecord::Structural {
                target_tokens,
                overlap_tokens,
                ..
            } => Chunking::Structural {
                target_tokens: target_tokens as usize,
                overlap_tokens: overlap_tokens as usize,
            },
        },
    };
    let opts = IndexOptions {
        hnsw: HnswParams {
            m: r.hnsw.m as usize,
            ef_construction: r.hnsw.ef_construction as usize,
            ef_search: r.hnsw.ef_search as usize,
            layered: r.hnsw.layered,
            ..HnswParams::default()
        },
        seal_threshold: r.seal_threshold as usize,
        // Not in the record: it's the presence of the collection's `lexical` row.
        lexical: false,
    };
    Ok((schema, opts))
}

pub(crate) fn index_config(schema: &Schema, opts: &IndexOptions) -> IndexConfig {
    let base = IndexConfig::new(schema.dim, schema.metric);
    IndexConfig {
        hnsw: opts.hnsw,
        seal_threshold: opts.seal_threshold,
        max_segment: (opts.seal_threshold * SEGMENT_GROWTH).max(1),
        quantization: match schema.quant {
            Quant::Int8 => Quantization::Int8,
            Quant::None => Quantization::None,
            Quant::Binary => Quantization::Binary,
        },
        // 1-bit distances order candidates more loosely: re-rank twice as
        // many (on Cohere 1M, recall@100 0.983 against int8's 0.972 at the
        // default, at about the same speed).
        rerank: match schema.quant {
            Quant::Binary => 2 * base.rerank,
            Quant::Int8 | Quant::None => base.rerank,
        },
        ..base
    }
}
