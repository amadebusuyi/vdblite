use vecfile_chunk::ChunkConfig;
use vecfile_core::codec::{metric_from_u8, metric_to_u8};
use vecfile_core::records::{ChunkingRecord, CollectionRecord, HnswRecord};
use vecfile_index::hnsw::HnswParams;
use vecfile_index::segment::Quantization;
use vecfile_index::{IndexConfig, Metric};

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
    /// int8 codes for the graph walk, f32 re-ranking. About 4x less memory.
    #[default]
    Int8,
    /// Full f32 vectors in memory. Use when int8 costs too much recall.
    None,
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
    if schema.dim == 0 || schema.dim > vecfile_index::distance::MAX_DIM {
        return Err(invalid(format!(
            "dim must be 1..={}",
            vecfile_index::distance::MAX_DIM
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
        quant: if r.quantization == 1 {
            Quant::None
        } else {
            Quant::Int8
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
    IndexConfig {
        hnsw: opts.hnsw,
        seal_threshold: opts.seal_threshold,
        max_segment: (opts.seal_threshold * 4).max(1),
        quantization: match schema.quant {
            Quant::Int8 => Quantization::Int8,
            Quant::None => Quantization::None,
        },
        ..IndexConfig::new(schema.dim, schema.metric)
    }
}
