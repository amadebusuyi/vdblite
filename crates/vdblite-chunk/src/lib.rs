//! Structure-aware chunking for vdblite.
//!
//! Content is split on structure first (headings, paragraphs, code blocks,
//! top-level definitions) and size second. Every chunk is a byte range of the
//! parent, and the ranges of a document's chunks tile it exactly, so the
//! parent rebuilds byte for byte and no text is ever copied.
//!
//! ```
//! use vdblite_chunk::{ChunkConfig, Format, chunk};
//!
//! let text = "# Guide\n\nIntro.\n\n## Install\n\nRun the installer.\n";
//! let doc = chunk(text, Format::Markdown, &ChunkConfig::default());
//! let rebuilt: String = doc.chunks.iter().map(|c| &text[c.range.clone()]).collect();
//! assert_eq!(rebuilt, text);
//! assert_eq!(doc.heading_path(text, &doc.chunks[1]), ["Guide", "Install"]);
//! ```

mod format;
mod pack;
mod parse;

use std::ops::Range;

pub use format::{Format, detect};

/// Bumped whenever chunking output changes for the same input, so stored
/// documents are re-chunked on their next write (see `vdblite`'s content
/// fingerprint). Chunks whose text is unchanged still keep their vectors.
///
/// History: 1 = one-line code declarations no longer start sections.
pub const VERSION: u64 = 1;

/// Counts tokens for sizing chunks.
pub trait Tokenizer: Send + Sync {
    fn count(&self, text: &str) -> usize;
}

/// Approximates tokens as bytes / 4, rounded up.
#[derive(Clone, Copy, Debug, Default)]
pub struct ApproxTokenizer;

impl Tokenizer for ApproxTokenizer {
    fn count(&self, text: &str) -> usize {
        text.len().div_ceil(4)
    }
}

/// Chunk sizing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkConfig {
    /// Blocks are merged until a chunk reaches about this many tokens.
    pub target_tokens: usize,
    /// A single block is only split when it alone exceeds this.
    pub max_tokens: usize,
    /// Tokens of the previous chunk prepended (as a range) for embedding.
    pub overlap_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        ChunkConfig {
            target_tokens: 400,
            max_tokens: 800,
            overlap_tokens: 50,
        }
    }
}

/// A heading and the span of text it governs (including subsections).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    /// 1 for `#`/`<h1>`/top-level code items, up to 6.
    pub level: u8,
    /// The heading's own text, without markup where it can be trimmed.
    pub heading: Range<usize>,
    /// From the heading to just before the next heading of equal or higher rank.
    pub range: Range<usize>,
    /// The enclosing section, if any.
    pub parent: Option<u32>,
}

/// One searchable piece of a document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// Position in the document, from 0.
    pub ordinal: u32,
    /// The chunk's own bytes; consecutive chunks' ranges are contiguous.
    pub range: Range<usize>,
    /// Start of the overlap borrowed from the previous chunk; the embedded
    /// text is `overlap_start..range.end`. Equals `range.start` when none.
    pub overlap_start: usize,
    /// The innermost section the chunk sits in.
    pub section: Option<u32>,
    /// Tokens in `range`.
    pub tokens: u32,
    /// Hash of the embedding input; unchanged hash means the vector can be reused.
    pub hash: u64,
}

/// A chunked document: its sections and chunks, all as byte ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunked {
    pub format: Format,
    pub sections: Vec<Section>,
    pub chunks: Vec<Chunk>,
}

impl Chunked {
    /// Headings from the outermost section down to the chunk's own.
    pub fn heading_path<'t>(&self, text: &'t str, chunk: &Chunk) -> Vec<&'t str> {
        self.section_path(text, chunk.section)
    }

    /// Headings from the outermost section down to `section`.
    pub fn section_path<'t>(&self, text: &'t str, section: Option<u32>) -> Vec<&'t str> {
        let mut path = Vec::new();
        let mut cur = section;
        while let Some(i) = cur {
            let s = &self.sections[i as usize];
            path.push(text[s.heading.clone()].trim());
            cur = s.parent;
        }
        path.reverse();
        path
    }

    /// The text to embed for a chunk: its heading path, a blank line, then
    /// the overlap and the chunk itself. HTML tags are stripped.
    pub fn embed_text(&self, text: &str, chunk: &Chunk) -> String {
        embed_text(
            self.format,
            &self.heading_path(text, chunk),
            &text[chunk.overlap_start..chunk.range.end],
        )
    }

    /// The byte range of the section a chunk belongs to, or the whole text
    /// when it sits before any heading.
    pub fn section_range(&self, text: &str, chunk: &Chunk) -> Range<usize> {
        chunk
            .section
            .map_or(0..text.len(), |s| self.sections[s as usize].range.clone())
    }
}

fn embed_text(format: Format, path: &[&str], body: &str) -> String {
    let mut out = String::with_capacity(body.len() + 64);
    let clean = |s: &str| {
        if format == Format::Html {
            parse::strip_tags(s)
        } else {
            s.to_owned()
        }
    };
    if !path.is_empty() {
        let joined: Vec<String> = path.iter().map(|p| clean(p).trim().to_owned()).collect();
        out.push_str(&joined.join(" > "));
        out.push_str("\n\n");
    }
    out.push_str(&clean(body));
    out
}

/// Chunks `text` with the default [`ApproxTokenizer`].
pub fn chunk(text: &str, format: Format, cfg: &ChunkConfig) -> Chunked {
    chunk_with(text, format, cfg, &ApproxTokenizer)
}

/// Chunks `text`, counting tokens with `tok`.
pub fn chunk_with(text: &str, format: Format, cfg: &ChunkConfig, tok: &dyn Tokenizer) -> Chunked {
    assert!(cfg.target_tokens > 0, "target_tokens must be positive");
    assert!(
        cfg.max_tokens >= cfg.target_tokens,
        "max_tokens must be >= target_tokens"
    );
    let parsed = parse::parse(text, format);
    let ranges = pack::pack(text, &parsed.blocks, &parsed.sections, cfg, tok);
    let mut doc = Chunked {
        format,
        sections: parsed.sections,
        chunks: Vec::with_capacity(ranges.len()),
    };
    for (i, (range, section)) in ranges.into_iter().enumerate() {
        let overlap_start = if i == 0 {
            range.start
        } else {
            pack::overlap_start(
                text,
                doc.chunks[i - 1].range.start,
                range.start,
                cfg.overlap_tokens,
                tok,
            )
        };
        let tokens = tok.count(&text[range.clone()]) as u32;
        doc.chunks.push(Chunk {
            ordinal: i as u32,
            range,
            overlap_start,
            section,
            tokens,
            hash: 0,
        });
    }
    for i in 0..doc.chunks.len() {
        let embed = doc.embed_text(text, &doc.chunks[i]);
        doc.chunks[i].hash = xxhash_rust::xxh3::xxh3_64(embed.as_bytes());
    }
    doc
}

/// The whole text as a single chunk (for collections that don't chunk).
pub fn single(text: &str, format: Format, tok: &dyn Tokenizer) -> Chunked {
    let mut doc = Chunked {
        format,
        sections: Vec::new(),
        chunks: Vec::new(),
    };
    if !text.is_empty() {
        let mut c = Chunk {
            ordinal: 0,
            range: 0..text.len(),
            overlap_start: 0,
            section: None,
            tokens: tok.count(text) as u32,
            hash: 0,
        };
        c.hash = xxhash_rust::xxh3::xxh3_64(doc.embed_text(text, &c).as_bytes());
        doc.chunks.push(c);
    }
    doc
}

#[cfg(test)]
mod tests;
