//! The keyword index: BM25 over chunk text, for hybrid search.
//!
//! Embeddings are good at meaning and poor at exact strings (identifiers,
//! error codes, rare names). Hybrid search runs BM25 next to the vector
//! search and merges the two rankings with Reciprocal Rank Fusion.
//!
//! Each chunk's indexed text is its heading path plus its own bytes (not the
//! overlap, which belongs to the previous chunk). Postings are written in the
//! same transaction as the chunk, so they can't drift from the vectors.

use std::collections::{BTreeMap, HashMap};

use redb::{ReadableTable, Table};
use roaring::RoaringBitmap;
use vecfile_core::records::{LexicalStats, SectionRecord, decode, encode};

use crate::error::Result;

/// Keyword-index layout version. 2: term counts are weighted, whole words 2
/// and identifier parts 1, so "info" ranks `info()` above `ChunkInfo`.
pub(crate) const INDEX_VERSION: u32 = 2;

/// Weight of a whole token, and of each identifier part, in term counts.
const WHOLE: u32 = 2;
const PART: u32 = 1;

const K1: f32 = 1.2;
const B: f32 = 0.75;
const MAX_TERM_LEN: usize = 64;

/// Common English words that carry no search signal.
const STOPWORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been",
    "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of",
    "on", "or", "our", "she", "so", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "too", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
];

fn is_stopword(t: &str) -> bool {
    STOPWORDS.binary_search(&t).is_ok()
}

/// Stats for an empty keyword index of the current layout.
pub(crate) fn empty_stats() -> LexicalStats {
    LexicalStats {
        chunks: 0,
        total_len: 0,
        version: INDEX_VERSION,
    }
}

/// Decodes a collection's keyword-index stats; `None` if the index uses an
/// older layout (or the bytes predate versioning) and must be rebuilt.
pub(crate) fn read_stats(bytes: &[u8]) -> Option<LexicalStats> {
    decode::<LexicalStats>(bytes, "lexical stats")
        .ok()
        .filter(|s| s.version == INDEX_VERSION)
}

/// Splits `text` into lowercase terms, each with a weight. Identifiers also
/// yield their parts at a lower weight: `SealedSegment` gives
/// `sealedsegment` (whole), `sealed` and `segment` (parts).
pub(crate) fn tokenize(text: &str, mut emit: impl FnMut(String, u32)) {
    let mut push = |t: &str, weight: u32| {
        let t = t.to_lowercase();
        if t.chars().count() >= 2 && t.len() <= MAX_TERM_LEN && !is_stopword(&t) {
            emit(t, weight);
        }
    };
    for raw in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        let raw = raw.trim_matches('_');
        if raw.is_empty() {
            continue;
        }
        push(raw, WHOLE);
        let parts = split_identifier(raw);
        if parts.len() > 1 {
            for p in parts {
                push(p, PART);
            }
        }
    }
}

/// Parts of an identifier at underscores, lower-to-upper case changes and
/// letter/digit boundaries.
fn split_identifier(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    for w in 0..chars.len() {
        let (i, c) = chars[w];
        if c == '_' {
            if i > start {
                parts.push(&s[start..i]);
            }
            start = i + 1;
            continue;
        }
        if w > 0 && i > start {
            let prev = chars[w - 1].1;
            let upper_after_lower = c.is_uppercase() && prev.is_lowercase();
            // "HTTPServer" -> "HTTP", "Server": split before the last capital of a run.
            let acronym_end = c.is_uppercase()
                && prev.is_uppercase()
                && chars.get(w + 1).is_some_and(|n| n.1.is_lowercase());
            let digit_edge = c.is_ascii_digit() != prev.is_ascii_digit() && prev != '_';
            if upper_after_lower || acronym_end || digit_edge {
                parts.push(&s[start..i]);
                start = i;
            }
        }
    }
    if start < s.len() {
        parts.push(&s[start..]);
    }
    parts
}

/// Weighted term counts of a chunk's indexed text, and its weighted length.
pub(crate) fn term_counts(text: &str) -> (BTreeMap<String, u32>, u32) {
    let mut counts = BTreeMap::new();
    let mut len = 0u32;
    tokenize(text, |t, w| {
        *counts.entry(t).or_insert(0) += w;
        len += w;
    });
    (counts, len)
}

/// The distinct terms of a query.
pub(crate) fn query_terms(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    tokenize(text, |t, _| {
        if !out.contains(&t) {
            out.push(t);
        }
    });
    out
}

/// The text a chunk is indexed under: its heading path, then its bytes.
pub(crate) fn chunk_text(
    content: &str,
    sections: &[SectionRecord],
    section: Option<u32>,
    range: std::ops::Range<usize>,
) -> String {
    let mut path = Vec::new();
    let mut cur = section;
    while let Some(i) = cur {
        let Some(s) = sections.get(i as usize) else {
            break;
        };
        path.push(
            content
                .get(s.heading.0 as usize..s.heading.1 as usize)
                .unwrap_or("")
                .trim(),
        );
        cur = s.parent.filter(|&p| p < i);
    }
    path.reverse();
    let mut text = path.join(" ");
    text.push('\n');
    text.push_str(content.get(range).unwrap_or(""));
    text
}

pub(crate) struct Tables<'t> {
    pub postings: Table<'t, (u32, &'static str, u32), u64>,
    pub df: Table<'t, (u32, &'static str), u32>,
    pub chunk_terms: Table<'t, (u32, u32), &'static [u8]>,
}

/// Adds a chunk to the keyword index.
pub(crate) fn index_chunk(
    t: &mut Tables<'_>,
    coll: u32,
    chunk: u32,
    text: &str,
    stats: &mut LexicalStats,
) -> Result<()> {
    let (counts, len) = term_counts(text);
    for (term, &tf) in &counts {
        t.postings.insert(
            (coll, term.as_str(), chunk),
            (u64::from(tf) << 32) | u64::from(len),
        )?;
        let df = t.df.get((coll, term.as_str()))?.map_or(0, |v| v.value());
        t.df.insert((coll, term.as_str()), df + 1)?;
    }
    let terms: Vec<&String> = counts.keys().collect();
    t.chunk_terms
        .insert((coll, chunk), encode(&terms).as_slice())?;
    stats.chunks += 1;
    stats.total_len += u64::from(len);
    Ok(())
}

/// Removes a chunk from the keyword index (a no-op if it isn't there).
pub(crate) fn unindex_chunk(
    t: &mut Tables<'_>,
    coll: u32,
    chunk: u32,
    stats: &mut LexicalStats,
) -> Result<()> {
    let Some(bytes) = t.chunk_terms.remove((coll, chunk))? else {
        return Ok(());
    };
    let terms: Vec<String> = decode(bytes.value(), "chunk terms")?;
    drop(bytes);
    let mut len = 0u64;
    for term in &terms {
        if let Some(v) = t.postings.remove((coll, term.as_str(), chunk))? {
            let packed = v.value();
            len = packed & 0xFFFF_FFFF;
        }
        let df = t.df.get((coll, term.as_str()))?.map_or(0, |v| v.value());
        if df <= 1 {
            t.df.remove((coll, term.as_str()))?;
        } else {
            t.df.insert((coll, term.as_str()), df - 1)?;
        }
    }
    stats.chunks = stats.chunks.saturating_sub(1);
    stats.total_len = stats.total_len.saturating_sub(len);
    Ok(())
}

/// The `n` chunks with the highest BM25 score for `query`, best first.
/// Chunks outside `filter` are skipped.
pub(crate) fn search(
    postings: &impl ReadableTable<(u32, &'static str, u32), u64>,
    df: &impl ReadableTable<(u32, &'static str), u32>,
    stats: LexicalStats,
    coll: u32,
    query: &str,
    filter: Option<&RoaringBitmap>,
    n: usize,
) -> Result<Vec<(u32, f32)>> {
    let terms = query_terms(query);
    if terms.is_empty() || stats.chunks == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let total = stats.chunks as f32;
    let avg_len = (stats.total_len as f32 / total).max(1.0);
    let mut scores: HashMap<u32, f32> = HashMap::new();
    for term in &terms {
        let Some(d) = df.get((coll, term.as_str()))?.map(|v| v.value()) else {
            continue;
        };
        let d = d as f32;
        let idf = (1.0 + (total - d + 0.5) / (d + 0.5)).ln();
        for row in postings.range((coll, term.as_str(), 0)..=(coll, term.as_str(), u32::MAX))? {
            let (k, v) = row?;
            let chunk = k.value().2;
            if filter.is_some_and(|f| !f.contains(chunk)) {
                continue;
            }
            let packed = v.value();
            // Counts are in weighted units where a whole word is WHOLE.
            let tf = (packed >> 32) as f32 / WHOLE as f32;
            let len = (packed & 0xFFFF_FFFF) as f32;
            let s = idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * len / avg_len));
            *scores.entry(chunk).or_insert(0.0) += s;
        }
    }
    let mut ranked: Vec<(u32, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(n);
    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        let mut v = Vec::new();
        tokenize(s, |t, _| v.push(t));
        v
    }

    #[test]
    fn tokenizes_prose_and_identifiers() {
        assert_eq!(toks("The quick fox"), ["quick", "fox"]);
        assert_eq!(
            toks("SealedSegment"),
            ["sealedsegment", "sealed", "segment"]
        );
        assert_eq!(
            toks("MAX_DIM = 65_536"),
            ["max_dim", "max", "dim", "65_536", "65", "536"]
        );
        assert_eq!(
            toks("HTTPServer v2"),
            ["httpserver", "http", "server", "v2"]
        );
        assert_eq!(toks("a I x"), Vec::<String>::new());
        assert_eq!(toks("Café résumé"), ["café", "résumé"]);
    }

    #[test]
    fn whole_words_outweigh_identifier_parts() {
        let (counts, len) = term_counts("info ChunkInfo");
        assert_eq!(counts["info"], WHOLE + PART);
        assert_eq!(counts["chunkinfo"], WHOLE);
        assert_eq!(counts["chunk"], PART);
        assert_eq!(len, 2 * WHOLE + 2 * PART);
    }

    #[test]
    fn outdated_stats_are_rejected() {
        let old = LexicalStats {
            chunks: 3,
            total_len: 9,
            version: 1,
        };
        assert!(read_stats(&encode(&old)).is_none());
        assert!(read_stats(&[3, 9]).is_none(), "pre-versioning bytes");
        assert_eq!(read_stats(&encode(&empty_stats())), Some(empty_stats()));
    }

    #[test]
    fn stopwords_are_sorted_for_binary_search() {
        assert!(STOPWORDS.windows(2).all(|w| w[0] < w[1]));
    }
}
