//! Greedy packing of blocks into chunks, splitting only oversized blocks.

use std::ops::Range;

use crate::format::floor_char_boundary;
use crate::parse::{Block, BlockKind};
use crate::{ChunkConfig, Section, Tokenizer};

/// Whether `inner` is `outer` or nested somewhere inside it.
fn is_within(sections: &[Section], inner: Option<u32>, outer: Option<u32>) -> bool {
    let Some(outer) = outer else {
        return false;
    };
    let mut cur = inner;
    while let Some(s) = cur {
        if s == outer {
            return true;
        }
        cur = sections[s as usize].parent;
    }
    false
}

/// Packs blocks into chunk ranges, each tagged with its section.
///
/// Blocks of the same section merge until the target size. A heading always
/// starts a chunk, but a chunk holding only headings absorbs the subsection
/// that follows, so "# Guide / ## Install / text" stays together. A block
/// larger than `max_tokens` is split at the finest boundary that fits.
pub(crate) fn pack(
    text: &str,
    blocks: &[Block],
    sections: &[Section],
    cfg: &ChunkConfig,
    tok: &dyn Tokenizer,
) -> Vec<(Range<usize>, Option<u32>)> {
    struct Cur {
        range: Range<usize>,
        section: Option<u32>,
        tokens: usize,
        only_headings: bool,
    }
    let mut out = Vec::new();
    let mut cur: Option<Cur> = None;
    let flush = |cur: &mut Option<Cur>, out: &mut Vec<(Range<usize>, Option<u32>)>| {
        if let Some(c) = cur.take() {
            out.push((c.range, c.section));
        }
    };

    for b in blocks {
        let t = tok.count(&text[b.range.clone()]);
        if t > cfg.max_tokens {
            flush(&mut cur, &mut out);
            let level = if b.kind == BlockKind::Code {
                Level::Lines
            } else {
                Level::Sentences
            };
            let mut pieces = Vec::new();
            split(text, b.range.clone(), level, cfg, tok, &mut pieces);
            out.extend(pieces.into_iter().map(|p| (p, b.section)));
            continue;
        }
        match cur.as_mut() {
            Some(c)
                if c.tokens + t <= cfg.target_tokens
                    && ((c.section == b.section && b.kind != BlockKind::Heading)
                        || (c.only_headings && is_within(sections, b.section, c.section))) =>
            {
                c.range.end = b.range.end;
                c.tokens += t;
                c.section = b.section;
                c.only_headings &= b.kind == BlockKind::Heading;
            }
            _ => {
                flush(&mut cur, &mut out);
                cur = Some(Cur {
                    range: b.range.clone(),
                    section: b.section,
                    tokens: t,
                    only_headings: b.kind == BlockKind::Heading,
                });
            }
        }
    }
    flush(&mut cur, &mut out);
    out
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Lines,
    Sentences,
    Words,
    Chars,
}

impl Level {
    fn finer(self) -> Level {
        match self {
            Level::Lines | Level::Sentences => Level::Words,
            Level::Words | Level::Chars => Level::Chars,
        }
    }
}

/// Cut points inside `range` (exclusive of its start, always including its end).
fn boundaries(text: &str, range: Range<usize>, level: Level) -> Vec<usize> {
    let s = &text[range.clone()];
    let base = range.start;
    let mut out = Vec::new();
    match level {
        Level::Lines => out.extend(s.match_indices('\n').map(|(i, _)| base + i + 1)),
        Level::Sentences => {
            let b = s.as_bytes();
            let mut i = 0;
            while i < b.len() {
                let c = b[i];
                i += 1;
                if c == b'\n' {
                    out.push(base + i);
                } else if matches!(c, b'.' | b'!' | b'?')
                    && i < b.len()
                    && b[i].is_ascii_whitespace()
                {
                    while i < b.len() && b[i].is_ascii_whitespace() && b[i] != b'\n' {
                        i += 1;
                    }
                    out.push(base + i);
                }
            }
        }
        Level::Words => {
            let b = s.as_bytes();
            for i in 1..b.len() {
                if b[i - 1].is_ascii_whitespace() && !b[i].is_ascii_whitespace() {
                    out.push(base + i);
                }
            }
        }
        Level::Chars => out.extend(s.char_indices().skip(1).map(|(i, _)| base + i)),
    }
    out.retain(|&p| p > range.start && p < range.end);
    out.push(range.end);
    out.dedup();
    out
}

/// Splits `range` into pieces of at most `target` tokens where possible,
/// falling back to finer boundaries for any piece that exceeds `max_tokens`.
fn split(
    text: &str,
    range: Range<usize>,
    level: Level,
    cfg: &ChunkConfig,
    tok: &dyn Tokenizer,
    out: &mut Vec<Range<usize>>,
) {
    let bounds = boundaries(text, range.clone(), level);
    let mut s = range.start;
    let mut i = 0;
    while s < range.end {
        // Largest j with count(s..bounds[j]) <= target, by binary search.
        let (mut lo, mut hi) = (i, bounds.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if tok.count(&text[s..bounds[mid]]) <= cfg.target_tokens {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo > i {
            let end = bounds[lo - 1];
            out.push(s..end);
            s = end;
            i = lo;
        } else {
            let end = bounds[i];
            if tok.count(&text[s..end]) <= cfg.max_tokens || level == Level::Chars {
                out.push(s..end);
            } else {
                split(text, s..end, level.finer(), cfg, tok, out);
            }
            s = end;
            i += 1;
        }
    }
}

/// Where the overlap borrowed from the previous chunk starts: about
/// `overlap_tokens` before `start`, moved forward to a word boundary, and
/// never before the previous chunk's start.
pub(crate) fn overlap_start(
    text: &str,
    prev_start: usize,
    start: usize,
    overlap_tokens: usize,
    tok: &dyn Tokenizer,
) -> usize {
    if overlap_tokens == 0 || start <= prev_start {
        return start;
    }
    let (mut lo, mut hi) = (prev_start, start);
    while lo < hi {
        let mid = floor_char_boundary(text, (lo + hi) / 2).max(lo);
        if mid == lo {
            break;
        }
        if tok.count(&text[mid..start]) <= overlap_tokens {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let mut p = if tok.count(&text[lo..start]) <= overlap_tokens {
        lo
    } else {
        hi
    };
    if p > prev_start && !text[..p].ends_with(char::is_whitespace) {
        // Start on a word, not mid-way through one.
        let window = &text[p..start];
        match window.find(|c: char| c.is_whitespace()) {
            Some(ws) => {
                let after = &window[ws..];
                let skip = after.len() - after.trim_start().len();
                p += ws + skip;
            }
            None => p = start,
        }
    }
    p.min(start)
}
