//! Parsing content into blocks and a section tree.
//!
//! Blocks tile the text: block `i` runs from its start to the next block's
//! start, so blank lines and whitespace trail the block before them. The first
//! block always starts at 0.

use std::ops::Range;

use crate::Section;
use crate::format::Format;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockKind {
    Heading,
    Paragraph,
    Code,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Block {
    pub range: Range<usize>,
    pub kind: BlockKind,
    pub section: Option<u32>,
}

pub(crate) struct Parsed {
    pub blocks: Vec<Block>,
    pub sections: Vec<Section>,
}

pub(crate) fn parse(text: &str, format: Format) -> Parsed {
    let mut b = Builder::default();
    match format {
        Format::Markdown => markdown(text, &mut b),
        Format::Code => code(text, &mut b),
        Format::Html => html(text, &mut b),
        Format::Text => plain(text, &mut b),
    }
    b.finish(text.len())
}

/// Collects block starts and maintains the open-section stack.
#[derive(Default)]
struct Builder {
    starts: Vec<(usize, BlockKind, Option<u32>)>,
    sections: Vec<Section>,
    open: Vec<u32>,
}

impl Builder {
    fn current(&self) -> Option<u32> {
        self.open.last().copied()
    }

    fn block(&mut self, start: usize, kind: BlockKind) {
        let section = self.current();
        self.starts.push((start, kind, section));
    }

    /// Closes every open section at or below `level` at `at`.
    fn close_to(&mut self, level: u8, at: usize) {
        while let Some(&top) = self.open.last() {
            if self.sections[top as usize].level < level {
                break;
            }
            self.sections[top as usize].range.end = at;
            self.open.pop();
        }
    }

    fn close_all(&mut self, at: usize) {
        self.close_to(0, at);
    }

    /// Opens a section whose heading starts a block at `start`.
    fn heading(&mut self, level: u8, heading: Range<usize>, start: usize) {
        self.close_to(level, start);
        let id = self.sections.len() as u32;
        self.sections.push(Section {
            level,
            heading,
            range: start..start,
            parent: self.current(),
        });
        self.open.push(id);
        self.block(start, BlockKind::Heading);
    }

    fn finish(mut self, len: usize) -> Parsed {
        self.close_all(len);
        if len > 0 && self.starts.first().is_none_or(|s| s.0 != 0) {
            // Leading whitespace (or a document with no blocks) joins the first block.
            match self.starts.first_mut() {
                Some(first) => first.0 = 0,
                None => self.starts.push((0, BlockKind::Paragraph, None)),
            }
        }
        let mut blocks = Vec::with_capacity(self.starts.len());
        for (i, &(start, kind, section)) in self.starts.iter().enumerate() {
            let end = self.starts.get(i + 1).map_or(len, |s| s.0);
            if end > start {
                blocks.push(Block {
                    range: start..end,
                    kind,
                    section,
                });
            }
        }
        Parsed {
            blocks,
            sections: self.sections,
        }
    }
}

/// Lines with their byte offsets, newlines included.
fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.split_inclusive('\n').scan(0, |off, line| {
        let start = *off;
        *off += line.len();
        Some((start, line))
    })
}

fn strip_eol(line: &str) -> &str {
    line.trim_end_matches(['\n', '\r'])
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start_matches([' ', '\t']).len()
}

/// Range of `inner` within `outer`, both slices of the same text at `base`.
fn sub_range(base: usize, outer: &str, inner: &str) -> Range<usize> {
    let start = base + (inner.as_ptr() as usize - outer.as_ptr() as usize);
    start..start + inner.len()
}

// ---------------------------------------------------------------- Markdown

fn atx_heading(line: &str) -> Option<(u8, &str)> {
    if indent(line) > 3 {
        return None;
    }
    let t = line.trim_start();
    let hashes = t.bytes().take_while(|&c| c == b'#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &t[hashes..];
    if !(rest.is_empty() || rest.starts_with([' ', '\t'])) {
        return None;
    }
    let mut content = rest.trim();
    // Optional closing sequence: " ##"
    let stripped = content.trim_end_matches('#');
    if stripped.len() < content.len() && (stripped.is_empty() || stripped.ends_with([' ', '\t'])) {
        content = stripped.trim_end();
    }
    Some((hashes as u8, content))
}

fn fence_open(line: &str) -> Option<(u8, usize)> {
    if indent(line) > 3 {
        return None;
    }
    let t = line.trim_start();
    let c = *t.as_bytes().first()?;
    if c != b'`' && c != b'~' {
        return None;
    }
    let n = t.bytes().take_while(|&x| x == c).count();
    (n >= 3 && !(c == b'`' && t[n..].contains('`'))).then_some((c, n))
}

fn fence_closes(line: &str, (c, n): (u8, usize)) -> bool {
    let t = line.trim();
    t.len() >= n && t.bytes().all(|x| x == c)
}

fn setext_level(line: &str) -> Option<u8> {
    if indent(line) > 3 {
        return None;
    }
    let t = line.trim();
    if t.is_empty() {
        return None;
    }
    if t.bytes().all(|c| c == b'=') {
        Some(1)
    } else if t.len() >= 2 && t.bytes().all(|c| c == b'-') {
        Some(2)
    } else {
        None
    }
}

fn markdown(text: &str, b: &mut Builder) {
    let mut fence: Option<(u8, usize)> = None;
    let mut prev_blank = true;
    // The open paragraph: its start, first line and line count.
    let mut para: Option<(usize, &str, usize)> = None;
    for (off, raw) in lines(text) {
        let line = strip_eol(raw);
        if let Some(f) = fence {
            if fence_closes(line, f) {
                fence = None;
            }
            prev_blank = false;
            continue;
        }
        if line.trim().is_empty() {
            prev_blank = true;
            para = None;
            continue;
        }
        if let Some((level, content)) = atx_heading(line) {
            b.heading(level, sub_range(off, raw, content), off);
            para = None;
        } else if let Some(f) = fence_open(line) {
            b.block(off, BlockKind::Code);
            fence = Some(f);
            para = None;
        } else if let (Some(level), Some((start, first, 1))) = (setext_level(line), para) {
            // "Title\n=====": the one-line paragraph above becomes a heading.
            b.starts.pop();
            b.heading(level, sub_range(start, first, first.trim()), start);
            para = None;
        } else if let Some(p) = para.as_mut().filter(|_| !prev_blank) {
            p.2 += 1;
        } else {
            b.block(off, BlockKind::Paragraph);
            para = Some((off, line, 1));
        }
        prev_blank = false;
    }
}

// ---------------------------------------------------------------- Code

const DEF_KEYWORDS: &[&str] = &[
    "fn ",
    "pub ",
    "async ",
    "def ",
    "class ",
    "impl",
    "struct ",
    "enum ",
    "trait ",
    "mod ",
    "func ",
    "function ",
    "export ",
    "interface ",
    "type ",
    "const ",
    "static ",
    "var ",
    "let ",
    "unsafe ",
    "public ",
    "private ",
    "protected ",
    "abstract ",
    "package ",
    "namespace ",
    "template",
    "union ",
    "object ",
    "macro_rules!",
];

fn is_preamble(line: &str) -> bool {
    let t = line.trim_start();
    ["#[", "#!", "@", "//", "/*", "*", "#", "--", "\"\"\"", "'''"]
        .iter()
        .any(|p| t.starts_with(p))
}

fn is_closer(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with(['}', ')', ']']) || t == "end" || t.starts_with("end ")
}

/// Code is split at blank lines followed by an unindented line, so each block
/// is roughly one top-level item with its doc comment and attributes. Items
/// that start with a definition keyword become level-1 sections.
fn code(text: &str, b: &mut Builder) {
    let mut prev_blank = true;
    let mut block_start = None;
    let mut found_def = false;
    for (off, raw) in lines(text) {
        let line = strip_eol(raw);
        if line.trim().is_empty() {
            prev_blank = true;
            continue;
        }
        let top_level = indent(line) == 0 && !is_closer(line);
        if prev_blank && top_level || block_start.is_none() {
            block_start = Some(off);
            found_def = false;
            b.close_all(off);
            b.block(off, BlockKind::Code);
        }
        if !found_def && indent(line) == 0 && !is_preamble(line) {
            found_def = true;
            // One-line declarations (`mod x;`, `pub use a::b;`, `const N: u8 = 1;`)
            // are not worth a section of their own: left section-less, they
            // merge with their neighbours instead of becoming tiny chunks.
            let one_liner = line.trim_end().ends_with(';') || line.starts_with("pub use ");
            if !one_liner && DEF_KEYWORDS.iter().any(|k| line.starts_with(k)) {
                let start = block_start.expect("set above");
                let head = line.trim_end().trim_end_matches(['{', ':']).trim_end();
                b.starts.pop();
                b.heading(1, sub_range(off, raw, head), start);
                b.starts.last_mut().expect("just pushed").1 = BlockKind::Code;
            }
        }
        prev_blank = false;
    }
}

// ---------------------------------------------------------------- HTML

const BLOCK_TAGS: &[&str] = &[
    "p",
    "div",
    "pre",
    "ul",
    "ol",
    "li",
    "section",
    "article",
    "table",
    "blockquote",
    "header",
    "footer",
    "main",
    "nav",
    "aside",
    "figure",
    "dl",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
];

/// Splits HTML at opening block-level tags; `<hN>` opens a section and
/// `<pre>` is kept whole. Script and style bodies never start blocks.
fn html(text: &str, b: &mut Builder) {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut i = 0;
    while let Some(rel) = lower[i..].find('<') {
        let at = i + rel;
        let name_start = at + 1;
        let name_len = bytes[name_start..]
            .iter()
            .take_while(|c| c.is_ascii_alphanumeric())
            .count();
        let name = &lower[name_start..name_start + name_len];
        let tag_end = lower[at..].find('>').map_or(text.len(), |e| at + e + 1);
        i = tag_end.max(at + 1);
        if name.is_empty() {
            continue;
        }
        if name == "script" || name == "style" || name == "pre" {
            let close = format!("</{name}");
            let end = lower[i..].find(&close).map_or(text.len(), |e| i + e);
            if name == "pre" {
                b.block(at, BlockKind::Code);
            }
            i = end;
            continue;
        }
        if !BLOCK_TAGS.contains(&name) {
            continue;
        }
        if let [b'h', d] = name.as_bytes()
            && (b'1'..=b'6').contains(d)
        {
            let close = format!("</{name}");
            let inner_end = lower[i..].find(&close).map_or(text.len(), |e| i + e);
            let inner = &text[i.min(inner_end)..inner_end];
            let trimmed = inner.trim();
            let heading = sub_range(i.min(inner_end), inner, trimmed);
            b.heading(d - b'0', heading, at);
            i = inner_end;
        } else {
            b.block(at, BlockKind::Paragraph);
        }
    }
}

/// Removes tags and decodes the common entities, for embedding HTML.
pub(crate) fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    for (from, to) in [
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", "\""),
        ("&#39;", "'"),
        ("&nbsp;", " "),
        ("&amp;", "&"),
    ] {
        if out.contains(from) {
            out = out.replace(from, to);
        }
    }
    out
}

// ---------------------------------------------------------------- Text

fn plain(text: &str, b: &mut Builder) {
    let mut prev_blank = true;
    for (off, raw) in lines(text) {
        if strip_eol(raw).trim().is_empty() {
            prev_blank = true;
            continue;
        }
        if prev_blank {
            b.block(off, BlockKind::Paragraph);
        }
        prev_blank = false;
    }
}
