# Phase 4 — Chunking engine

Phase 4 splits documents into searchable chunks that are byte ranges of the
parent, never copies.

## Decisions

**Blocks tile the text.** Every parser produces blocks whose ranges are
contiguous and cover the whole text (whitespace trails the block before it).
Chunks are unions of consecutive blocks or pieces of one, so chunk ranges
tile the document too, and the parent rebuilds byte for byte by concatenation.
Property tests check this for random documents in all four formats.

**Parsers are small and line-based.** Markdown: ATX and setext headings,
fenced code (never split, `#` inside is not a heading), paragraphs.
HTML: opening block tags split; `<hN>` opens a section; `<pre>`, `<script>`
and `<style>` bodies are opaque. Code: a blank line followed by an unindented,
non-closing line starts a new top-level item; doc comments and attributes stay
with their item; items starting with a definition keyword become sections.
Plain text: blank-line paragraphs.

**tree-sitter is not used yet.** The guide put it behind a feature flag; the
heuristic code splitter covers brace and indentation languages well enough
for v1 without a C toolchain dependency. It's the obvious upgrade if code
retrieval quality disappoints.

**Packing.** Blocks of the same section merge up to `target_tokens`. A
heading starts a chunk, but a chunk holding only headings absorbs the
subsections nested under it (never a sibling), so `# Guide / ## Install /
text` stays together. Only a block over `max_tokens` (2x target) is split:
at lines (code) or sentences, then words, then characters, using binary
search over cut points so huge blocks stay O(n log n).

**Overlap is a range.** `overlap_start` points about `overlap_tokens` back
into the previous chunk, snapped forward to a word start. The embedding input
is `heading path + "\n\n" + text[overlap_start..end]`; HTML tags are stripped
from it. Nothing is stored twice.

**Hash the embedding input.** Each chunk's xxh3 hash covers exactly what is
embedded. An edit re-embeds only chunks whose input changed (the edited
chunk, the next one through its overlap, any whose heading path changed).

## Gate result

Passed: unit tests per format plus 800 random property-test documents across
all formats and sizes, with byte-exact rebuild, contiguity, size caps and
well-formed section trees. The guide's chunking-quality eval (hit rate on a
labelled question set from your own documents) needs your documents and an
embedding model, so it isn't in the repo yet.
