use proptest::prelude::*;

use super::*;

fn small() -> ChunkConfig {
    ChunkConfig {
        target_tokens: 20,
        max_tokens: 40,
        overlap_tokens: 5,
    }
}

fn texts<'t>(text: &'t str, doc: &Chunked) -> Vec<&'t str> {
    doc.chunks.iter().map(|c| &text[c.range.clone()]).collect()
}

/// Every invariant a chunked document must hold.
fn check(text: &str, doc: &Chunked, cfg: &ChunkConfig) {
    let mut pos = 0;
    for (i, c) in doc.chunks.iter().enumerate() {
        assert_eq!(c.ordinal as usize, i);
        assert_eq!(c.range.start, pos, "chunks must be contiguous");
        assert!(c.range.end > c.range.start, "chunks must be non-empty");
        assert!(c.overlap_start <= c.range.start);
        if i > 0 {
            assert!(c.overlap_start >= doc.chunks[i - 1].range.start);
        }
        assert!(text.is_char_boundary(c.range.start) && text.is_char_boundary(c.range.end));
        assert!(text.is_char_boundary(c.overlap_start));
        assert!(
            c.tokens as usize <= cfg.max_tokens,
            "chunk {i} has {} tokens > {}: {:?}",
            c.tokens,
            cfg.max_tokens,
            &text[c.range.clone()]
        );
        if let Some(s) = c.section {
            assert!((s as usize) < doc.sections.len());
        }
        pos = c.range.end;
    }
    assert_eq!(pos, text.len(), "chunks must cover the whole text");
    let rebuilt: String = texts(text, doc).concat();
    assert_eq!(rebuilt, text);
    for (i, s) in doc.sections.iter().enumerate() {
        assert!(
            s.range.start <= s.heading.start && s.heading.end <= s.range.end.max(s.heading.end)
        );
        assert!(s.range.end <= text.len());
        if let Some(p) = s.parent {
            assert!((p as usize) < i, "parents come first");
            let parent = &doc.sections[p as usize];
            assert!(parent.level < s.level);
            assert!(parent.range.start <= s.range.start && s.range.end <= parent.range.end);
        }
    }
}

#[test]
fn markdown_sections_and_paths() {
    let text = "Preamble line.\n\n# Guide\n\nIntro text.\n\n## Install\n\n### Linux\n\nUse apt.\n\n## Usage\n\nRun it.\n";
    let doc = chunk(text, Format::Markdown, &ChunkConfig::default());
    check(text, &doc, &ChunkConfig::default());
    let paths: Vec<Vec<&str>> = doc
        .chunks
        .iter()
        .map(|c| doc.heading_path(text, c))
        .collect();
    assert_eq!(
        paths,
        vec![
            vec![],
            vec!["Guide"],
            vec!["Guide", "Install", "Linux"],
            vec!["Guide", "Usage"]
        ]
    );
    // "## Install" carries no text of its own, so it joins its subsection.
    assert!(texts(text, &doc)[2].starts_with("## Install\n\n### Linux"));
    let usage = &doc.chunks[3];
    assert_eq!(
        &text[doc.section_range(text, usage)],
        "## Usage\n\nRun it.\n"
    );
}

#[test]
fn sibling_headings_do_not_merge() {
    let text = "# A\n\n# B\n\nbody\n";
    let doc = chunk(text, Format::Markdown, &ChunkConfig::default());
    assert_eq!(texts(text, &doc), ["# A\n\n", "# B\n\nbody\n"]);
}

#[test]
fn setext_headings_and_closing_hashes() {
    let text = "Title\n=====\n\nBody.\n\nSub ##\n---\n\nMore.\n\n## Closed ##\n\nEnd.\n";
    let doc = chunk(text, Format::Markdown, &ChunkConfig::default());
    check(text, &doc, &ChunkConfig::default());
    let heads: Vec<&str> = doc
        .sections
        .iter()
        .map(|s| &text[s.heading.clone()])
        .collect();
    assert_eq!(heads, ["Title", "Sub ##", "Closed"]);
    assert_eq!(doc.sections[1].parent, Some(0));
}

#[test]
fn code_fences_are_never_split_or_parsed() {
    let fence = format!(
        "```rust\n# not a heading\n{}\n```\n",
        "let x = 1;\n".repeat(5)
    );
    let text = format!("# Doc\n\n{fence}\nAfter.\n");
    let doc = chunk(&text, Format::Markdown, &small());
    check(&text, &doc, &small());
    assert_eq!(doc.sections.len(), 1, "a # inside a fence is not a heading");
    assert!(
        texts(&text, &doc)
            .iter()
            .any(|t| t.contains("# not a heading") && t.contains("```rust"))
    );
}

#[test]
fn oversized_paragraphs_split_at_sentences() {
    let para = "This is a sentence that has some words. ".repeat(30);
    let doc = chunk(&para, Format::Text, &small());
    check(&para, &doc, &small());
    assert!(doc.chunks.len() > 5);
    for t in &texts(&para, &doc)[..doc.chunks.len() - 1] {
        assert!(t.ends_with(". "), "split mid-sentence: {t:?}");
    }
}

#[test]
fn unbroken_text_falls_back_to_char_splits() {
    let text = "é".repeat(500);
    let doc = chunk(&text, Format::Text, &small());
    check(&text, &doc, &small());
}

#[test]
fn code_splits_at_top_level_items() {
    let text = "use std::io;\n\n/// Adds.\n#[inline]\npub fn add(a: i32, b: i32) -> i32 {\n    a + b\n\n    // blank line inside\n}\n\nstruct Point {\n    x: f32,\n}\n";
    let cfg = ChunkConfig {
        target_tokens: 8,
        max_tokens: 400,
        overlap_tokens: 0,
    };
    let doc = chunk(text, Format::Code, &cfg);
    check(text, &doc, &cfg);
    let heads: Vec<&str> = doc
        .sections
        .iter()
        .map(|s| &text[s.heading.clone()])
        .collect();
    assert_eq!(heads, ["pub fn add(a: i32, b: i32) -> i32", "struct Point"]);
    let t = texts(text, &doc);
    assert_eq!(t[0], "use std::io;\n\n");
    assert!(
        t[1].starts_with("/// Adds.\n#[inline]\npub fn add")
            && t[1].contains("// blank line inside\n}\n")
    );
}

#[test]
fn one_line_declarations_merge_instead_of_becoming_tiny_chunks() {
    let text = "mod a;\nmod b;\n\npub use crate::x::Y;\n\nconst N: usize = 4;\n\nfn main() {\n    run();\n}\n";
    let doc = chunk(text, Format::Code, &ChunkConfig::default());
    check(text, &doc, &ChunkConfig::default());
    let heads: Vec<&str> = doc
        .sections
        .iter()
        .map(|s| &text[s.heading.clone()])
        .collect();
    assert_eq!(heads, ["fn main()"]);
    assert_eq!(
        texts(text, &doc)[0],
        "mod a;\nmod b;\n\npub use crate::x::Y;\n\nconst N: usize = 4;\n\n"
    );
}

#[test]
fn html_headings_and_pre() {
    let text = "<html><body><h1>Guide</h1><p>Intro &amp; more.</p><h2 class=\"x\">Install</h2><pre>\n<p>not a block</p>\n</pre><p>Done.</p></body></html>";
    let doc = chunk(text, Format::Html, &ChunkConfig::default());
    check(text, &doc, &ChunkConfig::default());
    let heads: Vec<&str> = doc
        .sections
        .iter()
        .map(|s| &text[s.heading.clone()])
        .collect();
    assert_eq!(heads, ["Guide", "Install"]);
    let install = doc
        .chunks
        .iter()
        .find(|c| doc.heading_path(text, c) == ["Guide", "Install"])
        .unwrap();
    let embed = doc.embed_text(text, install);
    assert!(embed.starts_with("Guide > Install\n\n"));
    assert!(!embed.contains("<h2"));
}

#[test]
fn overlap_is_a_range_of_the_previous_chunk() {
    let text = "alpha beta gamma delta. ".repeat(20);
    let cfg = ChunkConfig {
        target_tokens: 20,
        max_tokens: 40,
        overlap_tokens: 6,
    };
    let doc = chunk(&text, Format::Text, &cfg);
    check(&text, &doc, &cfg);
    for w in doc.chunks.windows(2) {
        let (prev, c) = (&w[0], &w[1]);
        assert!(c.overlap_start < c.range.start, "expected some overlap");
        assert!(c.overlap_start >= prev.range.start);
        let overlap = &text[c.overlap_start..c.range.start];
        assert!(ApproxTokenizer.count(overlap) <= 6);
        assert!(
            !overlap.starts_with(' '),
            "overlap should start on a word: {overlap:?}"
        );
    }
}

#[test]
fn hashes_change_only_where_text_changes() {
    let a = "# T\n\nFirst para.\n\n## S\n\nSecond para.\n\n## U\n\nThird para.\n";
    let b = "# T\n\nFirst para.\n\n## S\n\nSecond para, edited.\n\n## U\n\nThird para.\n";
    let cfg = ChunkConfig {
        overlap_tokens: 0,
        ..ChunkConfig::default()
    };
    let (da, db) = (
        chunk(a, Format::Markdown, &cfg),
        chunk(b, Format::Markdown, &cfg),
    );
    let ha: Vec<u64> = da.chunks.iter().map(|c| c.hash).collect();
    let hb: Vec<u64> = db.chunks.iter().map(|c| c.hash).collect();
    assert_eq!(ha.len(), 3);
    assert_eq!(ha[0], hb[0]);
    assert_ne!(ha[1], hb[1]);
    assert_eq!(ha[2], hb[2]);
}

#[test]
fn empty_and_whitespace_only() {
    assert!(chunk("", Format::Markdown, &small()).chunks.is_empty());
    let ws = "\n\n   \n";
    let doc = chunk(ws, Format::Markdown, &small());
    check(ws, &doc, &small());
    assert_eq!(doc.chunks.len(), 1);
}

#[test]
fn detection() {
    assert_eq!(
        detect("# Title\n\nSome text\n- item\n", None),
        Format::Markdown
    );
    assert_eq!(
        detect("<!DOCTYPE html><html><body></body></html>", None),
        Format::Html
    );
    assert_eq!(
        detect("fn main() {\n    println!(\"hi\");\n}\n", None),
        Format::Code
    );
    assert_eq!(
        detect("Just a plain letter.\n\nRegards.", None),
        Format::Text
    );
    assert_eq!(detect("anything", Some("notes.md")), Format::Markdown);
    assert_eq!(detect("anything", Some("text/html")), Format::Html);
    assert_eq!(detect("anything", Some("lib.rs")), Format::Code);
}

#[test]
fn single_chunk_mode() {
    let doc = single("whole doc", Format::Text, &ApproxTokenizer);
    assert_eq!(doc.chunks.len(), 1);
    assert_eq!(doc.chunks[0].range, 0..9);
    assert!(single("", Format::Text, &ApproxTokenizer).chunks.is_empty());
}

fn fragment() -> impl Strategy<Value = String> {
    prop_oneof![
        (1usize..7, "[a-zA-Z ]{0,20}").prop_map(|(n, t)| format!("{} {t}\n", "#".repeat(n))),
        "[a-z]{1,12}( [a-z]{1,12}){0,30}[.!?]? ?".prop_map(|s| format!("{s}\n")),
        Just("\n".to_string()),
        Just("\n\n".to_string()),
        "[a-z ]{0,40}".prop_map(|s| format!("```\n{s}\n# x\n```\n")),
        Just("Title\n===\n".to_string()),
        Just("---\n".to_string()),
        "[a-zé漢]{50,400}".prop_map(|s| s),
        "    [a-z(){};]{1,30}\n".prop_map(|s| s),
        "fn [a-z]{1,8}\\(\\) \\{\n".prop_map(|s| s),
        "<(p|h1|h3|div|pre)>[a-z ]{0,30}</(p|h1|h3|div|pre)>".prop_map(|s| s),
        Just("}\n".to_string()),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn invariants_hold_for_any_input(
        parts in prop::collection::vec(fragment(), 0..40),
        format in prop::sample::select(vec![Format::Markdown, Format::Code, Format::Html, Format::Text]),
        target in 4usize..60,
        extra in 0usize..60,
        overlap in 0usize..20,
    ) {
        let text = parts.concat();
        let cfg = ChunkConfig { target_tokens: target, max_tokens: target + extra, overlap_tokens: overlap };
        let doc = chunk(&text, format, &cfg);
        check(&text, &doc, &cfg);
    }

    #[test]
    fn arbitrary_unicode_never_panics(text in "\\PC{0,600}", target in 1usize..40) {
        let cfg = ChunkConfig { target_tokens: target, max_tokens: target * 2, overlap_tokens: 3 };
        for format in [Format::Markdown, Format::Code, Format::Html, Format::Text] {
            let doc = chunk(&text, format, &cfg);
            check(&text, &doc, &cfg);
        }
    }
}
