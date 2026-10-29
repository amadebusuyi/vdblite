//! Any text, any format, any sizes: chunks must tile the text exactly.
#![no_main]

use libfuzzer_sys::fuzz_target;
use vdblite_chunk::{ChunkConfig, Format, chunk};

fuzz_target!(|data: &[u8]| {
    let Some((&head, rest)) = data.split_first() else { return };
    let Ok(text) = std::str::from_utf8(rest) else { return };
    let format = [Format::Markdown, Format::Code, Format::Html, Format::Text][head as usize % 4];
    let target = 1 + (head as usize >> 2) % 64;
    let cfg = ChunkConfig { target_tokens: target, max_tokens: target * 2, overlap_tokens: target / 4 };
    let doc = chunk(text, format, &cfg);
    let mut pos = 0;
    for c in &doc.chunks {
        assert_eq!(c.range.start, pos);
        assert!(c.range.end > c.range.start);
        assert!(c.tokens as usize <= cfg.max_tokens);
        pos = c.range.end;
    }
    assert_eq!(pos, text.len());
});
