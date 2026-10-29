#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};

use vdblite::{BoxError, Embedder};

pub const DIM: usize = 64;

/// A deterministic bag-of-words embedder: each word adds weight to a few
/// hashed dimensions. Texts that share words get similar vectors.
#[derive(Default)]
pub struct WordEmbedder {
    pub calls: AtomicUsize,
    pub texts: AtomicUsize,
}

pub fn embed_one(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; DIM];
    for w in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        let h = xxhash(&w.to_lowercase());
        v[(h % DIM as u64) as usize] += 1.0;
        v[((h >> 16) % DIM as u64) as usize] += 0.5;
    }
    v[DIM - 1] += 0.01; // never all-zero
    v
}

fn xxhash(s: &str) -> u64 {
    // FNV-1a: stable and dependency-free.
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

impl Embedder for WordEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, BoxError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.texts.fetch_add(texts.len(), Ordering::Relaxed);
        Ok(texts.iter().map(|t| embed_one(t)).collect())
    }
}

impl WordEmbedder {
    pub fn texts_embedded(&self) -> usize {
        self.texts.load(Ordering::Relaxed)
    }
}

/// A markdown document with a few sections about `topic`.
pub fn article(topic: &str, n: usize) -> String {
    let mut s = format!("# {topic} guide\n\nAn introduction to {topic} number {n}.\n\n");
    for i in 0..3 {
        s.push_str(&format!(
            "## Part {i}\n\nThis part covers {topic} detail {i} for article {n}. It explains {topic} carefully.\n\n"
        ));
    }
    s
}
