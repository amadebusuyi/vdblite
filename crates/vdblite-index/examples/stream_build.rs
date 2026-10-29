//! A streamed segment build against a one-shot one, on the same vectors.
//!
//! ```text
//! cargo run --release -p vdblite-index --example stream_build -- \
//!     --base base.f32 [--dim 768] [--n 1000000] [--batch 10000] [--oneshot]
//! ```
//!
//! `base.f32` holds the vectors row by row. Feeds them to a
//! [`SegmentBuilder`] `--batch` rows at a time and reports how long the adds
//! and the finish took; with `--oneshot`, builds the same segment in one
//! call for comparison.

use std::time::Instant;

use rayon::prelude::*;
use vdblite_index::Metric;
use vdblite_index::hnsw::HnswParams;
use vdblite_index::segment::{Quantization, SealedSegment, SegmentBuilder, prepare};

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut base, mut dim, mut n, mut batch, mut oneshot) =
        (String::new(), 768usize, usize::MAX, 10_000usize, false);
    while let Some(a) = args.next() {
        let mut num = |what: &str| -> usize {
            args.next()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{what} N"))
        };
        match a.as_str() {
            "--base" => base = args.next().expect("--base FILE"),
            "--dim" => dim = num("--dim"),
            "--n" => n = num("--n"),
            "--batch" => batch = num("--batch"),
            "--oneshot" => oneshot = true,
            other => panic!("unknown argument {other}"),
        }
    }
    let bytes = std::fs::read(&base).unwrap_or_else(|e| panic!("{base}: {e}"));
    let rows = (bytes.len() / 4 / dim).min(n);
    let mut data = vec![0f32; rows * dim];
    bytemuck::cast_slice_mut::<f32, u8>(&mut data).copy_from_slice(&bytes[..rows * dim * 4]);
    drop(bytes);
    data.par_chunks_mut(dim)
        .for_each(|v| prepare(Metric::Cosine, v));
    let ids: Vec<u32> = (0..rows as u32).collect();
    let params = HnswParams::default();

    if oneshot {
        let t = Instant::now();
        let seg = SealedSegment::build(&ids, &data, dim, Metric::Cosine, &params);
        println!(
            "one-shot: {rows} rows in {:.1?} ({} rows)",
            t.elapsed(),
            seg.len()
        );
        return;
    }
    let t = Instant::now();
    let mut b = SegmentBuilder::new(dim, Metric::Cosine, &params, Quantization::Int8);
    for (i, chunk) in data.chunks(batch * dim).enumerate() {
        b.add(&ids[i * batch..i * batch + chunk.len() / dim], chunk);
    }
    let added = t.elapsed();
    let seg = b.finish();
    println!(
        "streamed in batches of {batch}: adds {added:.1?}, finish {:.1?}, total {:.1?} ({} rows)",
        t.elapsed() - added,
        t.elapsed(),
        seg.len()
    );
}
