//! Arbitrary bytes decoded as a segment's arrays must be rejected or yield a
//! segment that searches without panicking.
#![no_main]

use libfuzzer_sys::fuzz_target;
use vecfile_index::Metric;
use vecfile_index::hnsw::GraphParts;
use vecfile_index::segment::{SealedSegment, SearchParams, SegmentParts, SegmentScratch};

fn take<'a>(data: &mut &'a [u8], n: usize) -> &'a [u8] {
    let n = n.min(data.len());
    let (a, b) = data.split_at(n);
    *data = b;
    a
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) % 64).collect()
}

fuzz_target!(|data: &[u8]| {
    let mut d = data;
    let hdr = take(&mut d, 4);
    if hdr.len() < 4 {
        return;
    }
    let dim = 1 + hdr[0] as usize % 4;
    let n = hdr[1] as usize % 16;
    let m = 2 + hdr[2] as u32 % 3;
    let max_level = hdr[3] % 3;
    let levels: Vec<u8> = take(&mut d, n).iter().map(|b| b % 3).collect();
    let level0 = words(take(&mut d, 4 * n * (2 * m as usize + 1)));
    let upper_off = words(take(&mut d, 4 * n));
    let upper = words(take(&mut d, 64));
    let parts = SegmentParts {
        dim,
        metric: Metric::L2,
        graph: GraphParts { m, entry: hdr[3] as u32 % 16, max_level, levels, level0, upper_off, upper },
        quant_offset: vec![0.0; dim],
        quant_scale: 1.0,
        codes: take(&mut d, n * dim).iter().map(|&b| b as i8).collect(),
        corr: vec![0.0; n],
        vectors: Vec::new(),
        ids: (0..n as u32).collect(),
    };
    if let Ok(seg) = SealedSegment::from_parts(parts) {
        let mut out = Vec::new();
        let p = SearchParams { n: 4, ef: 8, brute_force_below: 0.02 };
        seg.candidates(&vec![0.5; dim], &p, None, None, &mut SegmentScratch::default(), &mut out);
    }
});
