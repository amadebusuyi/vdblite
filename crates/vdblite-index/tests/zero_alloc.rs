//! The per-segment query path must not allocate once its scratch is warm.
//!
//! A counting global allocator records allocations made on the test thread
//! while `TRACKING` is set.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use vdblite_index::Metric;
use vdblite_index::hnsw::HnswParams;
use vdblite_index::rng::SplitMix64;
use vdblite_index::segment::{SealedSegment, SearchParams, SegmentScratch};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
}

// SAFETY: forwards every call to the system allocator unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.with(Cell::get) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING.with(Cell::get) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn segment_search_allocates_nothing_when_warm() {
    let dim = 32;
    let n = 5000;
    let mut r = SplitMix64::new(9);
    let data: Vec<f32> = (0..n * dim).map(|_| r.next_f64_open() as f32).collect();
    let ids: Vec<u32> = (0..n as u32).collect();
    let seg = SealedSegment::build(&ids, &data, dim, Metric::L2, &HnswParams::default());

    let params = SearchParams {
        n: 40,
        ef: 64,
        brute_force_below: 0.02,
    };
    let mut scratch = SegmentScratch::default();
    let mut out = Vec::with_capacity(64);
    let queries: Vec<&[f32]> = data.chunks_exact(dim).take(100).collect();

    // Warm up: buffers grow to their working size.
    for q in &queries {
        out.clear();
        seg.candidates(q, &params, None, None, &mut scratch, &mut out);
    }

    ALLOCS.store(0, Ordering::Relaxed);
    TRACKING.with(|t| t.set(true));
    for q in &queries {
        out.clear();
        seg.candidates(q, &params, None, None, &mut scratch, &mut out);
    }
    TRACKING.with(|t| t.set(false));
    assert_eq!(
        ALLOCS.load(Ordering::Relaxed),
        0,
        "allocations on the warm query path"
    );
    assert_eq!(out.len(), 40);
}
