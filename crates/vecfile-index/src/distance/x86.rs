//! AVX2+FMA kernels for x86_64.
//!
//! f32 kernels run four independent 8-lane accumulators to hide FMA latency.
//! int8 kernels widen to i16 and use `vpmaddwd`, which multiplies pairs of
//! i16 lanes and adds adjacent products into i32 lanes; the sums are exact.

use std::arch::x86_64::*;

use super::{Kernels, scalar};

pub(super) const KERNELS: Kernels = Kernels {
    name: "avx2",
    dot_f32,
    l2_sq_f32,
    dot_i8,
    l2_sq_i8,
};

// The safe wrappers below are only reachable through `KERNELS`, which
// `Kernels::avx2()` hands out after checking for AVX2 and FMA at runtime.

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    // SAFETY: the CPU supports AVX2+FMA (see above) and the lengths match.
    unsafe { dot_f32_avx2(a, b) }
}

fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    // SAFETY: the CPU supports AVX2+FMA (see above) and the lengths match.
    unsafe { l2_sq_f32_avx2(a, b) }
}

fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    // SAFETY: the CPU supports AVX2 (see above) and the lengths match.
    unsafe { dot_i8_avx2(a, b) }
}

fn l2_sq_i8(a: &[i8], b: &[i8]) -> u32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    // SAFETY: the CPU supports AVX2 (see above) and the lengths match.
    unsafe { l2_sq_i8_avx2(a, b) }
}

/// # Safety
/// Requires AVX2 and FMA, and `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every load reads 8 floats at offset i with i + 8 <= n.
    unsafe {
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for (j, acc) in acc.iter_mut().enumerate() {
                let off = i + 8 * j;
                *acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(off)),
                    _mm256_loadu_ps(pb.add(off)),
                    *acc,
                );
            }
            i += 32;
        }
        while i + 8 <= n {
            acc[0] = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i)),
                _mm256_loadu_ps(pb.add(i)),
                acc[0],
            );
            i += 8;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        hsum_ps(sum) + scalar::dot_f32(&a[i..], &b[i..])
    }
}

/// # Safety
/// Requires AVX2 and FMA, and `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_sq_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every load reads 8 floats at offset i with i + 8 <= n.
    unsafe {
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for (j, acc) in acc.iter_mut().enumerate() {
                let off = i + 8 * j;
                let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(off)), _mm256_loadu_ps(pb.add(off)));
                *acc = _mm256_fmadd_ps(d, d, *acc);
            }
            i += 32;
        }
        while i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            acc[0] = _mm256_fmadd_ps(d, d, acc[0]);
            i += 8;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        hsum_ps(sum) + scalar::l2_sq_f32(&a[i..], &b[i..])
    }
}

/// # Safety
/// Requires AVX2 and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every load reads 16 bytes at offset i with i + 16 <= n.
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            acc0 = _mm256_add_epi32(
                acc0,
                _mm256_madd_epi16(load_i16(pa.add(i)), load_i16(pb.add(i))),
            );
            acc1 = _mm256_add_epi32(
                acc1,
                _mm256_madd_epi16(load_i16(pa.add(i + 16)), load_i16(pb.add(i + 16))),
            );
            i += 32;
        }
        if i + 16 <= n {
            acc0 = _mm256_add_epi32(
                acc0,
                _mm256_madd_epi16(load_i16(pa.add(i)), load_i16(pb.add(i))),
            );
            i += 16;
        }
        hsum_epi32(_mm256_add_epi32(acc0, acc1)) + scalar::dot_i8(&a[i..], &b[i..])
    }
}

/// # Safety
/// Requires AVX2 and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
unsafe fn l2_sq_i8_avx2(a: &[i8], b: &[i8]) -> u32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every load reads 16 bytes at offset i with i + 16 <= n.
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let d0 = _mm256_sub_epi16(load_i16(pa.add(i)), load_i16(pb.add(i)));
            let d1 = _mm256_sub_epi16(load_i16(pa.add(i + 16)), load_i16(pb.add(i + 16)));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d0, d0));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d1, d1));
            i += 32;
        }
        if i + 16 <= n {
            let d0 = _mm256_sub_epi16(load_i16(pa.add(i)), load_i16(pb.add(i)));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d0, d0));
            i += 16;
        }
        // Lane sums stay below 2^31 for dims <= MAX_DIM; the total may not, so
        // it is reinterpreted as u32 (the wrapping add is exact modulo 2^32).
        (hsum_epi32(_mm256_add_epi32(acc0, acc1)) as u32)
            .wrapping_add(scalar::l2_sq_i8(&a[i..], &b[i..]))
    }
}

/// Loads 16 int8 values and sign-extends them to 16 i16 lanes.
///
/// # Safety
/// Requires AVX2, and 16 readable bytes at `p`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_i16(p: *const i8) -> __m256i {
    // SAFETY: the caller guarantees 16 readable bytes at `p`.
    unsafe { _mm256_cvtepi8_epi16(_mm_loadu_si128(p.cast())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn hsum_ps(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0b01));
    _mm_cvtss_f32(s)
}

/// Wrapping horizontal sum of eight i32 lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum_epi32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s)
}
