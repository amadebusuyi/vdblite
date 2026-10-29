//! NEON kernels for aarch64.
//!
//! f32 kernels run four independent 4-lane FMA accumulators. int8 dot widens
//! with `smull` (i8×i8 fits i16) and pair-accumulates into i32; int8 L2 widens
//! the differences to i16 and multiply-accumulates into i32 with `smlal`.

use std::arch::aarch64::*;

use super::{Kernels, scalar};

pub(super) const KERNELS: Kernels = Kernels {
    name: "neon",
    dot_f32,
    l2_sq_f32,
    dot_i8,
    l2_sq_i8,
    dot_f32_i8,
    l2_sq_f32_i8,
    // `count_ones` is a NEON `cnt` on aarch64 already.
    binary_ip4: scalar::binary_ip4,
};

// The safe wrappers below are only reachable through `KERNELS`, which
// `Kernels::neon()` hands out after checking for NEON at runtime.

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    // SAFETY: the CPU supports NEON (see above) and the lengths match.
    unsafe { dot_f32_neon(a, b) }
}

fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    // SAFETY: the CPU supports NEON (see above) and the lengths match.
    unsafe { l2_sq_f32_neon(a, b) }
}

fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    // SAFETY: the CPU supports NEON (see above) and the lengths match.
    unsafe { dot_i8_neon(a, b) }
}

fn l2_sq_i8(a: &[i8], b: &[i8]) -> u32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    // SAFETY: the CPU supports NEON (see above) and the lengths match.
    unsafe { l2_sq_i8_neon(a, b) }
}

fn dot_f32_i8(a: &[f32], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    // SAFETY: the CPU supports NEON (see above) and the lengths match.
    unsafe { dot_f32_i8_neon(a, b) }
}

fn l2_sq_f32_i8(a: &[f32], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    // SAFETY: the CPU supports NEON (see above) and the lengths match.
    unsafe { l2_sq_f32_i8_neon(a, b) }
}

/// Eight int8 values at `p`, widened to two f32x4 halves.
///
/// # Safety
/// Requires NEON and 8 readable bytes at `p`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn load_i8_f32x8(p: *const i8) -> (float32x4_t, float32x4_t) {
    // SAFETY: the caller guarantees 8 readable bytes at `p`.
    let w = vmovl_s8(unsafe { vld1_s8(p) });
    (
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))),
        vcvtq_f32_s32(vmovl_high_s16(w)),
    )
}

/// # Safety
/// Requires NEON and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
unsafe fn dot_f32_i8_neon(a: &[f32], b: &[i8]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every load reads 8 floats and 8 bytes at offset i with i + 8 <= n.
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i + 16 <= n {
            let (c0, c1) = load_i8_f32x8(pb.add(i));
            let (c2, c3) = load_i8_f32x8(pb.add(i + 8));
            acc[0] = vfmaq_f32(acc[0], vld1q_f32(pa.add(i)), c0);
            acc[1] = vfmaq_f32(acc[1], vld1q_f32(pa.add(i + 4)), c1);
            acc[2] = vfmaq_f32(acc[2], vld1q_f32(pa.add(i + 8)), c2);
            acc[3] = vfmaq_f32(acc[3], vld1q_f32(pa.add(i + 12)), c3);
            i += 16;
        }
        while i + 8 <= n {
            let (c0, c1) = load_i8_f32x8(pb.add(i));
            acc[0] = vfmaq_f32(acc[0], vld1q_f32(pa.add(i)), c0);
            acc[1] = vfmaq_f32(acc[1], vld1q_f32(pa.add(i + 4)), c1);
            i += 8;
        }
        let sum = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        vaddvq_f32(sum) + scalar::dot_f32_i8(&a[i..], &b[i..])
    }
}

/// # Safety
/// Requires NEON and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
unsafe fn l2_sq_f32_i8_neon(a: &[f32], b: &[i8]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every load reads 8 floats and 8 bytes at offset i with i + 8 <= n.
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 2];
        let mut i = 0;
        while i + 8 <= n {
            let (c0, c1) = load_i8_f32x8(pb.add(i));
            let d0 = vsubq_f32(vld1q_f32(pa.add(i)), c0);
            let d1 = vsubq_f32(vld1q_f32(pa.add(i + 4)), c1);
            acc[0] = vfmaq_f32(acc[0], d0, d0);
            acc[1] = vfmaq_f32(acc[1], d1, d1);
            i += 8;
        }
        vaddvq_f32(vaddq_f32(acc[0], acc[1])) + scalar::l2_sq_f32_i8(&a[i..], &b[i..])
    }
}

/// # Safety
/// Requires NEON and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every load reads 4 floats at offset i with i + 4 <= n.
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i + 16 <= n {
            for (j, acc) in acc.iter_mut().enumerate() {
                let off = i + 4 * j;
                *acc = vfmaq_f32(*acc, vld1q_f32(pa.add(off)), vld1q_f32(pb.add(off)));
            }
            i += 16;
        }
        while i + 4 <= n {
            acc[0] = vfmaq_f32(acc[0], vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            i += 4;
        }
        let sum = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        vaddvq_f32(sum) + scalar::dot_f32(&a[i..], &b[i..])
    }
}

/// # Safety
/// Requires NEON and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
unsafe fn l2_sq_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every load reads 4 floats at offset i with i + 4 <= n.
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i + 16 <= n {
            for (j, acc) in acc.iter_mut().enumerate() {
                let off = i + 4 * j;
                let d = vsubq_f32(vld1q_f32(pa.add(off)), vld1q_f32(pb.add(off)));
                *acc = vfmaq_f32(*acc, d, d);
            }
            i += 16;
        }
        while i + 4 <= n {
            let d = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc[0] = vfmaq_f32(acc[0], d, d);
            i += 4;
        }
        let sum = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
        vaddvq_f32(sum) + scalar::l2_sq_f32(&a[i..], &b[i..])
    }
}

/// # Safety
/// Requires NEON and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
unsafe fn dot_i8_neon(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every load reads 16 bytes at offset i with i + 16 <= n.
    unsafe {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut i = 0;
        while i + 16 <= n {
            let x = vld1q_s8(pa.add(i));
            let y = vld1q_s8(pb.add(i));
            acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
            acc1 = vpadalq_s16(acc1, vmull_high_s8(x, y));
            i += 16;
        }
        vaddvq_s32(vaddq_s32(acc0, acc1)) + scalar::dot_i8(&a[i..], &b[i..])
    }
}

/// # Safety
/// Requires NEON and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
unsafe fn l2_sq_i8_neon(a: &[i8], b: &[i8]) -> u32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // SAFETY: every load reads 16 bytes at offset i with i + 16 <= n.
    unsafe {
        let mut acc = [vdupq_n_s32(0); 4];
        let mut i = 0;
        while i + 16 <= n {
            let x = vld1q_s8(pa.add(i));
            let y = vld1q_s8(pb.add(i));
            let lo = vsubl_s8(vget_low_s8(x), vget_low_s8(y));
            let hi = vsubl_high_s8(x, y);
            acc[0] = vmlal_s16(acc[0], vget_low_s16(lo), vget_low_s16(lo));
            acc[1] = vmlal_high_s16(acc[1], lo, lo);
            acc[2] = vmlal_s16(acc[2], vget_low_s16(hi), vget_low_s16(hi));
            acc[3] = vmlal_high_s16(acc[3], hi, hi);
            i += 16;
        }
        // Lane sums stay below 2^31 for dims <= MAX_DIM; the total is summed as
        // u32 so it cannot overflow.
        let sum = vaddq_s32(vaddq_s32(acc[0], acc[1]), vaddq_s32(acc[2], acc[3]));
        vaddvq_u32(vreinterpretq_u32_s32(sum)).wrapping_add(scalar::l2_sq_i8(&a[i..], &b[i..]))
    }
}
