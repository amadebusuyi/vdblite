//! Distance kernels for f32 and int8 vectors.
//!
//! Every kernel exists as a portable scalar reference ([`scalar`]) and, where
//! the CPU allows, as an AVX2+FMA (x86_64) or NEON (aarch64) version. The best
//! set is picked once, on first use, and cached in [`kernels()`].
//!
//! Set `VECFILE_KERNELS=scalar` in the environment to force the scalar set,
//! which is useful when bisecting a suspected SIMD bug.
//!
//! All kernels panic if the two slices differ in length. int8 kernels are exact
//! for dimensions up to [`MAX_DIM`].

use std::sync::OnceLock;

pub mod scalar;

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

/// Largest dimension for which the int8 kernels cannot overflow their 32-bit sums.
pub const MAX_DIM: usize = 65_536;

/// How similarity is measured within a collection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    /// `1 - a·b`. Vectors must be unit length; see [`normalize`].
    Cosine,
    /// Negated inner product, `-(a·b)`.
    Dot,
    /// Squared Euclidean distance. The square root is skipped because it
    /// does not change the ranking.
    L2,
}

impl Metric {
    /// Distance between `a` and `b` under this metric, where smaller means closer.
    #[inline]
    pub fn distance(self, k: &Kernels, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Cosine => 1.0 - (k.dot_f32)(a, b),
            Metric::Dot => -(k.dot_f32)(a, b),
            Metric::L2 => (k.l2_sq_f32)(a, b),
        }
    }
}

/// One complete set of distance kernels for a given instruction set.
///
/// The function pointers are safe to call: a SIMD set can only be obtained
/// through a constructor that has checked the CPU supports it.
#[derive(Clone, Copy)]
pub struct Kernels {
    /// Short name of the instruction set, e.g. `"avx2"`.
    pub name: &'static str,
    pub dot_f32: fn(&[f32], &[f32]) -> f32,
    pub l2_sq_f32: fn(&[f32], &[f32]) -> f32,
    pub dot_i8: fn(&[i8], &[i8]) -> i32,
    pub l2_sq_i8: fn(&[i8], &[i8]) -> u32,
}

impl std::fmt::Debug for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernels").field("name", &self.name).finish()
    }
}

impl Kernels {
    /// The portable reference kernels.
    pub const SCALAR: Kernels = Kernels {
        name: "scalar",
        dot_f32: scalar::dot_f32,
        l2_sq_f32: scalar::l2_sq_f32,
        dot_i8: scalar::dot_i8,
        l2_sq_i8: scalar::l2_sq_i8,
    };

    /// AVX2+FMA kernels, if this is an x86_64 CPU that supports them.
    pub fn avx2() -> Option<Kernels> {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Some(x86::KERNELS);
        }
        None
    }

    /// NEON kernels, if this is an aarch64 CPU (NEON is mandatory there).
    pub fn neon() -> Option<Kernels> {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Some(neon::KERNELS);
        }
        None
    }

    /// Every kernel set this CPU can run, scalar first.
    pub fn available() -> Vec<Kernels> {
        [Some(Kernels::SCALAR), Kernels::avx2(), Kernels::neon()]
            .into_iter()
            .flatten()
            .collect()
    }

    /// The fastest kernel set this CPU supports, honouring `VECFILE_KERNELS=scalar`.
    pub fn detect() -> Kernels {
        if std::env::var("VECFILE_KERNELS").is_ok_and(|v| v.eq_ignore_ascii_case("scalar")) {
            return Kernels::SCALAR;
        }
        Kernels::avx2()
            .or_else(Kernels::neon)
            .unwrap_or(Kernels::SCALAR)
    }
}

/// The kernel set chosen for this process, detected on first call.
#[inline]
pub fn kernels() -> &'static Kernels {
    static KERNELS: OnceLock<Kernels> = OnceLock::new();
    KERNELS.get_or_init(Kernels::detect)
}

/// Inner product of two f32 vectors.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    (kernels().dot_f32)(a, b)
}

/// Squared Euclidean distance between two f32 vectors.
#[inline]
pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    (kernels().l2_sq_f32)(a, b)
}

/// Inner product of two int8 vectors, computed exactly.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    (kernels().dot_i8)(a, b)
}

/// Squared Euclidean distance between two int8 vectors, computed exactly.
#[inline]
pub fn l2_sq_i8(a: &[i8], b: &[i8]) -> u32 {
    (kernels().l2_sq_i8)(a, b)
}

/// Scales `v` to unit length in place. A zero vector is left unchanged.
pub fn normalize(v: &mut [f32]) {
    let norm = dot_f32(v, v).sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        v.iter_mut().for_each(|x| *x *= inv);
    }
}
