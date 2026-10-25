//! Property tests for the distance kernels.
//!
//! SIMD kernels must match the scalar reference: exactly for int8, and within
//! a relative 1e-5 for f32 (the sums are reassociated, so the last bits move).

use proptest::prelude::*;
use vecfile_index::distance::{Kernels, Metric, normalize, scalar};

/// Lengths cover empty, every SIMD tail remainder, and past 768.
fn f32_pair() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    (0usize..1100).prop_flat_map(|n| {
        (
            prop::collection::vec(-1.0f32..1.0, n),
            prop::collection::vec(-1.0f32..1.0, n),
        )
    })
}

fn i8_pair() -> impl Strategy<Value = (Vec<i8>, Vec<i8>)> {
    (0usize..1100).prop_flat_map(|n| {
        (
            prop::collection::vec(any::<i8>(), n),
            prop::collection::vec(any::<i8>(), n),
        )
    })
}

fn f32_i8_pair() -> impl Strategy<Value = (Vec<f32>, Vec<i8>)> {
    (0usize..1100).prop_flat_map(|n| {
        (
            prop::collection::vec(-200.0f32..200.0, n),
            prop::collection::vec(any::<i8>(), n),
        )
    })
}

/// Tolerance scaled by the sum of absolute terms, which bounds the rounding
/// error of any summation order.
fn close(got: f32, want: f32, magnitude: f32) -> bool {
    (got - want).abs() <= 1e-5 * magnitude.max(1.0)
}

/// Exact references in wider types.
fn dot_exact(a: &[i8], b: &[i8]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

fn l2_exact(a: &[i8], b: &[i8]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (i64::from(x) - i64::from(y)).pow(2))
        .sum()
}

#[test]
fn at_least_scalar_is_available() {
    let names: Vec<_> = Kernels::available().iter().map(|k| k.name).collect();
    assert_eq!(names[0], "scalar");
    eprintln!("kernel sets on this CPU: {names:?}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn simd_f32_matches_scalar((a, b) in f32_pair()) {
        let dot_mag: f32 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
        let l2_mag = scalar::l2_sq_f32(&a, &b);
        let want_dot = scalar::dot_f32(&a, &b);
        for k in Kernels::available() {
            let got = (k.dot_f32)(&a, &b);
            prop_assert!(close(got, want_dot, dot_mag), "{} dot: {} vs {}", k.name, got, want_dot);
            let got = (k.l2_sq_f32)(&a, &b);
            prop_assert!(close(got, l2_mag, l2_mag), "{} l2: {} vs {}", k.name, got, l2_mag);
        }
    }

    #[test]
    fn simd_f32_i8_matches_scalar((a, b) in f32_i8_pair()) {
        let dot_mag: f32 = a.iter().zip(&b).map(|(x, &y)| (x * f32::from(y)).abs()).sum();
        let want_dot = scalar::dot_f32_i8(&a, &b);
        let want_l2 = scalar::l2_sq_f32_i8(&a, &b);
        for k in Kernels::available() {
            let got = (k.dot_f32_i8)(&a, &b);
            prop_assert!(close(got, want_dot, dot_mag), "{} dot: {} vs {}", k.name, got, want_dot);
            let got = (k.l2_sq_f32_i8)(&a, &b);
            prop_assert!(close(got, want_l2, want_l2), "{} l2: {} vs {}", k.name, got, want_l2);
        }
    }

    #[test]
    fn i8_kernels_are_exact((a, b) in i8_pair()) {
        for k in Kernels::available() {
            prop_assert_eq!(i64::from((k.dot_i8)(&a, &b)), dot_exact(&a, &b), "{} dot", k.name);
            prop_assert_eq!(i64::from((k.l2_sq_i8)(&a, &b)), l2_exact(&a, &b), "{} l2", k.name);
        }
    }

    #[test]
    fn scalar_f32_properties((a, b) in f32_pair()) {
        prop_assert_eq!(scalar::l2_sq_f32(&a, &a), 0.0);
        prop_assert_eq!(scalar::l2_sq_f32(&a, &b), scalar::l2_sq_f32(&b, &a));
        prop_assert_eq!(scalar::dot_f32(&a, &b), scalar::dot_f32(&b, &a));
        prop_assert!(scalar::l2_sq_f32(&a, &b) >= 0.0);
        // |a - b|^2 = |a|^2 + |b|^2 - 2 a·b
        let expanded = scalar::dot_f32(&a, &a) + scalar::dot_f32(&b, &b) - 2.0 * scalar::dot_f32(&a, &b);
        let mag = scalar::dot_f32(&a, &a) + scalar::dot_f32(&b, &b);
        prop_assert!((scalar::l2_sq_f32(&a, &b) - expanded).abs() <= 1e-4 * mag.max(1.0));
    }

    #[test]
    fn i8_extremes_do_not_overflow(n in 0usize..=4096, x in prop::sample::select(vec![i8::MIN, i8::MAX])) {
        let a = vec![x; n];
        let b = vec![if x == i8::MIN { i8::MAX } else { i8::MIN }; n];
        for k in Kernels::available() {
            prop_assert_eq!(i64::from((k.dot_i8)(&a, &a)), dot_exact(&a, &a));
            prop_assert_eq!(i64::from((k.l2_sq_i8)(&a, &b)), l2_exact(&a, &b));
        }
    }

    #[test]
    fn cosine_of_normalized_self_is_zero(mut v in prop::collection::vec(-1.0f32..1.0, 1..800)) {
        prop_assume!(v.iter().any(|x| x.abs() > 1e-3));
        normalize(&mut v);
        for k in Kernels::available() {
            prop_assert!(Metric::Cosine.distance(&k, &v, &v).abs() < 1e-5);
        }
    }
}

#[test]
fn i8_max_dim_does_not_overflow() {
    use vecfile_index::distance::MAX_DIM;
    let a = vec![i8::MIN; MAX_DIM];
    let b = vec![i8::MAX; MAX_DIM];
    for k in Kernels::available() {
        assert_eq!(
            i64::from((k.l2_sq_i8)(&a, &b)),
            l2_exact(&a, &b),
            "{}",
            k.name
        );
        assert_eq!(
            i64::from((k.dot_i8)(&a, &a)),
            dot_exact(&a, &a),
            "{}",
            k.name
        );
    }
}

#[test]
#[should_panic(expected = "length mismatch")]
fn mismatched_lengths_panic() {
    vecfile_index::distance::dot_f32(&[1.0, 2.0], &[1.0]);
}
