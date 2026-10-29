//! Portable reference kernels. SIMD kernels are tested against these.

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

#[inline]
pub fn l2_sq_i8(a: &[i8], b: &[i8]) -> u32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = i32::from(x) - i32::from(y);
            (d * d) as u32
        })
        .sum()
}

/// Inner product of an f32 vector and an int8 vector.
#[inline]
pub fn dot_f32_i8(a: &[f32], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter().zip(b).map(|(x, &y)| x * f32::from(y)).sum()
}

/// Squared Euclidean distance between an f32 vector and an int8 vector.
#[inline]
pub fn l2_sq_f32_i8(a: &[f32], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, &y)| {
            let d = x - f32::from(y);
            d * d
        })
        .sum()
}

/// See [`super::Kernels::binary_ip4`].
#[inline]
pub fn binary_ip4(code: &[u64], planes: &[u64]) -> u32 {
    let w = code.len();
    assert_eq!(planes.len(), 4 * w, "four planes per code");
    (0..4)
        .map(|j| {
            let ones: u32 = code
                .iter()
                .zip(&planes[j * w..(j + 1) * w])
                .map(|(c, p)| (c & p).count_ones())
                .sum();
            ones << j
        })
        .sum()
}
