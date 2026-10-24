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
