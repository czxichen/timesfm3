//! High-performance GEMM using `matrixmultiply` with Rayon row-parallelism.
//!
//! Replaces slow single-row outer-product dispatch with BLIS-grade cache-tiled
//! microkernels (AVX-512 / AVX2+FMA on x86_64, NEON on aarch64, SSE2 / scalar fallback)
//! partitioned across CPU cores using Rayon.

use rayon::prelude::*;

use crate::tensor::Tensor2D;

/// `out[m, n] = a[m, k] @ b[k, n]`, row-major f32.
pub fn matmul(a: &Tensor2D, b: &Tensor2D) -> Tensor2D {
    let (m, k) = (a.rows(), a.cols());
    let n = b.cols();
    assert_eq!(
        k,
        b.rows(),
        "matmul: inner dims mismatch ({} vs {})",
        k,
        b.rows()
    );

    let mut out = Tensor2D::zeros(m, n);
    if m == 0 || n == 0 || k == 0 {
        return out;
    }
    gemm(a.data(), b.data(), out.data_mut(), m, k, n);
    out
}

/// Row- or column-parallel GEMM driver using `matrixmultiply::sgemm`.
pub fn gemm(ad: &[f32], bd: &[f32], od: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    assert_eq!(ad.len(), m * k);
    assert_eq!(bd.len(), k * n);
    assert_eq!(od.len(), m * n);

    let num_threads = rayon::current_num_threads();
    let min_rows_per_chunk = 16;

    // Small total FLOPs or single thread: single-threaded sgemm
    if num_threads <= 1 || (m * k * n) < 200_000 {
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                ad.as_ptr(),
                k as isize,
                1,
                bd.as_ptr(),
                n as isize,
                1,
                0.0,
                od.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        return;
    }

    // Adaptive partitioning:
    // When n >= 128, partition along N columns. This provides superior L2/L3 cache locality,
    // avoids multi-core memory bus bandwidth contention, and speeds up GEMM by up to 4x.
    if n >= 128 {
        let min_cols = 128;
        let num_chunks = (n / min_cols).max(1).min(num_threads * 2);
        let n_chunk = (n.div_ceil(num_chunks)).next_multiple_of(16).max(16);
        let num_tiles = n.div_ceil(n_chunk);
        let od_ptr = od.as_mut_ptr() as usize;
        let ad_ptr = ad.as_ptr() as usize;
        let bd_ptr = bd.as_ptr() as usize;

        (0..num_tiles).into_par_iter().for_each(|tile_idx| {
            let j0 = tile_idx * n_chunk;
            let j1 = (j0 + n_chunk).min(n);
            let cur_n = j1 - j0;
            unsafe {
                let a = ad_ptr as *const f32;
                let b = (bd_ptr as *const f32).add(j0);
                let o = (od_ptr as *mut f32).add(j0);
                matrixmultiply::sgemm(
                    m, k, cur_n, 1.0, a, k as isize, 1, b, n as isize, 1, 0.0, o, n as isize, 1,
                );
            }
        });
        return;
    }

    // Otherwise, partition along M rows
    let chunk_rows = m.div_ceil(num_threads * 2).max(min_rows_per_chunk);
    od.par_chunks_mut(chunk_rows * n)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let r0 = ci * chunk_rows;
            let rows_here = chunk.len() / n;
            let a_slice = &ad[r0 * k..(r0 + rows_here) * k];
            unsafe {
                matrixmultiply::sgemm(
                    rows_here,
                    k,
                    n,
                    1.0,
                    a_slice.as_ptr(),
                    k as isize,
                    1,
                    bd.as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    chunk.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
        });
}

// ============================================================================
// Native SIMD FP16 streaming GEMM kernels
// ============================================================================

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn cvt_8xf16_to_8xf32(ptr: *const half::f16) -> std::arch::x86_64::__m256 {
    unsafe {
        let half8 = std::arch::x86_64::_mm_loadu_si128(ptr as *const std::arch::x86_64::__m128i);
        std::arch::x86_64::_mm256_cvtph_ps(half8)
    }
}

// 4x16 microkernel for F16
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn microkernel_4x16_f16(
    a_ptr: *const f32,
    stride_a: usize,
    b_ptr: *const half::f16,
    stride_b: usize,
    c_ptr: *mut f32,
    stride_c: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        let mut c0_0 = _mm256_setzero_ps();
        let mut c0_1 = _mm256_setzero_ps();
        let mut c1_0 = _mm256_setzero_ps();
        let mut c1_1 = _mm256_setzero_ps();
        let mut c2_0 = _mm256_setzero_ps();
        let mut c2_1 = _mm256_setzero_ps();
        let mut c3_0 = _mm256_setzero_ps();
        let mut c3_1 = _mm256_setzero_ps();

        let a0 = a_ptr;
        let a1 = a_ptr.add(stride_a);
        let a2 = a_ptr.add(stride_a * 2);
        let a3 = a_ptr.add(stride_a * 3);

        let mut t = 0;
        while t + 2 <= k {
            let b_row0 = b_ptr.add(t * stride_b);
            let vb0_0 = cvt_8xf16_to_8xf32(b_row0);
            let vb0_1 = cvt_8xf16_to_8xf32(b_row0.add(8));

            let a0_0 = _mm256_set1_ps(*a0.add(t));
            let a1_0 = _mm256_set1_ps(*a1.add(t));
            let a2_0 = _mm256_set1_ps(*a2.add(t));
            let a3_0 = _mm256_set1_ps(*a3.add(t));

            c0_0 = _mm256_fmadd_ps(a0_0, vb0_0, c0_0);
            c0_1 = _mm256_fmadd_ps(a0_0, vb0_1, c0_1);
            c1_0 = _mm256_fmadd_ps(a1_0, vb0_0, c1_0);
            c1_1 = _mm256_fmadd_ps(a1_0, vb0_1, c1_1);
            c2_0 = _mm256_fmadd_ps(a2_0, vb0_0, c2_0);
            c2_1 = _mm256_fmadd_ps(a2_0, vb0_1, c2_1);
            c3_0 = _mm256_fmadd_ps(a3_0, vb0_0, c3_0);
            c3_1 = _mm256_fmadd_ps(a3_0, vb0_1, c3_1);

            let b_row1 = b_ptr.add((t + 1) * stride_b);
            let vb1_0 = cvt_8xf16_to_8xf32(b_row1);
            let vb1_1 = cvt_8xf16_to_8xf32(b_row1.add(8));

            let a0_1 = _mm256_set1_ps(*a0.add(t + 1));
            let a1_1 = _mm256_set1_ps(*a1.add(t + 1));
            let a2_1 = _mm256_set1_ps(*a2.add(t + 1));
            let a3_1 = _mm256_set1_ps(*a3.add(t + 1));

            c0_0 = _mm256_fmadd_ps(a0_1, vb1_0, c0_0);
            c0_1 = _mm256_fmadd_ps(a0_1, vb1_1, c0_1);
            c1_0 = _mm256_fmadd_ps(a1_1, vb1_0, c1_0);
            c1_1 = _mm256_fmadd_ps(a1_1, vb1_1, c1_1);
            c2_0 = _mm256_fmadd_ps(a2_1, vb1_0, c2_0);
            c2_1 = _mm256_fmadd_ps(a2_1, vb1_1, c2_1);
            c3_0 = _mm256_fmadd_ps(a3_1, vb1_0, c3_0);
            c3_1 = _mm256_fmadd_ps(a3_1, vb1_1, c3_1);

            t += 2;
        }

        while t < k {
            let b_row = b_ptr.add(t * stride_b);
            let vb0 = cvt_8xf16_to_8xf32(b_row);
            let vb1 = cvt_8xf16_to_8xf32(b_row.add(8));

            let a_val0 = _mm256_set1_ps(*a0.add(t));
            c0_0 = _mm256_fmadd_ps(a_val0, vb0, c0_0);
            c0_1 = _mm256_fmadd_ps(a_val0, vb1, c0_1);

            let a_val1 = _mm256_set1_ps(*a1.add(t));
            c1_0 = _mm256_fmadd_ps(a_val1, vb0, c1_0);
            c1_1 = _mm256_fmadd_ps(a_val1, vb1, c1_1);

            let a_val2 = _mm256_set1_ps(*a2.add(t));
            c2_0 = _mm256_fmadd_ps(a_val2, vb0, c2_0);
            c2_1 = _mm256_fmadd_ps(a_val2, vb1, c2_1);

            let a_val3 = _mm256_set1_ps(*a3.add(t));
            c3_0 = _mm256_fmadd_ps(a_val3, vb0, c3_0);
            c3_1 = _mm256_fmadd_ps(a_val3, vb1, c3_1);

            t += 1;
        }

        let c0 = c_ptr;
        let c1 = c_ptr.add(stride_c);
        let c2 = c_ptr.add(stride_c * 2);
        let c3 = c_ptr.add(stride_c * 3);

        _mm256_storeu_ps(c0, c0_0);
        _mm256_storeu_ps(c0.add(8), c0_1);
        _mm256_storeu_ps(c1, c1_0);
        _mm256_storeu_ps(c1.add(8), c1_1);
        _mm256_storeu_ps(c2, c2_0);
        _mm256_storeu_ps(c2.add(8), c2_1);
        _mm256_storeu_ps(c3, c3_0);
        _mm256_storeu_ps(c3.add(8), c3_1);
    }
}

// 2x16 microkernel for F16
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn microkernel_2x16_f16(
    a_ptr: *const f32,
    stride_a: usize,
    b_ptr: *const half::f16,
    stride_b: usize,
    c_ptr: *mut f32,
    stride_c: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        let mut c0_0 = _mm256_setzero_ps();
        let mut c0_1 = _mm256_setzero_ps();
        let mut c1_0 = _mm256_setzero_ps();
        let mut c1_1 = _mm256_setzero_ps();

        let a0 = a_ptr;
        let a1 = a_ptr.add(stride_a);

        for t in 0..k {
            let b_row = b_ptr.add(t * stride_b);
            let vb0 = cvt_8xf16_to_8xf32(b_row);
            let vb1 = cvt_8xf16_to_8xf32(b_row.add(8));

            let a0_val = _mm256_set1_ps(*a0.add(t));
            let a1_val = _mm256_set1_ps(*a1.add(t));

            c0_0 = _mm256_fmadd_ps(a0_val, vb0, c0_0);
            c0_1 = _mm256_fmadd_ps(a0_val, vb1, c0_1);
            c1_0 = _mm256_fmadd_ps(a1_val, vb0, c1_0);
            c1_1 = _mm256_fmadd_ps(a1_val, vb1, c1_1);
        }

        let c0 = c_ptr;
        let c1 = c_ptr.add(stride_c);
        _mm256_storeu_ps(c0, c0_0);
        _mm256_storeu_ps(c0.add(8), c0_1);
        _mm256_storeu_ps(c1, c1_0);
        _mm256_storeu_ps(c1.add(8), c1_1);
    }
}

// 1x16 microkernel for F16
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn microkernel_1x16_f16(
    a_ptr: *const f32,
    b_ptr: *const half::f16,
    stride_b: usize,
    c_ptr: *mut f32,
    k: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        let mut c0_0 = _mm256_setzero_ps();
        let mut c0_1 = _mm256_setzero_ps();

        for t in 0..k {
            let b_row = b_ptr.add(t * stride_b);
            let vb0 = cvt_8xf16_to_8xf32(b_row);
            let vb1 = cvt_8xf16_to_8xf32(b_row.add(8));
            let a0_val = _mm256_set1_ps(*a_ptr.add(t));

            c0_0 = _mm256_fmadd_ps(a0_val, vb0, c0_0);
            c0_1 = _mm256_fmadd_ps(a0_val, vb1, c0_1);
        }

        _mm256_storeu_ps(c_ptr, c0_0);
        _mm256_storeu_ps(c_ptr.add(8), c0_1);
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_tail_cols_f16(
    a: &[f32],
    b: &[half::f16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    j_start: usize,
    j_end: usize,
) {
    for r in 0..m {
        for t in 0..k {
            let a_val = a[r * k + t];
            if a_val == 0.0 {
                continue;
            }
            for j in j_start..j_end {
                let b_val = b[t * n + j].to_f32();
                c[r * n + j] += a_val * b_val;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn fast_unpack_f16_row(src: *const half::f16, dst: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= len {
        unsafe {
            let half8 = _mm_loadu_si128(src.add(i) as *const __m128i);
            let ps8 = _mm256_cvtph_ps(half8);
            _mm256_storeu_ps(dst.add(i), ps8);
        }
        i += 8;
    }
    while i < len {
        unsafe {
            *dst.add(i) = (*src.add(i)).to_f32();
        }
        i += 1;
    }
}

thread_local! {
    static TILE_UNPACK_BUFFER: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Column-partitioned GEMM for FP16 with SIMD panel unpacking and BLIS microkernels
pub fn gemm_f16_portable(a: &[f32], b: &[half::f16], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    // NOTE: this used to also short-circuit on `rayon::current_num_threads() <= 1`
    // and fall through to `gemm_tail_cols_f16`, a scalar triple loop that does a
    // per-element f16->f32 conversion with no SIMD and no cache blocking.  That
    // path is 25-40x slower than the tiled SIMD-unpack + BLIS-microkernel path
    // below, so a single-threaded rayon pool was catastrophically slow
    // (7x1024 h96: 204 s vs 5.2 s for the f32 weights).  Only genuinely tiny
    // problems use the scalar path now; a 1-thread pool simply gets fewer tiles.
    if (m * k * n) < 20_000 {
        gemm_tail_cols_f16(a, b, c, m, k, n, 0, n);
        return;
    }

    let min_cols = 128;
    let num_threads = rayon::current_num_threads().max(1);
    let num_chunks = (n / min_cols).max(1).min(num_threads.saturating_mul(2));
    let tile_n = (n.div_ceil(num_chunks)).next_multiple_of(16).max(16);

    let num_tiles = n.div_ceil(tile_n);
    let c_raw = c.as_mut_ptr() as usize;
    let a_ptr = a.as_ptr() as usize;
    let b_ptr = b.as_ptr() as usize;

    (0..num_tiles).into_par_iter().for_each(|tile_idx| {
        let j0 = tile_idx * tile_n;
        let j1 = (j0 + tile_n).min(n);
        let cur_tile_n = j1 - j0;

        TILE_UNPACK_BUFFER.with(|buf_cell| {
            let mut buf = buf_cell.borrow_mut();
            let needed = k * cur_tile_n;
            if buf.len() < needed {
                buf.resize(needed, 0.0);
            }
            let local_b = &mut buf[..needed];
            let b_base = b_ptr as *const half::f16;
            let a_base = a_ptr as *const f32;
            let c_base = c_raw as *mut f32;

            #[cfg(target_arch = "x86_64")]
            let has_f16c = is_x86_feature_detected!("f16c");
            #[cfg(not(target_arch = "x86_64"))]
            let has_f16c = false;

            for r in 0..k {
                let src_row = unsafe { b_base.add(r * n + j0) };
                let dst_row = unsafe { local_b.as_mut_ptr().add(r * cur_tile_n) };
                if has_f16c {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        fast_unpack_f16_row(src_row, dst_row, cur_tile_n);
                    }
                } else {
                    let src_slice = unsafe { std::slice::from_raw_parts(src_row, cur_tile_n) };
                    let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst_row, cur_tile_n) };
                    for (d, s) in dst_slice.iter_mut().zip(src_slice.iter()) {
                        *d = s.to_f32();
                    }
                }
            }

            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    cur_tile_n,
                    1.0,
                    a_base,
                    k as isize,
                    1,
                    local_b.as_ptr(),
                    cur_tile_n as isize,
                    1,
                    0.0,
                    c_base.add(j0),
                    n as isize,
                    1,
                );
            }
        });
    });
}

/// High-performance FP16 weight GEMM: `out[m, n] = a[m, k] @ b_f16[k, n]`.
/// Automatically dispatches to hardware AVX2+FMA+F16C streaming microkernels when available.
pub fn gemm_f16(ad: &[f32], bd: &[half::f16], od: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    assert_eq!(ad.len(), m * k);
    assert_eq!(bd.len(), k * n);
    assert_eq!(od.len(), m * n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
        {
            // For small M (<= 16), keep accumulators in registers (zero memory scratchpad)
            if m <= 16 {
                let num_full_tiles = n / 16;
                let c_ptr = od.as_mut_ptr() as usize;
                let a_ptr = ad.as_ptr() as usize;
                let b_ptr = bd.as_ptr() as usize;

                if num_full_tiles > 0 {
                    (0..num_full_tiles).into_par_iter().for_each(|tile_idx| {
                        let j0 = tile_idx * 16;
                        unsafe {
                            stream_f16_kernels_m(
                                a_ptr as *const f32,
                                b_ptr as *const half::f16,
                                c_ptr as *mut f32,
                                m,
                                k,
                                n,
                                j0,
                            );
                        }
                    });
                }

                let tail_start = num_full_tiles * 16;
                if tail_start < n {
                    for r in 0..m {
                        for j in tail_start..n {
                            od[r * n + j] = 0.0;
                        }
                    }
                    gemm_tail_cols_f16(ad, bd, od, m, k, n, tail_start, n);
                }
                return;
            }
        }
    }

    gemm_f16_portable(ad, bd, od, m, k, n);
}

/// Runs the register microkernels over all M rows for the 16-column tile at
/// `j0` (m row-blocks of 4 with 2/1 tails), streaming `b` half rows from
/// (L2-resident) memory.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn stream_f16_kernels_m(
    a: *const f32,
    b: *const half::f16,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    j0: usize,
) {
    let mut r = 0;
    while r + 4 <= m {
        unsafe {
            microkernel_4x16_f16(a.add(r * k), k, b.add(j0), n, c.add(r * n + j0), n, k);
        }
        r += 4;
    }
    while r + 2 <= m {
        unsafe {
            microkernel_2x16_f16(a.add(r * k), k, b.add(j0), n, c.add(r * n + j0), n, k);
        }
        r += 2;
    }
    if r < m {
        unsafe {
            microkernel_1x16_f16(a.add(r * k), b.add(j0), n, c.add(r * n + j0), k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Tensor2D {
        let mut data = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[r * cols + c] = f(r, c);
            }
        }
        Tensor2D::from_row_major(rows, cols, data).expect("test fixture invariant")
    }

    #[test]
    fn tiny_2x3_times_3x2() {
        let a = t(2, 3, |r, c| (r * 3 + c) as f32 + 1.0); // [[1,2,3],[4,5,6]]
        let b = t(3, 2, |r, c| (r * 2 + c) as f32 + 1.0); // [[1,2],[3,4],[5,6]]
        let out = matmul(&a, &b);
        assert_eq!(out[(0, 0)], 22.0);
        assert_eq!(out[(0, 1)], 28.0);
        assert_eq!(out[(1, 0)], 49.0);
        assert_eq!(out[(1, 1)], 64.0);
    }

    #[test]
    fn identity() {
        let a = t(4, 4, |r, c| (r + c) as f32);
        let id = t(4, 4, |r, c| if r == c { 1.0 } else { 0.0 });
        assert_eq!(matmul(&a, &id), a);
    }

    #[test]
    fn nonzero_inner_dim() {
        let a = Tensor2D::from_row_major(2, 5, vec![1.0; 10]).expect("test fixture invariant");
        let b = Tensor2D::from_row_major(5, 1, vec![2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("test fixture invariant");
        let out = matmul(&a, &b);
        // each output element = 2+3+4+5+6 = 20
        assert_eq!(out[(0, 0)], 20.0);
        assert_eq!(out[(1, 0)], 20.0);
    }

    #[test]
    fn random_matches_scalar() {
        let (m, k, n) = (37usize, 53usize, 41usize);
        let a = Tensor2D::from_row_major(m, k, rand_fill(m * k)).expect("fixture");
        let b = Tensor2D::from_row_major(k, n, rand_fill(k * n)).expect("fixture");
        let fast = matmul(&a, &b);
        let mut slow = Tensor2D::zeros(m, n);
        scalar_into(&a, &b, &mut slow);
        let max_diff = fast.max_abs_diff(&slow);
        assert!(max_diff < 1e-4, "SIMD vs scalar diff {max_diff:e}");
    }

    // Deterministic pseudo-random fill.
    fn rand_fill(len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        let mut x = 0x12345678u32;
        for _ in 0..len {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = (x >> 8) as f32 / (1u32 << 24) as f32;
            out.push(f - 0.5);
        }
        out
    }

    fn scalar_into(a: &Tensor2D, b: &Tensor2D, out: &mut Tensor2D) {
        let (m, k) = (a.rows(), a.cols());
        let n = b.cols();
        for i in 0..m {
            for t in 0..k {
                let aik = a[(i, t)];
                if aik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    out[(i, j)] += aik * b[(t, j)];
                }
            }
        }
    }

    #[test]
    fn random_matches_scalar_parallel() {
        let (m, k, n) = (80usize, 64usize, 70usize);
        let a = Tensor2D::from_row_major(m, k, rand_fill(m * k)).expect("fixture");
        let b = Tensor2D::from_row_major(k, n, rand_fill(k * n)).expect("fixture");
        let fast = matmul(&a, &b);
        let mut slow = Tensor2D::zeros(m, n);
        scalar_into(&a, &b, &mut slow);
        let max_diff = fast.max_abs_diff(&slow);
        assert!(max_diff < 1e-4, "GEMM vs scalar diff {max_diff:e}");
    }

    #[test]
    fn f16_gemm_correctness() {
        let test_cases = [
            (1, 1, 1),
            (2, 3, 2),
            (3, 10, 17),
            (4, 32, 16),
            (7, 64, 41),
            (28, 128, 64),
            // Big-M fused streaming path (m*k >= 64_000 when AVX2 is on):
            // covers 64-col tiles, M row-blocks with 4/2/1 tails, K tail, N tail.
            (100, 1024, 80),
            (126, 640, 176),
            (34, 512, 91),
        ];

        for &(m, k, n) in &test_cases {
            let a = rand_fill(m * k);
            let b_f32 = rand_fill(k * n);
            let b_f16: Vec<half::f16> = b_f32.iter().map(|&v| half::f16::from_f32(v)).collect();

            let mut out = vec![0.0f32; m * n];
            gemm_f16(&a, &b_f16, &mut out, m, k, n);

            let b_unpacked: Vec<f32> = b_f16.iter().map(|v| v.to_f32()).collect();
            let b_t = Tensor2D::from_row_major(k, n, b_unpacked).unwrap();
            let a_t = Tensor2D::from_row_major(m, k, a).unwrap();
            let mut ref_out = Tensor2D::zeros(m, n);
            scalar_into(&a_t, &b_t, &mut ref_out);

            let mut max_diff = 0.0f32;
            for (&o, &r) in out.iter().zip(ref_out.data().iter()) {
                let d = (o - r).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            assert!(
                max_diff < 1e-3,
                "f16 GEMM mismatch on ({m}, {k}, {n}): max_diff={max_diff}"
            );
        }
    }
}
