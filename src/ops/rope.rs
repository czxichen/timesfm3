//! Rotary positional embedding (RoPE), matching `RotaryPositionalEmbedding`
//! in the official transformer.py.

use crate::tensor::Tensor2D;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Stateless RoPE — no learnable parameters (timescale table is derived from
/// the fixed min/max schedule).
#[derive(Debug, Clone)]
pub struct Rope {
    dims: usize,
    /// Precomputed per-half-dim timescale: ts[k] = min * (max/min)^(2k/dims).
    timescale: Vec<f64>,
    /// Precomputed cos/sin tables for positions [0..max_precomputed).
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
    max_precomputed: usize,
}

impl Rope {
    pub fn new(dims: usize, min_timescale: f32, max_timescale: f32) -> Rope {
        let half = dims / 2;
        let mut timescale = Vec::with_capacity(half);
        let (min64, max64) = (min_timescale as f64, max_timescale as f64);
        for k in 0..half {
            let fraction = 2.0 * k as f64 / dims as f64;
            timescale.push(min64 * (max64 / min64).powf(fraction));
        }

        let max_precomputed = 512;
        let mut cos_table = Vec::with_capacity(max_precomputed * half);
        let mut sin_table = Vec::with_capacity(max_precomputed * half);

        for pos in 0..max_precomputed {
            for &ts in &timescale {
                let angle = pos as f64 / ts;
                let (sin, cos) = angle.sin_cos();
                sin_table.push(sin as f32);
                cos_table.push(cos as f32);
            }
        }

        Rope {
            dims,
            timescale,
            cos_table,
            sin_table,
            max_precomputed,
        }
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Applies RoPE to the last `dims` columns of `x` in place, using
    /// `position` (i32 per row). If `position` is empty, uses `0..rows`.
    pub fn forward(&self, x: &mut Tensor2D, position: &[i32]) {
        assert_eq!(
            x.cols(),
            self.dims,
            "RoPE: input cols {} != dims {}",
            x.cols(),
            self.dims
        );
        for (r, row) in x.data_mut().chunks_mut(self.dims).enumerate() {
            let pos = if position.is_empty() {
                r as i32
            } else {
                position[r]
            };
            self.rotate(row, pos);
        }
    }

    /// Rotates a single row of length `dims` in place.
    #[inline]
    pub fn rotate(&self, row: &mut [f32], pos: i32) {
        let half = self.dims / 2;
        if pos >= 0 && (pos as usize) < self.max_precomputed {
            let p = pos as usize;
            let offset = p * half;
            let cos = &self.cos_table[offset..offset + half];
            let sin = &self.sin_table[offset..offset + half];

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && half == 40
                {
                    unsafe {
                        self.rotate_avx2_40(row.as_mut_ptr(), cos.as_ptr(), sin.as_ptr());
                        return;
                    }
                }
            }

            for k in 0..half {
                let c = cos[k];
                let s = sin[k];
                let a = row[k];
                let b = row[k + half];
                row[k] = a * c - b * s;
                row[k + half] = b * c + a * s;
            }
            return;
        }

        // Fallback for positions outside precomputed table or negative
        for k in 0..half {
            let angle = pos as f64 / self.timescale[k];
            let (sin, cos) = angle.sin_cos();
            let a = row[k] as f64;
            let b = row[k + half] as f64;
            row[k] = (a * cos - b * sin) as f32;
            row[k + half] = (b * cos + a * sin) as f32;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn rotate_avx2_40(&self, row: *mut f32, cos: *const f32, sin: *const f32) {
        unsafe {
            for i in 0..5 {
                let idx = i * 8;
                let va = _mm256_loadu_ps(row.add(idx));
                let vb = _mm256_loadu_ps(row.add(idx + 40));
                let vc = _mm256_loadu_ps(cos.add(idx));
                let vs = _mm256_loadu_ps(sin.add(idx));

                // new_a = va * vc - vb * vs
                let new_a = _mm256_fmsub_ps(va, vc, _mm256_mul_ps(vb, vs));
                // new_b = vb * vc + va * vs
                let new_b = _mm256_fmadd_ps(vb, vc, _mm256_mul_ps(va, vs));

                _mm256_storeu_ps(row.add(idx), new_a);
                _mm256_storeu_ps(row.add(idx + 40), new_b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_position_is_identity() {
        let mut x = Tensor2D::from_row_major(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("fixture");
        let rope = Rope::new(4, 1.0, 10000.0);
        let pos = vec![0i32, 0];
        rope.forward(&mut x, &pos);
        // cos(0)=1, sin(0)=0 → identity.
        assert_eq!(x.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn preserves_norm() {
        let mut x =
            Tensor2D::from_row_major(1, 8, vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
                .expect("fixture");
        let rope = Rope::new(8, 1.0, 10000.0);
        let pos = vec![3i32];
        rope.forward(&mut x, &pos);
        // RoPE is an orthogonal transform per pair → norm preserved.
        let norm: f32 = x.data().iter().map(|v| v * v).sum();
        let orig: f32 = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]
            .iter()
            .map(|v| v * v)
            .sum();
        assert!((norm - orig).abs() < 1e-4);
    }
}
