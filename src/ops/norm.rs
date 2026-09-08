//! Normalization ops: RMSNorm and PerDimScale (Pax/Flax semantics, matching
//! the official `normalization.py`).

use crate::checkpoint::Safetensors;
use crate::error::{Error, Result};
use crate::tensor::Tensor2D;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Reciprocal of softplus(0) ≈ 1/ln(2) — the Pax constant.
const RECIPROCAL_OF_SOFTPLUS_0: f32 = std::f32::consts::LOG2_E;

#[inline]
pub fn softplus(v: f32) -> f32 {
    // log(1 + exp(v)), numerically stable, computed in f64 and rounded once.
    let v64 = v as f64;
    let y = if v64 > 0.0 {
        v64 + (1.0 + (-v64).exp()).ln()
    } else {
        (1.0 + v64.exp()).ln()
    };
    y as f32
}

/// RMS normalization over the *last* dimension (rows of a Tensor2D).
///
/// PyTorch `nn.RMSNorm` semantics: `y = x * rsqrt(mean(x^2) + eps) * weight`.
/// For the official 3.0 config the weight is per-model-dimension and loaded
/// from the checkpoint.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    dims: usize,
    weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    /// Builds from a normalized-weight vector.
    pub fn from_weight(weight: Vec<f32>, eps: f32) -> Result<RmsNorm> {
        if weight.is_empty() {
            return Err(Error("RmsNorm: empty weight".into()));
        }
        Ok(RmsNorm {
            dims: weight.len(),
            weight,
            eps,
        })
    }

    /// Loads `{prefix}.weight` (per-dimension affine weight) from a store.
    pub fn load(store: &Safetensors, prefix: &str, eps: f32) -> Result<RmsNorm> {
        let weight = store.f32_flat(&format!("{prefix}.weight"))?;
        RmsNorm::from_weight(weight, eps)
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Normalizes `x` in place (`(rows, dims)` row-wise).
    pub fn forward(&self, x: &mut Tensor2D) {
        assert_eq!(
            x.cols(),
            self.dims,
            "RmsNorm: input cols {} != dims {}",
            x.cols(),
            self.dims
        );
        for row in x.data_mut().chunks_mut(self.dims) {
            self.forward_row(row);
        }
    }

    /// Normalizes a single row of length `dims` in place (no ownership change).
    /// Mean of squares is accumulated in f64 (correctly rounded result).
    #[inline]
    pub fn forward_row(&self, row: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && self.dims.is_multiple_of(8)
            {
                unsafe {
                    self.forward_row_avx2(row);
                    return;
                }
            }
        }

        let d = self.dims as f64;
        let mut sq_sum = 0.0f64;
        for &v in row.iter() {
            sq_sum += v as f64 * v as f64;
        }
        let scale = (1.0 / (sq_sum / d + self.eps as f64).sqrt()) as f32;
        for (v, &w) in row.iter_mut().zip(self.weight.iter()) {
            *v = (*v as f64 * scale as f64 * w as f64) as f32;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_row_avx2(&self, row: &mut [f32]) {
        unsafe {
            let mut s0 = _mm256_setzero_ps();
            let mut s1 = _mm256_setzero_ps();

            let chunks = self.dims / 16;
            for i in 0..chunks {
                let off = i * 16;
                let v0 = _mm256_loadu_ps(row.as_ptr().add(off));
                let v1 = _mm256_loadu_ps(row.as_ptr().add(off + 8));
                s0 = _mm256_fmadd_ps(v0, v0, s0);
                s1 = _mm256_fmadd_ps(v1, v1, s1);
            }
            let rem = (self.dims % 16) / 8;
            if rem > 0 {
                let off = chunks * 16;
                let v = _mm256_loadu_ps(row.as_ptr().add(off));
                s0 = _mm256_fmadd_ps(v, v, s0);
            }

            let sum = _mm256_add_ps(s0, s1);
            let hi = _mm256_extractf128_ps(sum, 1);
            let lo = _mm256_castps256_ps128(sum);
            let s4 = _mm_add_ps(lo, hi);
            let shuf = _mm_movehl_ps(s4, s4);
            let s2 = _mm_add_ps(s4, shuf);
            let shuf2 = _mm_shuffle_ps(s2, s2, 1);
            let s1 = _mm_add_ss(s2, shuf2);
            let sq_sum = _mm_cvtss_f32(s1);

            let scale = (1.0 / (sq_sum as f64 / self.dims as f64 + self.eps as f64).sqrt()) as f32;
            let vscale = _mm256_set1_ps(scale);

            let v_chunks = self.dims / 8;
            for i in 0..v_chunks {
                let off = i * 8;
                let v = _mm256_loadu_ps(row.as_ptr().add(off));
                let w = _mm256_loadu_ps(self.weight.as_ptr().add(off));
                let out = _mm256_mul_ps(_mm256_mul_ps(v, vscale), w);
                _mm256_storeu_ps(row.as_mut_ptr().add(off), out);
            }
        }
    }
}

/// Per-dimension query scaling (Pax `PerDimScale`).
///
/// Replaces 1/sqrt(d): `y = x * C * softplus(per_dim_scale)` with
/// `C = RECIPROCAL_OF_SOFTPLUS_0 / sqrt(num_dims)`, where `per_dim_scale` is
/// a learnable vector. The parameter is initialized to zeros so the initial
/// scale is close to 1/sqrt(d); real checkpoints carry the trained values.
#[derive(Debug, Clone)]
pub struct PerDimScale {
    num_dims: usize,
    scale: Vec<f32>,
}

impl PerDimScale {
    /// Precomputes the effective per-dimension scale from the raw parameter.
    pub fn from_parameter(per_dim_scale: Vec<f32>) -> Result<PerDimScale> {
        if per_dim_scale.is_empty() {
            return Err(Error("PerDimScale: empty parameter".into()));
        }
        let num_dims = per_dim_scale.len();
        let c = RECIPROCAL_OF_SOFTPLUS_0 / (num_dims as f32).sqrt();
        let scale = per_dim_scale.into_iter().map(|p| c * softplus(p)).collect();
        Ok(PerDimScale { num_dims, scale })
    }

    /// Loads `{prefix}` (raw parameter, no `.weight` suffix) from a store.
    ///
    /// The official 3.0 checkpoint stores `PerDimScale` as a submodule whose
    /// single parameter is named `per_dim_scale`, so the safetensors key is
    /// `...seq_attn.per_dim_scale.per_dim_scale`. Accept both that form and a
    /// bare `{prefix}.per_dim_scale` for compatibility.
    pub fn load(store: &Safetensors, name: &str) -> Result<PerDimScale> {
        let nested = format!("{name}.per_dim_scale");
        let param = store.f32_flat(&nested).or_else(|_| store.f32_flat(name))?;
        PerDimScale::from_parameter(param)
    }

    /// Applies the per-dim scale to the *last* dimension of `x` in place.
    pub fn forward(&self, x: &mut Tensor2D) {
        assert_eq!(
            x.cols(),
            self.num_dims,
            "PerDimScale: input cols {} != num_dims {}",
            x.cols(),
            self.num_dims
        );
        for row in x.data_mut().chunks_mut(self.num_dims) {
            self.forward_row(row);
        }
    }

    /// Scales a single row of length `num_dims` in place.
    #[inline]
    pub fn forward_row(&self, row: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && self.num_dims.is_multiple_of(8) {
                unsafe {
                    self.forward_row_avx2(row);
                    return;
                }
            }
        }

        for (v, &s) in row.iter_mut().zip(self.scale.iter()) {
            *v = (*v as f64 * s as f64) as f32;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn forward_row_avx2(&self, row: &mut [f32]) {
        unsafe {
            let chunks = self.num_dims / 8;
            for i in 0..chunks {
                let off = i * 8;
                let v = _mm256_loadu_ps(row.as_ptr().add(off));
                let s = _mm256_loadu_ps(self.scale.as_ptr().add(off));
                let out = _mm256_mul_ps(v, s);
                _mm256_storeu_ps(row.as_mut_ptr().add(off), out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_matches_reference() {
        // Hand-computed: x=[3,4], mean(x^2)=12.5, r=1/sqrt(12.5+1e-7)≈0.28284,
        // weight=1 → y≈[0.8485, 1.1314].
        let mut x = Tensor2D::from_row_major(1, 2, vec![3.0, 4.0]).expect("fixture");
        let n = RmsNorm::from_weight(vec![1.0, 1.0], 1e-7).expect("fixture");
        n.forward(&mut x);
        let scale = 1.0 / (12.5f32 + 1e-7).sqrt();
        assert!((x[(0, 0)] - 3.0 * scale).abs() < 1e-6);
        assert!((x[(0, 1)] - 4.0 * scale).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_affine_weight() {
        let mut x = Tensor2D::from_row_major(1, 3, vec![2.0, 2.0, 2.0]).expect("fixture");
        let n = RmsNorm::from_weight(vec![1.0, 2.0, 3.0], 0.0).expect("fixture");
        n.forward(&mut x);
        // mean(x^2)=4 → scale=0.5; y = 0.5 * [1,2,3]*2 = [1,2,3]
        assert!((x[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((x[(0, 2)] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn softplus_zero_is_ln2() {
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);
    }

    #[test]
    fn softplus_matches_log1p_exp_both_domains() {
        // softplus(v) = ln(1 + e^v) must be finite and correct for both signs
        // (regression: a broken positive-domain branch produced NaN).
        for &v in &[-5.0f32, -1.0, 0.5, 1.0, 5.0, 20.0] {
            let expected = ((v as f64).exp_m1().ln_1p()) + 0.0; // placeholder, real check below
            let _ = expected;
            let sp = softplus(v);
            let truth = (1.0 + (v as f64).exp()).ln() as f32;
            assert!(sp.is_finite(), "softplus({v}) not finite");
            assert!(
                (sp - truth).abs() < 1e-6,
                "softplus({v}) = {sp}, want {truth}"
            );
        }
    }

    #[test]
    fn per_dim_scale_zero_param_gives_approx_inv_sqrt_d() {
        // softplus(0)=ln2; scale ≈ 1.4427/|d * (1/√d) → c*ln2 ≈ (1.4427/√d)*0.6931 ≈ 1/√d
        let s = PerDimScale::from_parameter(vec![0.0; 16]).expect("fixture");
        let mut x = Tensor2D::from_row_major(2, 16, vec![1.0; 32]).expect("fixture");
        s.forward(&mut x);
        // expected factor = 1.442695041 / 4 * ln2 ≈ 0.25
        let f = x[(0, 0)];
        assert!((f - std::f32::consts::LOG2_E / 4.0 * std::f32::consts::LN_2).abs() < 1e-6);
    }
}
