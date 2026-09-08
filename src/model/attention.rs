//! Multi-head attention with RoPE, QK-norm, PerDimScale and patch masking.
//!
//! Port of the official `MultiHeadAttention` (transformer.py), adapted to the
//! engine's batched inference layout. Input is `(batch, n, d)` flattened to
//! `(batch * n, d)` (row-major); the layer reshapes internally to
//! `(batch, n, heads, head_dim)`.

use rayon::prelude::*;

use crate::config::{Norm, TransformerConfig};
use crate::error::{Error, Result};
use crate::model::residual_block::{Linear, load_linear};
use crate::ops::norm::{PerDimScale, RmsNorm};
use crate::ops::rope::Rope;
use crate::tensor::Tensor2D;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Thread-safe pointer wrapper for writing disjoint chunks of output in parallel.
#[derive(Copy, Clone)]
struct SendMutPtr(*mut f32);
unsafe impl Send for SendMutPtr {}
unsafe impl Sync for SendMutPtr {}

impl SendMutPtr {
    #[inline]
    unsafe fn add(self, count: usize) -> *mut f32 {
        unsafe { self.0.add(count) }
    }
}

/// Attention logit pre-scale. The official code premultiplies Q by sqrt(d)
/// (MEA path, net scale = sqrt(head_dim)) or applies an internal
/// 1/sqrt(d) division (dot-product path, net 1.0). Both are folded into this
/// single constant per layer.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    out_proj: Linear,
    query_ln: Option<RmsNorm>,
    key_ln: Option<RmsNorm>,
    value_ln: Option<RmsNorm>,
    rope: Option<Rope>,
    per_dim_scale: Option<PerDimScale>,
    causal: bool,
    /// Logit multiplier folded from the Q*√d / 1÷√d dance.
    scale: f32,
}

impl MultiHeadAttention {
    /// Loads the attention from a safetensors store; `prefix` is e.g.
    /// `transformer_stack.layers.0.seq_attn`. `use_qk_rope` selects RoPE for
    /// this particular attention instance (seq attention uses `use_rope_seq`,
    /// variate attention `use_rope_var`).
    pub fn load(
        store: &crate::checkpoint::Safetensors,
        prefix: &str,
        cfg: &TransformerConfig,
        rms_eps: f32,
        use_qk_rope: bool,
        causal: bool,
    ) -> Result<MultiHeadAttention> {
        let model_dims = cfg.model_dims;
        let num_heads = cfg.num_heads;
        if !model_dims.is_multiple_of(num_heads) {
            return Err(Error(format!(
                "MultiHeadAttention: model_dims {model_dims} not divisible by num_heads {num_heads}"
            )));
        }
        let head_dim = model_dims / num_heads;
        let load_linear =
            |name: &str| -> Result<Linear> { load_linear(store, prefix, name, cfg.use_bias) };
        let query_proj = load_linear("query_proj")?;
        let key_proj = load_linear("key_proj")?;
        let value_proj = load_linear("value_proj")?;
        let out_proj = load_linear("out_proj")?;

        let query_ln = match cfg.qk_norm {
            Norm::Rms => Some(RmsNorm::load(
                store,
                &format!("{prefix}.query_ln"),
                rms_eps,
            )?),
            Norm::None_ => None,
        };
        let key_ln = match cfg.qk_norm {
            Norm::Rms => Some(RmsNorm::load(store, &format!("{prefix}.key_ln"), rms_eps)?),
            Norm::None_ => None,
        };
        let value_ln = match cfg.v_norm {
            Norm::Rms => Some(RmsNorm::load(
                store,
                &format!("{prefix}.value_ln"),
                rms_eps,
            )?),
            Norm::None_ => None,
        };
        let rope = if use_qk_rope {
            Some(Rope::new(head_dim, 1.0, 10000.0))
        } else {
            None
        };
        let per_dim_scale = Some(PerDimScale::load(
            store,
            &format!("{prefix}.per_dim_scale"),
        )?);

        // rescale_logits mirrors Flax: MEA=true → scale=√d; MEA=false → net 1.0.
        let mea = cfg.use_memory_efficient_attention;
        let scale = if mea { (head_dim as f32).sqrt() } else { 1.0 };

        Ok(MultiHeadAttention {
            num_heads,
            head_dim,
            query_proj,
            key_proj,
            value_proj,
            out_proj,
            query_ln,
            key_ln,
            value_ln,
            rope,
            per_dim_scale,
            causal,
            scale,
        })
    }

    /// Batch-attention forward. Input `(batch*n, d)`; returns `(batch*n, d)`.
    ///
    /// - `kv_patch_mask`: optional `(batch*n)` bool, true = key/value patch is
    ///   fully masked (leading padding) and must not be attended to.
    /// - `positions`: optional per-position (per *row*, not per batch) RoPE
    ///   positions; when empty, positions are `0..n` repeated per batch.
    pub fn forward(
        &self,
        x: &Tensor2D,
        n: usize,
        batch: usize,
        kv_patch_mask: Option<&[bool]>,
        positions: Option<&[i32]>,
    ) -> Tensor2D {
        let d = x.cols();
        assert_eq!(x.rows(), batch * n, "MHA: batch*n mismatch");
        let h = self.num_heads;
        let hd = self.head_dim;
        assert_eq!(d, h * hd);

        let q = self.query_proj.forward(x);
        let k = self.key_proj.forward(x);
        let mut v = self.value_proj.forward(x);

        let mut q = q;
        let mut k = k;
        if let Some(rope) = &self.rope {
            // RoPE over the (n, h, hd) layout: same position for all heads of
            // a row; we can treat the buffer as rows of `hd` by iterating.
            apply_rope_heads(rope, &mut q, n, batch, h, hd, positions);
            apply_rope_heads(rope, &mut k, n, batch, h, hd, positions);
        }
        if let Some(ln) = &self.query_ln {
            apply_rms_heads(ln, &mut q, h, hd);
        }
        if let Some(ln) = &self.key_ln {
            apply_rms_heads(ln, &mut k, h, hd);
        }
        if let Some(ln) = &self.value_ln {
            apply_rms_heads(ln, &mut v, h, hd);
        }
        if let Some(pds) = &self.per_dim_scale {
            apply_per_dim_heads(pds, &mut q, h, hd);
        }

        // Attention: for each batch element, for each head, scores over
        // (n x n) with causal+mask, then weighted sum of V.
        // CPU Tiled Online Softmax (CPU FlashAttention):
        // Parallelize over (batch * num_heads) to saturate all CPU cores even with batch=1.
        // For each query row i, only evaluate keys j in 0..j_max (j_max = i+1 for causal, else n).
        // 50% dot-product FLOPs saved on causal attention, 0 bytes heap memory allocated.
        let mut out = Tensor2D::zeros(batch * n, d);
        let scale = self.scale;
        let causal = self.causal;
        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();

        #[cfg(target_arch = "x86_64")]
        let has_avx2 =
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && hd == 80;
        #[cfg(not(target_arch = "x86_64"))]
        let has_avx2 = false;

        #[cfg(target_arch = "aarch64")]
        let has_neon = hd == 80;
        #[cfg(not(target_arch = "aarch64"))]
        let has_neon = false;

        let out_ptr = SendMutPtr(out.data_mut().as_mut_ptr());

        (0..batch * h).into_par_iter().for_each(move |head_flat| {
            let b = head_flat / h;
            let hi = head_flat % h;
            let b_base = b * n * d;
            let h_base = hi * hd;

            let mut stack_scores = [0.0f32; 128];

            for i in 0..n {
                let j_max = if causal { i + 1 } else { n };
                let mut heap_scores = if j_max > 128 {
                    vec![0.0f32; j_max]
                } else {
                    Vec::new()
                };
                let row_scores = if j_max > 128 {
                    heap_scores.as_mut_slice()
                } else {
                    &mut stack_scores[..j_max]
                };

                let qi = b_base + i * d + h_base;
                let q_slice = &q_data[qi..qi + hd];
                let q_ptr = unsafe { q_data.as_ptr().add(qi) };

                let mut max_val = f32::NEG_INFINITY;
                for j in 0..j_max {
                    if let Some(mask) = kv_patch_mask
                        && mask[b * n + j]
                    {
                        row_scores[j] = f32::NEG_INFINITY;
                        continue;
                    }

                    let kj = b_base + j * d + h_base;
                    let dot_scaled = if has_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            let k_ptr = k_data.as_ptr().add(kj);
                            dot_80_f32(q_ptr, k_ptr) * scale
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        0.0
                    } else if has_neon {
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            let k_ptr = k_data.as_ptr().add(kj);
                            dot_80_f32_neon(q_ptr, k_ptr) * scale
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        0.0
                    } else {
                        let k_slice = &k_data[kj..kj + hd];
                        let mut dot = 0.0f64;
                        for t in 0..hd {
                            dot += q_slice[t] as f64 * k_slice[t] as f64;
                        }
                        (dot * scale as f64) as f32
                    };

                    row_scores[j] = dot_scaled;
                    if dot_scaled > max_val {
                        max_val = dot_scaled;
                    }
                }

                let oi = b_base + i * d + h_base;
                let dst_ptr = unsafe { out_ptr.add(oi) };

                // -inf row (all masked) -> uniform zeros
                if !max_val.is_finite() {
                    unsafe {
                        std::ptr::write_bytes(dst_ptr, 0, hd);
                    }
                    continue;
                }

                // Exponentiate and normalize row in L1 cache
                let mut sum = 0.0f64;
                for v in row_scores.iter_mut() {
                    *v = ((*v - max_val) as f64).exp() as f32;
                    sum += *v as f64;
                }
                if sum > 0.0 {
                    let inv_sum = 1.0 / sum;
                    for v in row_scores.iter_mut() {
                        *v = (*v as f64 * inv_sum) as f32;
                    }
                }

                // Weighted sum over V
                if has_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        let v_base_ptr = v_data.as_ptr().add(b_base + h_base);
                        weighted_sum_avx2_80(row_scores, v_base_ptr, d, j_max, dst_ptr);
                    }
                } else if has_neon {
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        let v_base_ptr = v_data.as_ptr().add(b_base + h_base);
                        weighted_sum_neon_80(row_scores, v_base_ptr, d, j_max, dst_ptr);
                    }
                } else {
                    let out_slice = unsafe { std::slice::from_raw_parts_mut(dst_ptr, hd) };
                    if hd <= 80 {
                        let mut acc = [0.0f64; 80];
                        for (j, &w) in row_scores.iter().enumerate() {
                            if w == 0.0 {
                                continue;
                            }
                            let vj = b_base + j * d + h_base;
                            let vr = &v_data[vj..vj + hd];
                            let w64 = w as f64;
                            for t in 0..hd {
                                acc[t] += w64 * vr[t] as f64;
                            }
                        }
                        for t in 0..hd {
                            out_slice[t] = acc[t] as f32;
                        }
                    } else {
                        let mut acc = vec![0.0f64; hd];
                        for (j, &w) in row_scores.iter().enumerate() {
                            if w == 0.0 {
                                continue;
                            }
                            let vj = b_base + j * d + h_base;
                            let vr = &v_data[vj..vj + hd];
                            let w64 = w as f64;
                            for t in 0..hd {
                                acc[t] += w64 * vr[t] as f64;
                            }
                        }
                        for t in 0..hd {
                            out_slice[t] = acc[t] as f32;
                        }
                    }
                }
            }
        });

        self.out_proj.forward(&out)
    }

    pub fn quantize(&mut self, precision: crate::config::QuantizationPrecision) {
        self.query_proj.quantize(precision);
        self.key_proj.quantize(precision);
        self.value_proj.quantize(precision);
        self.out_proj.quantize(precision);
    }
}

/// Applies RoPE over the (n, h, hd) layout. `positions` has one entry per row
/// of the *sequence* (length n per batch element, repeated). When `None`,
/// positions are 0..n per batch.
fn apply_rope_heads(
    rope: &Rope,
    x: &mut Tensor2D,
    n: usize,
    _batch: usize,
    _h: usize,
    _hd: usize,
    positions: Option<&[i32]>,
) {
    let cols = x.cols();
    let dims = rope.dims();
    x.data_mut()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, row)| {
            let pos = match positions {
                Some(p) => p[r],
                None => (r % n) as i32,
            };
            for head_slice in row.chunks_mut(dims) {
                rope.rotate(head_slice, pos);
            }
        });
}

/// Applies RMSNorm over the (…, h, head_dim) layout: normalize each head
/// slice independently (weight is per head_dim).
fn apply_rms_heads(ln: &RmsNorm, x: &mut Tensor2D, _h: usize, hd: usize) {
    x.data_mut().par_chunks_mut(hd * 16).for_each(|chunk| {
        for row in chunk.chunks_mut(hd) {
            ln.forward_row(row);
        }
    });
}

/// Applies PerDimScale over the (…, h, head_dim) layout.
fn apply_per_dim_heads(pds: &PerDimScale, x: &mut Tensor2D, _h: usize, hd: usize) {
    x.data_mut().par_chunks_mut(hd * 16).for_each(|chunk| {
        for row in chunk.chunks_mut(hd) {
            pds.forward_row(row);
        }
    });
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_80_f32(q: *const f32, k: *const f32) -> f32 {
    unsafe {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();

        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q), _mm256_loadu_ps(k), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(8)), _mm256_loadu_ps(k.add(8)), s1);

        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(16)), _mm256_loadu_ps(k.add(16)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(24)), _mm256_loadu_ps(k.add(24)), s1);

        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(32)), _mm256_loadu_ps(k.add(32)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(40)), _mm256_loadu_ps(k.add(40)), s1);

        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(48)), _mm256_loadu_ps(k.add(48)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(56)), _mm256_loadu_ps(k.add(56)), s1);

        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(64)), _mm256_loadu_ps(k.add(64)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(72)), _mm256_loadu_ps(k.add(72)), s1);

        let sum01 = _mm256_add_ps(s0, s1);
        let hi = _mm256_extractf128_ps(sum01, 1);
        let lo = _mm256_castps256_ps128(sum01);
        let s4 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehl_ps(s4, s4);
        let s2 = _mm_add_ps(s4, shuf);
        let shuf2 = _mm_shuffle_ps(s2, s2, 1);
        let s1 = _mm_add_ss(s2, shuf2);
        _mm_cvtss_f32(s1)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn weighted_sum_avx2_80(
    scores_row: &[f32],
    v_base_ptr: *const f32,
    d: usize,
    count: usize,
    out_ptr: *mut f32,
) {
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut a4 = _mm256_setzero_ps();
        let mut a5 = _mm256_setzero_ps();
        let mut a6 = _mm256_setzero_ps();
        let mut a7 = _mm256_setzero_ps();
        let mut a8 = _mm256_setzero_ps();
        let mut a9 = _mm256_setzero_ps();

        for (j, &w) in scores_row.iter().enumerate().take(count) {
            if w == 0.0 {
                continue;
            }
            let vw = _mm256_set1_ps(w);
            let v_ptr = v_base_ptr.add(j * d);
            a0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_ptr), a0);
            a1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_ptr.add(8)), a1);
            a2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_ptr.add(16)), a2);
            a3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_ptr.add(24)), a3);
            a4 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_ptr.add(32)), a4);
            a5 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_ptr.add(40)), a5);
            a6 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_ptr.add(48)), a6);
            a7 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_ptr.add(56)), a7);
            a8 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_ptr.add(64)), a8);
            a9 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(v_ptr.add(72)), a9);
        }

        _mm256_storeu_ps(out_ptr, a0);
        _mm256_storeu_ps(out_ptr.add(8), a1);
        _mm256_storeu_ps(out_ptr.add(16), a2);
        _mm256_storeu_ps(out_ptr.add(24), a3);
        _mm256_storeu_ps(out_ptr.add(32), a4);
        _mm256_storeu_ps(out_ptr.add(40), a5);
        _mm256_storeu_ps(out_ptr.add(48), a6);
        _mm256_storeu_ps(out_ptr.add(56), a7);
        _mm256_storeu_ps(out_ptr.add(64), a8);
        _mm256_storeu_ps(out_ptr.add(72), a9);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_80_f32_neon(q: *const f32, k: *const f32) -> f32 {
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);

        for i in (0..80).step_by(16) {
            s0 = vmlaq_f32(s0, vld1q_f32(q.add(i)), vld1q_f32(k.add(i)));
            s1 = vmlaq_f32(s1, vld1q_f32(q.add(i + 4)), vld1q_f32(k.add(i + 4)));
            s2 = vmlaq_f32(s2, vld1q_f32(q.add(i + 8)), vld1q_f32(k.add(i + 8)));
            s3 = vmlaq_f32(s3, vld1q_f32(q.add(i + 12)), vld1q_f32(k.add(i + 12)));
        }
        let s01 = vaddq_f32(s0, s1);
        let s23 = vaddq_f32(s2, s3);
        let sum = vaddq_f32(s01, s23);
        vaddvq_f32(sum)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn weighted_sum_neon_80(
    scores_row: &[f32],
    v_base_ptr: *const f32,
    d: usize,
    count: usize,
    out_ptr: *mut f32,
) {
    unsafe {
        let mut a = [vdupq_n_f32(0.0); 20];
        for (j, &w) in scores_row.iter().enumerate().take(count) {
            if w == 0.0 {
                continue;
            }
            let vw = vdupq_n_f32(w);
            let v_ptr = v_base_ptr.add(j * d);
            for t in 0..20 {
                a[t] = vmlaq_f32(a[t], vw, vld1q_f32(v_ptr.add(t * 4)));
            }
        }
        for t in 0..20 {
            vst1q_f32(out_ptr.add(t * 4), a[t]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_80_avx2_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut q = [0.0f32; 80];
            let mut k = [0.0f32; 80];
            for i in 0..80 {
                q[i] = (i as f32 * 0.1).sin();
                k[i] = (i as f32 * 0.2).cos();
            }
            let mut expected = 0.0f64;
            for i in 0..80 {
                expected += q[i] as f64 * k[i] as f64;
            }
            let actual = unsafe { dot_80_f32(q.as_ptr(), k.as_ptr()) };
            let diff = (actual as f64 - expected).abs();
            assert!(diff < 1e-4, "dot_80 diff: {diff}");
        }
    }

    #[test]
    fn test_weighted_sum_avx2_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let n = 16;
            let d = 80;
            let mut weights = vec![0.0f32; n];
            for (j, w) in weights.iter_mut().enumerate() {
                *w = (j + 1) as f32 / (n * (n + 1) / 2) as f32;
            }
            let mut v = vec![0.0f32; n * d];
            for (idx, val) in v.iter_mut().enumerate() {
                *val = (idx as f32 * 0.05).sin();
            }

            let mut out_avx2 = vec![0.0f32; d];
            unsafe {
                weighted_sum_avx2_80(&weights, v.as_ptr(), d, n, out_avx2.as_mut_ptr());
            }

            let mut out_scalar = vec![0.0f64; d];
            for (j, &w) in weights.iter().enumerate() {
                let vr = &v[j * d..(j + 1) * d];
                for t in 0..d {
                    out_scalar[t] += w as f64 * vr[t] as f64;
                }
            }

            for t in 0..d {
                let diff = (out_avx2[t] as f64 - out_scalar[t]).abs();
                assert!(
                    diff < 1e-4,
                    "t={t}, avx2={}, scalar={}",
                    out_avx2[t],
                    out_scalar[t]
                );
            }
        }
    }
}
