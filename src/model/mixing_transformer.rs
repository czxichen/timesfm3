//! MixingTransformer (official `MixingTransformer` + `StackedMixingTransformer`)
//! with sequence attention, variate attention, and FFN per layer.

use rayon::prelude::*;

use crate::checkpoint::Safetensors;
use crate::config::{Activation, StackedTransformersConfig};
use crate::error::{Error, Result};
use crate::model::attention::MultiHeadAttention;
use crate::model::residual_block::{Linear, load_linear};
use crate::ops::activations::{relu, silu};
use crate::ops::add_into;
use crate::ops::norm::RmsNorm;
use crate::tensor::Tensor2D;

/// Transposes a flat `(b*v*n, d)` buffer (row-major over (b, v, n, d)) into
/// `(b*n*v, d)` row-major over (b, n, v, d): permutes the v/n axes.
pub fn transpose_bvnd_to_bnvd_into(
    x: &Tensor2D,
    out: &mut Tensor2D,
    b: usize,
    v: usize,
    n: usize,
    d: usize,
) {
    assert_eq!(x.rows(), b * v * n);
    assert_eq!(x.cols(), d);
    assert_eq!(out.rows(), b * n * v);
    assert_eq!(out.cols(), d);
    for bbi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                let src_row = (bbi * v + vi) * n + ni;
                let dst_row = (bbi * n + ni) * v + vi;
                let s = &x.data()[src_row * d..(src_row + 1) * d];
                out.data_mut()[dst_row * d..(dst_row + 1) * d].copy_from_slice(s);
            }
        }
    }
}

pub fn transpose_bvnd_to_bnvd(x: &Tensor2D, b: usize, v: usize, n: usize, d: usize) -> Tensor2D {
    let mut out = Tensor2D::zeros(b * n * v, d);
    transpose_bvnd_to_bnvd_into(x, &mut out, b, v, n, d);
    out
}

/// Inverse of `transpose_bvnd_to_bnvd`.
pub fn transpose_bnvd_to_bvnd_into(
    x: &Tensor2D,
    out: &mut Tensor2D,
    b: usize,
    v: usize,
    n: usize,
    d: usize,
) {
    assert_eq!(x.rows(), b * n * v);
    assert_eq!(x.cols(), d);
    assert_eq!(out.rows(), b * v * n);
    assert_eq!(out.cols(), d);
    for bbi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                let src_row = (bbi * n + ni) * v + vi;
                let dst_row = (bbi * v + vi) * n + ni;
                let s = &x.data()[src_row * d..(src_row + 1) * d];
                out.data_mut()[dst_row * d..(dst_row + 1) * d].copy_from_slice(s);
            }
        }
    }
}

pub fn transpose_bnvd_to_bvnd(x: &Tensor2D, b: usize, v: usize, n: usize, d: usize) -> Tensor2D {
    let mut out = Tensor2D::zeros(b * v * n, d);
    transpose_bnvd_to_bvnd_into(x, &mut out, b, v, n, d);
    out
}

/// Permutes a (b, v, n) bool mask to (b, n, v).
pub fn transpose_mask_bvn_to_bnv(mask: &[bool], b: usize, v: usize, n: usize) -> Vec<bool> {
    let mut out = vec![false; b * n * v];
    for bbi in 0..b {
        for vi in 0..v {
            for ni in 0..n {
                out[(bbi * n + ni) * v + vi] = mask[(bbi * v + vi) * n + ni];
            }
        }
    }
    out
}

/// Reusable scratchpad workspace for MixingTransformer forward passes,
/// eliminating repeated heap allocations across all transformer layers.
pub struct TransformerWorkspace {
    pub buf_bvn_1: Tensor2D,
    pub buf_bvn_2: Tensor2D,
    pub buf_bnv_1: Tensor2D,
    pub ff_hidden: Tensor2D,
    pub var_mask: Vec<bool>,
}

impl TransformerWorkspace {
    pub fn new(
        b: usize,
        v: usize,
        n: usize,
        d: usize,
        hidden_dims: usize,
        patch_mask: &[bool],
    ) -> Self {
        Self {
            buf_bvn_1: Tensor2D::zeros(b * v * n, d),
            buf_bvn_2: Tensor2D::zeros(b * v * n, d),
            buf_bnv_1: Tensor2D::zeros(b * n * v, d),
            ff_hidden: Tensor2D::zeros(b * v * n, hidden_dims),
            var_mask: transpose_mask_bvn_to_bnv(patch_mask, b, v, n),
        }
    }
}

/// One MixingTransformer layer: sequence attention → variate attention → FFN,
/// each with pre/post RMSNorm and a residual connection.
#[derive(Debug, Clone)]
pub struct MixingTransformer {
    pre_seq_attn_ln: RmsNorm,
    post_seq_attn_ln: RmsNorm,
    seq_attn: MultiHeadAttention,
    var_attn: Option<(Option<RmsNorm>, Option<RmsNorm>, MultiHeadAttention)>,
    pre_ff_ln: RmsNorm,
    post_ff_ln: RmsNorm,
    ff0: Linear,
    ff1: Linear,
    ff_activation: Activation,
}

impl MixingTransformer {
    pub fn load(
        store: &Safetensors,
        prefix: &str,
        cfg: &StackedTransformersConfig,
        rms_eps: f32,
        use_variate_attention: bool,
    ) -> Result<MixingTransformer> {
        let t = &cfg.transformer;
        let d = t.model_dims;
        let hidden = t.hidden_dims;

        let load_ln = |name: &str| -> Result<RmsNorm> {
            RmsNorm::load(store, &format!("{prefix}.{name}"), rms_eps)
        };
        let seq_attn = MultiHeadAttention::load(
            store,
            &format!("{prefix}.seq_attn"),
            t,
            rms_eps,
            t.use_rope_seq,
            t.causal_attention,
        )?;

        let var_attn = if use_variate_attention {
            // Variate attention is always non-causal (official: causal=False).
            let attn = MultiHeadAttention::load(
                store,
                &format!("{prefix}.var_attn"),
                t,
                rms_eps,
                t.use_rope_var,
                false,
            )?;
            Some((
                Some(load_ln("pre_var_attn_ln")?),
                Some(load_ln("post_var_attn_ln")?),
                attn,
            ))
        } else {
            None
        };

        let ff0 = load_linear(store, prefix, "ff0", t.use_bias)?;
        let ff1 = load_linear(store, prefix, "ff1", t.use_bias)?;
        if ff0.in_features() != d || ff0.out_features() != hidden {
            return Err(Error(format!(
                "MixingTransformer: ff0 shape ({}->{}) != config ({}->{})",
                ff0.in_features(),
                ff0.out_features(),
                d,
                hidden
            )));
        }
        if ff1.in_features() != hidden || ff1.out_features() != d {
            return Err(Error(format!(
                "MixingTransformer: ff1 shape ({}->{}) != config ({}->{})",
                ff1.in_features(),
                ff1.out_features(),
                hidden,
                d
            )));
        }

        Ok(MixingTransformer {
            pre_seq_attn_ln: load_ln("pre_seq_attn_ln")?,
            post_seq_attn_ln: load_ln("post_seq_attn_ln")?,
            seq_attn,
            var_attn,
            pre_ff_ln: load_ln("pre_ff_ln")?,
            post_ff_ln: load_ln("post_ff_ln")?,
            ff0,
            ff1,
            ff_activation: t.ff_activation,
        })
    }

    /// Forward over a flat `(b*v*n, d)` embeddings tensor.
    ///
    /// `patch_mask` is `(b, v, n)` bool (true = fully-masked patch) already
    /// transformed by the caller into the *effective* (cumprod-ified) mask.
    /// Forward over a flat `(b*v*n, d)` embeddings tensor using a preallocated workspace.
    pub fn forward_with_workspace(
        &self,
        x: &mut Tensor2D,
        b: usize,
        v: usize,
        n: usize,
        patch_mask: &[bool],
        ws: &mut TransformerWorkspace,
    ) {
        let d = x.cols();

        // --- Sequence attention (flat (b*v, n, d)) ---
        ws.buf_bvn_1.copy_from(x);
        self.pre_seq_attn_ln.forward(&mut ws.buf_bvn_1);
        let mut h1 = self
            .seq_attn
            .forward(&ws.buf_bvn_1, n, b * v, Some(patch_mask), None);
        self.post_seq_attn_ln.forward(&mut h1);
        add_into(&mut h1, x);

        // --- Variate attention (flat (b*n, v, d)) ---
        if let Some((pre_ln, post_ln, attn)) = &self.var_attn {
            transpose_bvnd_to_bnvd_into(&h1, &mut ws.buf_bnv_1, b, v, n, d);
            if let Some(ln) = pre_ln {
                ln.forward(&mut ws.buf_bnv_1);
            }
            let var_out = attn.forward(&ws.buf_bnv_1, v, b * n, Some(&ws.var_mask), None);
            transpose_bnvd_to_bvnd_into(&var_out, &mut ws.buf_bvn_1, b, v, n, d);
            if let Some(ln) = post_ln {
                ln.forward(&mut ws.buf_bvn_1);
            }
            add_into(&mut ws.buf_bvn_1, &h1);
        } else {
            ws.buf_bvn_1.copy_from(&h1);
        }

        // --- FFN ---
        ws.buf_bvn_2.copy_from(&ws.buf_bvn_1);
        self.pre_ff_ln.forward(&mut ws.buf_bvn_2);
        self.ff0.forward_into(&ws.buf_bvn_2, &mut ws.ff_hidden);
        match self.ff_activation {
            Activation::Relu => relu(&mut ws.ff_hidden),
            Activation::Swish => silu(&mut ws.ff_hidden),
            Activation::Identity => {}
        }
        self.ff1.forward_into(&ws.ff_hidden, &mut ws.buf_bvn_2);
        self.post_ff_ln.forward(&mut ws.buf_bvn_2);
        add_into(&mut ws.buf_bvn_2, &ws.buf_bvn_1);

        x.copy_from(&ws.buf_bvn_2);
    }

    pub fn quantize(&mut self, precision: crate::config::QuantizationPrecision) {
        self.seq_attn.quantize(precision);
        if let Some((_, _, attn)) = &mut self.var_attn {
            attn.quantize(precision);
        }
        self.ff0.quantize(precision);
        self.ff1.quantize(precision);
    }

    /// Forward over a flat `(b*v*n, d)` embeddings tensor.
    ///
    /// `patch_mask` is `(b, v, n)` bool (true = fully-masked patch) already
    /// transformed by the caller into the *effective* (cumprod-ified) mask.
    pub fn forward(
        &self,
        x: &Tensor2D,
        b: usize,
        v: usize,
        n: usize,
        patch_mask: &[bool],
    ) -> Tensor2D {
        let mut out = x.clone();
        let mut ws =
            TransformerWorkspace::new(b, v, n, x.cols(), self.ff0.out_features(), patch_mask);
        self.forward_with_workspace(&mut out, b, v, n, patch_mask, &mut ws);
        out
    }
}

/// Stacked MixingTransformer layers.
#[derive(Debug, Clone)]
pub struct StackedMixingTransformer {
    pub config: StackedTransformersConfig,
    pub layers: Vec<MixingTransformer>,
}

impl StackedMixingTransformer {
    pub fn load(
        store: &Safetensors,
        cfg: &StackedTransformersConfig,
        rms_eps: f32,
    ) -> Result<Self> {
        const USE_VARIATE_ATTENTION: bool = true; // official 3.0 config
        let layers = (0..cfg.num_layers)
            .into_par_iter()
            .map(|i| {
                MixingTransformer::load(
                    store,
                    &format!("transformer_stack.layers.{i}"),
                    cfg,
                    rms_eps,
                    USE_VARIATE_ATTENTION,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(StackedMixingTransformer {
            config: cfg.clone(),
            layers,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor2D,
        b: usize,
        v: usize,
        n: usize,
        patch_mask: &[bool],
    ) -> Tensor2D {
        let mut out = x.clone();
        let hidden_dims = self.config.transformer.hidden_dims;
        let mut ws = TransformerWorkspace::new(b, v, n, x.cols(), hidden_dims, patch_mask);
        let dump = crate::model::timesfm::is_dump_enabled();
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_with_workspace(&mut out, b, v, n, patch_mask, &mut ws);
            if dump {
                crate::model::timesfm::dbg_dump(&format!("layer_{i}"), out.data());
            }
        }
        out
    }

    pub fn quantize(&mut self, precision: crate::config::QuantizationPrecision) {
        for layer in &mut self.layers {
            layer.quantize(precision);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_transposes() {
        // (b=1, v=2, n=3): idx = vi*3 + ni
        let m = vec![true, false, false, false, true, false];
        let t = transpose_mask_bvn_to_bnv(&m, 1, 2, 3);
        // out (b, n, v): idx = ni*2 + vi
        let expect = vec![true, false, false, true, false, false];
        assert_eq!(t, expect);
    }

    #[test]
    fn tensor_transpose_roundtrip() {
        let (b, v, n, d) = (1usize, 2usize, 3usize, 2usize);
        let data: Vec<f32> = (0..b * v * n * d).map(|i| i as f32).collect();
        let x = Tensor2D::from_row_major(b * v * n, d, data).expect("fixture");
        let t = transpose_bvnd_to_bnvd(&x, b, v, n, d);
        let back = transpose_bnvd_to_bvnd(&t, b, v, n, d);
        assert_eq!(back, x);
    }
}
