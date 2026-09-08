//! Pre-transformer residual block (official `ResidualBlock` in dense.py).
//!
//! 2-layer MLP with a linear residual branch:
//! `out = output(activation(hidden(x))) + residual(x)`
//! (with `residual_layer` replaced by identity when `identity_skip`).

use rayon::prelude::*;

use crate::checkpoint::Safetensors;
use crate::config::{Activation, Norm, QuantizationPrecision, ResidualBlockConfig};
use crate::error::{Error, Result};
use crate::ops::activations::{relu, silu};
use crate::ops::norm::RmsNorm;
use crate::ops::{add_into, gemm};
use crate::tensor::Tensor2D;

/// Weight storage format for Linear layers.
#[derive(Debug, Clone)]
pub enum WeightStorage {
    F32(Tensor2D),
    F16 {
        rows: usize,
        cols: usize,
        data: Vec<half::f16>,
    },
}

impl WeightStorage {
    #[inline]
    pub fn rows(&self) -> usize {
        match self {
            WeightStorage::F32(t) => t.rows(),
            WeightStorage::F16 { rows, .. } => *rows,
        }
    }

    #[inline]
    pub fn cols(&self) -> usize {
        match self {
            WeightStorage::F32(t) => t.cols(),
            WeightStorage::F16 { cols, .. } => *cols,
        }
    }

    pub fn to_f16(&self) -> Self {
        match self {
            WeightStorage::F16 { .. } => self.clone(),
            WeightStorage::F32(t) => {
                let data: Vec<half::f16> =
                    t.data().iter().map(|&v| half::f16::from_f32(v)).collect();
                WeightStorage::F16 {
                    rows: t.rows(),
                    cols: t.cols(),
                    data,
                }
            }
        }
    }

    pub fn to_f32(&self) -> Self {
        match self {
            WeightStorage::F32(_) => self.clone(),
            WeightStorage::F16 { rows, cols, data } => {
                let mut f32_data = vec![0.0f32; data.len()];
                unpack_f16_into(data, &mut f32_data);
                WeightStorage::F32(
                    Tensor2D::from_row_major(*rows, *cols, f32_data).expect("invariants"),
                )
            }
        }
    }

    pub fn quantize(&mut self, precision: QuantizationPrecision) {
        match precision {
            QuantizationPrecision::F32 => {
                if !matches!(self, WeightStorage::F32(_)) {
                    *self = self.to_f32();
                }
            }
            QuantizationPrecision::F16 => {
                if !matches!(self, WeightStorage::F16 { .. }) {
                    *self = self.to_f16();
                }
            }
        }
    }
}

/// Unpacks a slice of `half::f16` to `f32` in parallel using Rayon when beneficial.
pub fn unpack_f16_into(src: &[half::f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    let chunk_size = 65536;
    if src.len() < chunk_size || rayon::current_num_threads() <= 1 {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = s.to_f32();
        }
    } else {
        dst.par_chunks_mut(chunk_size)
            .zip(src.par_chunks(chunk_size))
            .for_each(|(dst_chunk, src_chunk)| {
                for (d, s) in dst_chunk.iter_mut().zip(src_chunk.iter()) {
                    *d = s.to_f32();
                }
            });
    }
}

/// A fused linear layer. `weight` is stored in inference layout `(in, out)`
/// (transposed once from torch's `(out, in)` checkpoint layout at load).
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: WeightStorage,
    pub bias: Option<Vec<f32>>,
}

impl Linear {
    /// Builds a linear layer from already-transposed weights.
    pub fn from_weight(weight: Tensor2D, bias: Option<Vec<f32>>) -> Result<Linear> {
        if let Some(b) = &bias
            && b.len() != weight.cols()
        {
            return Err(Error(format!(
                "Linear: bias len {} != out features {}",
                b.len(),
                weight.cols()
            )));
        }
        Ok(Linear {
            weight: WeightStorage::F32(weight),
            bias,
        })
    }

    /// Builds a linear layer from arbitrary WeightStorage.
    pub fn from_storage(weight: WeightStorage, bias: Option<Vec<f32>>) -> Result<Linear> {
        if let Some(b) = &bias
            && b.len() != weight.cols()
        {
            return Err(Error(format!(
                "Linear: bias len {} != out features {}",
                b.len(),
                weight.cols()
            )));
        }
        Ok(Linear { weight, bias })
    }

    #[inline]
    pub fn in_features(&self) -> usize {
        self.weight.rows()
    }

    #[inline]
    pub fn out_features(&self) -> usize {
        self.weight.cols()
    }

    pub fn quantize(&mut self, precision: QuantizationPrecision) {
        self.weight.quantize(precision);
    }

    /// `out = x @ W + b` with `x` of shape `(m, in_features)` and `out` preallocated to `(m, out_features)`.
    pub fn forward_into(&self, x: &Tensor2D, out: &mut Tensor2D) {
        assert_eq!(
            x.cols(),
            self.in_features(),
            "Linear: input cols {} != in_features {}",
            x.cols(),
            self.in_features()
        );
        assert_eq!(out.rows(), x.rows());
        assert_eq!(out.cols(), self.out_features());

        match &self.weight {
            WeightStorage::F32(w) => {
                gemm::gemm(
                    x.data(),
                    w.data(),
                    out.data_mut(),
                    x.rows(),
                    self.in_features(),
                    self.out_features(),
                );
            }
            WeightStorage::F16 { data, .. } => {
                gemm::gemm_f16(
                    x.data(),
                    data,
                    out.data_mut(),
                    x.rows(),
                    self.in_features(),
                    self.out_features(),
                );
            }
        }

        if let Some(bias) = &self.bias {
            let cols = out.cols();
            for row in out.data_mut().chunks_mut(cols) {
                for (v, &b) in row.iter_mut().zip(bias.iter()) {
                    *v += b;
                }
            }
        }
    }

    /// Backward compatibility wrapper for `forward_into`.
    pub fn forward_into_with_unpack(
        &self,
        x: &Tensor2D,
        out: &mut Tensor2D,
        _unpack_buf: &mut Vec<f32>,
    ) {
        self.forward_into(x, out);
    }

    /// `y = x @ W + b` with `x` of shape `(m, in_features)`.
    pub fn forward(&self, x: &Tensor2D) -> Tensor2D {
        let mut y = Tensor2D::zeros(x.rows(), self.out_features());
        self.forward_into(x, &mut y);
        y
    }
}

/// The pre-transformer residual block.
#[derive(Debug, Clone)]
pub struct ResidualBlock {
    pub config: ResidualBlockConfig,
    hidden_layer: Linear,
    output_layer: Linear,
    residual_layer: Option<Linear>,
    pre_norm: Option<RmsNorm>,
}

impl ResidualBlock {
    /// Loads the block from a safetensors store under `prefix`.
    ///
    /// Expects `{prefix}.hidden_layer.weight`, `{prefix}.output_layer.weight`
    /// and (unless `identity_skip`) `{prefix}.residual_layer.weight`, in torch
    /// `(out, in)` layout. If `prenorm` is rms, also loads
    /// `{prefix}.pre_norm.weight`.
    pub fn load(
        store: &Safetensors,
        prefix: &str,
        config: &ResidualBlockConfig,
        rms_eps: f32,
    ) -> Result<ResidualBlock> {
        let hidden_layer = load_linear(store, prefix, "hidden_layer", config.use_bias)?;
        if hidden_layer.out_features() != config.hidden_dims {
            return Err(Error(format!(
                "ResidualBlock: hidden_layer out {} != config.hidden_dims {}",
                hidden_layer.out_features(),
                config.hidden_dims
            )));
        }
        let output_layer = load_linear(store, prefix, "output_layer", config.use_bias)?;
        if output_layer.in_features() != config.hidden_dims
            || output_layer.out_features() != config.output_dims
        {
            return Err(Error(format!(
                "ResidualBlock: output_layer shape ({}->{}) != config ({}->{})",
                output_layer.in_features(),
                output_layer.out_features(),
                config.hidden_dims,
                config.output_dims
            )));
        }
        let residual_layer = if config.identity_skip {
            None
        } else {
            let rl = load_linear(store, prefix, "residual_layer", config.use_bias)?;
            if rl.out_features() != config.output_dims {
                return Err(Error(format!(
                    "ResidualBlock: residual_layer out {} != config.output_dims {}",
                    rl.out_features(),
                    config.output_dims
                )));
            }
            Some(rl)
        };
        let pre_norm = match config.prenorm {
            Norm::Rms => Some(RmsNorm::load(
                store,
                &format!("{prefix}.pre_norm"),
                rms_eps,
            )?),
            Norm::None_ => None,
        };
        Ok(ResidualBlock {
            config: config.clone(),
            hidden_layer,
            output_layer,
            residual_layer,
            pre_norm,
        })
    }

    /// Builds the block from pre-assembled layers (used by tests and embedders).
    pub fn from_layers(
        config: ResidualBlockConfig,
        hidden_layer: Linear,
        output_layer: Linear,
        residual_layer: Option<Linear>,
        pre_norm: Option<RmsNorm>,
    ) -> ResidualBlock {
        ResidualBlock {
            config,
            hidden_layer,
            output_layer,
            residual_layer,
            pre_norm,
        }
    }

    /// Forward pass. `x` shape `(m, in)`; returns `(m, output_dims)`.
    pub fn forward(&self, x: &Tensor2D) -> Tensor2D {
        let mut pre_norm_buf;
        let hidden_input = match &self.pre_norm {
            Some(n) => {
                pre_norm_buf = x.clone();
                n.forward(&mut pre_norm_buf);
                &pre_norm_buf
            }
            None => x,
        };
        let mut hidden = self.hidden_layer.forward(hidden_input);
        match self.config.activation {
            Activation::Relu => relu(&mut hidden),
            Activation::Swish => silu(&mut hidden),
            Activation::Identity => {}
        }
        let mut out = self.output_layer.forward(&hidden);
        match &self.residual_layer {
            Some(rl) => add_into(&mut out, &rl.forward(x)),
            None => add_into(&mut out, x),
        }
        out
    }

    #[inline]
    pub fn input_dims(&self) -> usize {
        self.hidden_layer.in_features()
    }

    #[inline]
    pub fn output_dims(&self) -> usize {
        self.output_layer.out_features()
    }

    pub fn quantize(&mut self, precision: QuantizationPrecision) {
        self.hidden_layer.quantize(precision);
        self.output_layer.quantize(precision);
        if let Some(rl) = &mut self.residual_layer {
            rl.quantize(precision);
        }
    }
}

/// Loads one linear layer (`{prefix}.{name}.weight` / optional `.bias`),
/// transposing the torch `(out, in)` weight into `(in, out)` inference layout.
pub(crate) fn load_linear(
    store: &Safetensors,
    prefix: &str,
    name: &str,
    use_bias: bool,
) -> Result<Linear> {
    // `name` empty ⇒ the layer itself is the weight name (e.g. output_head).
    let key = if name.is_empty() {
        prefix.to_owned()
    } else {
        format!("{prefix}.{name}")
    };
    let weight_key = format!("{key}.weight");
    let bias = if use_bias {
        Some(store.f32_flat(&format!("{key}.bias"))?)
    } else {
        None
    };

    // Dtype-aware load: half checkpoints load straight into half storage
    // (transposed in half too), skipping the decode->f32->re-quantize round
    // trip that tripled assembly memory traffic for quantized checkpoints.
    let wt_dtype = store.tensor_info(&weight_key).map(|i| i.dtype);
    let weight = match wt_dtype {
        Some(crate::checkpoint::DType::F16) => {
            let (r, c, half_data) = store.f16_tensor(&weight_key)?;
            let data = crate::tensor::transpose_f16(&half_data, r, c);
            WeightStorage::F16 {
                rows: c,
                cols: r,
                data,
            }
        }
        _ => WeightStorage::F32(store.f32_tensor(&weight_key)?.transpose()),
    };
    Linear::from_storage(weight, bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A residual block with fixed simple weights:
    /// hidden: (3 -> 2), output: (2 -> 1), residual: (3 -> 1).
    fn tiny_block() -> ResidualBlock {
        let cfg = ResidualBlockConfig {
            hidden_dims: 2,
            output_dims: 1,
            use_bias: true,
            activation: Activation::Relu,
            identity_skip: false,
            prenorm: Norm::None_,
        };
        // weight stored (in, out) inference layout
        let h = Linear::from_weight(
            Tensor2D::from_row_major(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .expect("test fixture invariant"),
            Some(vec![0.0, 0.0]),
        )
        .expect("test fixture invariant");
        let o = Linear::from_weight(
            Tensor2D::from_row_major(2, 1, vec![1.0, 1.0]).expect("test fixture invariant"),
            Some(vec![0.0]),
        )
        .expect("test fixture invariant");
        let r = Linear::from_weight(
            Tensor2D::from_row_major(3, 1, vec![1.0, 1.0, 1.0]).expect("test fixture invariant"),
            Some(vec![0.0]),
        )
        .expect("test fixture invariant");
        ResidualBlock::from_layers(cfg, h, o, Some(r), None)
    }

    #[test]
    fn forward_small() {
        let block = tiny_block();
        let x =
            Tensor2D::from_row_major(1, 3, vec![1.0, -2.0, 3.0]).expect("test fixture invariant");
        // hidden = x @ W_h, W_h rows: [[1,0],[0,1],[1,1]]
        //   x W_h = [1*1 + (-2)*0 + 3*1, 1*0 + (-2)*1 + 3*1] = [4, 1]
        // relu -> [4, 1]
        // output = sum = 5
        // residual = x W_r = 1 - 2 + 3 = 2
        // out = 5 + 2 = 7
        let out = block.forward(&x);
        assert_eq!(out.data(), &[7.0]);
    }

    #[test]
    fn quantization_forward_f16() {
        let mut block = tiny_block();
        let x =
            Tensor2D::from_row_major(1, 3, vec![1.0, -2.0, 3.0]).expect("test fixture invariant");

        // FP16 test
        block.quantize(QuantizationPrecision::F16);
        let out_f16 = block.forward(&x);
        assert!((out_f16[(0, 0)] - 7.0).abs() < 1e-3);

        // Dequantize back to FP32
        block.quantize(QuantizationPrecision::F32);
        let out_f32 = block.forward(&x);
        assert!((out_f32[(0, 0)] - 7.0).abs() < 1e-6);
    }
}
