//! Framework-agnostic model configuration (subset of the official configs.py).

use crate::error::{Error, Result};
use crate::json::Json;

/// Activation functions supported by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    #[default]
    Relu,
    Swish,
    Identity,
}

impl Activation {
    pub fn from_name(name: &str) -> Option<Activation> {
        match name {
            "relu" => Some(Activation::Relu),
            "swish" | "silu" => Some(Activation::Swish),
            "none" => Some(Activation::Identity),
            _ => None,
        }
    }
}

/// Normalization flavor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Norm {
    #[default]
    None_,
    Rms,
}

impl Norm {
    pub fn from_name(name: &str) -> Option<Norm> {
        match name {
            "rms" => Some(Norm::Rms),
            "none" => Some(Norm::None_),
            _ => None,
        }
    }
}

/// Quantization precision for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantizationPrecision {
    #[default]
    F32,
    F16,
}

impl QuantizationPrecision {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Some(Self::F32),
            "f16" | "fp16" => Some(Self::F16),
            _ => None,
        }
    }
}

/// Config for the pre-transformer residual block
/// (official: hidden=1280, output=1280, bias=false, relu, no prenorm).
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualBlockConfig {
    pub hidden_dims: usize,
    pub output_dims: usize,
    pub use_bias: bool,
    pub activation: Activation,
    /// If true, the residual branch is the identity instead of a linear layer.
    pub identity_skip: bool,
    /// rms pre-norm before the hidden layer (false for the official 3.0 block).
    pub prenorm: Norm,
}

impl Default for ResidualBlockConfig {
    fn default() -> Self {
        Self {
            hidden_dims: 1280,
            output_dims: 1280,
            use_bias: false,
            activation: Activation::Relu,
            identity_skip: false,
            prenorm: Norm::None_,
        }
    }
}

/// Per-layer transformer configuration (official `TransformerConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct TransformerConfig {
    pub model_dims: usize,
    pub hidden_dims: usize,
    pub num_heads: usize,
    pub attention_norm: Norm,
    pub feedforward_norm: Norm,
    pub qk_norm: Norm,
    pub use_rope_seq: bool,
    pub use_rope_var: bool,
    pub use_bias: bool,
    pub ff_activation: Activation,
    pub v_norm: Norm,
    pub causal_attention: bool,
    /// True = PyTorch SDPA path (scale = sqrt(head_dim));
    /// False = manual path (Q pre-multiplied by sqrt(head_dim), mask bias -1e9).
    pub use_sdpa: bool,
    pub use_memory_efficient_attention: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            model_dims: 1280,
            hidden_dims: 1280,
            num_heads: 16,
            attention_norm: Norm::Rms,
            feedforward_norm: Norm::Rms,
            qk_norm: Norm::Rms,
            use_rope_seq: true,
            use_rope_var: false,
            use_bias: false,
            ff_activation: Activation::Relu,
            v_norm: Norm::None_,
            causal_attention: true,
            use_sdpa: true,
            use_memory_efficient_attention: true,
        }
    }
}

/// Stacked transformer config (official `StackedTransformersConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct StackedTransformersConfig {
    pub num_layers: usize,
    pub transformer: TransformerConfig,
}

impl Default for StackedTransformersConfig {
    fn default() -> Self {
        Self {
            num_layers: 20,
            transformer: TransformerConfig::default(),
        }
    }
}

/// Full model config, mirroring the official `TimesFM3Torch` attributes and
/// the HF `config.json` (read from the checkpoint at load time).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub input_patch_len: usize,
    pub output_patch_len: usize,
    pub num_quantiles: usize,
    pub median_quantile_index: usize,
    pub residual_block_config: ResidualBlockConfig,
    pub transformer_config: StackedTransformersConfig,
    pub use_variate_attention: bool,
    pub value_clip: f32,
    pub use_stitching: bool,
    pub use_linear_detrending: bool,
    pub linear_detrending_threshold: f32,
    pub use_iterative_cpm_revin: bool,
    pub use_frozen_running_stats: bool,
    pub input_transform: String,
    /// RMSNorm epsilon. PyTorch `nn.RMSNorm` default when eps is None uses
    /// `torch.finfo(x.dtype).eps` (f32: ~1.19e-7).
    pub rms_eps: f32,
    /// output_patch_len / input_patch_len.
    pub rolls: usize,
    /// Weight storage precision (F32, F16).
    pub precision: QuantizationPrecision,
    /// True when the checkpoint's own `config.json` declared a `precision`
    /// field. In that case the per-tensor dtypes on disk are authoritative
    /// (the checkpoint may be *mixed* — e.g. most weights f16 with a few
    /// sensitive tensors kept f32) and the loader must NOT force-convert
    /// everything to `precision` at assembly time.
    pub precision_declared: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_patch_len: 32,
            output_patch_len: 64,
            num_quantiles: 9,
            median_quantile_index: 4,
            residual_block_config: ResidualBlockConfig::default(),
            transformer_config: StackedTransformersConfig::default(),
            use_variate_attention: true,
            value_clip: 1e20,
            use_stitching: true,
            use_linear_detrending: true,
            linear_detrending_threshold: 0.5,
            use_iterative_cpm_revin: true,
            use_frozen_running_stats: false,
            input_transform: "identity".into(),
            rms_eps: f32::EPSILON,
            rolls: 2,
            precision: QuantizationPrecision::F32,
            precision_declared: false,
        }
    }
}

impl ModelConfig {
    /// Validates cross-field invariants (mirrors the model `__init__` checks).
    pub fn validate(&self) -> Result<()> {
        if !self.output_patch_len.is_multiple_of(self.input_patch_len) {
            return Err(Error(format!(
                "output_patch_len {} must be a multiple of input_patch_len {}",
                self.output_patch_len, self.input_patch_len
            )));
        }
        if self.residual_block_config.output_dims != self.transformer_config.transformer.model_dims
        {
            return Err(Error(
                "ResidualBlock output_dims must match Transformer model_dims".into(),
            ));
        }
        if self.num_quantiles == 0 || self.median_quantile_index >= self.num_quantiles {
            return Err(Error(
                "median_quantile_index must be < num_quantiles".into(),
            ));
        }
        Ok(())
    }

    /// Overlays fields parsed from the HF `config.json` of the checkpoint.
    /// Unknown/missing fields simply keep the defaults.
    pub fn from_config_json(&mut self, root: &Json) -> Result<()> {
        let obj = root
            .as_obj()
            .ok_or_else(|| Error("config.json: not an object".into()))?;

        fn get<'a>(obj: &'a [(String, Json)], k: &str) -> Option<&'a Json> {
            obj.iter().find(|(n, _)| n == k).map(|(_, v)| v)
        }

        if let Some(Json::Int(v)) = get(obj, "input_patch_len") {
            self.input_patch_len = *v as usize;
        }
        if let Some(Json::Int(v)) = get(obj, "output_patch_len") {
            self.output_patch_len = *v as usize;
        }
        if let Some(v) = get(obj, "value_clip") {
            self.value_clip = json_f32(v);
        }
        if let Some(Json::Bool(b)) = get(obj, "use_variate_attention") {
            self.use_variate_attention = *b;
        }
        if let Some(Json::Bool(b)) = get(obj, "use_stitching") {
            self.use_stitching = *b;
        }
        if let Some(Json::Bool(b)) = get(obj, "use_linear_detrending") {
            self.use_linear_detrending = *b;
        }
        if let Some(v) = get(obj, "linear_detrending_threshold") {
            self.linear_detrending_threshold = json_f32(v);
        }
        if let Some(Json::Bool(b)) = get(obj, "use_iterative_cpm_revin") {
            self.use_iterative_cpm_revin = *b;
        }
        if let Some(Json::Bool(b)) = get(obj, "use_frozen_running_stats") {
            self.use_frozen_running_stats = *b;
        }
        if let Some(Json::Str(s)) = get(obj, "input_transform") {
            self.input_transform = s.clone();
        }
        if let Some(Json::Str(s)) = get(obj, "precision").or_else(|| get(obj, "dtype"))
            && let Some(p) = QuantizationPrecision::parse(s)
        {
            self.precision = p;
            self.precision_declared = true;
        }
        if let Some(Json::Arr(a)) = get(obj, "quantiles") {
            self.num_quantiles = a.len();
            // Official forecaster clamps median index after loading the model
            // (median_quantile_index defaults to 4, valid only for 9 quantiles).
            if self.median_quantile_index >= self.num_quantiles {
                self.median_quantile_index = self.num_quantiles / 2;
            }
        }

        if let Some(Json::Obj(rc)) = get(obj, "residual_block_config") {
            let rb = &mut self.residual_block_config;
            if let Some(Json::Int(v)) = get(rc, "hidden_dims") {
                rb.hidden_dims = *v as usize;
            }
            if let Some(Json::Int(v)) = get(rc, "output_dims") {
                rb.output_dims = *v as usize;
            }
            if let Some(Json::Bool(b)) = get(rc, "use_bias") {
                rb.use_bias = *b;
            }
            if let Some(Json::Bool(b)) = get(rc, "identity_skip") {
                rb.identity_skip = *b;
            }
            if let Some(Json::Str(s)) = get(rc, "activation") {
                rb.activation = Activation::from_name(s).unwrap_or_default();
            }
            if let Some(Json::Str(s)) = get(rc, "prenorm") {
                rb.prenorm = Norm::from_name(s).unwrap_or_default();
            }
        }

        if let Some(Json::Obj(sc)) = get(obj, "transformer_config") {
            if let Some(Json::Int(v)) = get(sc, "num_layers") {
                self.transformer_config.num_layers = *v as usize;
            }
            if let Some(Json::Obj(tc)) = get(sc, "transformer") {
                let t = &mut self.transformer_config.transformer;
                if let Some(Json::Int(v)) = get(tc, "model_dims") {
                    t.model_dims = *v as usize;
                }
                if let Some(Json::Int(v)) = get(tc, "hidden_dims") {
                    t.hidden_dims = *v as usize;
                }
                if let Some(Json::Int(v)) = get(tc, "num_heads") {
                    t.num_heads = *v as usize;
                }
                if let Some(Json::Bool(b)) = get(tc, "use_bias") {
                    t.use_bias = *b;
                }
                if let Some(Json::Bool(b)) = get(tc, "use_rope_seq") {
                    t.use_rope_seq = *b;
                }
                if let Some(Json::Bool(b)) = get(tc, "use_rope_var") {
                    t.use_rope_var = *b;
                }
                if let Some(Json::Bool(b)) = get(tc, "causal_attention") {
                    t.causal_attention = *b;
                }
                if let Some(Json::Bool(b)) = get(tc, "use_sdpa") {
                    t.use_sdpa = *b;
                }
                if let Some(Json::Bool(b)) = get(tc, "use_memory_efficient_attention") {
                    t.use_memory_efficient_attention = *b;
                }
                if let Some(Json::Str(s)) = get(tc, "ff_activation") {
                    t.ff_activation = Activation::from_name(s).unwrap_or_default();
                }
                if let Some(Json::Str(s)) = get(tc, "attention_norm") {
                    t.attention_norm = Norm::from_name(s).unwrap_or_default();
                }
                if let Some(Json::Str(s)) = get(tc, "feedforward_norm") {
                    t.feedforward_norm = Norm::from_name(s).unwrap_or_default();
                }
                if let Some(Json::Str(s)) = get(tc, "qk_norm") {
                    t.qk_norm = Norm::from_name(s).unwrap_or_default();
                }
                if let Some(Json::Str(s)) = get(tc, "v_norm") {
                    t.v_norm = Norm::from_name(s).unwrap_or_default();
                }
            }
        }

        self.rolls = self.output_patch_len / self.input_patch_len;
        Ok(())
    }
}

fn json_f32(v: &Json) -> f32 {
    match v {
        Json::Int(i) => *i as f32,
        Json::Float(f) => *f as f32,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_json_parses_official_shape() {
        let src = br#"{
          "input_patch_len": 32, "output_patch_len": 64,
          "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
          "residual_block_config": {"hidden_dims": 1280, "output_dims": 1280,
            "use_bias": false, "activation": "relu", "prenorm": "none"},
          "transformer_config": {"num_layers": 20, "transformer": {
            "model_dims": 1280, "hidden_dims": 1280, "num_heads": 16,
            "use_rope_seq": true, "use_rope_var": false, "qk_norm": "rms",
            "ff_activation": "relu", "use_sdpa": true}}
        }"#;
        let root = Json::parse(src).unwrap();
        let mut cfg = ModelConfig::default();
        cfg.from_config_json(&root).unwrap();
        assert_eq!(cfg.input_patch_len, 32);
        assert_eq!(cfg.num_quantiles, 9);
        assert_eq!(cfg.transformer_config.num_layers, 20);
        assert_eq!(cfg.transformer_config.transformer.model_dims, 1280);
        assert_eq!(cfg.transformer_config.transformer.qk_norm, Norm::Rms);
        assert!(cfg.validate().is_ok());
    }
}
