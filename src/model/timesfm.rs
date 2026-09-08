//! TimesFM3 model: preprocessing, forward, and non-autoregressive decode.
//!
//! Faithful port of the official `model.py` (inference-only) to flat row-major
//! buffers. Layout invariant: a `(b, v, n, p)` tensor is stored as one flat
//! row-major array with `bv = b*v` merged as a single axis, i.e.
//! `(b*v*n, p)` ≡ `(b*v, n*p)` — identical memory. Rows must be built in
//! batch-major, within-batch variate-major order so this equivalence holds.

use crate::checkpoint::Safetensors;
use crate::config::ModelConfig;
use crate::error::{Error, Result};
use crate::model::mixing_transformer::StackedMixingTransformer;
use crate::model::residual_block::{Linear, ResidualBlock, load_linear};
use crate::pipeline::{
    get_output_patch_via_roll, get_output_patch_via_roll_bool, get_running_stats, stitch_patches,
};
use crate::tensor::Tensor2D;

static DUMP_DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

/// Returns true if $TENSORSFM3_DUMP_DIR is configured.
#[inline]
pub fn is_dump_enabled() -> bool {
    DUMP_DIR
        .get_or_init(|| std::env::var("TENSORSFM3_DUMP_DIR").ok())
        .is_some()
}

/// Debug helper: dump an f32 buffer to `$TENSORSFM3_DUMP_DIR/<name>.bin`
/// (little-endian f32, row-major flat). No-op unless the env var is set.
pub fn dbg_dump(name: &str, data: &[f32]) {
    let Some(dir) = DUMP_DIR
        .get_or_init(|| std::env::var("TENSORSFM3_DUMP_DIR").ok())
        .as_ref()
    else {
        return;
    };
    use std::io::Write;
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Ok(mut f) = std::fs::File::create(std::path::Path::new(dir).join(format!("{name}.bin")))
    {
        let _ = f.write_all(&bytes);
    }
}

/// The TimesFM3 model (inference only).
#[derive(Debug, Clone)]
pub struct TimesFM3 {
    pub config: ModelConfig,
    residual_block: ResidualBlock,
    transformer: StackedMixingTransformer,
    output_head: Linear,
}

/// Result of the full forward: logits plus the RevIN running statistics used
/// for denormalization.
pub struct ForwardOutput {
    /// Logits `(b*v*n, o*q)` in RevIN-normalized space (before reverse RevIN).
    pub logits: Tensor2D,
    /// Running count of valid values per patch `(b*v, n)` flat.
    pub revin_n: Vec<f32>,
    /// Running mean `(b*v, n)` flat.
    pub revin_mean: Vec<f32>,
    /// Running std `(b*v, n)` flat.
    pub revin_std: Vec<f32>,
}

/// Output of preprocessing: residual-block output, patch mask, and RevIN stats.
pub struct PreprocessOutput {
    /// Residual block output `(b*v*n, d)`.
    pub resblock_out: Tensor2D,
    /// Patch mask `(b*v, n)`: true = fully-masked patch.
    pub patch_mask: Vec<bool>,
    /// Running count of valid values `(b*v, n)` flat.
    pub running_n: Vec<f32>,
    /// Running mean `(b*v, n)` flat.
    pub running_mean: Vec<f32>,
    /// Running std `(b*v, n)` flat.
    pub running_std: Vec<f32>,
}

impl TimesFM3 {
    pub fn transformer(&self) -> &StackedMixingTransformer {
        &self.transformer
    }

    pub fn load(store: &Safetensors, config: &ModelConfig) -> Result<TimesFM3> {
        crate::init_default_thread_pool();
        config.validate()?;
        let t = &config.transformer_config.transformer;

        let residual_block = ResidualBlock::load(
            store,
            "pre_transformer_resblock",
            &config.residual_block_config,
            config.rms_eps,
        )?;
        let expected_input = 2 * (config.input_patch_len + config.output_patch_len);
        if residual_block.input_dims() != expected_input {
            return Err(Error(format!(
                "TimesFM3: residual block input {} != expected 2*(p+o) = {expected_input}",
                residual_block.input_dims()
            )));
        }

        let transformer =
            StackedMixingTransformer::load(store, &config.transformer_config, config.rms_eps)?;

        let output_head = load_linear(store, "output_head", "", true)?;
        if output_head.in_features() != t.model_dims
            || output_head.out_features() != config.output_patch_len * config.num_quantiles
        {
            return Err(Error(format!(
                "TimesFM3: output_head ({}->{}) != config ({}->{})",
                output_head.in_features(),
                output_head.out_features(),
                t.model_dims,
                config.output_patch_len * config.num_quantiles
            )));
        }

        let mut model = TimesFM3 {
            config: config.clone(),
            residual_block,
            transformer,
            output_head,
        };
        // Apply runtime quantization ONLY when the checkpoint does not declare
        // a precision itself. A checkpoint that declares one (including mixed
        // f32/f16 checkpoints produced by `quantize --keep-f32`) is loaded
        // per-tensor-dtype already; force-converting here would defeat the
        // kept-f32 tensors.
        if !config.precision_declared
            && config.precision != crate::config::QuantizationPrecision::F32
        {
            model.quantize(config.precision);
        }
        Ok(model)
    }

    /// Quantizes (or dequantizes) all linear weight matrices across the model.
    pub fn quantize(&mut self, precision: crate::config::QuantizationPrecision) {
        self.residual_block.quantize(precision);
        self.transformer.quantize(precision);
        self.output_head.quantize(precision);
        self.config.precision = precision;
    }

    /// Preprocessing (official `_preprocess`): running RevIN stats, optional
    /// CPM masking, normalization, rolled future covariates, residual block.
    #[allow(clippy::too_many_arguments)]
    pub fn preprocess(
        &self,
        values: &[f32],
        masks: &[bool],
        patch_is_target: &[bool],
        b: usize,
        v: usize,
        n: usize,
        patch_cpm_mask: Option<&[bool]>, // (b, n)
        freeze_after: Option<usize>,
    ) -> Result<PreprocessOutput> {
        let cfg = &self.config;
        let p = cfg.input_patch_len;
        let rolls = cfg.rolls;
        let bv = b * v;

        let (running_n, mut running_mean, mut running_std) =
            get_running_stats(values, masks, bv, n, p);
        if let Some(fa) = freeze_after {
            let fa = fa.min(n - 1);
            for bi in 0..bv {
                let vm = running_mean[bi * n + fa];
                let vs = running_std[bi * n + fa];
                for ni in fa + 1..n {
                    running_mean[bi * n + ni] = vm;
                    running_std[bi * n + ni] = vs;
                }
            }
        }

        // CPM mask: additionally mask target variates at CPM positions.
        let mut masks_owned = masks.to_vec();
        if let Some(cpm) = patch_cpm_mask {
            for bvi in 0..bv {
                for ni in 0..n {
                    let bi = bvi / v;
                    if cpm[bi * n + ni] && patch_is_target[bvi * n + ni] {
                        let base = (bvi * n + ni) * p;
                        for j in 0..p {
                            masks_owned[base + j] = true;
                        }
                    }
                }
            }
        }
        let masks = &masks_owned;

        // RevIN normalize values per patch.
        let mut values_bvnp = values.to_vec();
        for bvi in 0..bv {
            for ni in 0..n {
                let mu = running_mean[bvi * n + ni];
                let sigma = running_std[bvi * n + ni];
                let safe = if sigma < 1e-6 { 1.0 } else { sigma };
                let base = (bvi * n + ni) * p;
                for j in 0..p {
                    let idx = base + j;
                    values_bvnp[idx] = if masks[idx] {
                        0.0
                    } else {
                        (values_bvnp[idx] - mu) / safe
                    };
                }
            }
        }

        // Rolled future covariates (values + normalization).
        let (mut values_fcov, wrap_mask) = get_output_patch_via_roll(&values_bvnp, bv, n, p, rolls);
        for bvi in 0..bv {
            for ni in 0..n {
                let mu = running_mean[bvi * n + ni];
                let sigma = running_std[bvi * n + ni];
                let safe = if sigma < 1e-6 { 1.0 } else { sigma };
                let base = (bvi * n + ni) * rolls * p;
                for j in 0..rolls * p {
                    values_fcov[base + j] = (values_fcov[base + j] - mu) / safe;
                }
            }
        }

        // Future-covariate masks.
        let (mut masks_fcov, _) = get_output_patch_via_roll_bool(masks, bv, n, p, rolls);
        for bvi in 0..bv {
            if patch_is_target[bvi * n] {
                for ni in 0..n {
                    let b = (bvi * n + ni) * rolls * p;
                    for j in 0..rolls * p {
                        masks_fcov[b + j] = true;
                    }
                }
            }
        }
        for idx in 0..masks_fcov.len() {
            if wrap_mask[idx] {
                masks_fcov[idx] = true;
            }
        }
        for idx in 0..values_fcov.len() {
            if masks_fcov[idx] {
                values_fcov[idx] = 0.0;
            }
        }

        // Concat values|fcov, masks|fcov_mask → residual block input.
        let total = p + rolls * p;
        let fcov_row = rolls * p; // values_fcov/masks_fcov row stride
        let mut resblock_in = vec![0.0f32; bv * n * 2 * total];
        for bvi in 0..bv {
            for ni in 0..n {
                // Three different strides: values (p), fcov (rolls*p),
                // resblock_in (2*total).
                let vals_base = (bvi * n + ni) * p;
                let fcov_base = (bvi * n + ni) * fcov_row;
                let out_base = (bvi * n + ni) * 2 * total;
                resblock_in[out_base..out_base + p]
                    .copy_from_slice(&values_bvnp[vals_base..vals_base + p]);
                resblock_in[out_base + p..out_base + total]
                    .copy_from_slice(&values_fcov[fcov_base..fcov_base + fcov_row]);
                for j in 0..p {
                    resblock_in[out_base + total + j] =
                        if masks[vals_base + j] { 1.0 } else { 0.0 };
                }
                for j in 0..fcov_row {
                    resblock_in[out_base + total + p + j] =
                        if masks_fcov[fcov_base + j] { 1.0 } else { 0.0 };
                }
            }
        }

        let rb_in = Tensor2D::from_row_major(bv * n, 2 * total, resblock_in)?;
        let rb_out = self.residual_block.forward(&rb_in);

        // Patch fully masked when every value and fcov point is masked.
        let mut patch_mask = vec![false; bv * n];
        for bvi in 0..bv {
            for ni in 0..n {
                let base = (bvi * n + ni) * p;
                let mut all = true;
                for j in 0..p {
                    if !masks[base + j] {
                        all = false;
                        break;
                    }
                }
                if all {
                    for j in 0..rolls * p {
                        if !masks_fcov[(bvi * n + ni) * rolls * p + j] {
                            all = false;
                            break;
                        }
                    }
                }
                patch_mask[bvi * n + ni] = all;
            }
        }

        Ok(PreprocessOutput {
            resblock_out: rb_out,
            patch_mask,
            running_n,
            running_mean,
            running_std,
        })
    }

    /// Full-sequence forward (official `forward`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        values: &[f32],
        masks: &[bool],
        patch_is_target: &[bool],
        b: usize,
        v: usize,
        n: usize,
        patch_cpm_mask: Option<&[bool]>,
        freeze_after: Option<usize>,
    ) -> Result<ForwardOutput> {
        let cfg = &self.config;
        let p = cfg.input_patch_len;
        let d = cfg.transformer_config.transformer.model_dims;
        let bv = b * v;

        let mut values = values.to_vec();
        for val in values.iter_mut() {
            if val.is_nan() {
                *val = 0.0;
            }
            *val = val.clamp(-cfg.value_clip, cfg.value_clip);
        }
        if values.len() != bv * n * p {
            return Err(Error(format!(
                "TimesFM3::forward: values len {} != b*v*n*p = {}",
                values.len(),
                bv * n * p
            )));
        }

        let pre = self.preprocess(
            &values,
            masks,
            patch_is_target,
            b,
            v,
            n,
            patch_cpm_mask,
            freeze_after,
        )?;
        dbg_dump("resblock_input", &values);
        dbg_dump("revin_mean", &pre.running_mean);
        dbg_dump("revin_std", &pre.running_std);
        dbg_dump("revin_n", &pre.running_n);
        let transformer_input = pre.resblock_out;
        dbg_dump("resblock_out", transformer_input.data());
        let raw_patch_mask = pre.patch_mask;
        let running_n = pre.running_n;
        let running_mean = pre.running_mean;
        let running_std = pre.running_std;

        // Effective patch mask = cumprod along n (leading fully-masked patches
        // hide everything after them).
        let mut effective = vec![false; bv * n];
        for bvi in 0..bv {
            let mut acc = true;
            for ni in 0..n {
                acc = acc && raw_patch_mask[bvi * n + ni];
                effective[bvi * n + ni] = acc;
            }
        }

        let transformer_out = self
            .transformer
            .forward(&transformer_input, b, v, n, &effective);
        debug_assert_eq!(transformer_out.cols(), d);
        dbg_dump("transformer_out", transformer_out.data());

        let logits = self.output_head.forward(&transformer_out);
        dbg_dump("logits_raw", logits.data());

        Ok(ForwardOutput {
            logits,
            revin_n: running_n,
            revin_mean: running_mean,
            revin_std: running_std,
        })
    }

    /// Non-autoregressive decode (official `decode`) → logits
    /// `(b, total_variates, horizon, q)` flat as `(b*total_variates*horizon*q)`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_shaped(
        &self,
        target: &Tensor2D, // (b*u, C)
        horizon: usize,
        past_only_covariates: Option<&Tensor2D>, // (b*v_po, C)
        past_future_covariates: Option<&Tensor2D>, // (b*w, C+horizon)
        mask: Option<&[bool]>,                   // (b, C)
        target_mask: Option<&[bool]>,            // (b*u, C)
        past_only_mask: Option<&[bool]>,         // (b*v_po, C)
        past_future_mask: Option<&[bool]>,       // (b*w, C+horizon)
        b: usize,
        u: usize,
        context: usize,
    ) -> Result<Tensor2D> {
        let cfg = &self.config;
        let p = cfg.input_patch_len;
        let o = cfg.output_patch_len;
        let q = cfg.num_quantiles;
        let rolls = cfg.rolls;

        if horizon == 0 {
            return Err(Error("decode: horizon must be > 0".into()));
        }
        if b == 0 {
            return Err(Error("decode: batch size b must be > 0".into()));
        }
        if let Some(t) = past_only_covariates
            && t.rows() % b != 0
        {
            return Err(Error(format!(
                "decode: past_only_covariates rows {} not divisible by batch size b={b}",
                t.rows()
            )));
        }
        if let Some(t) = past_future_covariates
            && t.rows() % b != 0
        {
            return Err(Error(format!(
                "decode: past_future_covariates rows {} not divisible by batch size b={b}",
                t.rows()
            )));
        }
        let v_po = past_only_covariates.map(|t| t.rows() / b).unwrap_or(0);
        let w_pf = past_future_covariates.map(|t| t.rows() / b).unwrap_or(0);
        let v_total = u + v_po + w_pf;

        // ---- 1. pad context left to multiple of input_patch_len ----
        let ctx_padding = (p - (context % p)) % p;
        let ctx_padded = context + ctx_padding;

        // ---- 2. horizon patch math ----
        let (num_forecast_patches, num_horizon_patches, padded_horizon, extract_len) =
            if cfg.use_stitching {
                let extract_len = (2 * p).min(o);
                let overlap = extract_len - p;
                let num_forecast_patches = (horizon.saturating_sub(overlap)).div_ceil(p).max(1);
                let num_horizon_patches = num_forecast_patches + rolls - 1;
                (
                    num_forecast_patches,
                    num_horizon_patches,
                    num_horizon_patches * p,
                    extract_len,
                )
            } else {
                let hor_padding = (o - (horizon % o)) % o;
                let padded_horizon = horizon + hor_padding;
                (0, padded_horizon / p, padded_horizon, o)
            };
        let num_context_patches = ctx_padded / p;
        let num_total_patches = num_context_patches + num_horizon_patches;
        let total_len = ctx_padded + padded_horizon;

        // ---- 3. build batched context + horizon (b, v_total, *) ----
        // Global mask (b, ctx_padded); true = left padding. Matches the
        // official decode: when `mask` is None a fresh mask is created and the
        // left-padding positions are marked masked.
        let msk = match mask {
            Some(m) => pad_left_bool(m, b, context, ctx_padded),
            None => {
                let mut m = vec![false; b * ctx_padded];
                for bi in 0..b {
                    m[bi * ctx_padded..bi * ctx_padded + ctx_padding].fill(true);
                }
                m
            }
        };
        let msk: Vec<bool> = msk;
        let tgt_m = target_mask.map(|m| pad_left_bool(m, b * u, context, ctx_padded));
        let po_m = past_only_mask.map(|m| pad_left_bool(m, b * v_po, context, ctx_padded));
        let pf_m =
            past_future_mask.map(|m| pad_left_pf_bool(m, b * w_pf, context, ctx_padded, horizon));

        let tgt_data = pad_left_f(target.data(), b * u, context, ctx_padded);
        let po_data =
            past_only_covariates.map(|t| pad_left_f(t.data(), b * v_po, context, ctx_padded));
        let pf_data = past_future_covariates
            .map(|t| pad_left_pf_f(t.data(), b * w_pf, context, ctx_padded, horizon));

        // ctx_vals (b, v_total, ctx_padded), ctx_masks bool same.
        let mut ctx_vals = vec![0.0f32; b * v_total * ctx_padded];
        let mut ctx_masks = vec![false; b * v_total * ctx_padded];
        for bi in 0..b {
            for vi in 0..u {
                let row = (bi * v_total + vi) * ctx_padded;
                ctx_vals[row..row + ctx_padded].copy_from_slice(
                    &tgt_data[(bi * u + vi) * ctx_padded..(bi * u + vi + 1) * ctx_padded],
                );
                let gm = &msk[bi * ctx_padded..(bi + 1) * ctx_padded];
                for c in 0..ctx_padded {
                    let masked = gm[c]
                        || tgt_m
                            .as_ref()
                            .is_some_and(|m| m[(bi * u + vi) * ctx_padded + c]);
                    ctx_masks[row + c] = masked;
                }
            }
            for vi in 0..v_po {
                let row = (bi * v_total + u + vi) * ctx_padded;
                if let Some(pd) = &po_data {
                    ctx_vals[row..row + ctx_padded].copy_from_slice(
                        &pd[(bi * v_po + vi) * ctx_padded..(bi * v_po + vi + 1) * ctx_padded],
                    );
                }
                let gm = &msk[bi * ctx_padded..(bi + 1) * ctx_padded];
                for c in 0..ctx_padded {
                    let masked = gm[c]
                        || po_m
                            .as_ref()
                            .is_some_and(|m| m[(bi * v_po + vi) * ctx_padded + c]);
                    ctx_masks[row + c] = masked;
                }
            }
            for vi in 0..w_pf {
                let row = (bi * v_total + u + v_po + vi) * ctx_padded;
                if let Some(pd) = &pf_data {
                    ctx_vals[row..row + ctx_padded].copy_from_slice(
                        &pd[(bi * w_pf + vi) * (ctx_padded + horizon)
                            ..(bi * w_pf + vi) * (ctx_padded + horizon) + ctx_padded],
                    );
                }
                let gm = &msk[bi * ctx_padded..(bi + 1) * ctx_padded];
                for c in 0..ctx_padded {
                    let masked = gm[c]
                        || pf_m
                            .as_ref()
                            .is_some_and(|m| m[(bi * w_pf + vi) * (ctx_padded + horizon) + c]);
                    ctx_masks[row + c] = masked;
                }
            }
        }

        // ---- 4. linear detrending per (b, v) row ----
        let mut m_trend = vec![0.0f32; b * v_total];
        let mut c_trend = vec![0.0f32; b * v_total];
        let mut apply_detrend = vec![false; b * v_total];
        if cfg.use_linear_detrending {
            let t: Vec<f32> = (0..ctx_padded)
                .map(|i| (i as f32 - (ctx_padded - 1) as f32) / ctx_padded as f32)
                .collect();
            let threshold = cfg.linear_detrending_threshold;
            for row in 0..b * v_total {
                let base = row * ctx_padded;
                // Regression sums in f64 (correctly rounded m/c).
                let mut n_v = 0.0f64;
                let (mut st, mut st2, mut sy, mut sty, mut sy2) =
                    (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for c in 0..ctx_padded {
                    if ctx_masks[base + c] {
                        continue;
                    }
                    let y = ctx_vals[base + c] as f64;
                    let t64 = t[c] as f64;
                    n_v += 1.0;
                    st += t64;
                    st2 += t64 * t64;
                    sy += y;
                    sty += t64 * y;
                    sy2 += y * y;
                }
                let mean_y = if n_v > 0.0 { sy / n_v } else { 0.0 };
                let std_orig = (sy2 / n_v.max(1.0) - mean_y * mean_y).max(0.0).sqrt();
                let det = n_v * st2 - st * st;
                let (m, c) = if det == 0.0 {
                    (0.0, mean_y)
                } else {
                    let m = (n_v * sty - st * sy) / det;
                    (m, (sy - m * st) / n_v)
                };
                m_trend[row] = m as f32;
                c_trend[row] = c as f32;
                let (m32, c32) = (m as f32, c as f32);
                for cidx in 0..ctx_padded {
                    if !ctx_masks[base + cidx] {
                        ctx_vals[base + cidx] -= m32 * t[cidx] + c32;
                    }
                }
                let mut sd = 0.0f64;
                let mut sd2 = 0.0f64;
                for cidx in 0..ctx_padded {
                    if !ctx_masks[base + cidx] {
                        let d = ctx_vals[base + cidx] as f64;
                        sd += d;
                        sd2 += d * d;
                    }
                }
                let mean_d = sd / n_v.max(1.0);
                let std_det = (sd2 / n_v.max(1.0) - mean_d * mean_d).max(0.0).sqrt();
                apply_detrend[row] = (std_det as f32) < threshold * (std_orig as f32);
                if !apply_detrend[row] {
                    // restore original values
                    for cidx in 0..ctx_padded {
                        if !ctx_masks[base + cidx] {
                            ctx_vals[base + cidx] += m32 * t[cidx] + c32;
                        }
                    }
                }
            }
        }
        // zero masked ctx
        for idx in 0..ctx_masks.len() {
            if ctx_masks[idx] {
                ctx_vals[idx] = 0.0;
            }
        }

        // ---- 5. horizon values & masks ----
        // target+po horizon: zeros all-masked; pf: real future values.
        let mut hor_vals = vec![0.0f32; b * v_total * padded_horizon];
        let mut hor_masks = vec![true; b * v_total * padded_horizon];
        if let Some(pf_src) = &pf_data {
            for bi in 0..b {
                for vi in 0..w_pf {
                    let row = (bi * v_total + u + v_po + vi) * padded_horizon;
                    hor_masks[row..row + padded_horizon].fill(false);
                    let srow = (bi * w_pf + vi) * (ctx_padded + horizon) + ctx_padded;
                    for hh in 0..horizon {
                        hor_vals[row + hh] = pf_src[srow + hh];
                        if pf_m.as_ref().is_some_and(|m| m[srow + hh]) {
                            hor_masks[row + hh] = true;
                        }
                    }
                    // apply pf detrend if active
                    let prow = bi * v_total + u + v_po + vi;
                    if apply_detrend[prow] {
                        for hh in 0..horizon {
                            let trend =
                                m_trend[prow] * ((hh + 1) as f32 / context as f32) + c_trend[prow];
                            hor_vals[row + hh] -= trend;
                        }
                    }
                    for hh in 0..horizon {
                        if hor_masks[row + hh] {
                            hor_vals[row + hh] = 0.0;
                        }
                    }
                }
            }
        }

        // ---- 6. assemble all_vals/all_masks, patch, run forward ----
        let mut all_vals = vec![0.0f32; b * v_total * total_len];
        let mut all_masks = vec![false; b * v_total * total_len];
        for r in 0..b * v_total {
            let cbase = r * total_len;
            all_vals[cbase..cbase + ctx_padded]
                .copy_from_slice(&ctx_vals[r * ctx_padded..(r + 1) * ctx_padded]);
            all_masks[cbase..cbase + ctx_padded]
                .copy_from_slice(&ctx_masks[r * ctx_padded..(r + 1) * ctx_padded]);
            let hbase = r * total_len + ctx_padded;
            all_vals[hbase..hbase + padded_horizon]
                .copy_from_slice(&hor_vals[r * padded_horizon..(r + 1) * padded_horizon]);
            all_masks[hbase..hbase + padded_horizon]
                .copy_from_slice(&hor_masks[r * padded_horizon..(r + 1) * padded_horizon]);
        }

        // Patched views (b*v_total, num_total_patches, p) → flat (b*v_total*n, p).
        let bv = b * v_total;
        let mut values_bvnp = vec![0.0f32; bv * num_total_patches * p];
        let mut masks_bvnp = vec![false; bv * num_total_patches * p];
        for r in 0..bv {
            for ni in 0..num_total_patches {
                let dst = ((r * num_total_patches) + ni) * p;
                let src = r * total_len + ni * p;
                values_bvnp[dst..dst + p].copy_from_slice(&all_vals[src..src + p]);
                masks_bvnp[dst..dst + p].copy_from_slice(&all_masks[src..src + p]);
            }
        }

        // patch_is_target (b*v_total, num_total_patches): target+po = true.
        let mut patch_is_target = vec![false; bv * num_total_patches];
        for r in 0..bv {
            let vi = r % v_total;
            if vi < u + v_po {
                for ni in 0..num_total_patches {
                    patch_is_target[r * num_total_patches + ni] = true;
                }
            }
        }

        // horizon CPM mask (b, num_total_patches).
        let mut horizon_cpm_mask = vec![false; b * num_total_patches];
        for bi in 0..b {
            for ni in num_context_patches..num_total_patches {
                horizon_cpm_mask[bi * num_total_patches + ni] = true;
            }
        }

        let freeze_after = if cfg.use_frozen_running_stats {
            Some(num_context_patches.saturating_sub(1))
        } else {
            None
        };

        let out = self.forward(
            &values_bvnp,
            &masks_bvnp,
            &patch_is_target,
            b,
            v_total,
            num_total_patches,
            Some(&horizon_cpm_mask),
            freeze_after,
        )?;

        // ---- 6.5 CPM-RevIN refinement (official forward step, only when
        // enabled) — replaces RevIN stats at CPM (horizon) positions with
        // iteratively refined estimates before reverse RevIN. ----
        let (revin_mean, revin_std) = if cfg.use_iterative_cpm_revin {
            let (ref_mu, ref_sig) = crate::pipeline::cpm_iterative_revin_refine(
                &out.logits,
                &out.revin_n,
                &out.revin_mean,
                &out.revin_std,
                &horizon_cpm_mask,
                b,
                v_total,
                num_total_patches,
                p,
                cfg.rolls,
                cfg.num_quantiles,
                cfg.median_quantile_index,
                cfg.value_clip,
            );
            (ref_mu, ref_sig)
        } else {
            (out.revin_mean.clone(), out.revin_std.clone())
        };

        // ---- 7. reverse RevIN + clamp → (b, v_total, n, o, q) ----
        let n_patches = num_total_patches;
        let mut denorm = out.logits.into_data();
        let oq = o * q;
        for bvi in 0..bv {
            for ni in 0..n_patches {
                let mu = revin_mean[bvi * n_patches + ni];
                let sigma = revin_std[bvi * n_patches + ni];
                let base = (bvi * n_patches + ni) * oq;
                for v in denorm[base..base + oq].iter_mut() {
                    *v = (*v * sigma + mu).clamp(-cfg.value_clip, cfg.value_clip);
                }
            }
        }

        // ---- 8. extract forecast patches → stitch / chunk, cut to horizon ----
        let horizon_logits: Vec<f32> = if cfg.use_stitching {
            // forecast_indices = arange(nfp) + (nc-1); patch_preds extract_len
            let nfp = num_forecast_patches.max(1);
            let mut stitched = stitch_from(
                &denorm,
                bv,
                n_patches,
                num_context_patches,
                nfp,
                extract_len,
                p,
                q,
                o,
            );
            // cut to horizon
            stitched.truncate(bv * horizon * q);
            stitched
        } else {
            let _num_chunks = padded_horizon / o;
            let mut h = vec![0.0f32; bv * horizon * q];
            for bvi in 0..bv {
                for hh in 0..horizon {
                    let chunk = hh / o;
                    let rem = hh % o;
                    let fi = num_context_patches - 1 + chunk * rolls;
                    let src = &denorm[(bvi * n_patches + fi) * oq + rem * q..];
                    for qq in 0..q {
                        h[(bvi * horizon + hh) * q + qq] = src[qq];
                    }
                }
            }
            h
        };

        // ---- 9. add trend back (if detrending applied) ----
        let mut out_flat = horizon_logits;
        if cfg.use_linear_detrending {
            let mut new = vec![0.0f32; bv * horizon * q];
            for bvi in 0..bv {
                if apply_detrend[bvi] {
                    for hh in 0..horizon {
                        let trend =
                            m_trend[bvi] * ((hh + 1) as f32 / context as f32) + c_trend[bvi];
                        for qq in 0..q {
                            new[(bvi * horizon + hh) * q + qq] =
                                out_flat[(bvi * horizon + hh) * q + qq] + trend;
                        }
                    }
                } else {
                    let base = bvi * horizon * q;
                    new[base..base + horizon * q]
                        .copy_from_slice(&out_flat[base..base + horizon * q]);
                }
            }
            out_flat = new;
        }

        Tensor2D::from_row_major(bv * horizon, q, out_flat)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn pad_left_f(data: &[f32], rows: usize, width: usize, width_new: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * width_new];
    for r in 0..rows {
        let src = &data[r * width..(r + 1) * width];
        let dst = &mut out[r * width_new + (width_new - width)..(r + 1) * width_new];
        dst.copy_from_slice(src);
    }
    out
}

fn pad_left_bool(m: &[bool], rows: usize, width: usize, width_new: usize) -> Vec<bool> {
    let mut out = vec![true; rows * width_new];
    for r in 0..rows {
        let src = &m[r * width..(r + 1) * width];
        let dst = &mut out[r * width_new + (width_new - width)..(r + 1) * width_new];
        dst.copy_from_slice(src);
    }
    out
}

/// Left-pads past-future covariates `(rows, context+horizon)` →
/// `(rows, ctx_padded+horizon)` (future part stays).
fn pad_left_pf_f(
    data: &[f32],
    rows: usize,
    context: usize,
    ctx_padded: usize,
    horizon: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * (ctx_padded + horizon)];
    for r in 0..rows {
        let src = &data[r * (context + horizon)..(r + 1) * (context + horizon)];
        let dst = &mut out
            [r * (ctx_padded + horizon) + (ctx_padded - context)..(r + 1) * (ctx_padded + horizon)];
        dst.copy_from_slice(src);
    }
    out
}

fn pad_left_pf_bool(
    m: &[bool],
    rows: usize,
    context: usize,
    ctx_padded: usize,
    horizon: usize,
) -> Vec<bool> {
    let mut out = vec![true; rows * (ctx_padded + horizon)];
    for r in 0..rows {
        let src = &m[r * (context + horizon)..(r + 1) * (context + horizon)];
        let dst = &mut out
            [r * (ctx_padded + horizon) + (ctx_padded - context)..(r + 1) * (ctx_padded + horizon)];
        dst.copy_from_slice(src);
    }
    out
}

/// Extracts the forecast patches and stitches them (official stitching block).
#[allow(clippy::too_many_arguments)]
fn stitch_from(
    denorm: &[f32], // (b*v, n, o*q)
    bv: usize,
    n_patches: usize,
    num_context_patches: usize,
    num_forecast_patches: usize,
    extract_len: usize,
    patch_len: usize,
    q: usize,
    o: usize,
) -> Vec<f32> {
    let mut patch_preds = vec![0.0f32; bv * num_forecast_patches * extract_len * q];
    for bvi in 0..bv {
        for k in 0..num_forecast_patches {
            let fi = num_context_patches - 1 + k;
            let src_base = (bvi * n_patches + fi) * (o * q);
            let dst_base = (bvi * num_forecast_patches + k) * extract_len * q;
            patch_preds[dst_base..dst_base + extract_len * q]
                .copy_from_slice(&denorm[src_base..src_base + extract_len * q]);
        }
    }
    stitch_patches(
        &patch_preds,
        bv,
        num_forecast_patches,
        extract_len,
        patch_len,
        q,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_left_moves_data() {
        let out = pad_left_f(&[1.0, 2.0, 3.0, 4.0], 2, 2, 3);
        assert_eq!(out, vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn pad_left_bool_true() {
        let out = pad_left_bool(&[false, true, true, false], 2, 2, 3);
        assert_eq!(out, vec![true, false, true, true, true, false]);
    }

    #[test]
    fn pad_left_pf_keeps_future() {
        // context=2, pad to 3, horizon=2 → rows of 5
        let out = pad_left_pf_f(&[1.0, 2.0, 9.0, 8.0], 1, 2, 3, 2);
        assert_eq!(out, vec![0.0, 1.0, 2.0, 9.0, 8.0]);
    }
}

#[cfg(test)]
mod detrend_tests {
    use crate::tensor::Tensor2D;

    /// Runs just the detrending decision block (lines ~474-532 of decode_shaped)
    /// on a synthetic row and returns (apply, m, c).
    fn detrend_decision(
        ctx_vals: &Tensor2D,
        ctx_masks: &[bool],
        threshold: f32,
    ) -> (bool, f32, f32) {
        let ctx_padded = ctx_vals.cols();
        let t: Vec<f32> = (0..ctx_padded)
            .map(|i| (i as f32 - (ctx_padded - 1) as f32) / ctx_padded as f32)
            .collect();
        let row = 0usize;
        let base = row * ctx_padded;
        let (mut n_v, mut st, mut st2, mut sy, mut sty, mut sy2) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for c in 0..ctx_padded {
            if ctx_masks[base + c] {
                continue;
            }
            let y = ctx_vals[(row, c)];
            n_v += 1.0;
            st += t[c];
            st2 += t[c] * t[c];
            sy += y;
            sty += t[c] * y;
            sy2 += y * y;
        }
        let mean_y = sy / n_v;
        let std_orig = (sy2 / n_v - mean_y * mean_y).max(0.0).sqrt();
        let det = n_v * st2 - st * st;
        let (m, c0) = if det == 0.0 {
            (0.0, mean_y)
        } else {
            let m = (n_v * sty - st * sy) / det;
            (m, (sy - m * st) / n_v)
        };
        // std of residuals
        let mut sd2 = 0.0f32;
        for c in 0..ctx_padded {
            if ctx_masks[base + c] {
                continue;
            }
            let d = ctx_vals[(row, c)] - (m * t[c] + c0);
            sd2 += d * d;
        }
        let std_det = (sd2 / n_v).max(0.0).sqrt();
        (std_det < threshold * std_orig, m, c0)
    }

    #[test]
    fn linear_trend_triggers_detrend() {
        // y = 3t + 2 exactly: slope 3*ctx_padded/... in normalized t, apply=true
        let n = 64usize;
        let vals: Vec<f32> = (0..n)
            .map(|i| 3.0 * (i as f32 - (n - 1) as f32) / n as f32 + 2.0)
            .collect();
        let x = Tensor2D::from_slice(1, n, &vals).expect("fixture");
        let masks = vec![false; n];
        let (apply, m, c) = detrend_decision(&x, &masks, 0.5);
        assert!(apply, "clean linear trend must be detrended");
        // The normalized-t slope should be ~3.0 (t spans ~[-1, 1) at unit scale? t=(i-(n-1))/n ∈ [-0.984,0.0]).
        // Verify residual std ≈ 0 → m slope recovers the 3.0 gradient per t-unit.
        assert!(m.abs() > 0.0);
        assert!((m).abs() > 0.5, "slope magnitude {m}");
        assert!(c.abs() > 0.0);
    }

    #[test]
    fn white_noise_not_detrended() {
        let n = 64usize;
        let mut rng_state = 0x9e3779b9u32;
        let mut nxt = || {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng_state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        };
        let vals: Vec<f32> = (0..n).map(|_| nxt()).collect();
        let x = Tensor2D::from_slice(1, n, &vals).expect("fixture");
        let masks = vec![false; n];
        let (apply, _, _) = detrend_decision(&x, &masks, 0.5);
        // Noise has no linear component: residual std ≈ full std → no detrend.
        assert!(!apply, "white noise should NOT be detrended");
    }

    #[test]
    fn masked_points_are_ignored() {
        let n = 64usize;
        // Half the points masked (NaN-like); remaining follow a clean line.
        let mut vals = vec![0.0f32; n];
        for (i, v) in vals.iter_mut().enumerate() {
            if i % 2 == 0 {
                *v = 2.0 * (i as f32 - (n - 1) as f32) / n as f32;
            }
        }
        let mut masks = vec![false; n];
        for i in (1..n).step_by(2) {
            masks[i] = true;
        }
        let x = Tensor2D::from_slice(1, n, &vals).expect("fixture");
        let (apply, _, _) = detrend_decision(&x, &masks, 0.5);
        assert!(apply, "clean line through valid points must detrend");
    }
}
