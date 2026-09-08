//! High-level forecaster API (port of `timesfm3_forecaster.py`).
//!
//! Handles the numpy-level semantics: NaN stripping/interpolation, z-norm,
//! symmetric (flip) averaging, quantile sorting, non-negative clamping, per
//! batch-context padding, and median-quantile extraction.

use crate::error::{Error, Result};
use crate::model::timesfm::TimesFM3;
use crate::tensor::Tensor2D;

/// Maximum context length (official `_MAX_CONTEXT_LENGTH`).
pub const MAX_CONTEXT_LENGTH: usize = 15360;

/// Output for a single forecast call.
#[derive(Debug, Clone)]
pub struct ForecastResult {
    /// Point forecast (median quantile), `(num_variates, horizon)` flat.
    pub forecast: Vec<f32>,
    /// Full quantiles `(num_variates, horizon, num_quantiles)` flat, optional.
    pub quantiles: Option<Vec<f32>>,
    pub num_variates: usize,
    pub horizon: usize,
    pub num_quantiles: usize,
    /// Optional speculative decoding report if speculative mode was enabled.
    pub speculative_report: Option<crate::ops::speculative::SpeculativeReport>,
}

/// Options controlling a forecast.
#[derive(Debug, Clone)]
pub struct ForecastOptions {
    pub return_quantiles: bool,
    pub use_symmetric_averaging: bool,
    pub make_positive: bool,
    pub sort_quantiles: bool,
    pub use_znorm: bool,
    pub per_core_batch_size: usize,
    /// Sort and group series by length before batching to minimize left-padding distortion.
    pub use_bucket_batching: bool,
    /// Robust median/IQR z-normalization resilient against extreme spikes and outliers.
    pub use_robust_znorm: bool,
    /// Optional thread count override for isolated request execution in serving.
    pub num_threads: Option<usize>,
    /// Strictly non-crossing quantiles via Pool Adjacent Violators Algorithm (PAVA).
    pub use_isotonic_quantiles: bool,
    /// Optional seasonal period (e.g. 24 for hourly, 96 for 15-min, 7 for daily).
    /// Decomposes seasonal component before TimesFM and recomposes onto forecast.
    pub seasonal_period: Option<usize>,
    /// Exponential decay boundary smoothing to weld context to forecast and eliminate step jumps.
    pub use_boundary_smoothing: bool,
    /// Gaussian Speculative Decoding (Yu et al., 2024):
    /// Fast adaptive draft generator + tempered log-likelihood ratio verification.
    pub use_speculative: bool,
    /// Speculative likelihood ratio acceptance threshold (default 0.90).
    pub speculative_gamma: f32,
    /// Maximum Mahalanobis distance z-score for acceptance (default 2.0).
    pub speculative_max_z: f32,
}

impl Default for ForecastOptions {
    fn default() -> Self {
        Self {
            return_quantiles: true,
            use_symmetric_averaging: false,
            make_positive: false,
            sort_quantiles: true,
            use_znorm: false,
            per_core_batch_size: 32,
            use_bucket_batching: true,
            use_robust_znorm: false,
            num_threads: None,
            use_isotonic_quantiles: true,
            seasonal_period: None,
            use_boundary_smoothing: false,
            use_speculative: false,
            speculative_gamma: 0.90,
            speculative_max_z: 2.0,
        }
    }
}

impl TimesFM3 {
    /// Forecast a batch of time series.
    ///
    /// - `contexts`: list of `(num_variates, context_len)` f32 rows (flat).
    ///   Univariate series are passed as `(1, C)`.
    /// - `horizon`: forecast length.
    /// - `past_only_covariates[i]`: optional `(num_po, C)`.
    /// - `past_future_covariates[i]`: optional `(num_pf, C+horizon)`.
    ///
    /// Mirrors `predict_batch` (per-core batching + dynamic batch context).
    #[allow(clippy::too_many_arguments)]
    pub fn predict_batch(
        &self,
        contexts: &[Vec<f32>],
        ctx_dims: &[(usize, usize)], // (num_variates, context_len) per series
        horizon: usize,
        past_only_covariates: &[Option<Vec<f32>>],
        po_dims: &[Option<(usize, usize)>], // (num_po, C) per series
        past_future_covariates: &[Option<Vec<f32>>],
        pf_dims: &[Option<(usize, usize)>], // (num_pf, C+horizon) per series
        opts: &ForecastOptions,
    ) -> Result<Vec<ForecastResult>> {
        let cfg = &self.config;
        let p = cfg.input_patch_len;

        if contexts.is_empty() {
            return Ok(vec![]);
        }
        if horizon == 0 {
            return Err(Error("predict_batch: horizon must be > 0".into()));
        }
        if contexts.len() != ctx_dims.len() {
            return Err(Error(
                "predict_batch: contexts/ctx_dims length mismatch".into(),
            ));
        }

        // Thread pool isolation if custom thread count is requested
        if let Some(threads) = opts.num_threads
            && threads > 0
            && threads != rayon::current_num_threads()
            && let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(threads).build()
        {
            let mut opts_no_threads = opts.clone();
            opts_no_threads.num_threads = None;
            return pool.install(|| {
                self.predict_batch(
                    contexts,
                    ctx_dims,
                    horizon,
                    past_only_covariates,
                    po_dims,
                    past_future_covariates,
                    pf_dims,
                    &opts_no_threads,
                )
            });
        }

        // ---- preprocess per series: strip leading all-NaN, interpolate ----
        let mut cleaned: Vec<Vec<f32>> = Vec::with_capacity(contexts.len());
        let mut cleaned_dims: Vec<(usize, usize)> = Vec::with_capacity(contexts.len());
        let mut cleaned_po: Vec<Option<Vec<f32>>> = Vec::with_capacity(contexts.len());
        let mut cleaned_po_dims: Vec<Option<(usize, usize)>> = Vec::with_capacity(contexts.len());
        let mut cleaned_pf: Vec<Option<Vec<f32>>> = Vec::with_capacity(contexts.len());
        let mut cleaned_pf_dims: Vec<Option<(usize, usize)>> = Vec::with_capacity(contexts.len());

        let num_targets = ctx_dims[0].0;

        for (i, ctx) in contexts.iter().enumerate() {
            let (v_len, c_len) = ctx_dims[i];
            if v_len != num_targets {
                return Err(Error(format!(
                    "predict_batch: contexts[0] has {num_targets} variates, contexts[{i}] has {v_len}"
                )));
            }
            if ctx.len() != v_len * c_len {
                return Err(Error(format!(
                    "predict_batch: contexts[{i}] len {} != {}x{}",
                    ctx.len(),
                    v_len,
                    c_len
                )));
            }

            // Strip leading all-NaN columns.
            let mut first_valid = c_len;
            'col: for c in 0..c_len {
                for v in 0..v_len {
                    if ctx[v * c_len + c].is_nan() {
                        continue;
                    }
                    first_valid = c;
                    break 'col;
                }
            }
            let slice_start = first_valid.min(c_len);
            let new_c = c_len - slice_start;
            let stripped = if slice_start < c_len {
                let mut t = vec![0.0f32; v_len * new_c];
                for v in 0..v_len {
                    t[v * new_c..(v + 1) * new_c]
                        .copy_from_slice(&ctx[v * c_len + slice_start..v * c_len + c_len]);
                }
                t
            } else {
                if c_len == 0 {
                    ctx.clone()
                } else {
                    vec![0.0; v_len * c_len]
                }
            };
            let interped = interpolate_nan(&stripped, v_len, new_c).0;
            cleaned.push(interped);
            cleaned_dims.push((v_len, new_c));

            // covariates aligned to the stripped window
            let po_dims_i = po_dims[i]; // Copy
            match (&past_only_covariates[i], po_dims_i) {
                (Some(po), Some((po_v, _))) => {
                    let mut t = vec![0.0f32; po_v * new_c];
                    for v in 0..po_v {
                        t[v * new_c..(v + 1) * new_c]
                            .copy_from_slice(&po[v * c_len + slice_start..v * c_len + c_len]);
                    }
                    cleaned_po.push(Some(interpolate_nan(&t, po_v, new_c).0));
                    cleaned_po_dims.push(Some((po_v, new_c)));
                }
                _ => {
                    cleaned_po.push(None);
                    cleaned_po_dims.push(None);
                }
            }
            let pf_dims_i = pf_dims[i]; // Copy
            match (&past_future_covariates[i], pf_dims_i) {
                (Some(pf), Some((pf_v, pf_len))) => {
                    let future = pf_len - c_len; // = horizon typically
                    let mut t = vec![0.0f32; pf_v * (new_c + future)];
                    for v in 0..pf_v {
                        t[v * (new_c + future)..(v + 1) * (new_c + future)].copy_from_slice(
                            &pf[v * pf_len + slice_start..v * pf_len + c_len + (pf_len - c_len)],
                        );
                    }
                    cleaned_pf.push(Some(interpolate_nan(&t, pf_v, new_c + future).0));
                    cleaned_pf_dims.push(Some((pf_v, new_c + future)));
                }
                _ => {
                    cleaned_pf.push(None);
                    cleaned_pf_dims.push(None);
                }
            }
        }

        // ---- optional seasonal decomposition: extract periodic component before inference ----
        let mut seasonal_profiles: Vec<Vec<Vec<f32>>> = Vec::new();
        if let Some(period) = opts.seasonal_period
            && period >= 2
        {
            for (i, c_series) in cleaned.iter_mut().enumerate() {
                let (v_len, c_len) = cleaned_dims[i];
                let slice_start = ctx_dims[i].1 - c_len;
                let mut var_profs = Vec::with_capacity(v_len);
                for v in 0..v_len {
                    let row = &c_series[v * c_len..(v + 1) * c_len];
                    let (res, prof) = decompose_seasonal_with_offset(row, period, slice_start);
                    c_series[v * c_len..(v + 1) * c_len].copy_from_slice(&res);
                    var_profs.push(prof);
                }
                seasonal_profiles.push(var_profs);
            }
        }

        // ---- optional bucket batching: sort series by context length to minimize padding distortion ----
        let mut order: Vec<usize> = (0..contexts.len()).collect();
        if opts.use_bucket_batching && contexts.len() > 1 {
            order.sort_by_key(|&idx| cleaned_dims[idx].1);
            let mut new_cleaned = Vec::with_capacity(contexts.len());
            let mut new_cleaned_dims = Vec::with_capacity(contexts.len());
            let mut new_cleaned_po = Vec::with_capacity(contexts.len());
            let mut new_cleaned_po_dims = Vec::with_capacity(contexts.len());
            let mut new_cleaned_pf = Vec::with_capacity(contexts.len());
            let mut new_cleaned_pf_dims = Vec::with_capacity(contexts.len());

            for &idx in &order {
                new_cleaned.push(std::mem::take(&mut cleaned[idx]));
                new_cleaned_dims.push(cleaned_dims[idx]);
                new_cleaned_po.push(cleaned_po[idx].take());
                new_cleaned_po_dims.push(cleaned_po_dims[idx]);
                new_cleaned_pf.push(cleaned_pf[idx].take());
                new_cleaned_pf_dims.push(cleaned_pf_dims[idx]);
            }

            cleaned = new_cleaned;
            cleaned_dims = new_cleaned_dims;
            cleaned_po = new_cleaned_po;
            cleaned_po_dims = new_cleaned_po_dims;
            cleaned_pf = new_cleaned_pf;
            cleaned_pf_dims = new_cleaned_pf_dims;
        }

        // ---- z-norm (optional): normalize rows, keep stats for denorm ----
        // Official: stats per (series, variate) computed on the cleaned rows;
        // forecast/quantiles are denormalized AFTER sym merging, BEFORE
        // make_positive.
        let mut znorm_stats: Vec<Vec<(f32, f32)>> = Vec::new();
        if opts.use_znorm || opts.use_robust_znorm {
            for (data, (v_len, c_len)) in cleaned.iter_mut().zip(&cleaned_dims) {
                let mut stats = Vec::with_capacity(*v_len);
                for v in 0..*v_len {
                    let slice = &mut data[v * c_len..(v + 1) * c_len];
                    if opts.use_robust_znorm {
                        stats.push(robust_znorm_row(slice));
                    } else {
                        stats.push(znorm_row(slice));
                    }
                }
                znorm_stats.push(stats);
            }
        }

        // ---- symmetric averaging: double the query set ----
        let q_ctx = if opts.use_symmetric_averaging {
            let mut s = Vec::with_capacity(cleaned.len() * 2);
            let mut sd = Vec::with_capacity(cleaned_dims.len() * 2);
            let mut spo = Vec::with_capacity(cleaned_po.len() * 2);
            let mut spd = Vec::with_capacity(cleaned_po_dims.len() * 2);
            let mut spf = Vec::with_capacity(cleaned_pf.len() * 2);
            let mut sfd = Vec::with_capacity(cleaned_pf_dims.len() * 2);
            for (i, d) in cleaned.iter().enumerate() {
                s.push(d.clone());
                sd.push(cleaned_dims[i]);
                spo.push(cleaned_po[i].clone());
                spd.push(cleaned_po_dims[i]);
                spf.push(cleaned_pf[i].clone());
                sfd.push(cleaned_pf_dims[i]);
                // negated copy
                s.push(d.iter().map(|x| -x).collect());
                sd.push(cleaned_dims[i]);
                spo.push(
                    cleaned_po[i]
                        .as_ref()
                        .map(|v| v.iter().map(|x| -x).collect()),
                );
                spd.push(cleaned_po_dims[i]);
                spf.push(
                    cleaned_pf[i]
                        .as_ref()
                        .map(|v| v.iter().map(|x| -x).collect()),
                );
                sfd.push(cleaned_pf_dims[i]);
            }
            (s, sd, spo, spd, spf, sfd)
        } else {
            (
                cleaned,
                cleaned_dims,
                cleaned_po,
                cleaned_po_dims,
                cleaned_pf,
                cleaned_pf_dims,
            )
        };
        let (q_ctx, q_dims, q_po, q_po_dims, q_pf, q_pf_dims) = q_ctx;

        // ---- global_horizon (rounded to output patch) ----
        let o = cfg.output_patch_len;
        let global_horizon = horizon.div_ceil(o) * o;

        // ---- run in per-core batches with dynamic batch context ----
        let mut outputs: Vec<Tensor2D> = Vec::new();
        let per_core = opts.per_core_batch_size.max(1);
        let mut i = 0;
        while i < q_ctx.len() {
            let mut end = (i + per_core).min(q_ctx.len());
            if opts.use_bucket_batching && end > i + 1 {
                // Keep batch homogeneous: avoid pairing sequences with > 2x length disparity
                let min_len = q_dims[i].1.max(p);
                let step = if opts.use_symmetric_averaging { 2 } else { 1 };
                let mut cut = end;
                let mut k = i + step;
                while k < end {
                    if q_dims[k].1 > min_len * 2 {
                        cut = k;
                        break;
                    }
                    k += step;
                }
                end = cut;
            }
            let mut max_ctx = 0usize;
            for (_, d) in q_dims.iter().enumerate().take(end).skip(i) {
                max_ctx = max_ctx.max(d.1);
            }
            let mut batch_context = max_ctx.div_ceil(p) * p;
            batch_context = batch_context.min(MAX_CONTEXT_LENGTH.div_ceil(p) * p);
            batch_context = batch_context.max(p);

            // Pad each series to batch_context (left-pad, mask=true).
            let b = end - i;
            let bu = num_targets;
            let mut tgt = vec![0.0f32; b * bu * batch_context];
            let mut msk = vec![false; b * batch_context];
            let mut po_opt: Option<Vec<f32>> = None;
            let mut po_dim = (0usize, 0usize);
            let mut pf_opt: Option<Vec<f32>> = None;
            let mut pf_dim = (0usize, 0usize);

            for (bi, j) in (i..end).enumerate() {
                let (v_len, c_len) = q_dims[j];
                // Official Query.format: a context longer than the batch
                // context is truncated to its LAST batch_context points.
                let skip = c_len.saturating_sub(batch_context);
                let c_eff = c_len - skip;
                let pad = batch_context - c_eff;
                for v in 0..v_len {
                    let dst = (bi * bu + v) * batch_context + pad;
                    let base = v * c_len + skip;
                    let src = &q_ctx[j][base..base + c_eff];
                    if v < bu {
                        tgt[dst..dst + c_eff].copy_from_slice(src);
                    }
                }
                if pad > 0 {
                    for c in 0..pad {
                        msk[bi * batch_context + c] = true;
                    }
                }
                // po
                if let (Some(d), Some((v, _))) = (&q_po[j], q_po_dims[j]) {
                    po_dim = (v, batch_context);
                    if po_opt.is_none() {
                        po_opt = Some(vec![0.0f32; b * v * batch_context]);
                    }
                    if let Some(fill) = po_opt.as_mut() {
                        for vv in 0..v {
                            let dst_row = (bi * v + vv) * batch_context + pad;
                            let src_row = vv * c_len + skip;
                            fill[dst_row..dst_row + c_eff]
                                .copy_from_slice(&d[src_row..src_row + c_eff]);
                        }
                    }
                }
                // pf
                if let (Some(d), Some((v, pf_len))) = (&q_pf[j], q_pf_dims[j]) {
                    pf_dim = (v, batch_context + global_horizon);
                    if pf_opt.is_none() {
                        pf_opt = Some(vec![0.0f32; b * v * (batch_context + global_horizon)]);
                    }
                    if let Some(fill) = pf_opt.as_mut() {
                        let future = pf_len - c_len;
                        for vv in 0..v {
                            // past part (last c_eff points, aligned left)
                            let dst_row = (bi * v + vv) * (batch_context + global_horizon) + pad;
                            let src_row = vv * pf_len + skip;
                            fill[dst_row..dst_row + c_eff]
                                .copy_from_slice(&d[src_row..src_row + c_eff]);
                            // future part (unchanged position: starts at batch_context)
                            let dst_f =
                                (bi * v + vv) * (batch_context + global_horizon) + batch_context;
                            let src_f = src_row + c_len;
                            fill[dst_f..dst_f + future].copy_from_slice(&d[src_f..src_f + future]);
                        }
                    }
                }
            }

            // po/pf placeholder for batches without any covariates: skip.
            let po_t = match po_opt {
                Some(d) => Some(Tensor2D::from_slice(b * po_dim.0, po_dim.1, &d)?),
                None => None,
            };
            let pf_t = match pf_opt {
                Some(d) => Some(Tensor2D::from_slice(b * pf_dim.0, pf_dim.1, &d)?),
                None => None,
            };

            let tgt_t = Tensor2D::from_slice(b * bu, batch_context, &tgt)?;
            let out = self.decode_shaped(
                &tgt_t,
                global_horizon,
                po_t.as_ref(),
                pf_t.as_ref(),
                Some(&msk),
                None,
                None,
                None,
                b,
                bu,
                batch_context,
            )?;
            outputs.push(out);
            i = end;
        }

        // ---- post-process ----
        let mut results: Vec<ForecastResult> = Vec::with_capacity(num_targets);
        // decode_shaped returns (b, v_total, horizon, q) flattened to
        // (b*v_total*horizon, q). Unpack per query, then per (variate, horizon).
        let v_total = num_targets; // no covariates in this path; po/pf variates
        // would add to v_total via decode_shaped.
        let mut by_query: Vec<Vec<f32>> = Vec::new();
        let per_query = v_total * global_horizon * cfg.num_quantiles;
        for o in &outputs {
            let q_in_out = o.rows() / (v_total * global_horizon);
            for j in 0..q_in_out {
                let base = j * per_query;
                by_query.push(o.data()[base..base + per_query].to_vec());
            }
        }

        if opts.sort_quantiles {
            for qy in by_query.iter_mut() {
                for v in 0..v_total {
                    for h in 0..global_horizon {
                        let base = (v * global_horizon + h) * cfg.num_quantiles;
                        let slice = &mut qy[base..base + cfg.num_quantiles];
                        slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    }
                }
            }
        }

        if opts.use_symmetric_averaging {
            let mut merged: Vec<Vec<f32>> = Vec::with_capacity(by_query.len() / 2);
            for pair in by_query.chunks(2) {
                let (pos, neg) = (&pair[0], &pair[1]);
                let mut m = vec![0.0f32; per_query];
                for v in 0..v_total {
                    for h in 0..global_horizon {
                        let base = (v * global_horizon + h) * cfg.num_quantiles;
                        for qq in 0..cfg.num_quantiles {
                            // pos - flipped(neg): quantile qq pairs with neg[nq-1-qq]
                            let flipped = neg[base + (cfg.num_quantiles - 1 - qq)];
                            m[base + qq] = (pos[base + qq] - flipped) / 2.0;
                        }
                    }
                }
                merged.push(m);
            }
            by_query = merged;
        }

        // Output: one ForecastResult per query (symmetry pairs merged above).
        // forecast is (v_total, horizon) flat; quantiles
        // (v_total, horizon, num_quantiles) flat.
        let median = cfg.median_quantile_index;
        for (si, raw) in by_query.iter().enumerate() {
            let mut f = vec![0.0f32; v_total * horizon];
            let mut qs: Option<Vec<f32>> = if opts.return_quantiles {
                Some(vec![0.0f32; v_total * horizon * cfg.num_quantiles])
            } else {
                None
            };
            for v in 0..v_total {
                for h in 0..horizon {
                    let base = (v * global_horizon + h) * cfg.num_quantiles;
                    f[v * horizon + h] = raw[base + median];
                    if let Some(q) = &mut qs {
                        let dbase = (v * horizon + h) * cfg.num_quantiles;
                        q[dbase..dbase + cfg.num_quantiles]
                            .copy_from_slice(&raw[base..base + cfg.num_quantiles]);
                    }
                }
            }
            // z-norm denormalization (official: raw * sigma + mu per variate,
            // after sym averaging, before make_positive).
            if (opts.use_znorm || opts.use_robust_znorm)
                && let Some(stats) = znorm_stats.get(si)
            {
                for (idx, val) in f.iter_mut().enumerate() {
                    let (mu, sigma) = stats[idx / horizon];
                    *val = *val * sigma + mu;
                }
                if let Some(q) = &mut qs {
                    let nq = cfg.num_quantiles;
                    for (idx, val) in q.iter_mut().enumerate() {
                        let (mu, sigma) = stats[idx / (horizon * nq)];
                        *val = *val * sigma + mu;
                    }
                }
            }

            let orig_si = order[si];

            // Recompose seasonal profile if enabled
            if let Some(period) = opts.seasonal_period
                && period >= 2
                && let Some(var_profs) = seasonal_profiles.get(orig_si)
            {
                let raw_c = ctx_dims[orig_si].1;
                for v in 0..var_profs.len().min(v_total) {
                    let prof = &var_profs[v];
                    for h in 0..horizon {
                        let s = prof[(raw_c + h) % period];
                        f[v * horizon + h] += s;
                        if let Some(q) = &mut qs {
                            let nq = cfg.num_quantiles;
                            let base = (v * horizon + h) * nq;
                            for x in q[base..base + nq].iter_mut() {
                                *x += s;
                            }
                        }
                    }
                }
            }

            // Exponential decay boundary smoothing if enabled
            if opts.use_boundary_smoothing {
                let raw_c = ctx_dims[orig_si].1;
                for v in 0..num_targets.min(v_total) {
                    let raw_row = &contexts[orig_si][v * raw_c..(v + 1) * raw_c];
                    if let Some(&last_val) = raw_row.iter().rev().find(|x| !x.is_nan()) {
                        let f_slice = &mut f[v * horizon..(v + 1) * horizon];
                        let q_slice = qs.as_mut().map(|q| {
                            let nq = cfg.num_quantiles;
                            &mut q[v * horizon * nq..(v + 1) * horizon * nq]
                        });
                        apply_boundary_smoothing(
                            f_slice,
                            q_slice,
                            last_val,
                            horizon,
                            cfg.num_quantiles,
                        );
                    }
                }
            }

            if opts.make_positive {
                // clamp >= 0 per (series, variate) when that input row is
                // non-negative (official per-row `_is_nonnegative`).
                if let Some(&(v_in, c)) = ctx_dims.get(orig_si) {
                    for v in 0..v_in.min(v_total) {
                        let mut ok = true;
                        for idx in 0..c {
                            let x = contexts[orig_si][v * c + idx];
                            if x.is_nan() || x < 0.0 {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            for h in 0..horizon {
                                f[v * horizon + h] = f[v * horizon + h].max(0.0);
                            }
                            if let Some(q) = &mut qs {
                                let nq = cfg.num_quantiles;
                                for h in 0..horizon {
                                    let base = (v * horizon + h) * nq;
                                    for x in q[base..base + nq].iter_mut() {
                                        *x = x.max(0.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Guaranteed non-crossing monotonic quantiles via PAVA
            if opts.use_isotonic_quantiles
                && let Some(q) = &mut qs
            {
                let nq = cfg.num_quantiles;
                for v in 0..v_total {
                    for h in 0..horizon {
                        let base = (v * horizon + h) * nq;
                        pava_isotonic_regression(&mut q[base..base + nq]);
                    }
                }
            }

            // Optional Gaussian Speculative Decoding verification
            let mut spec_report = None;
            if opts.use_speculative {
                let raw_c = ctx_dims[orig_si].1;
                let raw_row = &contexts[orig_si][..raw_c];
                let draft = crate::ops::speculative::GaussianDraft::fit_and_predict(
                    raw_row,
                    horizon,
                    opts.seasonal_period,
                );
                let verifier = crate::ops::speculative::SpeculativeVerifier {
                    gamma: opts.speculative_gamma,
                    max_z: opts.speculative_max_z,
                };
                let report =
                    verifier.verify(&draft, &f[..horizon], qs.as_deref(), cfg.num_quantiles);
                spec_report = Some(report);
            }

            results.push(ForecastResult {
                forecast: f,
                quantiles: qs,
                num_variates: v_total,
                horizon,
                num_quantiles: cfg.num_quantiles,
                speculative_report: spec_report,
            });
        }
        if opts.use_bucket_batching && contexts.len() > 1 {
            let mut ordered_results = Vec::with_capacity(contexts.len());
            ordered_results.resize_with(contexts.len(), || None);
            for (orig_idx, res) in order.into_iter().zip(results) {
                ordered_results[orig_idx] = Some(res);
            }
            Ok(ordered_results
                .into_iter()
                .map(|r| r.expect("all results populated"))
                .collect())
        } else {
            Ok(results)
        }
    }
}

// ---------------------------------------------------------------------------
// numpy-level helpers
// ---------------------------------------------------------------------------

/// NaN removal: strips leading all-NaN columns then linearly interpolates
/// interior NaNs (official `linear_interpolation`). `ctx` is `(v, C)` flat.
fn interpolate_nan(ctx: &[f32], v: usize, c: usize) -> (Vec<f32>, usize) {
    // leading all-NaN strip already done by caller; here handle interior NaN.
    let mut out = ctx.to_vec();
    for vv in 0..v {
        let row = &mut out[vv * c..(vv + 1) * c];
        let mut has_nan = row.iter().any(|x| x.is_nan());
        if !has_nan {
            continue;
        }
        // find valid indices
        let valid: Vec<usize> = (0..c).filter(|&i| !row[i].is_nan()).collect();
        if valid.is_empty() {
            for x in row.iter_mut() {
                *x = 0.0;
            }
            continue;
        }
        let mut prev_valid = valid[0];
        let mut pending = Vec::new();
        for i in 0..c {
            if row[i].is_nan() {
                pending.push(i);
            } else {
                if !pending.is_empty() {
                    // linear interpolate between prev_valid and i
                    let last = i;
                    let n = pending.len() as f32;
                    let y0 = row[prev_valid];
                    let y1 = row[last];
                    for (k, &pi) in pending.iter().enumerate() {
                        let t = (k as f32 + 1.0) / (n + 1.0);
                        row[pi] = y0 + t * (y1 - y0);
                    }
                    pending.clear();
                }
                prev_valid = i;
            }
        }
        if !pending.is_empty() {
            // trailing NaNs after the last valid → ffill
            for &pi in &pending {
                row[pi] = row[prev_valid];
            }
        }
        has_nan = false;
    }
    (out, c)
}

fn znorm_row(row: &mut [f32]) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut n = 0.0f32;
    for &x in row.iter() {
        if !x.is_nan() {
            sum += x;
            n += 1.0;
        }
    }
    let mu = if n > 0.0 { sum / n } else { 0.0 };
    let mut var = 0.0f32;
    for &x in row.iter() {
        if !x.is_nan() {
            var += (x - mu) * (x - mu);
        }
    }
    let sigma = if n > 0.0 { (var / n).sqrt() } else { 1.0 };
    let sigma = if sigma < 1e-7 || !sigma.is_finite() {
        1.0
    } else {
        sigma
    };
    for x in row.iter_mut() {
        *x = (*x - mu) / sigma;
    }
    (mu, sigma)
}

/// Robust median/IQR z-score normalization resilient to extreme spikes/outliers.
fn robust_znorm_row(row: &mut [f32]) -> (f32, f32) {
    let mut valid: Vec<f32> = row.iter().copied().filter(|x| !x.is_nan()).collect();
    if valid.is_empty() {
        return (0.0, 1.0);
    }
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = valid.len();
    let median = if n % 2 == 1 {
        valid[n / 2]
    } else {
        0.5 * (valid[n / 2 - 1] + valid[n / 2])
    };
    let q25 = valid[n / 4];
    let q75 = valid[(3 * n) / 4];
    let iqr = q75 - q25;
    // Standard deviation equivalent for Gaussian distribution: IQR / 1.34898
    let mut sigma = iqr / 1.3489795;
    if sigma < 1e-7 || !sigma.is_finite() {
        // Near-zero IQR (e.g. identical/constant central values): set scale to 1.0
        sigma = 1.0;
    }
    for x in row.iter_mut() {
        if !x.is_nan() {
            *x = (*x - median) / sigma;
        }
    }
    (median, sigma)
}

/// Pool Adjacent Violators Algorithm (PAVA) for Isotonic Regression.
/// Computes non-decreasing sequence `q_0 <= q_1 <= ... <= q_{N-1}`
/// minimizing `sum_{i} (q_i - y_i)^2`.
pub fn pava_isotonic_regression(y: &mut [f32]) {
    let n = y.len();
    if n <= 1 {
        return;
    }
    struct Block {
        weight: f64,
        sum: f64,
        start: usize,
        len: usize,
    }
    let mut blocks: Vec<Block> = Vec::with_capacity(n);
    for (i, &val) in y.iter().enumerate() {
        let mut cur = Block {
            weight: 1.0,
            sum: if val.is_nan() { 0.0 } else { val as f64 },
            start: i,
            len: 1,
        };
        while let Some(top) = blocks.last() {
            let top_val = top.sum / top.weight;
            let cur_val = cur.sum / cur.weight;
            if top_val <= cur_val {
                break;
            }
            let top = blocks.pop().unwrap();
            cur = Block {
                weight: top.weight + cur.weight,
                sum: top.sum + cur.sum,
                start: top.start,
                len: top.len + cur.len,
            };
        }
        blocks.push(cur);
    }
    for b in blocks {
        let mean = (b.sum / b.weight) as f32;
        for val in y.iter_mut().skip(b.start).take(b.len) {
            *val = mean;
        }
    }
}

/// Decomposes a 1D time series into a zero-centered seasonal profile and residual:
/// `values[t] = residual[t] + seasonal[(phase_offset + t) % period]`.
pub fn decompose_seasonal_with_offset(
    values: &[f32],
    period: usize,
    phase_offset: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = values.len();
    if period < 2 || n < 2 * period {
        return (values.to_vec(), vec![0.0; period.max(1)]);
    }
    let mut sum = vec![0.0f64; period];
    let mut count = vec![0usize; period];

    for (t, &x) in values.iter().enumerate() {
        if !x.is_nan() {
            let phase = (phase_offset + t) % period;
            sum[phase] += x as f64;
            count[phase] += 1;
        }
    }

    let mut profile = vec![0.0f32; period];
    let mut overall_sum = 0.0f64;
    let mut valid_phases = 0usize;
    for p in 0..period {
        if count[p] > 0 {
            let mean = (sum[p] / count[p] as f64) as f32;
            profile[p] = mean;
            overall_sum += mean as f64;
            valid_phases += 1;
        }
    }
    let overall_mean = if valid_phases > 0 {
        (overall_sum / valid_phases as f64) as f32
    } else {
        0.0
    };
    for p in profile.iter_mut().take(period) {
        *p -= overall_mean;
    }

    let mut residual = Vec::with_capacity(n);
    for (t, &x) in values.iter().enumerate() {
        if x.is_nan() {
            residual.push(x);
        } else {
            let phase = (phase_offset + t) % period;
            residual.push(x - profile[phase]);
        }
    }
    (residual, profile)
}

/// Smooths the transition between context and forecast with an exponential decay weld:
/// `delta(h) = (last_val - forecast[0]) * exp(-h / tau)`
pub fn apply_boundary_smoothing(
    forecast: &mut [f32],
    mut quantiles: Option<&mut [f32]>,
    last_context_val: f32,
    horizon: usize,
    num_quantiles: usize,
) {
    if forecast.is_empty() || last_context_val.is_nan() {
        return;
    }
    let first_forecast = forecast[0];
    if first_forecast.is_nan() {
        return;
    }
    let gap = last_context_val - first_forecast;
    if gap.abs() < 1e-6 {
        return;
    }
    let tau = (horizon as f32 / 4.0).clamp(2.0, 12.0);
    for (h, f_val) in forecast.iter_mut().enumerate().take(horizon) {
        let decay = (-(h as f32) / tau).exp();
        let correction = gap * decay;
        *f_val += correction;
        if let Some(ref mut qs) = quantiles {
            let base = h * num_quantiles;
            for q in qs[base..base + num_quantiles].iter_mut() {
                *q += correction;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_znorm_resists_outliers() {
        // 100 normal values around 10.0, plus 1 huge outlier spike of 10000.0
        let mut row1 = vec![10.0f32; 100];
        row1[50] = 10000.0;
        let mut row2 = row1.clone();

        let (r_med, r_sig) = robust_znorm_row(&mut row1);
        let (z_mu, z_sig) = znorm_row(&mut row2);

        // Standard z-norm has a huge sigma due to the 10000 spike
        assert!(z_sig > 500.0);
        assert!(z_mu > 100.0);

        // Robust z-norm has median exactly 10.0 and stable sigma
        assert_eq!(r_med, 10.0);
        assert!(r_sig < 2.0);
        // Normal points stay close to 0.0 in robust z-norm
        assert_eq!(row1[0], 0.0);
        // In standard z-norm, normal points are shifted far away from 0
        assert!(row2[0] < -0.1);
    }

    #[test]
    fn forecast_options_defaults() {
        let opts = ForecastOptions::default();
        assert!(opts.use_bucket_batching);
        assert!(!opts.use_robust_znorm);
        assert_eq!(opts.per_core_batch_size, 32);
        assert!(opts.sort_quantiles);
        assert!(opts.return_quantiles);
        assert!(opts.use_isotonic_quantiles);
        assert!(opts.seasonal_period.is_none());
        assert!(!opts.use_boundary_smoothing);
    }

    #[test]
    fn test_pava_isotonic_regression() {
        // Monotonic input stays unchanged
        let mut q = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        pava_isotonic_regression(&mut q);
        assert_eq!(q, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Inverted crossing quantiles: [1.0, 3.0, 2.0, 4.0] -> [1.0, 2.5, 2.5, 4.0]
        let mut q_cross = vec![1.0, 3.0, 2.0, 4.0];
        pava_isotonic_regression(&mut q_cross);
        assert_eq!(q_cross, vec![1.0, 2.5, 2.5, 4.0]);

        // Severe inversion: [5.0, 1.0, 2.0] -> [8/3, 8/3, 8/3]
        let mut q_severe = vec![5.0, 1.0, 2.0];
        pava_isotonic_regression(&mut q_severe);
        assert!((q_severe[0] - 8.0 / 3.0).abs() < 1e-6);
        assert!((q_severe[1] - 8.0 / 3.0).abs() < 1e-6);
        assert!((q_severe[2] - 8.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_seasonal_decomposition() {
        // Exact period 4 seasonal cycle: [10, 20, 30, 40, 10, 20, 30, 40]
        let values = vec![10.0, 20.0, 30.0, 40.0, 10.0, 20.0, 30.0, 40.0];
        let (res, prof) = decompose_seasonal_with_offset(&values, 4, 0);
        assert_eq!(prof.len(), 4);
        // Profile sums to zero
        let sum: f32 = prof.iter().sum();
        assert!(sum.abs() < 1e-5);
        // Residual is flat line of mean = 25.0
        for x in &res {
            assert!((x - 25.0).abs() < 1e-5);
        }
        // Recomposition reconstructs exact original values
        for (t, &r) in res.iter().enumerate() {
            assert!((r + prof[t % 4] - values[t]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_boundary_smoothing() {
        let mut forecast = vec![90.0, 91.0, 92.0, 93.0];
        let mut quantiles = vec![
            80.0, 90.0, 100.0, 81.0, 91.0, 101.0, 82.0, 92.0, 102.0, 83.0, 93.0, 103.0,
        ];
        let last_context_val = 100.0;
        let gap = last_context_val - forecast[0]; // 10.0

        apply_boundary_smoothing(&mut forecast, Some(&mut quantiles), last_context_val, 4, 3);

        // At step 0: forecast should start close to 100.0 (decay=1.0)
        assert!((forecast[0] - 100.0).abs() < 1e-5);
        // Step 0 median quantile also starts at 100.0
        assert!((quantiles[1] - 100.0).abs() < 1e-5);
        // Later steps: correction diminishes monotonically
        assert!(forecast[3] - 93.0 < gap);
    }
}
