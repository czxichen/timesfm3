//! Determinism + `predict_batch` contract tests on the micro golden model.
//!
//! Dimensions covered:
//! - decode is bit-identical across repeated calls (rules out nondeterminism
//!   from threading / workspace reuse),
//! - decode rejects degenerate arguments (horizon 0, batch 0),
//! - `predict_batch` happy path: output shapes, finiteness, quantile
//!   monotonicity with isotonic correction enabled,
//! - `predict_batch` error paths: horizon 0, contexts/dims mismatch, and the
//!   documented empty-input shortcut,
//! - NaN-tolerant input: interior NaNs interpolate instead of failing.

use std::path::PathBuf;

use timesfm3::checkpoint::Safetensors;
use timesfm3::config::ModelConfig;
use timesfm3::forecast::{ForecastOptions, ForecastResult};
use timesfm3::json::Json;
use timesfm3::model::timesfm::TimesFM3;
use timesfm3::tensor::Tensor2D;

fn micro_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden/e2e_micro")
}

fn load_micro() -> TimesFM3 {
    let dir = micro_dir();
    let cfg_bytes = std::fs::read(dir.join("config.json")).expect("config.json");
    let root = Json::parse(&cfg_bytes).expect("parse config.json");
    let mut config = ModelConfig::default();
    config.from_config_json(&root).expect("overlay config");
    config.validate().expect("config valid");
    let store = Safetensors::load(&dir.join("model.safetensors")).expect("load weights");
    TimesFM3::load(&store, &config).expect("assemble model")
}

fn ramp(ctx_len: usize) -> Vec<f32> {
    (0..ctx_len).map(|i| (i as f32) * 0.25 - 2.0).collect()
}

fn sine(ctx_len: usize) -> Vec<f32> {
    (0..ctx_len)
        .map(|i| ((i as f32) * 0.35).sin() + 0.1 * (i as f32))
        .collect()
}

fn decode(model: &TimesFM3, ctx: &[f32], horizon: usize) -> Tensor2D {
    let target = Tensor2D::from_slice(1, ctx.len(), ctx).expect("target shape");
    model
        .decode_shaped(
            &target,
            horizon,
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            1,
            ctx.len(),
        )
        .expect("decode")
}

#[test]
fn decode_is_bit_identical_across_calls() {
    let model = load_micro();
    let ctx = ramp(16);
    let a = decode(&model, &ctx, 8);
    let b = decode(&model, &ctx, 8);
    assert_eq!(a.data().len(), b.data().len());
    assert!(
        a.data() == b.data(),
        "repeated decode must be bit-identical"
    );
    // And finite everywhere.
    assert!(a.data().iter().all(|v| v.is_finite()));
}

#[test]
fn decode_rejects_degenerate_args() {
    let model = load_micro();
    let ctx = ramp(16);
    let target = Tensor2D::from_slice(1, 16, &ctx).expect("target shape");
    assert!(
        model
            .decode_shaped(&target, 0, None, None, None, None, None, None, 1, 1, 16)
            .is_err(),
        "horizon=0 must fail"
    );
    assert!(
        model
            .decode_shaped(&target, 8, None, None, None, None, None, None, 0, 1, 16)
            .is_err(),
        "b=0 must fail"
    );
}

fn default_opts() -> ForecastOptions {
    ForecastOptions {
        use_symmetric_averaging: false,
        ..Default::default()
    }
}

fn check_result_shape(res: &[ForecastResult], n_var: usize, horizon: usize, n_q: usize) {
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].num_variates, n_var);
    assert_eq!(res[0].horizon, horizon);
    assert_eq!(res[0].num_quantiles, n_q);
    assert_eq!(res[0].forecast.len(), n_var * horizon);
    let q = res[0].quantiles.as_ref().expect("quantiles returned");
    assert_eq!(q.len(), n_var * horizon * n_q);
}

#[test]
fn predict_batch_shapes_finiteness_and_monotone_quantiles() {
    let model = load_micro();
    let ctx = sine(32);
    let res = model
        .predict_batch(
            &[ctx],
            &[(1, 32)],
            8,
            &[None],
            &[None],
            &[None],
            &[None],
            &default_opts(),
        )
        .expect("predict_batch");
    check_result_shape(&res, 1, 8, 3);

    assert!(
        res[0].forecast.iter().all(|v| v.is_finite()),
        "median forecast must be finite"
    );
    let q = res[0].quantiles.as_ref().unwrap();
    assert!(q.iter().all(|v| v.is_finite()), "quantiles must be finite");
    // Isotonic + sorted quantiles: q10 <= q50 <= q90 at every step.
    for h in 0..8 {
        let (q10, q50, q90) = (q[h * 3], q[h * 3 + 1], q[h * 3 + 2]);
        assert!(
            q10 <= q50 && q50 <= q90,
            "quantiles crossed at step {h}: {q10} {q50} {q90}"
        );
    }
}

#[test]
fn predict_batch_multivariate_and_longer_context() {
    let model = load_micro();
    let contexts = vec![sine(64), ramp(32)];
    let dims = vec![(1, 64), (1, 32)];
    let res = model
        .predict_batch(
            &contexts,
            &dims,
            8,
            &[None, None],
            &[None, None],
            &[None, None],
            &[None, None],
            &default_opts(),
        )
        .expect("predict_batch multivariate");
    assert_eq!(res.len(), 2);
    for r in &res {
        assert_eq!(r.forecast.len(), 8);
        assert!(r.forecast.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn predict_batch_error_paths() {
    let model = load_micro();
    let opts = default_opts();
    // Empty input is a documented no-op.
    let empty = model
        .predict_batch(&[], &[], 8, &[], &[], &[], &[], &opts)
        .expect("empty input must be Ok");
    assert!(empty.is_empty());

    // Zero horizon.
    assert!(
        model
            .predict_batch(
                &[sine(32)],
                &[(1, 32)],
                0,
                &[None],
                &[None],
                &[None],
                &[None],
                &opts
            )
            .is_err()
    );
    // contexts / dims length mismatch.
    assert!(
        model
            .predict_batch(
                &[sine(32)],
                &[(1, 32), (1, 32)],
                8,
                &[None],
                &[None],
                &[None],
                &[None],
                &opts
            )
            .is_err()
    );
}

#[test]
fn predict_batch_interpolates_interior_nans() {
    let model = load_micro();
    let mut ctx = sine(32);
    ctx[10] = f32::NAN;
    ctx[11] = f32::NAN;
    let res = model
        .predict_batch(
            &[ctx],
            &[(1, 32)],
            8,
            &[None],
            &[None],
            &[None],
            &[None],
            &default_opts(),
        )
        .expect("NaN input must not fail");
    check_result_shape(&res, 1, 8, 3);
    assert!(
        res[0].forecast.iter().all(|v| v.is_finite()),
        "forecast over NaN-interpolated context must be finite"
    );
}

#[test]
fn predict_batch_option_flags_all_run() {
    // Every post-processing flag combination must run without panicking and
    // keep output shapes (correctness of each flag is asserted in
    // numerics_properties; here we lock the plumbing).
    let model = load_micro();
    let flag_sets = [
        (true, None, false),
        (false, Some(8), false),
        (true, Some(8), true),
        (false, None, true),
    ];
    for (robust, seasonal, smooth) in flag_sets {
        let opts = ForecastOptions {
            use_symmetric_averaging: false,
            use_robust_znorm: robust,
            seasonal_period: seasonal,
            use_boundary_smoothing: smooth,
            ..Default::default()
        };
        let res = model
            .predict_batch(
                &[sine(48)],
                &[(1, 48)],
                8,
                &[None],
                &[None],
                &[None],
                &[None],
                &opts,
            )
            .expect("flag combo must run");
        check_result_shape(&res, 1, 8, 3);
        assert!(res[0].forecast.iter().all(|v| v.is_finite()));
    }
}
