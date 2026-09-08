//! Speed micro-benchmark used by the official-vs-Rust test report.
//!
//! Usage:
//!   bench_speed <ckpt_dir> <variates> <context> <horizon> <repeats>
//!              [--precision f32|f16|balanced]
//!
//! Prints one JSON line on stdout:
//!   {"shape":"24x1024","horizon":96,"precision":"f32","threads":8,
//!    "load_ms":..,"assemble_ms":..,"iters_ms":[..],"median_ms":..,"min_ms":..}
//!
//! `load_ms` covers config parse + safetensors mmap, `assemble_ms` covers
//! `TimesFM3::load` (weight views + RoPE tables), `iters_ms` are N repeated
//! `predict_batch` calls on the SAME loaded model (steady state).

use std::time::Instant;

use timesfm3::checkpoint::Safetensors;
use timesfm3::config::ModelConfig;
use timesfm3::error::Result;
use timesfm3::forecast::ForecastOptions;
use timesfm3::json::Json;
use timesfm3::model::timesfm::TimesFM3;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "usage: bench_speed <ckpt_dir> <variates> <context> <horizon> <repeats> \
             [--precision p]"
        );
        std::process::exit(2);
    }
    timesfm3::init_default_thread_pool();

    let ckpt_dir = std::path::PathBuf::from(&args[1]);
    let v: usize = args[2].parse().map_err(|_| "bad variates")?;
    let c: usize = args[3].parse().map_err(|_| "bad context")?;
    let horizon: usize = args[4].parse().map_err(|_| "bad horizon")?;
    let repeats: usize = args[5].parse().map_err(|_| "bad repeats")?;
    // optional 6th positional: batch size (number of identical series)
    let batch: usize = args
        .get(6)
        .filter(|a| !a.starts_with('-'))
        .and_then(|a| a.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let precision = args
        .iter()
        .position(|a| a == "--precision")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .or_else(|| std::env::var("TIMESFM_PRECISION").ok());

    // ---- load + assemble ----
    let t0 = Instant::now();
    let mut config = ModelConfig::default();
    // precision comes from the checkpoint's config.json (the quantiser writes
    // "f16" for both the full and the balanced preset); an explicit
    // --precision / TIMESFM_PRECISION override is applied AFTER the parse so
    // it is not clobbered.
    let cfg_bytes = std::fs::read(ckpt_dir.join("config.json"))?;
    config.from_config_json(&Json::parse(&cfg_bytes)?)?;
    if let Some(p) = precision.as_deref()
        && let Some(q) = timesfm3::config::QuantizationPrecision::parse(p)
    {
        config.precision = q;
    }
    config.validate()?;
    let store = Safetensors::load(&ckpt_dir.join("model.safetensors"))?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let model = TimesFM3::load(&store, &config)?;
    let assemble_ms = t1.elapsed().as_secs_f64() * 1e3;

    // ---- deterministic synthetic input (no I/O in the timed loop) ----
    let mut flat = Vec::with_capacity(v * c);
    for vi in 0..v {
        for i in 0..c {
            let t = i as f32;
            let x = 3.0 * ((t * 0.05) + vi as f32).sin()
                + 0.8 * ((t * 0.43) + 1.7).sin()
                + 0.01 * t
                + 0.2 * ((t * 0.017) + vi as f32 * 0.3).cos();
            flat.push(x);
        }
    }

    let opts = ForecastOptions {
        return_quantiles: true,
        use_symmetric_averaging: true,
        make_positive: false,
        sort_quantiles: true,
        use_znorm: false,
        per_core_batch_size: 32,
        use_bucket_batching: false,
        use_robust_znorm: false,
        num_threads: None,
        use_isotonic_quantiles: false,
        seasonal_period: None,
        use_boundary_smoothing: false,
        ..Default::default()
    };

    let contexts: Vec<Vec<f32>> = vec![flat; batch];
    let dims: Vec<(usize, usize)> = vec![(v, c); batch];
    let mut iters_ms: Vec<f64> = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t = Instant::now();
        let _ = model.predict_batch(
            &contexts,
            &dims,
            horizon,
            &vec![None; batch],
            &vec![None; batch],
            &vec![None; batch],
            &vec![None; batch],
            &opts,
        )?;
        iters_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut sorted = iters_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() { 0.0 } else { sorted[sorted.len() / 2] };
    let min = if sorted.is_empty() { 0.0 } else { sorted[0] };
    println!(
        "{{\"shape\":\"{}x{}\",\"batch\":{},\"horizon\":{},\"precision\":\"{}\",\"threads\":{},\
\"load_ms\":{:.2},\"assemble_ms\":{:.2},\"median_ms\":{:.2},\"min_ms\":{:.2},\
\"iters_ms\":[{}]}}",
        v * batch,
        c,
        batch,
        horizon,
        precision.as_deref().unwrap_or("f32"),
        rayon::current_num_threads(),
        load_ms,
        assemble_ms,
        median,
        min,
        iters_ms
            .iter()
            .map(|x| format!("{x:.1}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}
