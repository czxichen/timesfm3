//! Exports the Rust engine forecast as flat binary files for the official
//! comparison pipeline.
//!
//! Usage: export_forecast <checkpoint_dir> <horizon> <csv> <out_dir>
//!
//! Writes <out_dir>/median.bin, <out_dir>/quantiles.bin (f32 LE, shape
//! (v,H) and (v,H,9)), plus <out_dir>/meta.json.

use std::path::PathBuf;

use timesfm3::checkpoint::Safetensors;
use timesfm3::config::ModelConfig;
use timesfm3::error::Result;
use timesfm3::forecast::{ForecastOptions, MAX_CONTEXT_LENGTH};
use timesfm3::json::Json;
use timesfm3::model::timesfm::TimesFM3;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: export_forecast <ckpt_dir> <horizon> <csv> <out_dir>");
        std::process::exit(2);
    }
    let ckpt_dir = PathBuf::from(&args[1]);
    let horizon: usize = args[2].parse().map_err(|_| "bad horizon")?;
    let csv = PathBuf::from(&args[3]);
    let out_dir = PathBuf::from(&args[4]);
    std::fs::create_dir_all(&out_dir)?;

    let mut config = ModelConfig::default();
    let cfg_bytes = std::fs::read(ckpt_dir.join("config.json"))?;
    config.from_config_json(&Json::parse(&cfg_bytes)?)?;
    config.validate()?;

    let store = Safetensors::load(&ckpt_dir.join("model.safetensors"))?;
    let model = TimesFM3::load(&store, &config)?;

    // Same loader as the CLI: each CSV line is one variate.
    let text = std::fs::read_to_string(&csv)?;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let vals = line
            .split(',')
            .map(|t| {
                t.trim()
                    .parse::<f32>()
                    .map_err(|_| timesfm3::error::Error("bad num".into()))
            })
            .collect::<Result<Vec<f32>>>()?;
        rows.push(vals);
    }
    let v = rows.len();
    let c = rows[0].len();
    let mut flat = Vec::with_capacity(v * c);
    for r in &rows {
        flat.extend_from_slice(r);
    }
    let contexts = vec![flat];
    let ctx_dims = vec![(v, c)];

    let mut opts = ForecastOptions {
        return_quantiles: true,
        use_symmetric_averaging: true,
        make_positive: false,
        sort_quantiles: true,
        use_znorm: false,
        per_core_batch_size: 32,
        use_bucket_batching: false,
        use_robust_znorm: false,
        num_threads: None,
        use_isotonic_quantiles: true,
        seasonal_period: None,
        use_boundary_smoothing: false,
        ..Default::default()
    };
    if let Some(pos) = args.iter().position(|a| a == "--seasonal")
        && let Some(p_str) = args.get(pos + 1)
        && let Ok(p) = p_str.parse::<usize>()
    {
        opts.seasonal_period = Some(p);
    }
    if args.iter().any(|a| a == "--boundary-smooth") {
        opts.use_boundary_smoothing = true;
    }
    if args.iter().any(|a| a == "--no-isotonic") {
        opts.use_isotonic_quantiles = false;
    }
    if args.iter().any(|a| a == "--robust-znorm") {
        opts.use_robust_znorm = true;
    }
    if args.iter().any(|a| a == "--speculative") {
        opts.use_speculative = true;
    }
    let no_po = vec![None; 1];
    let empty_dim = vec![None; 1];
    let results = model.predict_batch(
        &contexts, &ctx_dims, horizon, &no_po, &empty_dim, &no_po, &empty_dim, &opts,
    )?;

    let r = &results[0];
    // median: (v, H) flat
    write_bin(&out_dir.join("median.bin"), &r.forecast)?;
    let q = r.quantiles.as_ref().expect("quantiles requested");
    write_bin(&out_dir.join("quantiles.bin"), q)?;
    let meta = format!(
        "{{\"variates\": {v}, \"horizon\": {horizon}, \"context_len\": {c}, \"quantiles\": 9, \"symmetric_averaging\": true, \"max_context\": {MAX_CONTEXT_LENGTH}}}"
    );
    std::fs::write(out_dir.join("meta.json"), meta)?;
    println!(
        "wrote {}/{{median.bin, quantiles.bin, meta.json}}",
        out_dir.display()
    );
    println!(
        "v0 median[:8] = {:?}",
        &r.forecast[..8.min(r.forecast.len())]
    );
    Ok(())
}

fn write_bin(path: &std::path::Path, data: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}
