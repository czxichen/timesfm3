//! Dumps Rust-engine intermediates for bitwise layer comparison against the
//! official torch hook dump (out_official/layers.npz).
//!
//! Usage: export_layers <ckpt_dir> <csv> <horizon> <dump_dir>
//! env TENSORSFM3_DUMP_DIR must equal <dump_dir> (checked by the model code).

use std::path::PathBuf;

use timesfm3::checkpoint::Safetensors;
use timesfm3::config::ModelConfig;
use timesfm3::error::Result;
use timesfm3::forecast::ForecastOptions;
use timesfm3::json::Json;
use timesfm3::model::timesfm::TimesFM3;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: export_layers <ckpt_dir> <csv> <horizon> <dump_dir>");
        std::process::exit(2);
    }
    let ckpt_dir = PathBuf::from(&args[1]);
    let csv = PathBuf::from(&args[2]);
    let horizon: usize = args[3].parse().map_err(|_| "bad horizon")?;
    let dump_dir = PathBuf::from(&args[4]);
    std::fs::create_dir_all(&dump_dir)?;
    // Safety: runs before any threads are spawned; no concurrent env access.
    unsafe {
        std::env::set_var("TENSORSFM3_DUMP_DIR", &dump_dir);
    }

    let mut config = ModelConfig::default();
    let cfg_bytes = std::fs::read(ckpt_dir.join("config.json"))?;
    config.from_config_json(&Json::parse(&cfg_bytes)?)?;
    config.validate()?;
    let store = Safetensors::load(&ckpt_dir.join("model.safetensors"))?;
    let model = TimesFM3::load(&store, &config)?;

    let text = std::fs::read_to_string(&csv)?;
    let rows: Vec<Vec<f32>> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .map(|l| {
            l.split(',')
                .map(|t| {
                    t.trim()
                        .parse::<f32>()
                        .map_err(|_| timesfm3::error::Error("bad num".into()))
                })
                .collect::<Result<Vec<f32>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let v = rows.len();
    let c = rows[0].len();
    let mut flat = Vec::with_capacity(v * c);
    for r in &rows {
        flat.extend_from_slice(r);
    }

    let opts = ForecastOptions {
        return_quantiles: true,
        use_symmetric_averaging: false, // single batch == official mirror batch0
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
    let no_cov = vec![None; 1];
    let empty_dim = vec![None; 1];
    let results = model.predict_batch(
        &[flat],
        &[(v, c)],
        horizon,
        &no_cov,
        &empty_dim,
        &no_cov,
        &empty_dim,
        &opts,
    )?;
    let r = &results[0];
    timesfm3::model::timesfm::dbg_dump("final_median", &r.forecast);
    let q = r.quantiles.as_ref().expect("quantiles");
    timesfm3::model::timesfm::dbg_dump("final_quantiles", q);
    println!("dumped intermediates -> {}", dump_dir.display());
    Ok(())
}
