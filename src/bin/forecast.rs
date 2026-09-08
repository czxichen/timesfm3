//! Forecasting example CLI.
//!
//! Usage:
//!   cargo run --release --bin forecast -- <checkpoint_dir> <horizon> [csv]
//!
//! Loads `config.json` + `model.safetensors` from a checkpoint directory and
//! forecasts the given series (or a small demo series when no csv is given).
//! Each CSV line is one `variates;series` — see README.

use std::path::PathBuf;
use std::time::Instant;

use timesfm3::checkpoint::Safetensors;
use timesfm3::config::ModelConfig;
use timesfm3::error::Result;
use timesfm3::forecast::{ForecastOptions, MAX_CONTEXT_LENGTH};
use timesfm3::json::Json;
use timesfm3::model::timesfm::TimesFM3;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: forecast <checkpoint_dir> <horizon> [context.json]");
        std::process::exit(2);
    }
    let ckpt_dir = PathBuf::from(&args[1]);
    let horizon: usize = args[2].parse().map_err(|_| "bad horizon")?;

    // ---- load config.json + weights ----
    let cfg_path = ckpt_dir.join("config.json");
    let t0 = Instant::now();
    let mut config = ModelConfig::default();
    if cfg_path.exists() {
        let cfg_bytes = std::fs::read(&cfg_path)?;
        let root = Json::parse(&cfg_bytes)?;
        config.from_config_json(&root)?;
    } else {
        eprintln!("warning: no config.json found — using model defaults");
    }
    config.validate()?;

    let st_path = ckpt_dir.join("model.safetensors");
    if !st_path.exists() {
        eprintln!("error: no model.safetensors in {}", ckpt_dir.display());
        std::process::exit(2);
    }
    if let Ok(p_str) = std::env::var("TIMESFM_PRECISION")
        && let Some(p) = timesfm3::config::QuantizationPrecision::parse(&p_str)
    {
        config.precision = p;
    }
    for (i, arg) in args.iter().enumerate() {
        if arg == "--precision"
            && i + 1 < args.len()
            && let Some(p) = timesfm3::config::QuantizationPrecision::parse(&args[i + 1])
        {
            config.precision = p;
        }
    }

    println!(
        "config: p={} o={} layers={} dims={} heads={} quantiles={} precision={:?}",
        config.input_patch_len,
        config.output_patch_len,
        config.transformer_config.num_layers,
        config.transformer_config.transformer.model_dims,
        config.transformer_config.transformer.num_heads,
        config.num_quantiles,
        config.precision
    );
    let store = Safetensors::load(&st_path)?;
    println!(
        "weights: {} tensors loaded in {:.2?}",
        store.len(),
        t0.elapsed()
    );

    let t1 = Instant::now();
    let model = TimesFM3::load(&store, &config)?;
    println!("model assembled in {:.2?}", t1.elapsed());

    // ---- load context from CSV (or demo series) ----
    let mut csv_path = None;
    let mut i = 3;
    while i < args.len() {
        if args[i] == "--threads" || args[i] == "--precision" || args[i] == "--seasonal" {
            i += 2;
        } else if args[i].starts_with("--") {
            i += 1;
        } else {
            csv_path = Some(PathBuf::from(&args[i]));
            break;
        }
    }

    let (context, dims) = if let Some(ref p) = csv_path {
        load_csv(p)?
    } else {
        // Demo: three sine series of different lengths.
        let (c1, d1) = sine(96);
        let (c2, d2) = sine(64);
        (vec![c1, c2], vec![d1, d2])
    };

    let mut opts = ForecastOptions {
        return_quantiles: true,
        use_symmetric_averaging: true,
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
        ..Default::default()
    };
    if args.iter().any(|a| a == "--no-bucket-batching") {
        opts.use_bucket_batching = false;
    }
    if args.iter().any(|a| a == "--robust-znorm") {
        opts.use_robust_znorm = true;
    }
    if args.iter().any(|a| a == "--no-isotonic") {
        opts.use_isotonic_quantiles = false;
    }
    if args.iter().any(|a| a == "--boundary-smooth") {
        opts.use_boundary_smoothing = true;
    }
    if let Some(pos) = args.iter().position(|a| a == "--seasonal")
        && let Some(p_str) = args.get(pos + 1)
        && let Ok(p) = p_str.parse::<usize>()
    {
        opts.seasonal_period = Some(p);
    }
    if let Some(pos) = args.iter().position(|a| a == "--threads")
        && let Some(t_str) = args.get(pos + 1)
        && let Ok(t) = t_str.parse::<usize>()
    {
        opts.num_threads = Some(t);
    }
    if args.iter().any(|a| a == "--speculative") {
        opts.use_speculative = true;
    }
    let no_po = vec![None; context.len()];
    let empty_dims = vec![None; context.len()];
    let no_pf = vec![None; context.len()];

    let t2 = Instant::now();
    let results = model.predict_batch(
        &context,
        &dims,
        horizon,
        &no_po,
        &empty_dims,
        &no_pf,
        &empty_dims,
        &opts,
    )?;
    println!("forecast took {:.2?}", t2.elapsed());

    for (i, r) in results.iter().enumerate() {
        println!(
            "=== series {i}: {} variates, horizon {} ===",
            r.num_variates, r.horizon
        );
        if let Some(ref rep) = r.speculative_report {
            println!(
                "  speculative: accepted {}/{} ({:.1}%), mean z-score = {:.3}",
                rep.accepted_len,
                horizon,
                rep.acceptance_rate * 100.0,
                rep.mean_z_score
            );
        }
        for v in 0..r.num_variates.min(7) {
            let base = v * r.horizon;
            let md = &r.forecast[base..base + r.horizon];
            let first = md
                .iter()
                .take(8)
                .map(|x| format!("{x:.3}"))
                .collect::<Vec<_>>()
                .join(" ");
            if let Some(q) = &r.quantiles {
                let qbase = v * r.horizon * r.num_quantiles;
                let q10 = (0..r.horizon)
                    .map(|h| format!("{:.2}", q[qbase + h * r.num_quantiles + 1]))
                    .collect::<Vec<_>>()
                    .join(" ");
                let q90 = (0..r.horizon)
                    .map(|h| format!("{:.2}", q[qbase + h * r.num_quantiles + 7]))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("  v{v}: median[:8] = {first}");
                println!("         q10 = {q10}");
                println!("         q90 = {q90}");
            } else {
                println!("  v{v}: {first}");
            }
        }
    }
    println!(
        "\n(max context used: {} of {MAX_CONTEXT_LENGTH})",
        dims.iter().map(|d| d.1).max().unwrap_or(0)
    );
    Ok(())
}

fn sine(len: usize) -> (Vec<f32>, (usize, usize)) {
    let data: Vec<f32> = (0..len)
        .map(|i| (i as f32 * 0.3).sin() + 0.01 * i as f32)
        .collect();
    (data, (1, len))
}

/// Loads a CSV of context series. One row per line, comma-separated floats;
/// multiple lines → multiple variates of one series.
#[allow(clippy::type_complexity)]
fn load_csv(path: &PathBuf) -> Result<(Vec<Vec<f32>>, Vec<(usize, usize)>)> {
    let text = std::fs::read_to_string(path)?;
    let rows: Vec<Vec<f32>> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .map(|l| {
            l.split(',')
                .map(|t| {
                    t.trim()
                        .parse::<f32>()
                        .map_err(|_| timesfm3::error::Error(format!("bad number: {t}")))
                })
                .collect::<Result<Vec<f32>>>()
        })
        .collect::<Result<_>>()?;
    let v = rows.len();
    let c = rows.first().map(|r| r.len()).unwrap_or(0);
    for r in &rows {
        if r.len() != c {
            return Err(timesfm3::error::Error(
                "CSV rows must have equal length".into(),
            ));
        }
    }
    let mut flat = Vec::with_capacity(v * c);
    for r in &rows {
        flat.extend_from_slice(r);
    }
    Ok((vec![flat], vec![(v, c)]))
}
