//! Runs ALL scenarios from tools/scenarios.json with the Rust engine.
//!
//! Loads the checkpoint ONCE, then for each scenario applies model-config
//! overrides, runs one predict_batch call, and writes
//! out_rust_sc/<name>/s{i}_median.bin, s{i}_quantiles.bin (f32 LE),
//! matching the official scenario driver's npz layout.
//!
//! Usage: export_scenarios <ckpt_dir> <scenarios_json> <out_dir>

use std::path::{Path, PathBuf};

use timesfm3::checkpoint::Safetensors;
use timesfm3::config::ModelConfig;
use timesfm3::error::{Error, Result};
use timesfm3::forecast::ForecastOptions;
use timesfm3::json::Json;
use timesfm3::model::timesfm::TimesFM3;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: export_scenarios <ckpt_dir> <scenarios_json> <out_dir> [scenario_name ...]"
        );
        std::process::exit(2);
    }
    let ckpt_dir = PathBuf::from(&args[1]);
    let manifest_path = PathBuf::from(&args[2]);
    let out_dir = PathBuf::from(&args[3]);
    let only: std::collections::HashSet<String> = args[4..].iter().cloned().collect();
    std::fs::create_dir_all(&out_dir)?;

    let mut base_config = ModelConfig::default();
    let cfg_bytes = std::fs::read(ckpt_dir.join("config.json"))?;
    base_config.from_config_json(&Json::parse(&cfg_bytes)?)?;
    base_config.validate()?;
    let store = Safetensors::load(&ckpt_dir.join("model.safetensors"))?;
    let mut model = TimesFM3::load(&store, &base_config)?;

    let root = Json::parse(&std::fs::read(&manifest_path)?)?;
    let root_map = root
        .as_obj()
        .ok_or_else(|| Error("manifest: not an object".into()))?;
    let scenarios =
        arr(root_map, "scenarios").ok_or_else(|| Error("manifest: no scenarios".into()))?;

    let mut timings = String::from("{\n");
    let mut first_entry = true;
    for (idx, sc) in scenarios.iter().enumerate() {
        let sc_map = sc
            .as_obj()
            .ok_or_else(|| Error(format!("scenario {idx}: not an object")))?;
        let name = str_field(sc_map, "name")
            .ok_or_else(|| Error("scenario: no name".into()))?
            .to_string();
        if !only.is_empty() && !only.contains(&name) {
            continue;
        }
        let horizon = int_field(sc_map, "horizon")
            .ok_or_else(|| Error("scenario: no horizon".into()))? as usize;
        let flags = obj(sc_map, "flags").ok_or_else(|| Error("scenario: no flags".into()))?;
        let series = arr(sc_map, "series").ok_or_else(|| Error("scenario: no series".into()))?;

        // reset to checkpoint config, then apply per-scenario overrides
        model.config = base_config.clone();
        if let Some(ovr) = obj(sc_map, "model_overrides") {
            for (k, v) in ovr {
                let Json::Bool(b) = v else {
                    return Err(Error(format!("override {k}: not a bool")));
                };
                match k.as_str() {
                    "use_stitching" => model.config.use_stitching = *b,
                    "use_linear_detrending" => model.config.use_linear_detrending = *b,
                    "use_iterative_cpm_revin" => model.config.use_iterative_cpm_revin = *b,
                    "use_frozen_running_stats" => model.config.use_frozen_running_stats = *b,
                    other => return Err(Error(format!("unknown override: {other}"))),
                }
            }
        }

        let mut contexts: Vec<Vec<f32>> = Vec::new();
        let mut dims: Vec<(usize, usize)> = Vec::new();
        for s in series {
            let s_map = s
                .as_obj()
                .ok_or_else(|| Error(format!("series of {name}: not an object")))?;
            let csv = str_field(s_map, "csv").ok_or_else(|| Error("series: no csv".into()))?;
            let v = int_field(s_map, "variates")
                .ok_or_else(|| Error("series: no variates".into()))? as usize;
            let c = int_field(s_map, "context_len")
                .ok_or_else(|| Error("series: no context_len".into()))?
                as usize;
            let text = std::fs::read_to_string(Path::new(&csv))?;
            let rows: Vec<Vec<f32>> = text
                .lines()
                .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
                .map(|l| {
                    l.split(',')
                        .map(|t| {
                            t.trim()
                                .parse::<f32>()
                                .map_err(|_| Error(format!("bad num in {csv}")))
                        })
                        .collect::<Result<Vec<f32>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            if rows.len() != v || rows[0].len() != c {
                return Err(Error(format!(
                    "{csv}: expected {v}x{c}, got {}x{}",
                    rows.len(),
                    rows.first().map_or(0, |r| r.len())
                )));
            }
            let mut flat = Vec::with_capacity(v * c);
            for r in &rows {
                flat.extend_from_slice(r);
            }
            contexts.push(flat);
            dims.push((v, c));
        }

        let opts = ForecastOptions {
            return_quantiles: true,
            use_symmetric_averaging: bool_field(flags, "use_symmetric_averaging"),
            make_positive: bool_field(flags, "make_positive"),
            sort_quantiles: bool_field(flags, "sort_quantiles"),
            use_znorm: bool_field(flags, "use_znorm"),
            per_core_batch_size: 32,
            use_bucket_batching: false,
            use_robust_znorm: false,
            num_threads: None,
            use_isotonic_quantiles: false,
            seasonal_period: None,
            use_boundary_smoothing: false,
            ..Default::default()
        };

        let n = contexts.len();
        let no_cov = vec![None; n];
        let empty_dim = vec![None; n];
        let t0 = std::time::Instant::now();
        let results = model.predict_batch(
            &contexts, &dims, horizon, &no_cov, &empty_dim, &no_cov, &empty_dim, &opts,
        )?;
        let elapsed = t0.elapsed().as_secs_f32();

        let sdir = out_dir.join(&name);
        std::fs::create_dir_all(&sdir)?;
        for (i, r) in results.iter().enumerate() {
            write_bin(&sdir.join(format!("s{i}_median.bin")), &r.forecast)?;
            let q = r.quantiles.as_ref().expect("quantiles requested");
            write_bin(&sdir.join(format!("s{i}_quantiles.bin")), q)?;
        }
        let meta = format!("{{\"horizon\": {horizon}, \"quantiles\": 9, \"series\": {n}}}");
        std::fs::write(sdir.join("meta.json"), meta)?;
        let prefix = if first_entry { "" } else { ",\n" };
        first_entry = false;
        timings.push_str(&format!("{prefix}  \"{name}\": {elapsed:.2}"));
        let shapes: Vec<String> = dims.iter().map(|(v, c)| format!("{v}x{c}")).collect();
        println!(
            "[rust] {name:<20} h={horizon:<4} {} {elapsed:7.2}s",
            shapes.join("+"),
        );
    }
    timings.push_str("\n}\n");
    std::fs::write(out_dir.join("timings.json"), timings)?;
    println!("done -> {}", out_dir.display());
    Ok(())
}

fn write_bin(path: &Path, data: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

// ---- minimal manifest accessors over the crate's zero-dep JSON parser ----

type Map = [(String, Json)];

fn obj<'a>(m: &'a Map, key: &str) -> Option<&'a Map> {
    m.iter()
        .find(|(n, _)| n == key)
        .and_then(|(_, v)| v.as_obj())
}

fn arr<'a>(m: &'a Map, key: &str) -> Option<&'a [Json]> {
    m.iter()
        .find(|(n, _)| n == key)
        .and_then(|(_, v)| v.as_arr())
}

fn str_field<'a>(m: &'a Map, key: &str) -> Option<&'a str> {
    m.iter()
        .find(|(n, _)| n == key)
        .and_then(|(_, v)| v.as_str())
}

fn int_field(m: &Map, key: &str) -> Option<u64> {
    m.iter()
        .find(|(n, _)| n == key)
        .and_then(|(_, v)| v.as_u64())
}

fn bool_field(m: &Map, key: &str) -> bool {
    m.iter()
        .find(|(n, _)| n == key)
        .is_some_and(|(_, v)| matches!(v, Json::Bool(true)))
}
