//! Offline model weight quantization tool for TimesFM 3.0.
//!
//! Converts standard FP32 safetensors checkpoints into compact FP16
//! checkpoints (reducing disk and memory footprint by 50% with near-zero accuracy loss).
//! Optional `--keep-f32` keeps the given weight-name prefixes in FP32 (mixed
//! precision), which materially reduces the quantization error introduced by
//! sensitive tensors (e.g. `output_head`, `pre_transformer_resblock`).
//! `--preset balanced` is the measured sweet spot: keep the output head, the
//! pre-transformer residual block, and the last 3 transformer layers in FP32
//! (error -80% for +21% size); see report/eval_v8_precision.md.
//!
//! Usage:
//!   cargo run --release --bin quantize -- <src_ckpt_dir> <dst_ckpt_dir> <f16> [--keep-f32 p1,p2]
//!   cargo run --release --bin quantize -- ckpt ckpt_f16_highacc f16 --preset balanced

use std::path::PathBuf;
use std::time::Instant;

use timesfm3::checkpoint::{DType, Safetensors, TensorView, write_safetensors};
use timesfm3::config::QuantizationPrecision;
use timesfm3::error::{Error, Result};

struct ConvertedTensor {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Tensor-name prefixes that stay in FP32 (mixed-precision quantize).
fn parse_keep_f32(args: &[String]) -> Result<Vec<String>> {
    // `--preset balanced`: measured sweet spot for error-vs-size
    // (head + resblock + last 3 transformer layers stay FP32).
    if let Some(pos) = args.iter().position(|a| a == "--preset") {
        let name = args
            .get(pos + 1)
            .ok_or_else(|| Error("--preset requires a name".into()))?;
        if name == "balanced" {
            return Ok(vec![
                "output_head".into(),
                "pre_transformer_resblock".into(),
                "transformer_stack.layers.17.".into(),
                "transformer_stack.layers.18.".into(),
                "transformer_stack.layers.19.".into(),
            ]);
        }
        if name == "ultra" || name == "high_accuracy" {
            return Ok(vec![
                "output_head".into(),
                "pre_transformer_resblock".into(),
                "transformer_stack.layers.15.".into(),
                "transformer_stack.layers.16.".into(),
                "transformer_stack.layers.17.".into(),
                "transformer_stack.layers.18.".into(),
                "transformer_stack.layers.19.".into(),
            ]);
        }
        return Err(Error(format!(
            "unknown preset '{name}': expected 'balanced' or 'ultra'"
        )));
    }
    if let Some(pos) = args.iter().position(|a| a == "--keep-f32") {
        let list = args
            .get(pos + 1)
            .ok_or_else(|| Error("--keep-f32 requires a comma-separated prefix list".into()))?;
        Ok(list
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect())
    } else {
        Ok(Vec::new())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let keep_f32 = parse_keep_f32(&args)?;
    if args.len() < 4 {
        eprintln!(
            "usage: quantize <src_ckpt_dir> <dst_ckpt_dir> <f16> [--keep-f32 p1,p2] [--preset balanced|ultra]"
        );
        std::process::exit(2);
    }
    let src_dir = PathBuf::from(&args[1]);
    let dst_dir = PathBuf::from(&args[2]);
    let precision = QuantizationPrecision::parse(&args[3])
        .ok_or_else(|| Error(format!("invalid precision '{}': expected 'f16'", args[3])))?;

    if precision == QuantizationPrecision::F32 {
        return Err(Error("target precision must be 'f16'".into()));
    }

    std::fs::create_dir_all(&dst_dir)?;

    // 1. Process and copy config.json
    let src_cfg_path = src_dir.join("config.json");
    if src_cfg_path.exists() {
        let mut cfg_content = std::fs::read_to_string(&src_cfg_path)?;
        let prec_str = match precision {
            QuantizationPrecision::F16 => "\"f16\"",
            _ => "\"f32\"",
        };
        if cfg_content.contains("\"precision\"") {
            // Already has precision
        } else if let Some(idx) = cfg_content.find('{') {
            cfg_content.insert_str(idx + 1, &format!("\n  \"precision\": {},", prec_str));
        }
        std::fs::write(dst_dir.join("config.json"), cfg_content)?;
        println!("config: updated config.json with precision={:?}", precision);
    }

    // 2. Load source model.safetensors
    let src_st_path = src_dir.join("model.safetensors");
    if !src_st_path.exists() {
        return Err(Error(format!("not found: {}", src_st_path.display())));
    }
    println!("loading weights from {}...", src_st_path.display());
    let t0 = Instant::now();
    let store = Safetensors::load(&src_st_path)?;
    println!("loaded {} tensors in {:.2?}", store.len(), t0.elapsed());
    if !keep_f32.is_empty() {
        println!("keep-f32 prefixes: {}", keep_f32.join(", "));
    }

    // 3. Convert weights
    println!("quantizing weights to {:?}...", precision);
    let t1 = Instant::now();
    let mut converted: Vec<ConvertedTensor> = Vec::with_capacity(store.len());

    let mut orig_total_bytes = 0usize;
    let mut quant_total_bytes = 0usize;

    // Deterministic order
    let mut names: Vec<String> = store.iter().map(|(k, _)| k.to_owned()).collect();
    names.sort();

    for name in &names {
        let info = store.tensor_info(name).unwrap();
        orig_total_bytes += info.byte_len;

        // Quantize 2D weight matrices (Linear weights). Keep 1D vectors in F32
        // to preserve norm precision. Tensors matching a `--keep-f32` prefix
        // also stay F32 (mixed precision for error-sensitive weights).
        let kept = keep_f32.iter().any(|p| name.starts_with(p.as_str()));
        let is_2d_weight = info.shape.len() == 2 && name.ends_with(".weight");

        if is_2d_weight && !kept {
            let data_bytes = match precision {
                QuantizationPrecision::F16 => {
                    let f16_data = store.f16_flat(name)?;
                    f16_data
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect::<Vec<u8>>()
                }
                _ => unreachable!(),
            };

            quant_total_bytes += data_bytes.len();
            let target_dtype = match precision {
                QuantizationPrecision::F16 => DType::F16,
                _ => unreachable!(),
            };

            converted.push(ConvertedTensor {
                name: name.clone(),
                dtype: target_dtype,
                shape: info.shape.clone(),
                data: data_bytes,
            });
        } else {
            // Keep original F32 (or raw) bytes
            let f32_data = store.f32_flat(name)?;
            let data_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            quant_total_bytes += data_bytes.len();

            converted.push(ConvertedTensor {
                name: name.clone(),
                dtype: DType::F32,
                shape: info.shape.clone(),
                data: data_bytes,
            });
        }
    }
    println!(
        "quantized {} tensors in {:.2?}",
        converted.len(),
        t1.elapsed()
    );

    // 4. Write output safetensors
    let views: Vec<TensorView> = converted
        .iter()
        .map(|t| TensorView {
            name: &t.name,
            dtype: t.dtype,
            shape: &t.shape,
            data: &t.data,
        })
        .collect();

    let dst_st_path = dst_dir.join("model.safetensors");
    println!("writing output to {}...", dst_st_path.display());
    let t2 = Instant::now();
    write_safetensors(&dst_st_path, &views)?;
    println!("written in {:.2?}", t2.elapsed());

    let orig_mb = orig_total_bytes as f64 / 1_048_576.0;
    let quant_mb = quant_total_bytes as f64 / 1_048_576.0;
    let savings = (1.0 - quant_mb / orig_mb) * 100.0;
    println!(
        "\nSuccess! Original: {:.2} MB -> Quantized: {:.2} MB ({:.1}% reduction)",
        orig_mb, quant_mb, savings
    );

    Ok(())
}
