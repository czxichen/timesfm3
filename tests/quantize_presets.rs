//! Quantization preset coverage on the tiny golden model (`golden/e2e_micro`).
//!
//! Dimensions covered:
//! - full-F16 dtype census (every 2D `.weight` -> F16, 1D tensors stay F32),
//! - `--preset balanced` / `--preset ultra` keep sensitive prefixes in F32,
//! - custom `--keep-f32` prefix filtering,
//! - `config.json` precision stamping,
//! - CLI error paths (no args / bad precision / missing src / unknown preset),
//! - decode parity: quantized full-F16 model vs FP32 on identical input.
//!
//! Uses the real `quantize` binary via `CARGO_BIN_EXE_quantize` so the CLI
//! arg parsing is exercised too. The micro checkpoint is 13 KB, so this
//! suite runs in well under a second.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use timesfm3::checkpoint::{DType, Safetensors};
use timesfm3::config::ModelConfig;
use timesfm3::json::Json;
use timesfm3::model::timesfm::TimesFM3;
use timesfm3::tensor::Tensor2D;

fn micro_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden/e2e_micro")
}

fn quantize_bin() -> &'static str {
    env!("CARGO_BIN_EXE_quantize")
}

/// Unique scratch dir per test (Windows-safe, no shell involved).
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("timesfm3_qt_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn run_quantize(src: &Path, dst: &Path, extra: &[&str]) -> Output {
    let mut cmd = Command::new(quantize_bin());
    cmd.arg(src).arg(dst).arg("f16");
    for a in extra {
        cmd.arg(a);
    }
    cmd.output().expect("spawn quantize binary")
}

/// Census of output dtypes: (f16_names, f32_names).
fn census(dir: &Path) -> (HashSet<String>, HashSet<String>) {
    let store = Safetensors::load(&dir.join("model.safetensors")).expect("load output");
    let mut f16 = HashSet::new();
    let mut f32 = HashSet::new();
    for (name, info) in store.iter() {
        match info.dtype {
            DType::F16 => {
                f16.insert(name.to_owned());
            }
            DType::F32 => {
                f32.insert(name.to_owned());
            }
            ref other => panic!("unexpected dtype {other:?} for {name}"),
        }
    }
    (f16, f32)
}

/// Expected 2D `.weight` tensors of the source checkpoint (mirrors the
/// quantizer rule: only 2D `*.weight` matrices are converted).
fn src_2d_weights() -> HashSet<String> {
    let store = Safetensors::load(&micro_dir().join("model.safetensors")).expect("load src");
    store
        .iter()
        .filter(|(name, info)| info.shape.len() == 2 && name.ends_with(".weight"))
        .map(|(name, _)| name.to_owned())
        .collect()
}

fn src_all_names() -> HashSet<String> {
    let store = Safetensors::load(&micro_dir().join("model.safetensors")).expect("load src");
    store.iter().map(|(n, _)| n.to_owned()).collect()
}

fn kept_by_prefixes(names: &HashSet<String>, prefixes: &[&str]) -> HashSet<String> {
    names
        .iter()
        .filter(|n| prefixes.iter().any(|p| n.starts_with(p)))
        .cloned()
        .collect()
}

fn assert_precision_stamped(dst: &Path) {
    let cfg = std::fs::read_to_string(dst.join("config.json")).expect("output config.json");
    assert!(
        cfg.contains("\"precision\""),
        "output config.json must declare precision"
    );
    assert!(cfg.contains("f16"), "output precision must be f16");
}

#[test]
fn full_f16_converts_every_2d_weight() {
    let dst = scratch("full");
    let out = run_quantize(&micro_dir(), &dst, &[]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let want_f16 = src_2d_weights();
    assert!(!want_f16.is_empty(), "micro model must have 2D weights");
    let all = src_all_names();
    let (got_f16, got_f32) = census(&dst);

    assert_eq!(got_f16, want_f16, "every 2D .weight must be F16");
    let want_f32: HashSet<String> = all.difference(&want_f16).cloned().collect();
    assert_eq!(
        got_f32, want_f32,
        "1D tensors (bias/norm/scales) must stay F32"
    );
    assert_precision_stamped(&dst);
    let _ = std::fs::remove_dir_all(&dst);
}

#[test]
fn balanced_preset_keeps_sensitive_layers_f32() {
    let dst = scratch("balanced");
    let out = run_quantize(&micro_dir(), &dst, &["--preset", "balanced"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let all = src_all_names();
    let w2d = src_2d_weights();
    // Balanced prefixes; layers.17-19 match nothing on the 2-layer micro
    // model (asserted below), so the F32 set is head + resblock + 1D tensors.
    let keep_prefixes = [
        "output_head",
        "pre_transformer_resblock",
        "transformer_stack.layers.17.",
        "transformer_stack.layers.18.",
        "transformer_stack.layers.19.",
    ];
    let kept_2d = kept_by_prefixes(&w2d, &keep_prefixes);
    assert!(
        !kept_2d.is_empty(),
        "balanced must keep at least head+resblock in F32"
    );
    assert!(
        kept_2d
            .iter()
            .all(|n| n.starts_with("output_head") || n.starts_with("pre_transformer_resblock")),
        "on the 2-layer micro model only head+resblock may be kept, got: {kept_2d:?}"
    );
    let want_f32: HashSet<String> = all
        .difference(&w2d)
        .cloned()
        .collect::<HashSet<_>>()
        .union(&kept_2d)
        .cloned()
        .collect();
    let want_f16: HashSet<String> = w2d.difference(&kept_2d).cloned().collect();
    assert!(
        !want_f16.is_empty(),
        "most layer weights must still quantize"
    );

    let (got_f16, got_f32) = census(&dst);
    assert_eq!(got_f16, want_f16);
    assert_eq!(got_f32, want_f32);
    // Spot-check the sensitive tensors explicitly.
    for n in [
        "output_head.weight",
        "pre_transformer_resblock.hidden_layer.weight",
    ] {
        assert!(got_f32.contains(n), "{n} must stay F32 under balanced");
    }
    assert!(
        got_f16.contains("transformer_stack.layers.1.ff1.weight"),
        "late-layer weights must quantize under balanced"
    );
    assert_precision_stamped(&dst);
    let _ = std::fs::remove_dir_all(&dst);
}

#[test]
fn ultra_preset_is_a_superset_of_balanced_on_disk() {
    let dst_b = scratch("ultra_b");
    let dst_u = scratch("ultra_u");
    let out_b = run_quantize(&micro_dir(), &dst_b, &["--preset", "balanced"]);
    let out_u = run_quantize(&micro_dir(), &dst_u, &["--preset", "ultra"]);
    assert!(out_b.status.success() && out_u.status.success());

    let (b_f16, b_f32) = census(&dst_b);
    let (u_f16, u_f32) = census(&dst_u);
    // Ultra keeps head + resblock + last 5 layers; on the 2-layer micro model
    // the F32 set coincides with balanced (both keep head+resblock only).
    assert_eq!(
        b_f32, u_f32,
        "micro: ultra F32 set must equal balanced F32 set"
    );
    assert_eq!(b_f16, u_f16);
    assert!(u_f32.contains("output_head.weight"));
    let _ = std::fs::remove_dir_all(&dst_b);
    let _ = std::fs::remove_dir_all(&dst_u);
}

#[test]
fn custom_keep_f32_prefix_filters_layers() {
    let dst = scratch("keep");
    let out = run_quantize(
        &micro_dir(),
        &dst,
        &["--keep-f32", "transformer_stack.layers.0."],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let (got_f16, got_f32) = census(&dst);
    for n in &got_f32 {
        if src_2d_weights().contains(n) {
            assert!(
                n.starts_with("transformer_stack.layers.0."),
                "only layer-0 2D weights may stay F32, got {n}"
            );
        }
    }
    assert!(
        got_f16.contains("transformer_stack.layers.1.ff1.weight"),
        "layer-1 weights must quantize with a layer-0 keep filter"
    );
    assert!(
        got_f32.contains("transformer_stack.layers.0.ff1.weight"),
        "layer-0 weights must stay F32 with a layer-0 keep filter"
    );
    let _ = std::fs::remove_dir_all(&dst);
}

#[test]
fn cli_rejects_bad_inputs() {
    // No args -> usage, exit 2.
    let out = Command::new(quantize_bin())
        .output()
        .expect("spawn quantize");
    assert_eq!(out.status.code(), Some(2));

    // Unknown preset.
    let dst = scratch("bad_preset");
    let out = run_quantize(&micro_dir(), &dst, &["--preset", "extreme"]);
    assert!(!out.status.success(), "unknown preset must fail");

    // Invalid precision argument.
    let dst2 = scratch("bad_prec");
    let out = Command::new(quantize_bin())
        .arg(micro_dir())
        .arg(&dst2)
        .arg("int8")
        .output()
        .expect("spawn quantize");
    assert!(!out.status.success(), "non-f16 precision must fail");

    // Missing source checkpoint.
    let dst3 = scratch("missing_src");
    let out = Command::new(quantize_bin())
        .arg(micro_dir().join("does_not_exist"))
        .arg(&dst3)
        .arg("f16")
        .output()
        .expect("spawn quantize");
    assert!(!out.status.success(), "missing src must fail");
    let _ = std::fs::remove_dir_all(&dst);
}

fn load_micro_model(dir: &Path) -> (TimesFM3, ModelConfig) {
    let cfg_bytes = std::fs::read(dir.join("config.json")).expect("config.json");
    let root = Json::parse(&cfg_bytes).expect("parse config.json");
    let mut config = ModelConfig::default();
    config.from_config_json(&root).expect("overlay config");
    config.validate().expect("config valid");
    let store = Safetensors::load(&dir.join("model.safetensors")).expect("load weights");
    let model = TimesFM3::load(&store, &config).expect("assemble model");
    (model, config)
}

#[test]
fn quantized_full_f16_decodes_close_to_f32() {
    let dst = scratch("parity");
    let out = run_quantize(&micro_dir(), &dst, &[]);
    assert!(out.status.success());

    let (f32_model, _) = load_micro_model(&micro_dir());
    let (f16_model, _) = load_micro_model(&dst);

    // Deterministic ramp input, C=16 (multiple of input_patch_len=4).
    let ctx: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let target = Tensor2D::from_slice(1, 16, &ctx).expect("target shape");
    let d32 = f32_model
        .decode_shaped(&target, 8, None, None, None, None, None, None, 1, 1, 16)
        .expect("f32 decode");
    let d16 = f16_model
        .decode_shaped(&target, 8, None, None, None, None, None, None, 1, 1, 16)
        .expect("f16 decode");
    assert_eq!(d32.data().len(), d16.data().len());

    let mut max_abs = 0.0f32;
    for (a, b) in d32.data().iter().zip(d16.data()) {
        max_abs = max_abs.max((a - b).abs());
    }
    println!("micro f32-vs-f16 decode max_abs = {max_abs:.3e}");
    assert!(
        max_abs < 5e-2,
        "quantized decode drifted too far: {max_abs:.3e}"
    );
    let _ = std::fs::remove_dir_all(&dst);
}

/// Shape census of the micro checkpoint itself (locks the fixture contract
/// the preset tests above rely on).
#[test]
fn micro_fixture_contract() {
    let store = Safetensors::load(&micro_dir().join("model.safetensors")).expect("load src");
    let by_name: HashMap<&str, _> = store.iter().collect();
    // 1D tensors must exist (they pin the "stay F32" rule).
    assert!(by_name.contains_key("output_head.bias"));
    assert_eq!(by_name["output_head.bias"].shape.len(), 1);
    assert!(by_name.contains_key("transformer_stack.layers.0.seq_attn.per_dim_scale"));
    // 2D weights exist in every layer + head + resblock.
    for n in [
        "output_head.weight",
        "pre_transformer_resblock.hidden_layer.weight",
        "transformer_stack.layers.0.ff0.weight",
        "transformer_stack.layers.1.ff1.weight",
    ] {
        assert_eq!(by_name[n].shape.len(), 2, "{n} must be a 2D matrix");
    }
}
