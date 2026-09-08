//! Property tests for pure numerical helpers + parsing/validation edges.
//!
//! Dimensions covered (no model weights needed, all sub-second):
//! - PAVA isotonic regression: monotonicity, idempotence, sorted-input
//!   identity, degenerate lengths,
//! - seasonal decomposition: residual+profile roundtrip, short-series
//!   fallback, NaN passthrough,
//! - boundary smoothing: gap weld + decay, empty/NaN no-ops,
//! - JSON parser error paths + accessor type mismatches,
//! - `ModelConfig::validate` rejection rules + `QuantizationPrecision::parse`,
//! - `DType::parse` + `write_safetensors` dtype allowlist,
//! - safetensors write->read roundtrip (F32 values, F16 bits) + malformed
//!   header rejection (unknown dtype, reversed offsets),
//! - `Tensor2D` shape errors + double-transpose identity.

use timesfm3::checkpoint::{DType, Safetensors, TensorView, write_safetensors};
use timesfm3::config::{ModelConfig, QuantizationPrecision};
use timesfm3::forecast::{
    apply_boundary_smoothing, decompose_seasonal_with_offset, pava_isotonic_regression,
};
use timesfm3::json::Json;
use timesfm3::tensor::Tensor2D;

// ---------- PAVA ----------

fn is_nondecreasing(y: &[f32]) -> bool {
    y.windows(2).all(|w| w[0] <= w[1])
}

#[test]
fn pava_fixes_crossed_quantiles() {
    let mut y = vec![3.0, 1.0, 2.0, 2.0, 0.5, 4.0];
    pava_isotonic_regression(&mut y);
    assert!(
        is_nondecreasing(&y),
        "PAVA output must be non-decreasing: {y:?}"
    );
    // Least-squares solution preserves the overall mean.
    let mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
    assert!((mean - (3.0 + 1.0 + 2.0 + 2.0 + 0.5 + 4.0) / 6.0).abs() < 1e-5);
}

#[test]
fn pava_is_identity_on_sorted_and_idempotent() {
    let mut sorted = vec![0.5, 1.0, 1.0, 2.5, 9.0];
    pava_isotonic_regression(&mut sorted);
    assert_eq!(sorted, vec![0.5, 1.0, 1.0, 2.5, 9.0]);

    let mut y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    pava_isotonic_regression(&mut y);
    assert!(is_nondecreasing(&y));
    let once = y.clone();
    pava_isotonic_regression(&mut y);
    assert_eq!(y, once, "PAVA must be idempotent");
}

#[test]
fn pava_degenerate_lengths_do_not_panic() {
    let mut empty: Vec<f32> = vec![];
    pava_isotonic_regression(&mut empty);
    assert!(empty.is_empty());
    let mut one = vec![7.0];
    pava_isotonic_regression(&mut one);
    assert_eq!(one, vec![7.0]);
}

// ---------- seasonal decomposition ----------

#[test]
fn seasonal_decomposition_roundtrips() {
    let values: Vec<f32> = (0..32)
        .map(|t| (t as f32 * 0.7).sin() * 3.0 + t as f32 * 0.1)
        .collect();
    let (residual, profile) = decompose_seasonal_with_offset(&values, 8, 3);
    assert_eq!(residual.len(), values.len());
    assert_eq!(profile.len(), 8);
    // Zero-centered profile.
    let mean = profile.iter().sum::<f32>() / profile.len() as f32;
    assert!(
        mean.abs() < 1e-4,
        "profile must be zero-centered, mean={mean}"
    );
    // residual[t] + profile[(offset+t) % period] == values[t].
    for (t, (&r, &v)) in residual.iter().zip(&values).enumerate() {
        let back = r + profile[(3 + t) % 8];
        assert!((back - v).abs() < 1e-4, "roundtrip failed at {t}");
    }
}

#[test]
fn seasonal_short_series_falls_back() {
    let values = vec![1.0, 2.0, 3.0];
    let (residual, profile) = decompose_seasonal_with_offset(&values, 8, 0);
    assert_eq!(residual, values, "too-short series must pass through");
    assert!(profile.iter().all(|&p| p == 0.0));
}

#[test]
fn seasonal_passes_nans_through() {
    let mut values: Vec<f32> = (0..24).map(|t| t as f32).collect();
    values[5] = f32::NAN;
    let (residual, _) = decompose_seasonal_with_offset(&values, 6, 0);
    assert!(residual[5].is_nan());
    assert!(residual.iter().filter(|v| v.is_nan()).count() == 1);
}

// ---------- boundary smoothing ----------

#[test]
fn boundary_smoothing_welds_first_step_and_decays() {
    let mut fcst = vec![0.0f32; 8];
    apply_boundary_smoothing(&mut fcst, None, 10.0, 8, 1);
    assert!(
        (fcst[0] - 10.0).abs() < 1e-5,
        "first step must weld to context"
    );
    // Exponential decay: later corrections strictly shrink.
    let deltas: Vec<f32> = fcst.to_vec();
    for h in 1..8 {
        assert!(deltas[h] < deltas[h - 1] && deltas[h] > 0.0);
    }
}

#[test]
fn boundary_smoothing_noops() {
    // Empty forecast.
    let mut empty: Vec<f32> = vec![];
    apply_boundary_smoothing(&mut empty, None, 5.0, 0, 1);
    // NaN context.
    let mut fcst = vec![1.0, 2.0, 3.0];
    apply_boundary_smoothing(&mut fcst, None, f32::NAN, 3, 1);
    assert_eq!(fcst, vec![1.0, 2.0, 3.0]);
    // Negligible gap (< 1e-6) leaves values untouched.
    let mut fcst2 = vec![5.0, 6.0];
    apply_boundary_smoothing(&mut fcst2, None, 5.0 + 1e-7, 2, 1);
    assert_eq!(fcst2, vec![5.0, 6.0]);
}

// ---------- JSON ----------

#[test]
fn json_rejects_malformed_input() {
    for src in [
        "{", "[1,", "{\"a\":}", "\"abc", "{'a':1}", "tru", "nul", "12x", "", "   ", "\x00",
    ] {
        assert!(Json::parse(src.as_bytes()).is_err(), "must reject {src:?}");
    }
}

#[test]
fn json_documents_trailing_comma_tolerance() {
    // The lightweight parser tolerates trailing commas (it is more lenient
    // than strict JSON). Lock the values so a future strictness change is
    // a deliberate, visible decision.
    let arr = Json::parse(b"[1, 2,]").expect("trailing comma in array");
    let items = arr.as_arr().unwrap();
    assert_eq!(items.len(), 2);
    let obj = Json::parse(br#"{"a":1,}"#).expect("trailing comma in object");
    let fields = obj.as_obj().unwrap();
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].0, "a");
}

#[test]
fn json_accessors_reject_wrong_types() {
    let v = Json::parse(br#"{"n": 3, "s": "x", "a": [1]}"#).unwrap();
    let obj = v.as_obj().unwrap();
    assert!(v.as_arr().is_none());
    assert!(v.as_str().is_none());
    assert!(v.as_u64().is_none());
    let n = &obj.iter().find(|(k, _)| k == "n").unwrap().1;
    assert_eq!(n.as_u64(), Some(3));
    assert!(n.as_str().is_none());
    assert!(n.as_obj().is_none());
    // Floats are not integers.
    let f = Json::parse(b"1.5").unwrap();
    assert!(f.as_u64().is_none());
    // Escapes + unicode in strings.
    let s = Json::parse(br#""a\nb\u00e9""#).unwrap();
    assert_eq!(s.as_str(), Some("a\nbé"));
}

// ---------- config ----------

#[test]
fn config_validate_accepts_default_rejects_broken() {
    ModelConfig::default()
        .validate()
        .expect("default must validate");

    let bad_patch = ModelConfig {
        output_patch_len: 33,
        ..Default::default()
    }; // not a multiple of 32
    assert!(bad_patch.validate().is_err());

    let mut bad_dims = ModelConfig::default();
    bad_dims.residual_block_config.output_dims = 64;
    assert!(bad_dims.validate().is_err());

    let bad_q0 = ModelConfig {
        num_quantiles: 0,
        ..Default::default()
    };
    assert!(bad_q0.validate().is_err());

    let mut bad_med = ModelConfig::default();
    bad_med.median_quantile_index = bad_med.num_quantiles; // out of range
    assert!(bad_med.validate().is_err());

    // A non-object config.json must fail the overlay.
    let mut cfg = ModelConfig::default();
    let arr = Json::parse(b"[1,2]").unwrap();
    assert!(cfg.from_config_json(&arr).is_err());
}

#[test]
fn quantization_precision_parse() {
    assert_eq!(
        QuantizationPrecision::parse("f16"),
        Some(QuantizationPrecision::F16)
    );
    assert_eq!(
        QuantizationPrecision::parse("F16"),
        Some(QuantizationPrecision::F16)
    );
    assert_eq!(
        QuantizationPrecision::parse("fp16"),
        Some(QuantizationPrecision::F16)
    );
    assert_eq!(
        QuantizationPrecision::parse("f32"),
        Some(QuantizationPrecision::F32)
    );
    assert_eq!(
        QuantizationPrecision::parse("fp32"),
        Some(QuantizationPrecision::F32)
    );
    assert_eq!(QuantizationPrecision::parse("int8"), None);
    assert_eq!(QuantizationPrecision::parse(""), None);
    assert_eq!(QuantizationPrecision::default(), QuantizationPrecision::F32);
}

// ---------- checkpoint dtypes / roundtrip ----------

#[test]
fn dtype_parse_allowlist() {
    for (s, ok) in [
        ("F32", true),
        ("F16", true),
        ("F64", true),
        ("I32", true),
        ("BOOL", true),
        ("BF16", false),
        ("f32", false),
        ("", false),
    ] {
        assert_eq!(DType::parse(s).is_some(), ok, "dtype {s:?}");
    }
}

fn write_tmp(name: &str, tensors: &[TensorView]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("timesfm3_{}_{}", std::process::id(), name));
    let _ = std::fs::remove_file(&path);
    write_safetensors(&path, tensors).expect("write");
    path
}

#[test]
fn safetensors_write_read_roundtrip_f32_and_f16() {
    let w: Vec<f32> = vec![1.0, -2.5, 3.25, 4.0, 5.5, 6.75];
    let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    // F16 bits without the `half` crate: 1.0=0x3C00, -2.0=0xC000,
    // 0.5=0x3800, +inf=0x7C00.
    let h_bits: Vec<u16> = vec![0x3C00, 0xC000, 0x3800, 0x7C00];
    let h_bytes: Vec<u8> = h_bits.iter().flat_map(|v| v.to_le_bytes()).collect();

    let views = vec![
        TensorView {
            name: "w",
            dtype: DType::F32,
            shape: &[2, 3],
            data: &w_bytes,
        },
        TensorView {
            name: "h",
            dtype: DType::F16,
            shape: &[2, 2],
            data: &h_bytes,
        },
    ];
    let path = write_tmp("rt.safetensors", &views);
    let st = Safetensors::load(&path).expect("reload");
    assert_eq!(st.len(), 2);

    let back = st.f32_flat("w").expect("read w");
    assert_eq!(back, w);
    let t = st.f32_tensor("w").expect("read w tensor");
    assert_eq!((t.rows(), t.cols()), (2, 3));

    let hf = st.f32_flat("h").expect("read h as f32");
    assert_eq!(hf, vec![1.0, -2.0, 0.5, f32::INFINITY]);
    let info = st.tensor_info("h").expect("info h");
    assert_eq!(info.dtype, DType::F16);
    assert_eq!(info.byte_len, 8);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn write_safetensors_rejects_unsupported_dtype() {
    let data = vec![0u8; 8];
    for dtype in [DType::F64, DType::U8, DType::I8] {
        let path = std::env::temp_dir().join(format!(
            "timesfm3_reject_{}_{:?}",
            std::process::id(),
            dtype
        ));
        let views = vec![TensorView {
            name: "x",
            dtype,
            shape: &[2],
            data: &data,
        }];
        assert!(
            write_safetensors(&path, &views).is_err(),
            "dtype {dtype:?} must be rejected"
        );
        assert!(!path.exists());
    }
}

#[test]
fn safetensors_rejects_unknown_dtype_and_reversed_offsets() {
    // Unknown dtype string.
    let header = "{\"w\": {\"dtype\": \"BF16\", \"shape\": [2], \"data_offsets\": [0, 4]}}";
    let mut blob = Vec::new();
    blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
    blob.extend_from_slice(header.as_bytes());
    blob.extend_from_slice(&[0u8; 4]);
    assert!(Safetensors::parse_bytes(&blob).is_err());

    // Reversed offsets.
    let header2 = "{\"w\": {\"dtype\": \"F32\", \"shape\": [1], \"data_offsets\": [4, 0]}}";
    let mut blob2 = Vec::new();
    blob2.extend_from_slice(&(header2.len() as u64).to_le_bytes());
    blob2.extend_from_slice(header2.as_bytes());
    blob2.extend_from_slice(&[0u8; 4]);
    assert!(Safetensors::parse_bytes(&blob2).is_err());
}

// ---------- Tensor2D ----------

#[test]
fn tensor_shape_errors_and_double_transpose() {
    assert!(Tensor2D::from_row_major(2, 3, vec![1.0; 5]).is_err());
    assert!(Tensor2D::from_slice(2, 2, &[1.0, 2.0, 3.0]).is_err());

    let t = Tensor2D::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let tt = t.transpose().transpose();
    assert_eq!(tt.data(), t.data());
    assert_eq!((tt.rows(), tt.cols()), (2, 3));
    // Spot-check the single transpose layout: row-major (2,3) -> (3,2).
    let one = t.transpose();
    assert_eq!((one.rows(), one.cols()), (3, 2));
    assert_eq!(one.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let a = Tensor2D::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b = Tensor2D::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    assert_eq!(a.max_abs_diff(&b), 0.0);
    let c = Tensor2D::from_row_major(2, 2, vec![1.0, 0.5, 0.0, 1.0]).unwrap();
    assert!((a.max_abs_diff(&c) - 0.5).abs() < 1e-9);
}
