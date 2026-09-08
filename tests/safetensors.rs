//! End-to-end exercises of the safetensors parser (in-memory blobs + files).

use std::path::PathBuf;

use timesfm3::checkpoint::{DType, Safetensors};

/// A two-tensor F32 safetensors blob, hand-assembled.
fn sample_blob() -> Vec<u8> {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
    let b: Vec<f32> = vec![10.0, 20.0]; // [2]
    let a_off = 0usize;
    let a_end = a.len() * 4;
    let b_off = a_end;
    let b_end = b_off + b.len() * 4;
    let header = format!(
        "{{\"a\": {{\"dtype\": \"F32\", \"shape\": [2, 3], \"data_offsets\": [{a_off}, {a_end}]}}, \"b\": {{\"dtype\": \"F32\", \"shape\": [2], \"data_offsets\": [{b_off}, {b_end}]}}, \"__metadata__\": {{\"k\": \"v\"}}}}",
        a_off = a_off,
        a_end = a_end,
        b_off = b_off,
        b_end = b_end,
    );
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    out.extend_from_slice(&b.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    out
}

#[test]
fn parses_in_memory_blob() {
    let st = Safetensors::parse_bytes(&sample_blob()).expect("test fixture invariant");
    assert_eq!(st.len(), 2);

    let a = st.f32_tensor("a").expect("test fixture invariant");
    assert_eq!((a.rows(), a.cols()), (2, 3));
    assert_eq!(a[(0, 0)], 1.0);
    assert_eq!(a[(1, 2)], 6.0);

    let b = st.f32_flat("b").expect("test fixture invariant");
    assert_eq!(b, vec![10.0, 20.0]);

    let info_a = st.tensor_info("a").expect("test fixture invariant");
    assert_eq!(info_a.dtype, DType::F32);
    assert_eq!(info_a.shape, vec![2, 3]);
    assert_eq!(info_a.byte_len, 24);
}

#[test]
fn missing_tensor_is_an_error() {
    let st = Safetensors::parse_bytes(&sample_blob()).expect("test fixture invariant");
    assert!(st.f32_tensor("nope").is_err());
    assert!(st.f32_flat("nope").is_err());
    assert!(st.tensor_info("nope").is_none());
}

#[test]
fn rejects_truncated_files() {
    assert!(Safetensors::parse_bytes(b"\x01\x00\x00\x00\x00\x00\x00\x00").is_err());
    let mut bogus = vec![0u8; 16];
    bogus[..8].copy_from_slice(&1_000_000u64.to_le_bytes());
    assert!(Safetensors::parse_bytes(&bogus).is_err());
}

#[test]
fn loads_golden_fixture_file() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden");
    let st = Safetensors::load(&root.join("residual_block_full/weights.safetensors"))
        .expect("test fixture invariant");
    assert!(st.len() >= 3); // hidden + output + residual (no bias)
    let w = st
        .f32_tensor("pre_transformer_resblock.output_layer.weight")
        .expect("test fixture invariant");
    assert_eq!((w.rows(), w.cols()), (1280, 1280));
    assert!(
        st.tensor_info("pre_transformer_resblock.hidden_layer.bias")
            .is_none()
    );
}
