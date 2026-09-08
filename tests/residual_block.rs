//! Golden cross-check: Rust residual block vs numpy-generated reference.

use std::path::PathBuf;

use timesfm3::checkpoint::Safetensors;
use timesfm3::config::{Activation, Norm, ResidualBlockConfig};
use timesfm3::model::residual_block::ResidualBlock;
use timesfm3::tensor::Tensor2D;

fn golden_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden")
}

fn read_f32_bin(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read bin fixture");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

fn run_fixture(name: &str, m: usize, inp: usize, hidden: usize, out: usize, use_bias: bool) {
    let root = golden_root().join(name);

    let store = Safetensors::load(&root.join("weights.safetensors"))
        .unwrap_or_else(|e| panic!("{name}: load safetensors: {e}"));
    let cfg = ResidualBlockConfig {
        hidden_dims: hidden,
        output_dims: out,
        use_bias,
        activation: Activation::Relu,
        identity_skip: false,
        prenorm: Norm::None_,
    };
    let block = ResidualBlock::load(&store, "pre_transformer_resblock", &cfg, f32::EPSILON)
        .unwrap_or_else(|e| panic!("{name}: load block: {e}"));

    let input = Tensor2D::from_row_major(m, inp, read_f32_bin(&root.join("input.bin")))
        .expect("input fixture shape");
    let expected = read_f32_bin(&root.join("expected.bin"));

    let got = block.forward(&input).data().to_vec();
    assert_eq!(got.len(), expected.len(), "{name}: output length");

    let mut max_abs = 0.0f32;
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        let diff = (g - e).abs();
        max_abs = max_abs.max(diff);
        // Tolerance: 1e-3 absolute + 1e-4 relative — generous enough for f32
        // accumulation-order differences, tight enough to catch real bugs.
        let allowed = 1e-3 + 1e-4 * e.abs();
        assert!(
            diff <= allowed,
            "{name}: element {i}: got {g} != ref {e} (diff {diff:e} > {allowed:e})"
        );
    }
    println!("{name}: ok, max_abs_diff = {max_abs:.3e}");
}

#[test]
fn residual_block_official_dims() {
    // Official config: in=192 (2*(32+64)), hidden=1280, out=1280, no bias.
    run_fixture("residual_block_full", 4, 192, 1280, 1280, false);
}

#[test]
fn residual_block_with_bias_small() {
    run_fixture("residual_block_bias", 3, 9, 24, 16, true);
}
