//! Ops for the TimesFM 3.0 inference graph.

pub mod activations;
pub mod gemm;
pub mod norm;
pub mod rope;
pub mod softmax;
pub mod speculative;

use crate::tensor::Tensor2D;

/// `dst += src` elementwise (shapes must match exactly).
pub fn add_into(dst: &mut Tensor2D, src: &Tensor2D) {
    assert_eq!(
        (dst.rows(), dst.cols()),
        (src.rows(), src.cols()),
        "add_into: shape mismatch ({}x{}, {}x{})",
        dst.rows(),
        dst.cols(),
        src.rows(),
        src.cols()
    );
    for (d, &s) in dst.data_mut().iter_mut().zip(src.data()) {
        *d += s;
    }
}
