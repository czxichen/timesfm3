//! Elementwise activations (in-place, so no temporaries on the hot path).

use crate::tensor::Tensor2D;

/// ReLU applied in place.
pub fn relu(t: &mut Tensor2D) {
    for v in t.data_mut() {
        *v = v.max(0.0);
    }
}

/// SiLU (swish): x * sigmoid(x), applied in place (f64 exp, single rounding).
pub fn silu(t: &mut Tensor2D) {
    for v in t.data_mut() {
        let x = *v as f64;
        *v = (x / (1.0 + (-x).exp())) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_clips_negatives() {
        let mut t = Tensor2D::from_row_major(1, 4, vec![-1.0, 0.5, 0.0, -3.0])
            .expect("test fixture invariant");
        relu(&mut t);
        assert_eq!(t.data(), &[0.0, 0.5, 0.0, 0.0]);
    }

    #[test]
    fn silu_shape_preserved() {
        let mut t = Tensor2D::from_row_major(2, 2, vec![-1.0, 0.0, 1.0, 2.0])
            .expect("test fixture invariant");
        silu(&mut t);
        assert_eq!(t.rows(), 2);
        assert_eq!(t.cols(), 2);
    }
}
