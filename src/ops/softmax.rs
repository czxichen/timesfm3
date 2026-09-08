//! Row-wise softmax over the last dimension (in place), with optional mask.

use crate::tensor::Tensor2D;

/// Stable row-softmax of `(rows, cols)` in place. `mask` (optional, one bool
/// per element) marks positions that are forced to 0 (e.g. causal / patch
/// masking). Masking is applied as `-inf` before exp, matching PyTorch's
/// `scaled_dot_product_attention` bool-mask semantics.
pub fn softmax_rows(x: &mut Tensor2D, mask: Option<&[bool]>) {
    let cols = x.cols();
    let rows = x.rows();
    let data = x.data_mut();
    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];
        let row_mask = mask.map(|m| &m[r * cols..(r + 1) * cols]);
        let mut max = f32::NEG_INFINITY;
        for (j, &v) in row.iter().enumerate() {
            if row_mask.is_none_or(|m| !m[j]) {
                max = max.max(v);
            }
        }
        // If every position is masked the row becomes uniform-ish zeros; keep
        // the same convention as -inf masking (exp(-inf)=0).
        let mut sum = 0.0f64;
        for (j, v) in row.iter_mut().enumerate() {
            let masked = row_mask.is_some_and(|m| m[j]);
            let e = if masked || !max.is_finite() {
                0.0f32
            } else {
                ((*v - max) as f64).exp() as f32
            };
            *v = e;
            sum += e as f64;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v = (*v as f64 / sum) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_rows() {
        let mut x = Tensor2D::from_row_major(2, 2, vec![1.0, 1.0, 0.0, 100.0]).expect("fixture");
        softmax_rows(&mut x, None);
        assert!((x[(0, 0)] - 0.5).abs() < 1e-6);
        assert!((x[(1, 1)]).abs() > 0.999999);
    }

    #[test]
    fn masked_positions_zero() {
        let mut x = Tensor2D::from_row_major(1, 3, vec![1.0, 1.0, 1.0]).expect("fixture");
        let mask = vec![false, true, false];
        softmax_rows(&mut x, Some(&mask));
        assert_eq!(x[(0, 0)], 0.5);
        assert_eq!(x[(0, 1)], 0.0);
        assert_eq!(x[(0, 2)], 0.5);
    }
}
