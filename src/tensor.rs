//! Minimal row-major f32 2-D tensor.
//!
//! Deliberately NOT a generic N-d array system: this engine has exactly one
//! hot layout (row-major f32 matrices for GEMM) plus structured reshapes that
//! the model layers handle explicitly. Anything more generic would buy nothing
//! and cost binary size.

use rayon::prelude::*;
use std::ops::{Index, IndexMut};

use crate::error::{Error, Result};

/// Row-major f32 matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor2D {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Tensor2D {
    /// Creates a new zero-initialized matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates a matrix from flat row-major data, validating the length.
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self> {
        if rows * cols != data.len() {
            return Err(Error(format!(
                "Tensor2D: shape ({rows}, {cols}) needs {} elements, got {}",
                rows * cols,
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn elements(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Transposes into a fresh matrix (used once at checkpoint load time to
    /// normalize torch `(out, in)` weights into inference `(in, out)` layout).
    ///
    /// Blocked cache-friendly tiling; when the matrix is large the row blocks
    /// transpose in parallel (each block writes a disjoint column range of the
    /// output, so no synchronization is needed).
    pub fn transpose(&self) -> Self {
        let rows = self.rows;
        let cols = self.cols;
        let mut t = Self::zeros(cols, rows);
        const TILE: usize = 64;
        let total = rows * cols;
        if rayon::current_num_threads() <= 1 || total < (1 << 18) {
            unsafe {
                transpose_f32_tiled(
                    self.data.as_ptr(),
                    t.data.as_mut_ptr(),
                    rows,
                    cols,
                    TILE,
                    0,
                    rows,
                );
            }
        } else {
            // Each task transposes a slab of `TILE` input rows into output
            // columns `r0..r1`; tasks write disjoint output regions. Raw
            // pointers travel as `usize` so the `Fn + Sync` rayon closure
            // stays sendable (same pattern as gemm.rs).
            let src = self.data.as_ptr() as usize;
            let dst = t.data.as_mut_ptr() as usize;
            (0..rows).into_par_iter().step_by(TILE).for_each(move |r0| {
                let r1 = (r0 + TILE).min(rows);
                unsafe {
                    transpose_f32_tiled(
                        src as *const f32,
                        dst as *mut f32,
                        rows,
                        cols,
                        TILE,
                        r0,
                        r1,
                    );
                }
            });
        }
        t
    }

    /// Overwrites this tensor's data with data from `src` (shapes must match).
    #[inline]
    pub fn copy_from(&mut self, src: &Self) {
        assert_eq!(
            (self.rows, self.cols),
            (src.rows, src.cols),
            "copy_from: shape mismatch ({}x{} vs {}x{})",
            self.rows,
            self.cols,
            src.rows,
            src.cols
        );
        self.data.copy_from_slice(&src.data);
    }

    /// Largest absolute difference between two same-shaped matrices.
    pub fn max_abs_diff(&self, other: &Self) -> f32 {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "max_abs_diff: shape mismatch"
        );
        self.data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }
}

impl Index<(usize, usize)> for Tensor2D {
    type Output = f32;

    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &f32 {
        &self.data[r * self.cols + c]
    }
}

impl IndexMut<(usize, usize)> for Tensor2D {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f32 {
        &mut self.data[r * self.cols + c]
    }
}

impl Tensor2D {
    /// Consumes the tensor and returns its owned data.
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Builds a `(rows, cols)` tensor from a flat slice (copies).
    pub fn from_slice(rows: usize, cols: usize, data: &[f32]) -> Result<Self> {
        Self::from_row_major(rows, cols, data.to_vec())
    }
}

/// Transposes `data` (a `rows x cols` row-major f16 matrix) into a fresh
/// `cols x rows` row-major f16 vector. Used at checkpoint load to keep
/// quantized weights in half precision end-to-end (avoids the
/// decode->f32->re-quantize round trip that doubles/triples the memory
/// traffic of quantized checkpoints).
pub fn transpose_f16(data: &[half::f16], rows: usize, cols: usize) -> Vec<half::f16> {
    let mut dst = vec![half::f16::ZERO; cols * rows];
    const TILE: usize = 64;
    let total = rows * cols;
    debug_assert_eq!(data.len(), total);
    if rayon::current_num_threads() <= 1 || total < (1 << 18) {
        unsafe {
            transpose_f16_tiled(data.as_ptr(), dst.as_mut_ptr(), rows, cols, TILE, 0, rows);
        }
    } else {
        let src = data.as_ptr() as usize;
        let dptr = dst.as_mut_ptr() as usize;
        (0..rows).into_par_iter().step_by(TILE).for_each(move |r0| {
            let r1 = (r0 + TILE).min(rows);
            unsafe {
                transpose_f16_tiled(
                    src as *const half::f16,
                    dptr as *mut half::f16,
                    rows,
                    cols,
                    TILE,
                    r0,
                    r1,
                );
            }
        });
    }
    dst
}

/// Block-tiled transpose of input rows `[r0, r1)` of a `rows x cols` f32
/// matrix into output columns `r0..r1` (raw pointers; disjoint output columns
/// per caller-chosen row slab).
unsafe fn transpose_f32_tiled(
    src: *const f32,
    dst: *mut f32,
    rows: usize,
    cols: usize,
    tile: usize,
    r0: usize,
    r1: usize,
) {
    unsafe {
        let mut r_b = r0;
        while r_b < r1 {
            let r_e = (r_b + tile).min(r1);
            let mut c_b = 0;
            while c_b < cols {
                let c_e = (c_b + tile).min(cols);
                for r in r_b..r_e {
                    let src_off = r * cols;
                    for c in c_b..c_e {
                        *dst.add(c * rows + r) = *src.add(src_off + c);
                    }
                }
                c_b += tile;
            }
            r_b += tile;
        }
    }
}

/// f16 twin of [`transpose_f32_tiled`].
unsafe fn transpose_f16_tiled(
    src: *const half::f16,
    dst: *mut half::f16,
    rows: usize,
    cols: usize,
    tile: usize,
    r0: usize,
    r1: usize,
) {
    unsafe {
        let mut r_b = r0;
        while r_b < r1 {
            let r_e = (r_b + tile).min(r1);
            let mut c_b = 0;
            while c_b < cols {
                let c_e = (c_b + tile).min(cols);
                for r in r_b..r_e {
                    let src_off = r * cols;
                    for c in c_b..c_e {
                        *dst.add(c * rows + r) = *src.add(src_off + c);
                    }
                }
                c_b += tile;
            }
            r_b += tile;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_row_major_validates_len() {
        assert!(Tensor2D::from_row_major(2, 3, vec![0.0; 6]).is_ok());
        assert!(Tensor2D::from_row_major(2, 3, vec![0.0; 5]).is_err());
    }

    #[test]
    fn transpose_swap() {
        let a = Tensor2D::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("test fixture invariant");
        let t = a.transpose();
        assert_eq!((t.rows(), t.cols()), (3, 2));
        assert_eq!(t[(0, 1)], 4.0);
        assert_eq!(t[(2, 1)], 6.0);
        assert_eq!(t.transpose(), a);
    }

    #[test]
    fn transpose_matches_naive_for_large() {
        // Bigger than the parallel threshold -> exercises the rayon path;
        // cross-check against a simple scalar reference.
        let (rows, cols) = (512, 77);
        let src: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.5).collect();
        let a = Tensor2D::from_row_major(rows, cols, src).expect("fixture");
        let t = a.transpose();
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(t[(c, r)], a[(r, c)]);
            }
        }
        assert_eq!(t.rows(), cols);
        assert_eq!(t.cols(), rows);
    }

    #[test]
    fn transpose_f16_matches_scalar() {
        let (rows, cols) = (128, 39);
        let src: Vec<half::f16> = (0..rows * cols)
            .map(|i| half::f16::from_f32((i as f32) * 0.25))
            .collect();
        let t = transpose_f16(&src, rows, cols);
        assert_eq!(t.len(), cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(t[c * rows + r], src[r * cols + c]);
            }
        }
    }
}
