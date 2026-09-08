//! Inference pipeline helpers (RevIN stats, rolling future covariates,
//! patch stitching, CPM-RevIN refinement).
//!
//! Ports of the official `util.py` / `cpm_revin_refine.py` functions, adapted
//! to flat row-major `Tensor2D` buffers. All shapes are `(b, v, n, p)`-style
//! flattened to `(b*v, n*p)` etc. — every function takes explicit dims.

use crate::tensor::Tensor2D;

/// Tolerance governing the "safe division" clamp in RevIN (official `_TOLERANCE`).
const TOLERANCE: f32 = 1e-6;

fn make_safe_for_division(values: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(values) {
        *o = if v < TOLERANCE { 1.0 } else { v };
    }
}

/// Incremental running-stat update for one patch (official `update_running_stats`).
/// `x` is `(b, v, p)`, `mask` bool `(b, v, p)` (true = masked).
/// Returns (new_n, new_mu, new_sigma) each `(b, v)` flat.
pub fn update_running_stats(
    n_prev: &[f32],
    mu_prev: &[f32],
    sigma_prev: &[f32],
    x: &[f32],
    mask: &[bool],
    p: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let bv = x.len() / p;
    let mut new_n = vec![0.0f32; bv];
    let mut new_mu = vec![0.0f32; bv];
    let mut new_sigma = vec![0.0f32; bv];

    for i in 0..bv {
        let base = i * p;
        // All statistics are accumulated in f64 and rounded to f32 once, so
        // each output is the correctly rounded value of the exact statistic.
        let mut inc_n = 0.0f64;
        let mut inc_sum = 0.0f64;
        for j in 0..p {
            if !mask[base + j] {
                inc_n += 1.0;
                inc_sum += x[base + j] as f64;
            }
        }
        let inc_mu = if inc_n == 0.0 { 0.0 } else { inc_sum / inc_n };

        let mut diff_sq = 0.0f64;
        for j in 0..p {
            if !mask[base + j] {
                let d = x[base + j] as f64 - inc_mu;
                diff_sq += d * d;
            }
        }
        let inc_var = if inc_n == 0.0 { 0.0 } else { diff_sq / inc_n };
        let inc_sigma = inc_var.sqrt();

        let n_prev64 = n_prev[i] as f64;
        let mu_prev64 = mu_prev[i] as f64;
        let sigma_prev64 = sigma_prev[i] as f64;
        let nn = n_prev64 + inc_n;
        new_n[i] = nn as f32;
        let nm = if nn == 0.0 {
            0.0
        } else {
            (n_prev64 * mu_prev64 + inc_mu * inc_n) / nn
        };
        new_mu[i] = nm as f32;
        new_sigma[i] = if nn == 0.0 {
            0.0
        } else {
            ((n_prev64 * sigma_prev64 * sigma_prev64
                + inc_n * inc_sigma * inc_sigma
                + n_prev64 * (mu_prev64 - nm) * (mu_prev64 - nm)
                + inc_n * (inc_mu - nm) * (inc_mu - nm))
                / nn)
                .sqrt() as f32
        };
    }
    (new_n, new_mu, new_sigma)
}

/// Cumulative per-patch running stats over the patch axis (official
/// `get_running_stats`). Values `(b*v, n*p)`, mask `(b*v, n*p)`.
/// Returns (running_n, running_mu, running_sigma) each `(b*v, n)` flat.
pub fn get_running_stats(
    values: &[f32],
    masks: &[bool],
    bv: usize,
    n: usize,
    p: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Output layout: (bv, n) flat, i.e. index bv*n_i + bv_i — the same layout
    // callers use (``revin_mean[bvi * n_patches + ni]``). Note the historical
    // bug this fixes: it used to emit (n, bv), which is identical only when
    // bv == 1 and silently corrupted multivariate runs.
    let mut all_n = vec![0.0f32; bv * n];
    let mut all_mu = vec![0.0f32; bv * n];
    let mut all_sigma = vec![0.0f32; bv * n];
    let mut cur_n = vec![0.0f32; bv];
    let mut cur_mu = vec![0.0f32; bv];
    let mut cur_sigma = vec![0.0f32; bv];

    for i in 0..n {
        // Slice `(b*v, p)` at patch i from the `(b*v, n*p)` layout.
        let mut x = vec![0.0f32; bv * p];
        let mut m = vec![false; bv * p];
        for bi in 0..bv {
            let row = bi * n * p + i * p;
            x[bi * p..(bi + 1) * p].copy_from_slice(&values[row..row + p]);
            m[bi * p..(bi + 1) * p].copy_from_slice(&masks[row..row + p]);
        }
        let (nn, nm, ns) = update_running_stats(&cur_n, &cur_mu, &cur_sigma, &x, &m, p);
        cur_n = nn;
        cur_mu = nm;
        cur_sigma = ns;
        for bi in 0..bv {
            all_n[bi * n + i] = cur_n[bi];
            all_mu[bi * n + i] = cur_mu[bi];
            all_sigma[bi * n + i] = cur_sigma[bi];
        }
    }
    (all_n, all_mu, all_sigma)
}

/// RevIN normalize/denormalize with broadcasting of per-row stats.
/// `x` `(rows, d)`; `mu`, `sigma` `(rows,)` (or length 1 for broadcast).
/// Matches official `revin`.
pub fn revin(x: &mut Tensor2D, mu: &[f32], sigma: &[f32], reverse: bool) {
    let rows = x.rows();
    assert!(mu.len() == rows || mu.len() == 1);
    let mut safe = vec![0.0f32; sigma.len()];
    make_safe_for_division(sigma, &mut safe);
    for r in 0..rows {
        let m = if mu.len() == 1 { mu[0] } else { mu[r] };
        let s = if sigma.len() == 1 { safe[0] } else { safe[r] };
        let cols = x.cols();
        for v in x.data_mut()[r * cols..(r + 1) * cols].iter_mut() {
            if reverse {
                *v = *v * s + m;
            } else {
                *v = (*v - m) / s;
            }
        }
    }
}

/// Creates future-covariate patches by rolling the patched input along the
/// patch axis (official `get_output_patch_via_roll`).
///
/// `x` `(b*v, n*p)` patched values; returns `(rolled: (b*v, n*rolls*p),
/// wrap_mask: Vec<bool> same shape)`.
pub fn get_output_patch_via_roll(
    x: &[f32],
    bv: usize,
    n: usize,
    p: usize,
    rolls: usize,
) -> (Vec<f32>, Vec<bool>) {
    // rolling_mat[bv][n][rolls+1][p]
    let mut rolling = vec![0.0f32; bv * n * (rolls + 1) * p];
    for bi in 0..bv {
        for ni in 0..n {
            let src = &x[(bi * n + ni) * p..(bi * n + ni + 1) * p];
            let dst = &mut rolling
                [((bi * n + ni) * (rolls + 1)) * p..((bi * n + ni) * (rolls + 1) + 1) * p];
            dst.copy_from_slice(src);
        }
    }
    // For r in 0..rolls: rolling[..][r+1] = shift(rolling[..][r], -1 along n)
    let mut next = vec![0.0f32; bv * n * p];
    for r in 0..rolls {
        for bi in 0..bv {
            for ni in 0..n {
                let src_ni = (ni + 1) % n;
                let src_base = ((bi * n + src_ni) * (rolls + 1) + r) * p;
                let dst_base = (bi * n + ni) * p;
                next[dst_base..dst_base + p].copy_from_slice(&rolling[src_base..src_base + p]);
            }
        }
        for bi in 0..bv {
            for ni in 0..n {
                let dst = ((bi * n + ni) * (rolls + 1) + r + 1) * p;
                rolling[dst..dst + p]
                    .copy_from_slice(&next[(bi * n + ni) * p..(bi * n + ni + 1) * p]);
            }
        }
    }
    // result[bv][n][rolls*p]: take roll axis 1..=rolls
    let mut result = vec![0.0f32; bv * n * rolls * p];
    let mut wrap = vec![false; bv * n * rolls * p];
    for bi in 0..bv {
        for ni in 0..n {
            for r in 0..rolls {
                let src_base = ((bi * n + ni) * (rolls + 1) + r + 1) * p;
                let dst_base = (bi * n + ni) * rolls * p + r * p;
                result[dst_base..dst_base + p].copy_from_slice(&rolling[src_base..src_base + p]);
                // wrap mask: source_patch = ni + 1 + point//p >= n  → beyond the last patch
                for j in 0..rolls * p {
                    let sp = ni + 1 + j / p;
                    wrap[(bi * n + ni) * rolls * p + j] = sp >= n;
                }
            }
        }
    }
    (result, wrap)
}

/// Boolean variant of `get_output_patch_via_roll` for mask rolling.
/// `x` `(b*v, n*p)` bool; returns `(b*v, n*rolls*p)` bool + wrap mask.
pub fn get_output_patch_via_roll_bool(
    x: &[bool],
    bv: usize,
    n: usize,
    p: usize,
    rolls: usize,
) -> (Vec<bool>, Vec<bool>) {
    let mut rolling = vec![false; bv * n * (rolls + 1) * p];
    for bi in 0..bv {
        for ni in 0..n {
            let src = &x[(bi * n + ni) * p..(bi * n + ni + 1) * p];
            let dst = &mut rolling
                [((bi * n + ni) * (rolls + 1)) * p..((bi * n + ni) * (rolls + 1) + 1) * p];
            dst.copy_from_slice(src);
        }
    }
    let mut next = vec![false; bv * n * p];
    for r in 0..rolls {
        for bi in 0..bv {
            for ni in 0..n {
                let src_ni = (ni + 1) % n;
                let src_base = ((bi * n + src_ni) * (rolls + 1) + r) * p;
                let dst_base = (bi * n + ni) * p;
                next[dst_base..dst_base + p].copy_from_slice(&rolling[src_base..src_base + p]);
            }
        }
        for bi in 0..bv {
            for ni in 0..n {
                let dst = ((bi * n + ni) * (rolls + 1) + r + 1) * p;
                rolling[dst..dst + p]
                    .copy_from_slice(&next[(bi * n + ni) * p..(bi * n + ni + 1) * p]);
            }
        }
    }
    let mut result = vec![false; bv * n * rolls * p];
    let mut wrap = vec![false; bv * n * rolls * p];
    for bi in 0..bv {
        for ni in 0..n {
            for r in 0..rolls {
                let src_base = ((bi * n + ni) * (rolls + 1) + r + 1) * p;
                let dst_base = (bi * n + ni) * rolls * p + r * p;
                result[dst_base..dst_base + p].copy_from_slice(&rolling[src_base..src_base + p]);
                for j in 0..rolls * p {
                    let sp = ni + 1 + j / p;
                    wrap[(bi * n + ni) * rolls * p + j] = sp >= n;
                }
            }
        }
    }
    (result, wrap)
}

/// Stitches overlapping patch predictions (official `stitch_patches`).
///
/// `patch_preds` `(b*v, num_patches, total_len*q)` (flattened over patch
/// length × quantiles); `patch_len` is the context patch length, and
/// `total_len - patch_len` is the overlap. Returns `(b*v, (num_patches-1)*patch_len + total_len, q)`.
pub fn stitch_patches(
    patch_preds: &[f32],
    bv: usize,
    num_patches: usize,
    total_len: usize,
    patch_len: usize,
    num_quantiles: usize,
) -> Vec<f32> {
    let overlap = total_len - patch_len;
    let out_len = num_patches * patch_len + overlap;
    let mut out = vec![0.0f32; bv * out_len * num_quantiles];

    for bi in 0..bv {
        let slice = |row: usize| -> &[f32] {
            &patch_preds[(bi * num_patches + row) * (total_len * num_quantiles)
                ..(bi * num_patches + row + 1) * (total_len * num_quantiles)]
        };
        let mut o =
            |row: usize, col: usize| -> usize { (bi * out_len + row) * num_quantiles + col };

        if num_patches == 1 {
            let src = slice(0);
            out[(bi * out_len) * num_quantiles..(bi * out_len + total_len) * num_quantiles]
                .copy_from_slice(src);
            continue;
        }

        // first_chunk = preds[0][:patch_len]
        copy_q(&mut out, &mut o, 0, slice(0), patch_len, num_quantiles);

        for k in 0..num_patches - 1 {
            let prev = slice(k);
            let next = slice(k + 1);
            // Output block for transition k starts at row (k+1)*patch_len.
            let row0 = (k + 1) * patch_len;
            // stitched overlap: w * prev[overlap:] + (1-w) * next[:overlap]
            // w = linspace(1,0,overlap) = 1 - j/(overlap-1) for overlap>1.
            let w_denom = (overlap - 1).max(1) as f64;
            for j in 0..overlap {
                let w = 1.0 - j as f64 / w_denom;
                for q in 0..num_quantiles {
                    let pv = prev[(patch_len + j) * num_quantiles + q];
                    let nx = next[j * num_quantiles + q];
                    out[o(row0 + j, q)] = (w * pv as f64 + (1.0 - w) * nx as f64) as f32;
                }
            }
            // middle = next[overlap:patch_len]
            for j in overlap..patch_len {
                for q in 0..num_quantiles {
                    out[o(row0 + j, q)] = next[j * num_quantiles + q];
                }
            }
        }
        // tail = preds[-1][patch_len:], placed after the first block + transitions
        let last = slice(num_patches - 1);
        let t0 = num_patches * patch_len;
        for j in patch_len..total_len {
            for q in 0..num_quantiles {
                out[o(t0 + j - patch_len, q)] = last[j * num_quantiles + q];
            }
        }
    }
    out
}

fn copy_q(
    out: &mut [f32],
    o: &mut impl FnMut(usize, usize) -> usize,
    row: usize,
    src: &[f32],
    ncols: usize,
    num_quantiles: usize,
) {
    for c in 0..ncols {
        for q in 0..num_quantiles {
            let dst = o(row + c, q);
            out[dst] = src[c * num_quantiles + q];
        }
    }
}

/// Iterative RevIN refinement for CPM-masked patches (official
/// `cpm_iterative_revin_refine` in cpm_revin_refine.py).
///
/// For each patch `i`, the running stats may be replaced at CPM positions by
/// stats that additionally incorporate the model's *estimated* values of all
/// preceding CPM patches. The estimate comes from the median quantile of the
/// raw logits, reverse-RevIN'd with the (possibly refined) stats, then
/// clamped to ±value_clip.
///
/// Inputs (all flat, `bv = b*v` rows): `raw_logits` `(bv, n, o*q)`,
/// `revin_n`/`revin_mu`/`revin_sigma` `(bv, n)`, `patch_cpm_mask` `(b, n)`.
///
/// Returns `(refined_mu, refined_sigma)`, each `(bv, n)`; non-CPM positions
/// keep the original values.
#[allow(clippy::too_many_arguments)]
pub fn cpm_iterative_revin_refine(
    raw_logits: &Tensor2D,
    revin_n: &[f32],
    revin_mu: &[f32],
    revin_sigma: &[f32],
    patch_cpm_mask: &[bool],
    b: usize,
    v: usize,
    n: usize,
    p: usize,
    rolls: usize,
    num_quantiles: usize,
    median_q_idx: usize,
    value_clip: f32,
) -> (Vec<f32>, Vec<f32>) {
    let bv = b * v;
    let oq = rolls * p * num_quantiles;
    debug_assert_eq!(raw_logits.rows(), bv * n);
    debug_assert_eq!(raw_logits.cols(), oq);

    let rl = raw_logits.data();

    // Extract median-quantile logits per (b, v, n, rolls, p).
    // index(b, v, ni, r, j) -> (bv*ni + r*p + j)*oq + (r*p + j)*q + median_q
    let mut median_logits = vec![0.0f32; bv * n * rolls * p];
    for ni in 0..n {
        for bvi in 0..bv {
            let base = (bvi * n + ni) * oq;
            for r in 0..rolls {
                for j in 0..p {
                    let idx = ((bvi * n + ni) * rolls + r) * p + j;
                    median_logits[idx] = rl[base + (r * p + j) * num_quantiles + median_q_idx];
                }
            }
        }
    }

    // Carry state.
    let mut carry_n = vec![0.0f32; bv];
    let mut carry_mu = vec![0.0f32; bv];
    let mut carry_sigma = vec![0.0f32; bv];
    // anchor_predicted_values: (b, v, rolls, p)
    let mut anchor = vec![0.0f32; b * v * rolls * p];
    let mut block_offset = vec![0usize; b];
    let mut refined_mu = vec![0.0f32; bv * n];
    let mut refined_sigma = vec![0.0f32; bv * n];

    // step_masks = all-false (predicted values count as valid).
    let step_masks = vec![false; bv * p];

    for ni in 0..n {
        let is_cpm: Vec<bool> = (0..b).map(|bi| patch_cpm_mask[bi * n + ni]).collect();

        // predicted_values_step[bvi][j] = anchor[bvi][block_offset[b]][j]
        let mut predicted = vec![0.0f32; bv * p];
        for (bi, &off) in block_offset.iter().enumerate() {
            for vi in 0..v {
                let bvi = bi * v + vi;
                let a_base = ((bi * v + vi) * rolls) * p + off * p;
                predicted[bvi * p..(bvi + 1) * p].copy_from_slice(&anchor[a_base..a_base + p]);
            }
        }

        // update running stats with the estimated patch (mask = all valid)
        let (new_n, new_mu, new_sigma) = update_running_stats(
            &carry_n,
            &carry_mu,
            &carry_sigma,
            &predicted,
            &step_masks,
            p,
        );

        // out_* = cpm ? new : actual
        let mut out_n = vec![0.0f32; bv];
        let mut out_mu = vec![0.0f32; bv];
        let mut out_sigma = vec![0.0f32; bv];
        for (bi, &cpm) in is_cpm.iter().enumerate() {
            for vi in 0..v {
                let bvi = bi * v + vi;
                out_n[bvi] = if cpm {
                    new_n[bvi]
                } else {
                    revin_n[bvi * n + ni]
                };
                out_mu[bvi] = if cpm {
                    new_mu[bvi]
                } else {
                    revin_mu[bvi * n + ni]
                };
                out_sigma[bvi] = if cpm {
                    new_sigma[bvi]
                } else {
                    revin_sigma[bvi * n + ni]
                };
            }
        }

        // new_block_offset[b] = cpm ? (offset+1)%rolls : 0
        let mut new_block_offset = vec![0usize; b];
        for (bi, &off) in block_offset.iter().enumerate() {
            new_block_offset[bi] = if is_cpm[bi] { (off + 1) % rolls } else { 0 };
        }

        // step_predicted = revin(current median logits, out_mu, out_sigma, reverse)
        //   = median_logits * sigma + mu ; clamp to ±value_clip
        // current_step_logits: (b, v, rolls, p)
        let mut step_pred = vec![0.0f32; bv * rolls * p];
        for bvi in 0..bv {
            let mu = out_mu[bvi] as f64;
            let sigma = out_sigma[bvi] as f64;
            let safe = if sigma < 1e-6 { 1.0 } else { sigma };
            let base = (bvi * n + ni) * rolls * p;
            for k in 0..rolls * p {
                let val = median_logits[base + k] as f64 * safe + mu;
                step_pred[bvi * rolls * p + k] = (val as f32).clamp(-value_clip, value_clip);
            }
        }

        // new anchor = (new_block_offset == 0) ? step_pred : anchor
        for (bi, &off) in new_block_offset.iter().enumerate() {
            if off == 0 {
                for vi in 0..v {
                    let bvi = bi * v + vi;
                    anchor[(bvi) * rolls * p..(bvi + 1) * rolls * p]
                        .copy_from_slice(&step_pred[bvi * rolls * p..(bvi + 1) * rolls * p]);
                }
            }
        }

        for bvi in 0..bv {
            refined_mu[bvi * n + ni] = out_mu[bvi];
            refined_sigma[bvi * n + ni] = out_sigma[bvi];
        }

        carry_n = out_n;
        carry_mu = out_mu;
        carry_sigma = out_sigma;
        block_offset = new_block_offset;
    }

    (refined_mu, refined_sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_stats_single_patch() {
        // one row, p=4: [1,2,3,4], no mask
        let (n, mu, sigma) = update_running_stats(
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[false; 4],
            4,
        );
        assert_eq!(n, vec![4.0]);
        assert!((mu[0] - 2.5).abs() < 1e-6);
        assert!((sigma[0] - 1.1180339).abs() < 1e-5);
    }

    #[test]
    fn roll_matches_manual() {
        // bv=1, n=3, p=2, rolls=2
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (rolled, wrap) = get_output_patch_via_roll(&x, 1, 3, 2, 2);
        // rolling_mat[r+1] = roll(rolling_mat[r], -1 along n):
        //   roll0 = [x1, x2, x0] = [3,4, 5,6, 1,2]
        //   roll1 = [x2, x0, x1] = [5,6, 1,2, 3,4]
        // result[ni] = roll0[ni] ⊕ roll1[ni]
        //   result[0] = [3,4,5,6]; result[1] = [5,6,1,2]; result[2] = [1,2,3,4]
        assert_eq!(
            &rolled[0..12],
            &[3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0]
        );
        // wrap: source_patch = ni + 1 + point//p >= n
        //   ni=0 -> 1 + [0,0,1,1] = [1,1,2,2] all < 3
        //   ni=1 -> 2 + ... = [2,2,3,3] -> [F,F,T,T]
        //   ni=2 -> 3 + ... = [3,3,4,4] all >= 3
        assert_eq!(
            wrap,
            vec![
                false, false, false, false, false, false, true, true, true, true, true, true
            ]
        );
    }

    #[test]
    fn stitch_two_patches() {
        // patch_len=2, total_len=3 (overlap 1), q=1, two patches
        // preds[0] = [a0, m0, o0], preds[1] = [o1, m1, t1]
        // out = [a0, m0, w*o0 + (1-w)*o1, m1, t1], w=1 at j=0 → o0
        let preds = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = stitch_patches(&preds, 1, 2, 3, 2, 1);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 5.0, 6.0]);
    }
}
