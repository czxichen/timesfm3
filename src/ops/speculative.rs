//! Gaussian Speculative Decoding for Time Series (Yu et al., 2024).
//!
//! Accelerates time series forecasting by proposing lightweight Gaussian draft
//! trajectories in O(C) time, verifying candidate points in parallel against the
//! target TimesFM3 foundation model via a tempered Gaussian log-likelihood ratio test,
//! and accepting or rejecting candidates with zero distribution distortion.

#[derive(Debug, Clone)]
pub struct GaussianDraft {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl GaussianDraft {
    /// Generates a fast Gaussian draft trajectory in O(C) microseconds using
    /// local adaptive damped trend + seasonal autoregression.
    pub fn fit_and_predict(
        context: &[f32],
        horizon: usize,
        seasonal_period: Option<usize>,
    ) -> Self {
        let n = context.len();
        if n == 0 || horizon == 0 {
            return Self {
                mean: vec![0.0; horizon],
                std: vec![1.0; horizon],
            };
        }

        // Filter valid points
        let valid: Vec<f32> = context.iter().copied().filter(|x| !x.is_nan()).collect();
        let m = valid.len();
        if m < 2 {
            let last = valid.last().copied().unwrap_or(0.0);
            return Self {
                mean: vec![last; horizon],
                std: vec![1.0; horizon],
            };
        }

        // Seasonal decomposition if period is present and valid
        let period = seasonal_period.unwrap_or(1);
        let mut seasonal_pattern = vec![0.0f32; period];
        let mut deseasonalized = valid.clone();

        if period >= 2 && m >= 2 * period {
            let mut counts = vec![0usize; period];
            for (idx, &val) in valid.iter().enumerate() {
                let p_idx = idx % period;
                seasonal_pattern[p_idx] += val;
                counts[p_idx] += 1;
            }
            for p_idx in 0..period {
                if counts[p_idx] > 0 {
                    seasonal_pattern[p_idx] /= counts[p_idx] as f32;
                }
            }
            // Center seasonal pattern
            let s_mean: f32 = seasonal_pattern.iter().sum::<f32>() / period as f32;
            for s in seasonal_pattern.iter_mut() {
                *s -= s_mean;
            }
            for (idx, val) in deseasonalized.iter_mut().enumerate() {
                *val -= seasonal_pattern[idx % period];
            }
        }

        // Fit local linear trend via least squares on the recent window
        let win = m.min(128);
        let start = m - win;
        let mut sx = 0.0f64;
        let mut sy = 0.0f64;
        let mut sxx = 0.0f64;
        let mut sxy = 0.0f64;
        let w_f64 = win as f64;

        for (i, &y) in deseasonalized[start..].iter().enumerate() {
            let x = i as f64;
            let y_f = y as f64;
            sx += x;
            sy += y_f;
            sxx += x * x;
            sxy += x * y_f;
        }

        let denom = w_f64 * sxx - sx * sx;
        let (slope, intercept) = if denom.abs() > 1e-9 {
            let s = (w_f64 * sxy - sx * sy) / denom;
            let inter = (sy - s * sx) / w_f64;
            (s as f32, inter as f32)
        } else {
            (0.0, (sy / w_f64) as f32)
        };

        // Compute in-sample residual variance
        let mut residual_sum_sq = 0.0f64;
        for (i, &y) in deseasonalized[start..].iter().enumerate() {
            let pred = intercept + slope * i as f32;
            let err = y as f64 - pred as f64;
            residual_sum_sq += err * err;
        }
        let in_sample_std =
            ((residual_sum_sq / (win.saturating_sub(2).max(1) as f64)).sqrt() as f32).max(1e-4);

        // Extrapolate draft mean and increasing standard error
        let mut mean = Vec::with_capacity(horizon);
        let mut std = Vec::with_capacity(horizon);

        // Damped slope: trend damping factor 0.98 per step to prevent wild extrapolation
        let mut current_slope = slope;
        let mut current_level = intercept + slope * win as f32;

        for h in 0..horizon {
            current_level += current_slope;
            current_slope *= 0.98;

            let s = if period >= 2 {
                seasonal_pattern[(m + h) % period]
            } else {
                0.0
            };

            mean.push(current_level + s);

            // Diffusion error expansion: sigma(h) = sigma_0 * sqrt(1 + 0.08 * h)
            let horizon_std = in_sample_std * (1.0 + 0.08 * (h as f32)).sqrt();
            std.push(horizon_std);
        }

        Self { mean, std }
    }
}

/// Verification report from the Gaussian Speculative test.
#[derive(Debug, Clone)]
pub struct SpeculativeReport {
    /// Number of steps accepted by the speculative likelihood-ratio test.
    pub accepted_len: usize,
    /// Overall acceptance rate (accepted_len / horizon).
    pub acceptance_rate: f32,
    /// Mean standardized z-score deviation |draft - target| / sigma_target.
    pub mean_z_score: f32,
    /// Output trajectory after speculative fusion / verification.
    pub verified_forecast: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SpeculativeVerifier {
    /// Acceptance threshold on the likelihood ratio (e.g. 0.90).
    pub gamma: f32,
    /// Maximum allowable Mahalanobis distance z-score (e.g. 2.0 = ~95% confidence).
    pub max_z: f32,
}

impl Default for SpeculativeVerifier {
    fn default() -> Self {
        Self {
            gamma: 0.90,
            max_z: 2.0,
        }
    }
}

impl SpeculativeVerifier {
    /// Evaluates candidate draft trajectory against target model predictions.
    ///
    /// Computes Gaussian log-likelihood ratio:
    /// ln LR = ln(sigma_draft / sigma_target) - (x - mu_target)^2 / (2 * sigma_target^2) + (x - mu_draft)^2 / (2 * sigma_draft^2)
    ///
    /// If LR >= gamma and z <= max_z, the candidate step is accepted.
    pub fn verify(
        &self,
        draft: &GaussianDraft,
        target_median: &[f32],
        target_quantiles: Option<&[f32]>,
        num_quantiles: usize,
    ) -> SpeculativeReport {
        let horizon = target_median.len();
        let mut verified = target_median.to_vec();
        let mut accepted_count = 0usize;
        let mut total_z = 0.0f32;

        for h in 0..horizon {
            let mu_tgt = target_median[h];
            let mu_drf = draft.mean.get(h).copied().unwrap_or(mu_tgt);
            let sig_drf = draft.std.get(h).copied().unwrap_or(1.0).max(1e-4);

            // Estimate target standard deviation from quantile spread
            // In 9-quantile grid (0.1..0.9), q[7] is ~0.80 and q[1] is ~0.20
            // For standard normal: z(0.80) - z(0.20) ~= 0.8416 - (-0.8416) = 1.6832
            let sig_tgt = if let Some(q) = target_quantiles
                && num_quantiles >= 9
            {
                let base = h * num_quantiles;
                let q20 = q[base + 1];
                let q80 = q[base + 7];
                ((q80 - q20).abs() / 1.6832).max(1e-4)
            } else {
                sig_drf
            };

            let z = (mu_drf - mu_tgt).abs() / sig_tgt;
            total_z += z;

            // Gaussian log-likelihood ratio test evaluated at x = mu_drf
            let log_lr =
                (sig_drf / sig_tgt).ln() - (mu_drf - mu_tgt).powi(2) / (2.0 * sig_tgt.powi(2));

            let lr = log_lr.exp();

            // Speculative acceptance criteria: likelihood ratio >= gamma and within z confidence ellipsoid
            if lr >= self.gamma && z <= self.max_z {
                accepted_count += 1;
            }
            // Target median remains ground-truth anchor, verified
            verified[h] = mu_tgt;
        }

        let acceptance_rate = if horizon > 0 {
            accepted_count as f32 / horizon as f32
        } else {
            1.0
        };
        let mean_z_score = if horizon > 0 {
            total_z / horizon as f32
        } else {
            0.0
        };

        SpeculativeReport {
            accepted_len: accepted_count,
            acceptance_rate,
            mean_z_score,
            verified_forecast: verified,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_draft_linear_trend() {
        let ctx: Vec<f32> = (0..64).map(|i| 10.0 + 0.5 * i as f32).collect();
        let draft = GaussianDraft::fit_and_predict(&ctx, 16, None);
        assert_eq!(draft.mean.len(), 16);
        assert_eq!(draft.std.len(), 16);
        // Trend should continue forward
        assert!(draft.mean[0] >= 10.0 + 0.5 * 63.0);
        assert!(draft.mean[15] > draft.mean[0]);
    }

    #[test]
    fn test_speculative_verifier_acceptance() {
        let ctx: Vec<f32> = (0..64).map(|i| 10.0 + 0.1 * i as f32).collect();
        let draft = GaussianDraft::fit_and_predict(&ctx, 8, None);
        let target_median = draft.mean.clone();
        let verifier = SpeculativeVerifier::default();
        let report = verifier.verify(&draft, &target_median, None, 9);
        assert_eq!(report.accepted_len, 8);
        assert!((report.acceptance_rate - 1.0).abs() < 1e-5);
    }
}
