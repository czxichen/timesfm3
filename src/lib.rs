//! timesfm3 — dedicated Rust inference engine for Google TimesFM 3.0.
//!
//! No tch-rs, no candle, no python: a small, dependency-light engine where
//! the hot path (GEMM) is pure Rust + rayon, and everything else is thin
//! structured code mirroring the official PyTorch inference graph.
//!
//! Phase status:
//!   Phase 1 (in progress):  tensor/ops foundation, safetensors loading,
//!                           ResidualBlock + golden tests.
//!   Phase 2 (planned):      MixingTransformer (seq+variate attention, RoPE,
//!                           RMSNorm, PerDimScale, FFN).
//!   Phase 3 (planned):      decode/stitch/detrend/CPM-RevIN pipeline.
//!   Phase 4 (planned):      multivariate + covariates + symmetric averaging.
//!   Phase 5 (planned):      GEMM SIMD microkernel + threading tuning.

pub mod checkpoint;
pub mod config;
pub mod error;
pub mod forecast;
pub mod json;
pub mod model;
pub mod ops;
pub mod pipeline;
pub mod tensor;

static INIT_THREAD_POOL: std::sync::Once = std::sync::Once::new();

/// Initializes the global Rayon thread pool to the optimal physical core count
/// unless explicitly overridden by `TIMESFM_NUM_THREADS` or `RAYON_NUM_THREADS`.
pub fn init_default_thread_pool() {
    INIT_THREAD_POOL.call_once(|| {
        if let Ok(s) = std::env::var("TIMESFM_NUM_THREADS")
            && let Ok(n) = s.parse::<usize>()
            && n > 0
        {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global();
            return;
        }
        if std::env::var("RAYON_NUM_THREADS").is_ok() {
            return;
        }
        // Avoid hyperthreading overhead on AVX2/FMA execution units
        let physical = num_cpus::get_physical().max(1);
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(physical)
            .build_global();
    });
}
