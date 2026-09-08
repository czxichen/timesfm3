# timesfm3

[简体中文](README.md) · **English**

<p align="center">
<img alt="Pure Rust" src="https://img.shields.io/badge/Rust-proprietary_engine-orange">
<img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue">
<img alt="Runtime deps" src="https://img.shields.io/badge/runtime_deps-zero-green">
<img alt="Binary" src="https://img.shields.io/badge/Release_Binary-~1.1MB-purple">
</p>

A high-performance **proprietary pure-Rust inference engine** purpose-built for Google **TimesFM 3.0**, the time-series foundation model.

Every time-series attention and math operator is implemented from scratch — **no `tch-rs`, no `candle`, no Python runtime**. Hot paths run on Rayon multithreading plus hand-written SIMD microkernels (x86_64 AVX2+FMA+F16C and aarch64 NEON). The statically linked Release binary is only ~**1.1 MB** and starts instantly.

---

## Table of Contents

- [Introduction](#introduction)
- [Model Zoo](#model-zoo)
- [Accuracy: bit-aligned with the official PyTorch](#accuracy-bit-aligned-with-the-official-pytorch)
- [Speed: vs. the official PyTorch implementation](#speed-vs-the-official-pytorch-implementation)
- [License](#license)

---

## Introduction

- **Target model**: Google TimesFM 3.0 — ~330M parameters, 20 dual-attention (sequence + variate) MixingTransformer layers, `model_dims=1280`, 16 heads × 80. Official FP32 safetensors weights are ~1.32 GB.
- **Design goals**:
  1. **Zero heavy dependencies** — no `libtorch`/CUDA shared libraries and no Python runtime, removing cold-start and deployment pain in embedded devices, slim containers, and Serverless; the full external dependency set is five conventional crates: `rayon`, `memmap2`, `matrixmultiply`, `num_cpus`, `half`.
  2. **Extremely lightweight** — inference-only: no backprop or gradient storage, preallocated hot-path memory, ~**1.1 MB** statically linked Release binary.
  3. **Strict official-accuracy alignment** — floating-point agreement down to $10^{-5}$ (see the accuracy benchmark below).
- **Highlights**:
  - **Self-written operators, no black boxes**: CPU tiled online softmax (FlashAttention-style; causal FLOPs halved), N-axis column-partitioned GEMM, hand-written RMSNorm / PerDimScale / static RoPE lookup tables;
  - **Native SIMD + multithreading**: x86_64 AVX2/F16C and aarch64 NEON kernels; Rayon-parallelized over the flattened batch×head axis, so even a single sequence saturates all cores;
  - **Zero-copy checkpoint loading**: `memmap2`-based safetensors parsing plus a built-in ~150-line recursive-descent JSON parser; FP16 weights are transposed directly in half domain, so half-precision checkpoints assemble in **~0.27 s**;
  - **Three high-fidelity precision tiers**: FP32 native (1.32 GB) · full FP16 (631 MB) · **FP16 Balanced** (730 MB, keeps FP32 on sensitive layers — worst-case error ~1/5 of full FP16, i.e. **≈5× higher precision for a 45% size cut; recommended**);
  - **Time-series-robust algorithms**: PAVA isotonic regression (prevents crossed quantiles), seasonal decomposition with phase-compenated re-fill, Robust Median/IQR normalization against spike outliers, CPM iterative correction, boundary smoothing;
  - **Cross-platform consistency**: the full 155-scenario matrix was re-run on macOS and Windows 10 (x86_64) with identical TOL/FAIL lists; every NaN triple produced by the official implementation is finite in this engine.

---

## Model Zoo

Quantized weights are published on **Hugging Face** and **ModelScope** (non-commercial derivatives of the official TimesFM weights, `timesfm-non-commercial-license-v1.0`):

| Tier | Size | Worst error (full grid Δ, ETTh1) | Hugging Face | ModelScope |
| :--- | :---: | :---: | :--- | :--- |
| **FP16 Balanced (recommended)** | 730 MB | $1.3 \times 10^{-3}$ (0.007%) | [timesfm3.0-balanced](https://huggingface.co/czxichen/timesfm3.0-balanced) | [timesfm3.0-balanced](https://modelscope.cn/models/czxichen/timesfm3.0-balanced) |
| **Full FP16** | 631 MB | $6.1 \times 10^{-3}$ (0.03%) | [timesfm3.0-f16](https://huggingface.co/czxichen/timesfm3.0-f16) | [timesfm3.0-f16](https://modelscope.cn/models/czxichen/timesfm3.0-f16) |

Get the official **FP32** weights from Google ([google-research/timesfm](https://github.com/google-research/timesfm), non-commercial license). After downloading and unzipping a tier you can forecast directly:

```bash
cargo run --release --bin forecast -- <ckpt_dir> <horizon> [context.csv]
# e.g. cargo run --release --bin forecast -- ckpt_balanced 96 data/etth1.csv
```

---

## Accuracy: bit-aligned with the official PyTorch

Scenario: **ETTh1, 7 variates × context 1024 → horizon 24**, compared value-by-value against the official PyTorch implementation; the Windows 10 re-run (i7-10700, torch 2.13, same scenario) is shown in the last column.

| Precision mode | Point-forecast median $\max\|\Delta\|$ | Full quantile grid $\max\|\Delta\|$ | Pearson r | Windows re-run $\max\|\Delta\|$ | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FP32 native** | **$7.6 \times 10^{-6}$** | **$1.0 \times 10^{-5}$** | **1.0000000** | 1.8e-05 / 2.7e-05 | Pure float summation-order noise; fully aligned |
| **Full FP16** | $3.7 \times 10^{-3}$ | $6.1 \times 10^{-3}$ | **1.0000000** | 3.7e-03 / 6.1e-03 | ~0.03% relative error; lossless in practice |
| **FP16 Balanced** | **$7.6 \times 10^{-4}$** | **$1.3 \times 10^{-3}$** | **1.0000000** | 7.6e-04 / 1.4e-03 | Only 0.007% relative error — **recommended** |

> Detailed 155-scenario matrix: [`report/accuracy_report.md`](report/accuracy_report.md).
> Windows re-run: TOL=133 and the FAIL list match macOS exactly; quantized tiers (vs. this engine's own f32): full FP16 worst $9.6 \times 10^{-3}$, Balanced worst $5.9 \times 10^{-3}$.

---

## Speed: vs. the official PyTorch implementation

Measured on two hardware environments (each compared against the official implementation on the same machine, same checkpoint, same inputs):

### Environment A · Intel Core i7-8750H (6C/12T) vs. official PyTorch 2.2

End-to-end cold wall-clock (load + model assembly + full prediction):

| Implementation | Wall clock | Speedup | Peak RSS |
| :--- | :---: | :---: | :---: |
| **Official PyTorch 2.2** | 5.16 s | 1.00x | ~2.69 GB |
| **This engine, FP32** | 1.96 s | **2.63x** | ~1.85 GB |
| **This engine, FP16 Balanced** | 1.70 s | **3.04x** | ~1.48 GB |
| **This engine, full FP16** | **1.65 s** | **3.13x** | **~1.37 GB(-49%)** |

### Environment B · Windows 10 · Intel Core i7-10700 (8C/16T) vs. official PyTorch 2.13

7 variates × ctx1024, horizon=96, `forecast` built-in timing medians:

| Implementation | End-to-end | Model assembly | Forward pass |
| :--- | :---: | :---: | :---: |
| **This engine, FP32** | 3.27 s | 948 ms | 1.51 s |
| **This engine, full FP16** | **2.42 s** | **446 ms(-53%)** | 1.72 s |
| **This engine, FP16 Balanced** | 2.56 s | 552 ms(-42%) | 1.77 s |

### Key takeaways

- **Assembly speed**: thanks to mmap zero-copy + concurrent transposition + direct half-domain layout, half-precision checkpoints assemble in ~**0.27 s**; end-to-end time is assembly-dominated.
- **Full throughput**: the 155-scenario matrix (one load each) — official torch 2.13 ≈ 352 s, this engine FP32 **277.1 s (1.27×)**, FP16 300.0 s, Balanced 303.4 s.
- **Note**: FP16 forward runs ~8% slower than FP32 (half unpacking overhead), consistent with the “+15%” observation on macOS, and is outweighed by the assembly gain.

---

## License

- **Engine source code**: Apache-2.0.
- **Model weights**: official TimesFM 3.0 pretrained weights are released under `timesfm-non-commercial-license-v1.0` (non-commercial research only).
