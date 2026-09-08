# timesfm3

**简体中文** · [English](README.en.md)

<p align="center">
<img alt="Pure Rust" src="https://img.shields.io/badge/Rust-纯Rust引擎-orange">
<img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue">
<img alt="Runtime deps" src="https://img.shields.io/badge/运行期依赖-零-green">
<img alt="Binary" src="https://img.shields.io/badge/Release_Binary-~1.1MB-purple">
</p>

专为 Google **TimesFM 3.0** 时间序列大基座模型打造的高性能**纯 Rust 推理引擎**。

**无 `tch-rs` / `candle` / Python 运行时依赖**;热路径基于 Rayon 多线程与 SIMD 微内核(x86_64 AVX2+FMA+F16C、aarch64 NEON),Release 静态二进制仅约 **1.1 MB**,冷启动即时就绪。

---

## 目录

- [简介](#简介)
- [模型下载 (Model Zoo)](#模型下载-model-zoo)
- [精度:与官方 PyTorch 对齐](#精度与官方-pytorch-对齐)
- [速度:与官方 PyTorch 对比](#速度与官方-pytorch-对比)
- [许可](#许可)

---

## 简介

- **目标模型**:Google TimesFM 3.0,约 3.3 亿参数,20 层双注意力(序列 + 变量)MixingTransformer,`model_dims=1280`、16 头 × 80 维;官方 FP32 safetensors 权重约 1.32 GB。
- **设计定位**:
  1. **零重型依赖** — 摆脱 `libtorch` / CUDA 动态库与 Python 运行时,直面嵌入式设备、轻量容器与 Serverless 的冷启动与部署痛点;全部外部依赖仅 `rayon`、`memmap2`、`matrixmultiply`、`num_cpus`、`half` 五个常规 crate。
  2. **极致轻量** — 纯推理导向,去除反向传播与梯度存储,热路径内存预分配;Release 静态链接体积约 **1.1 MB**。
  3. **严格对齐官方精度** — 浮点精度对齐至 $10^{-5}$ 级别(见下文精度基准)。
- **核心特性**:
  - **独立算子实现**:CPU Tiled Online Softmax(FlashAttention 思想,因果区算力减半)、N 轴列切分 GEMM、RMSNorm / PerDimScale / RoPE 静态预计算表;
  - **原生 SIMD + 多线程**:x86_64 AVX2/F16C 与 aarch64 NEON 双内核,Rayon 并行任务化(batch×head 展平,单序列也能跑满多核);
  - **零拷贝加载**:基于 `memmap2` 的 safetensors 内存映射解析 + 内置约 150 行递归下降 JSON 解析器;FP16 权重在 half 域直接完成转置布局,半精度检查点**秒级装配**;
  - **高保真量化三档**:FP32 原生(1.32 GB)· FP16 全量(631 MB)· FP16 混合 Balanced(730 MB,敏感层保留 FP32,worst-case 误差约为全量 FP16 的 1/5,**体积缩减 45% 同时精度提升约 5 倍,推荐首选**);
  - **时序稳健算法**:PAVA 保序回归(分位数单调防穿越)、季节分解与相位回填、Robust Median/IQR 抗脉冲归一化、CPM 迭代校正与边界平滑;
  - **跨平台一致性**:全量 155 场景矩阵在 macOS 与 Windows 10(x86_64)上分别复测,TOL / FAIL 名单完全一致,官方实现中的 NaN 三元组在引擎内全部为有限值。

---

## 模型下载 (Model Zoo)

量化权重已发布到 **Hugging Face** 与 **ModelScope**(均为 TimesFM 官方权重的非商用量化派生,遵循 `timesfm-non-commercial-license-v1.0`):

| 档位 | 体积 | 最坏误差(全分位网格 Δ,ETTh1) | Hugging Face | ModelScope |
| :--- | :---: | :---: | :--- | :--- |
| **FP16 Balanced(推荐)** | 730 MB | $1.3 \times 10^{-3}$(0.007%) | [timesfm3.0-balanced](https://huggingface.co/czxichen/timesfm3.0-balanced) | [timesfm3.0-balanced](https://modelscope.cn/models/czxichen/timesfm3.0-balanced) |
| **FP16 全量** | 631 MB | $6.1 \times 10^{-3}$(0.03%) | [timesfm3.0-f16](https://huggingface.co/czxichen/timesfm3.0-f16) | [timesfm3.0-f16](https://modelscope.cn/models/czxichen/timesfm3.0-f16) |

官方 **FP32 原权重**请从 Google 官方渠道获取([google-research/timesfm](https://github.com/google-research/timesfm),非商用许可)。下载任一站点的权重并解压后即可直接预测:

```bash
cargo run --release --bin forecast -- <ckpt_dir> <horizon> [context.csv]
# 例:cargo run --release --bin forecast -- ckpt_balanced 96 data/etth1.csv
```

---

## 精度:与官方 PyTorch 对齐

场景:经典电力变压器数据集 **ETTh1,7 变量 × 输入长度 1024 → 预测 24 步**,与官方 PyTorch 实现逐值位级对比;Windows 10 复测(i7-10700,torch 2.13,同场景)见末列。

| 精度模式 | 中位数点预测 $\max\|\Delta\|$ | 全分位网格 $\max\|\Delta\|$ | Pearson 相关系数 | Windows 复测 $\max\|\Delta\|$ | 说明 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FP32 原生** | **$7.6 \times 10^{-6}$** | **$1.0 \times 10^{-5}$** | **1.0000000** | 1.8e-05 / 2.7e-05 | 纯浮点顺序累加噪声,完全对齐 |
| **FP16 全量量化** | $3.7 \times 10^{-3}$ | $6.1 \times 10^{-3}$ | **1.0000000** | 3.7e-03 / 6.1e-03 | 相对误差约 0.03%,工程无损 |
| **FP16 混合 (Balanced)** | **$7.6 \times 10^{-4}$** | **$1.3 \times 10^{-3}$** | **1.0000000** | 7.6e-04 / 1.4e-03 | 相对误差仅 0.007%,**推荐首选** |

> 155 个时序场景矩阵详尽报告见 [`report/accuracy_report.md`](report/accuracy_report.md)。
> Windows 复测:155 场景 TOL=133、FAIL 名单与 macOS 完全一致;量化档(vs 引擎自身 f32):FP16 全量 worst $9.6 \times 10^{-3}$、Balanced worst $5.9 \times 10^{-3}$。

---

## 速度:与官方 PyTorch 对比

在以下两种硬件环境分别实测(均为各自环境的最新端到端状态,与官方实现同机对照、同 checkpoint、同输入):

### 环境 A · Intel Core i7-8750H(6 核 12 线程)vs 官方 PyTorch 2.2

端到端冷态墙钟(加载 + 模型装配 + 预测全周期):

| 实现方案 | 整体耗时 | 加速比 | 峰值内存驻留 |
| :--- | :---: | :---: | :---: |
| **官方 PyTorch 2.2** | 5.16 s | 1.00x | ~2.69 GB |
| **本引擎 FP32** | 1.96 s | **2.63x** | ~1.85 GB |
| **本引擎 FP16 混合 (Balanced)** | 1.70 s | **3.04x** | ~1.48 GB |
| **本引擎 FP16 全量** | **1.65 s** | **3.13x** | **~1.37 GB(-49%)** |

### 环境 B · Windows 10 · Intel Core i7-10700(8 核 16 线程)vs 官方 PyTorch 2.13

7 变量 × ctx1024、horizon=96,`forecast` 内嵌计时中位数:

| 实现方案 | 端到端墙钟 | 模型装配 | 前向预测 |
| :--- | :---: | :---: | :---: |
| **本引擎 FP32** | 3.27 s | 948 ms | 1.51 s |
| **本引擎 FP16 全量** | **2.42 s** | **446 ms(-53%)** | 1.72 s |
| **本引擎 FP16 混合 (Balanced)** | 2.56 s | 552 ms(-42%) | 1.77 s |

### 关键结论

- **装配速度**:得益于内存映射零拷贝 + 并发转置 + half 域直接布局,半精度检查点装配仅约 **0.27 s**,端到端由装配加速主导。
- **全量吞吐**:155 场景矩阵(各加载一次)——官方 torch 2.13 约 352 s,本引擎 FP32 **277.1 s(1.27×)**、FP16 300.0 s、Balanced 303.4 s。
- **注**:FP16 前向较 FP32 慢约 8%(half 解包开销),与 macOS 侧“+15%”结论同向,不抵消装配端优势。

---

## 许可

- **引擎源码**:Apache-2.0。
- **模型权重**:TimesFM 3.0 官方预训练权重遵循 `timesfm-non-commercial-license-v1.0`(仅限非商用研究)。
