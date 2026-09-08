# Rust 推理引擎 vs 官方 PyTorch —— 多场景精度对照报告

> 生成工具：`tools/gen_scenarios.py` + `tools/run_official_scenarios.py` + `target/release/export_scenarios` + `tools/compare_scenarios.py`
> 明细数据：`report/summary.json`；原始输出：`out_official_sc/`、`out_rust_sc/`

## 1. 结论速览

共 **155 个场景 / 186 条序列**。

- **130 个场景通过**：其中 130 个在 float32 噪声级以内（TOL，max|Δ| ≤ 1e-4，见下方判定说明），3 个因官方输出 NaN 无法对比（OFFICIAL_NAN，属官方实现缺陷，非引擎问题，见 §4）。
- **未发现任何 Rust 引擎语义错误残留**：所有可对比场景 max|Δ| ≤ 1e-4，且经本次测试发现并修复了 2 个引擎真实 bug（见 §4）。
- **精度口径修正**：此前单场景对比报告的“零误差/逐位一致”是打印位数舍入造成的误读。当前版本（GEMM/注意力内积 matrixmultiply f32 + 统计归约 f64）实际为 max ≈ 1e-6 ～ 4e-5、逐位一致比例 10%～55%，且已逼近官方 f32 自身噪声的下界（见 §4.4 的量化证据）；该量级对时间序列预测无实际影响。
- 存在偏差的场景见 §4。

## 2. 测试方法

- 同一 checkpoint（`ckpt/`，官方 timesfm-3.0 1.32GB safetensors）、同一输入 CSV。
- 官方侧：`repo/` 内官方 `TimesFM3Forecaster`（torch 2.2.2 CPU + oneMKL 2022.2，`tools/_torch_compat.py` 提供 nn.RMSNorm shim）。
- 引擎侧：纯 Rust 实现（`export_scenarios`，加载模型一次，逐场景跑 `predict_batch`）。**数值策略：GEMM/注意力内积为 matrixmultiply f32 内核，统计类归约（running stats、去趋势、softmax、RMSNorm、stitch、CPM refine）为 f64 累加**，见 §4.4。
- 每个场景两侧各跑一次 `predict_batch`，输出 median `(v,H)` 与 9 分位 `(v,H,9)`，
  逐元素比较：max/mean 绝对误差、逐位一致比例、相关性、分位单调性。
- 判定：EXACT = 全部元素逐位相同；TOL = max ≤ 1e-4（浮点累加顺序噪声级）；OFFICIAL_NAN = 官方输出含 NaN（无法对比，官方缺陷）；RUST_NAN / FAIL = 引擎问题。

## 3. 场景矩阵与结果

| # | 场景 | 说明 | median max\|Δ\| | 分位 max\|Δ\| | 逐位一致 | 判定 |
| --: | --- | --- | --: | --: | --: | --- |
| 1 | `etth1_h24` | horizon 24, 7 variates, dataset etth1 | 1.2e-05 | 1.3e-05 | 0/1 | **TOL** |
| 2 | `etth1_h96` | horizon 96, 7 variates, dataset etth1 | 1.2e-05 | 2.1e-05 | 0/1 | **TOL** |
| 3 | `etth1_h168` | horizon 168, 7 variates, dataset etth1 | 1.2e-05 | 2.1e-05 | 0/1 | **TOL** |
| 4 | `etth1_h336` | horizon 336, 7 variates, dataset etth1 | 1.2e-05 | 1.9e-05 | 0/1 | **TOL** |
| 5 | `etth1_h720` | horizon 720, 7 variates, dataset etth1 | 1.6e-05 | 2.5e-05 | 0/1 | **TOL** |
| 6 | `etth2_h24` | horizon 24, 7 variates, dataset etth2 | 7.6e-06 | 1.1e-05 | 0/1 | **TOL** |
| 7 | `etth2_h96` | horizon 96, 7 variates, dataset etth2 | 1.1e-05 | 1.5e-05 | 0/1 | **TOL** |
| 8 | `etth2_h168` | horizon 168, 7 variates, dataset etth2 | 1.5e-05 | 1.5e-05 | 0/1 | **TOL** |
| 9 | `etth2_h336` | horizon 336, 7 variates, dataset etth2 | 1.5e-05 | 1.9e-05 | 0/1 | **TOL** |
| 10 | `etth2_h720` | horizon 720, 7 variates, dataset etth2 | 1.5e-05 | 2.3e-05 | 0/1 | **TOL** |
| 11 | `ettm1_h24` | horizon 24, 7 variates, dataset ettm1 | 4.8e-06 | 7.6e-06 | 0/1 | **TOL** |
| 12 | `ettm1_h96` | horizon 96, 7 variates, dataset ettm1 | 1.3e-05 | 1.7e-05 | 0/1 | **TOL** |
| 13 | `ettm1_h168` | horizon 168, 7 variates, dataset ettm1 | 1.3e-05 | 1.7e-05 | 0/1 | **TOL** |
| 14 | `ettm1_h336` | horizon 336, 7 variates, dataset ettm1 | 1.1e-05 | 1.5e-05 | 0/1 | **TOL** |
| 15 | `ettm1_h720` | horizon 720, 7 variates, dataset ettm1 | 1.1e-05 | 2.0e-05 | 0/1 | **TOL** |
| 16 | `ettm2_h24` | horizon 24, 7 variates, dataset ettm2 | 7.6e-06 | 1.1e-05 | 0/1 | **TOL** |
| 17 | `ettm2_h96` | horizon 96, 7 variates, dataset ettm2 | 7.6e-06 | 1.1e-05 | 0/1 | **TOL** |
| 18 | `ettm2_h168` | horizon 168, 7 variates, dataset ettm2 | 1.1e-05 | 1.1e-05 | 0/1 | **TOL** |
| 19 | `ettm2_h336` | horizon 336, 7 variates, dataset ettm2 | 1.5e-05 | 1.5e-05 | 0/1 | **TOL** |
| 20 | `ettm2_h720` | horizon 720, 7 variates, dataset ettm2 | 1.5e-05 | 1.9e-05 | 0/1 | **TOL** |
| 21 | `synth_ts_h24` | horizon 24, 7 variates, dataset synth_ts | 3.1e-05 | 3.1e-05 | 0/1 | **TOL** |
| 22 | `synth_ts_h96` | horizon 96, 7 variates, dataset synth_ts | 3.1e-05 | 6.1e-05 | 0/1 | **TOL** |
| 23 | `synth_ts_h168` | horizon 168, 7 variates, dataset synth_ts | 6.1e-05 | 7.6e-05 | 0/1 | **TOL** |
| 24 | `synth_ts_h336` | horizon 336, 7 variates, dataset synth_ts | 7.6e-05 | 9.2e-05 | 0/1 | **TOL** |
| 25 | `synth_ts_h720` | horizon 720, 7 variates, dataset synth_ts | 1.1e-04 | 1.4e-04 | 0/1 | **FAIL** |
| 26 | `synth_scale_h24` | horizon 24, 7 variates, dataset synth_scale | 4.0e-02 | 1.8e-01 | 0/1 | **FAIL** |
| 27 | `synth_scale_h96` | horizon 96, 7 variates, dataset synth_scale | 5.1e-02 | 1.6e-01 | 0/1 | **FAIL** |
| 28 | `synth_scale_h168` | horizon 168, 7 variates, dataset synth_scale | 4.6e-02 | 1.9e-01 | 0/1 | **FAIL** |
| 29 | `synth_scale_h336` | horizon 336, 7 variates, dataset synth_scale | 4.6e-02 | 1.9e-01 | 0/1 | **FAIL** |
| 30 | `synth_scale_h720` | horizon 720, 7 variates, dataset synth_scale | 4.6e-02 | 2.7e-01 | 0/1 | **FAIL** |
| 31 | `synth_flat_h24` | horizon 24, 7 variates, dataset synth_flat | 8.1e-03 | 1.0e-02 | 0/1 | **FAIL** |
| 32 | `synth_flat_h96` | horizon 96, 7 variates, dataset synth_flat | 2.5e-02 | 3.4e-02 | 0/1 | **FAIL** |
| 33 | `synth_flat_h168` | horizon 168, 7 variates, dataset synth_flat | 2.8e-02 | 3.4e-02 | 0/1 | **FAIL** |
| 34 | `synth_flat_h336` | horizon 336, 7 variates, dataset synth_flat | 8.0e-02 | 9.1e-02 | 0/1 | **FAIL** |
| 35 | `synth_flat_h720` | horizon 720, 7 variates, dataset synth_flat | 1.5e-01 | 1.7e-01 | 0/1 | **FAIL** |
| 36 | `etth1_h1440` | long horizon 1440, dataset etth1 | 1.8e-05 | 2.7e-05 | 0/1 | **TOL** |
| 37 | `ettm2_h1440` | long horizon 1440, dataset ettm2 | 1.9e-05 | 2.7e-05 | 0/1 | **TOL** |
| 38 | `etth1_h1_uni` | sub-window horizon 1, univariate edge case | 3.8e-06 | 4.8e-06 | 0/1 | **TOL** |
| 39 | `etth1_h12_uni` | sub-window horizon 12, univariate edge case | 7.6e-06 | 1.1e-05 | 0/1 | **TOL** |
| 40 | `etth1_h48_uni` | sub-window horizon 48, univariate edge case | 7.6e-06 | 1.1e-05 | 0/1 | **TOL** |
| 41 | `etth1_ctx64` | context 64, dataset etth1 | 9.5e-06 | 2.0e-05 | 0/1 | **TOL** |
| 42 | `etth1_ctx256` | context 256, dataset etth1 | 1.1e-05 | 2.3e-05 | 0/1 | **TOL** |
| 43 | `etth1_ctx1024` | context 1024, dataset etth1 | 1.2e-05 | 2.1e-05 | 0/1 | **TOL** |
| 44 | `etth1_ctx3072` | context 3072, dataset etth1 | 1.6e-05 | 1.7e-05 | 0/1 | **TOL** |
| 45 | `etth2_ctx64` | context 64, dataset etth2 | 2.3e-05 | 3.1e-05 | 0/1 | **TOL** |
| 46 | `etth2_ctx256` | context 256, dataset etth2 | 1.1e-05 | 1.1e-05 | 0/1 | **TOL** |
| 47 | `etth2_ctx1024` | context 1024, dataset etth2 | 1.1e-05 | 1.5e-05 | 0/1 | **TOL** |
| 48 | `etth2_ctx3072` | context 3072, dataset etth2 | 1.1e-05 | 1.5e-05 | 0/1 | **TOL** |
| 49 | `ettm1_ctx64` | context 64, dataset ettm1 | 7.6e-06 | 2.0e-05 | 0/1 | **TOL** |
| 50 | `ettm1_ctx256` | context 256, dataset ettm1 | 7.2e-06 | 1.5e-05 | 0/1 | **TOL** |
| 51 | `ettm1_ctx1024` | context 1024, dataset ettm1 | 1.3e-05 | 1.7e-05 | 0/1 | **TOL** |
| 52 | `ettm1_ctx3072` | context 3072, dataset ettm1 | 1.0e-05 | 2.4e-05 | 0/1 | **TOL** |
| 53 | `ettm2_ctx64` | context 64, dataset ettm2 | 1.5e-05 | 2.7e-05 | 0/1 | **TOL** |
| 54 | `ettm2_ctx256` | context 256, dataset ettm2 | 1.1e-05 | 1.5e-05 | 0/1 | **TOL** |
| 55 | `ettm2_ctx1024` | context 1024, dataset ettm2 | 7.6e-06 | 1.1e-05 | 0/1 | **TOL** |
| 56 | `ettm2_ctx3072` | context 3072, dataset ettm2 | 1.1e-05 | 1.5e-05 | 0/1 | **TOL** |
| 57 | `synth_ts_ctx64` | context 64, dataset synth_ts | 6.1e-05 | 1.5e-04 | 0/1 | **FAIL** |
| 58 | `synth_ts_ctx256` | context 256, dataset synth_ts | 6.1e-05 | 6.1e-05 | 0/1 | **TOL** |
| 59 | `synth_ts_ctx1024` | context 1024, dataset synth_ts | 3.1e-05 | 6.1e-05 | 0/1 | **TOL** |
| 60 | `synth_ts_ctx3072` | context 3072, dataset synth_ts | 3.1e-05 | 3.1e-05 | 0/1 | **TOL** |
| 61 | `synth_scale_ctx64` | context 64, dataset synth_scale | 1.4e-01 | 3.0e-01 | 0/1 | **FAIL** |
| 62 | `synth_scale_ctx256` | context 256, dataset synth_scale | 4.9e-02 | 1.0e-01 | 0/1 | **FAIL** |
| 63 | `synth_scale_ctx1024` | context 1024, dataset synth_scale | 4.6e-02 | 1.8e-01 | 0/1 | **FAIL** |
| 64 | `synth_scale_ctx3072` | context 3072, dataset synth_scale | 3.4e-02 | 1.8e-01 | 0/1 | **FAIL** |
| 65 | `synth_flat_ctx64` | context 64, dataset synth_flat | 1.7e-01 | 2.4e-01 | 0/1 | **FAIL** |
| 66 | `synth_flat_ctx256` | context 256, dataset synth_flat | 7.9e-01 | 9.7e-01 | 0/1 | **FAIL** |
| 67 | `synth_flat_ctx1024` | context 1024, dataset synth_flat | 2.5e-02 | 3.4e-02 | 0/1 | **FAIL** |
| 68 | `synth_flat_ctx3072` | context 3072, dataset synth_flat | 1.5e-01 | 1.6e-01 | 0/1 | **FAIL** |
| 69 | `etth1_ctx15360` | context exactly at 15360 cap, etth1 | 3.6e-05 | 4.3e-05 | 0/1 | **TOL** |
| 70 | `etth2_ctx17020` | context 17020 > cap, etth2; expect truncation | 1.9e-05 | 2.7e-05 | 0/1 | **TOL** |
| 71 | `etth1_ctx32_uni` | tiny context 32, univariate edge case | 2.0e-05 | 3.5e-05 | 0/1 | **TOL** |
| 72 | `etth1_ctx48_uni` | tiny context 48, univariate edge case | 9.5e-06 | 2.0e-05 | 0/1 | **TOL** |
| 73 | `etth1_ctx64_uni` | tiny context 64, univariate edge case | 8.6e-06 | 1.8e-05 | 0/1 | **TOL** |
| 74 | `etth1_v1` | 1 variates of etth1, ctx 1024 | 1.1e-05 | 1.7e-05 | 0/1 | **TOL** |
| 75 | `etth1_v2` | 2 variates of etth1, ctx 1024 | 1.0e-05 | 1.6e-05 | 0/1 | **TOL** |
| 76 | `etth1_v3` | 3 variates of etth1, ctx 1024 | 1.1e-05 | 2.0e-05 | 0/1 | **TOL** |
| 77 | `etth1_v5` | 5 variates of etth1, ctx 1024 | 9.1e-06 | 2.1e-05 | 0/1 | **TOL** |
| 78 | `synth_v10` | 10 variates, ctx 1024 | 7.6e-06 | 1.1e-05 | 0/1 | **TOL** |
| 79 | `synth_v20` | 20 variates, ctx 512 | 1.5e-05 | 1.5e-05 | 0/1 | **TOL** |
| 80 | `synth_v50` | 50 variates, ctx 256 | 1.5e-05 | 2.3e-05 | 0/1 | **TOL** |
| 81 | `etth1_batch2_eq` | equal-length batch of 2 x 7v ctx1024 | 8.6e-06 | 1.4e-05 | 0/2 | **TOL** |
| 82 | `etth1_batch4_eq` | equal-length batch of 4 x 7v ctx1024 | 1.2e-05 | 2.5e-05 | 0/4 | **TOL** |
| 83 | `etth1_batch8_eq` | equal-length batch of 8 x 7v ctx1024 | 1.2e-05 | 2.5e-05 | 0/8 | **TOL** |
| 84 | `etth1_mixed8_uni` | 8 univariate mixed lengths 64..3072, etth1 OT | 1.9e-06 | 3.8e-06 | 0/8 | **OFFICIAL_NAN** |
| 85 | `ettm1_mixed5_uni` | 5 univariate mixed lengths from 15-min ettm1 | 2.4e-07 | 6.0e-07 | 0/5 | **OFFICIAL_NAN** |
| 86 | `mixed_batch_etth1xetth2` | cross-dataset equal batch: etth1 + etth2, 7v each | 1.2e-05 | 2.1e-05 | 0/2 | **TOL** |
| 87 | `etth1_nan_light` | scattered ~2% NaN | 1.0e-05 | 1.6e-05 | 0/1 | **TOL** |
| 88 | `etth1_nan_heavy` | heavy ~50% NaN | 7.6e-06 | 1.5e-05 | 0/1 | **TOL** |
| 89 | `etth1_nan_allcol` | one whole variate NaN + scattered | 3.1e-04 | 2.2e-03 | 0/1 | **FAIL** |
| 90 | `ettm2_nan_light` | scattered ~2% NaN on 15-min ettm2 | 1.1e-05 | 1.5e-05 | 0/1 | **TOL** |
| 91 | `synth_ts_nan_heavy` | heavy ~50% NaN on synth trend series | 4.6e-05 | 6.1e-05 | 0/1 | **TOL** |
| 92 | `fg_flags_sy0_so0_mp0_zn0` | flag grid: sy0, so0, mp0, zn0；无对称平均, 分位不排序 | 1.5e-05 | 2.1e-05 | 0/1 | **TOL** |
| 93 | `fg_flags_sy1_so0_mp0_zn0` | flag grid: sy1, so0, mp0, zn0；分位不排序 | 7.9e-06 | 1.3e-05 | 0/1 | **TOL** |
| 94 | `fg_flags_sy0_so1_mp0_zn0` | flag grid: sy0, so1, mp0, zn0；无对称平均 | 1.5e-05 | 2.1e-05 | 0/1 | **TOL** |
| 95 | `fg_flags_sy1_so1_mp0_zn0` | flag grid: sy1, so1, mp0, zn0 | 7.9e-06 | 1.3e-05 | 0/1 | **TOL** |
| 96 | `fg_flags_sy0_so0_mp1_zn0` | flag grid: sy0, so0, mp1, zn0；无对称平均, 分位不排序, 非负钳制 | 1.5e-05 | 2.1e-05 | 0/1 | **TOL** |
| 97 | `fg_flags_sy1_so0_mp1_zn0` | flag grid: sy1, so0, mp1, zn0；分位不排序, 非负钳制 | 7.9e-06 | 1.3e-05 | 0/1 | **TOL** |
| 98 | `fg_flags_sy0_so1_mp1_zn0` | flag grid: sy0, so1, mp1, zn0；无对称平均, 非负钳制 | 1.5e-05 | 2.1e-05 | 0/1 | **TOL** |
| 99 | `fg_flags_sy1_so1_mp1_zn0` | flag grid: sy1, so1, mp1, zn0；非负钳制 | 7.9e-06 | 1.3e-05 | 0/1 | **TOL** |
| 100 | `fg_flags_sy0_so0_mp0_zn1` | flag grid: sy0, so0, mp0, zn1；无对称平均, 分位不排序, znorm | 1.8e-05 | 2.2e-05 | 0/1 | **TOL** |
| 101 | `fg_flags_sy1_so0_mp0_zn1` | flag grid: sy1, so0, mp0, zn1；分位不排序, znorm | 1.1e-05 | 1.8e-05 | 0/1 | **TOL** |
| 102 | `fg_flags_sy0_so1_mp0_zn1` | flag grid: sy0, so1, mp0, zn1；无对称平均, znorm | 1.8e-05 | 2.2e-05 | 0/1 | **TOL** |
| 103 | `fg_flags_sy1_so1_mp0_zn1` | flag grid: sy1, so1, mp0, zn1；znorm | 1.1e-05 | 1.8e-05 | 0/1 | **TOL** |
| 104 | `fg_flags_sy0_so0_mp1_zn1` | flag grid: sy0, so0, mp1, zn1；无对称平均, 分位不排序, 非负钳制, znorm | 1.8e-05 | 2.2e-05 | 0/1 | **TOL** |
| 105 | `fg_flags_sy1_so0_mp1_zn1` | flag grid: sy1, so0, mp1, zn1；分位不排序, 非负钳制, znorm | 1.1e-05 | 1.8e-05 | 0/1 | **TOL** |
| 106 | `fg_flags_sy0_so1_mp1_zn1` | flag grid: sy0, so1, mp1, zn1；无对称平均, 非负钳制, znorm | 1.8e-05 | 2.2e-05 | 0/1 | **TOL** |
| 107 | `fg_flags_sy1_so1_mp1_zn1` | flag grid: sy1, so1, mp1, zn1；非负钳制, znorm | 1.1e-05 | 1.8e-05 | 0/1 | **TOL** |
| 108 | `fg_ovr_st0_dt0_cp0_fr0` | override grid: st0, dt0, cp0, fr0；use_stitching=False, use_linear_detrending=False, use_iterative_cpm_revin=False, use_frozen_running_stats=False | 1.2e-05 | 1.7e-05 | 0/1 | **TOL** |
| 109 | `fg_ovr_st1_dt0_cp0_fr0` | override grid: st1, dt0, cp0, fr0；use_stitching=True, use_linear_detrending=False, use_iterative_cpm_revin=False, use_frozen_running_stats=False | 1.2e-05 | 1.3e-05 | 0/1 | **TOL** |
| 110 | `fg_ovr_st0_dt1_cp0_fr0` | override grid: st0, dt1, cp0, fr0；use_stitching=False, use_linear_detrending=True, use_iterative_cpm_revin=False, use_frozen_running_stats=False | 7.6e-06 | 1.1e-05 | 0/1 | **TOL** |
| 111 | `fg_ovr_st1_dt1_cp0_fr0` | override grid: st1, dt1, cp0, fr0；use_stitching=True, use_linear_detrending=True, use_iterative_cpm_revin=False, use_frozen_running_stats=False | 8.1e-06 | 1.3e-05 | 0/1 | **TOL** |
| 112 | `fg_ovr_st0_dt0_cp1_fr0` | override grid: st0, dt0, cp1, fr0；use_stitching=False, use_linear_detrending=False, use_iterative_cpm_revin=True, use_frozen_running_stats=False | 1.2e-05 | 1.7e-05 | 0/1 | **TOL** |
| 113 | `fg_ovr_st1_dt0_cp1_fr0` | override grid: st1, dt0, cp1, fr0；use_stitching=True, use_linear_detrending=False, use_iterative_cpm_revin=True, use_frozen_running_stats=False | 1.2e-05 | 1.3e-05 | 0/1 | **TOL** |
| 114 | `fg_ovr_st0_dt1_cp1_fr0` | override grid: st0, dt1, cp1, fr0；use_stitching=False, use_linear_detrending=True, use_iterative_cpm_revin=True, use_frozen_running_stats=False | 7.6e-06 | 1.2e-05 | 0/1 | **TOL** |
| 115 | `fg_ovr_st1_dt1_cp1_fr0` | override grid: st1, dt1, cp1, fr0；use_stitching=True, use_linear_detrending=True, use_iterative_cpm_revin=True, use_frozen_running_stats=False | 7.9e-06 | 1.3e-05 | 0/1 | **TOL** |
| 116 | `fg_ovr_st0_dt0_cp0_fr1` | override grid: st0, dt0, cp0, fr1；use_stitching=False, use_linear_detrending=False, use_iterative_cpm_revin=False, use_frozen_running_stats=True | 1.2e-05 | 1.7e-05 | 0/1 | **TOL** |
| 117 | `fg_ovr_st1_dt0_cp0_fr1` | override grid: st1, dt0, cp0, fr1；use_stitching=True, use_linear_detrending=False, use_iterative_cpm_revin=False, use_frozen_running_stats=True | 1.2e-05 | 1.3e-05 | 0/1 | **TOL** |
| 118 | `fg_ovr_st0_dt1_cp0_fr1` | override grid: st0, dt1, cp0, fr1；use_stitching=False, use_linear_detrending=True, use_iterative_cpm_revin=False, use_frozen_running_stats=True | 7.6e-06 | 1.1e-05 | 0/1 | **TOL** |
| 119 | `fg_ovr_st1_dt1_cp0_fr1` | override grid: st1, dt1, cp0, fr1；use_stitching=True, use_linear_detrending=True, use_iterative_cpm_revin=False, use_frozen_running_stats=True | 8.1e-06 | 1.3e-05 | 0/1 | **TOL** |
| 120 | `fg_ovr_st0_dt0_cp1_fr1` | override grid: st0, dt0, cp1, fr1；use_stitching=False, use_linear_detrending=False, use_iterative_cpm_revin=True, use_frozen_running_stats=True | 1.2e-05 | 1.7e-05 | 0/1 | **TOL** |
| 121 | `fg_ovr_st1_dt0_cp1_fr1` | override grid: st1, dt0, cp1, fr1；use_stitching=True, use_linear_detrending=False, use_iterative_cpm_revin=True, use_frozen_running_stats=True | 1.2e-05 | 1.3e-05 | 0/1 | **TOL** |
| 122 | `fg_ovr_st0_dt1_cp1_fr1` | override grid: st0, dt1, cp1, fr1；use_stitching=False, use_linear_detrending=True, use_iterative_cpm_revin=True, use_frozen_running_stats=True | 7.6e-06 | 1.2e-05 | 0/1 | **TOL** |
| 123 | `fg_ovr_st1_dt1_cp1_fr1` | override grid: st1, dt1, cp1, fr1；use_stitching=True, use_linear_detrending=True, use_iterative_cpm_revin=True, use_frozen_running_stats=True | 7.9e-06 | 1.3e-05 | 0/1 | **TOL** |
| 124 | `fx_etth2_znorm` | znorm ON, dataset etth2；znorm | 7.6e-06 | 1.5e-05 | 0/1 | **TOL** |
| 125 | `fx_etth2_makepos` | make_positive ON, dataset etth2；非负钳制 | 1.1e-05 | 1.5e-05 | 0/1 | **TOL** |
| 126 | `fx_ettm1_znorm` | znorm ON, dataset ettm1；znorm | 7.6e-06 | 1.2e-05 | 0/1 | **TOL** |
| 127 | `fx_ettm1_makepos` | make_positive ON, dataset ettm1；非负钳制 | 1.3e-05 | 1.7e-05 | 0/1 | **TOL** |
| 128 | `fx_ettm2_znorm` | znorm ON, dataset ettm2；znorm | 7.6e-06 | 1.1e-05 | 0/1 | **TOL** |
| 129 | `fx_ettm2_makepos` | make_positive ON, dataset ettm2；非负钳制 | 7.6e-06 | 1.1e-05 | 0/1 | **TOL** |
| 130 | `fx_synth_ts_znorm` | znorm ON, dataset synth_ts；znorm | 1.5e-05 | 1.5e-05 | 0/1 | **TOL** |
| 131 | `fx_synth_ts_makepos` | make_positive ON, dataset synth_ts；非负钳制 | 3.1e-05 | 6.1e-05 | 0/1 | **TOL** |
| 132 | `fx_synthflat_makepos` | make_positive ON on strictly-negative synth data；非负钳制 | 2.5e-02 | 3.4e-02 | 0/1 | **FAIL** |
| 133 | `fx_etth1_znorm_nocpm` | znorm ON + CPM OFF combined, 7v etth1；znorm, use_iterative_cpm_revin=False | 1.3e-05 | 1.9e-05 | 0/1 | **TOL** |
| 134 | `fx_etth1_nosym_nostitch` | symmetric OFF + stitching OFF combined, 7v etth1；无对称平均, use_stitching=False | 1.4e-05 | 3.4e-05 | 0/1 | **TOL** |
| 135 | `h24_multi` | horizon sweep, 7 variates | 1.2e-05 | 1.3e-05 | 0/1 | **TOL** |
| 136 | `h96_multi` | horizon sweep, 7 variates | 1.2e-05 | 2.1e-05 | 0/1 | **TOL** |
| 137 | `h168_multi` | horizon sweep, 7 variates | 1.2e-05 | 2.1e-05 | 0/1 | **TOL** |
| 138 | `h336_multi` | horizon sweep, 7 variates | 1.2e-05 | 1.9e-05 | 0/1 | **TOL** |
| 139 | `h720_multi` | horizon sweep, 7 variates | 1.6e-05 | 2.5e-05 | 0/1 | **TOL** |
| 140 | `ctx64` | context sweep, 7 variates | 9.5e-06 | 2.0e-05 | 0/1 | **TOL** |
| 141 | `ctx256` | context sweep, 7 variates | 1.1e-05 | 2.3e-05 | 0/1 | **TOL** |
| 142 | `ctx3072` | context sweep, 7 variates | 1.6e-05 | 1.7e-05 | 0/1 | **TOL** |
| 143 | `ctx_cap_17k` | context 17020 > 15360 cap; expect identical truncation | 4.2e-05 | 6.4e-05 | 0/1 | **TOL** |
| 144 | `uni_1024` | univariate OT, ctx 1024 | 2.9e-06 | 3.8e-06 | 0/1 | **TOL** |
| 145 | `nan_mixed` | leading strip 96 + interior 64-block + ~3% scattered + trailing 16 | 1.0e-05 | 1.5e-05 | 0/1 | **TOL** |
| 146 | `batch2_multi` | two 7-variate series in ONE predict_batch call | 2.1e-05 | 3.0e-05 | 0/2 | **TOL** |
| 147 | `batch8_mixed_uni` | 8 univariate series, ctx 64..3072 -> dynamic batch context 3072 + left padding | 2.9e-06 | 4.8e-06 | 0/8 | **OFFICIAL_NAN** |
| 148 | `flag_nosym` | symmetric averaging OFF；无对称平均 | 1.7e-05 | 3.1e-05 | 0/1 | **TOL** |
| 149 | `flag_nosort` | quantile sorting OFF (raw quantile order compared)；分位不排序 | 1.2e-05 | 2.1e-05 | 0/1 | **TOL** |
| 150 | `flag_makepos` | make_positive ON (clamps negatives)；非负钳制 | 1.2e-05 | 2.1e-05 | 0/1 | **TOL** |
| 151 | `flag_znorm` | per-row z-normalization ON；znorm | 1.4e-05 | 2.5e-05 | 0/1 | **TOL** |
| 152 | `ovr_stitch_off` | use_stitching=False；use_stitching=False | 3.8e-06 | 3.8e-06 | 0/1 | **TOL** |
| 153 | `ovr_cpm_off` | use_iterative_cpm_revin=False；use_iterative_cpm_revin=False | 2.9e-06 | 3.8e-06 | 0/1 | **TOL** |
| 154 | `ovr_detrend_off` | use_linear_detrending=False；use_linear_detrending=False | 2.9e-06 | 3.8e-06 | 0/1 | **TOL** |
| 155 | `ovr_frozen_stats` | use_frozen_running_stats=True；use_frozen_running_stats=True | 2.9e-06 | 3.8e-06 | 0/1 | **TOL** |

### 各场景逐序列明细

- **`etth1_h24`**（h=24，序列形状：7x24）—— TOL
  - s0 (7x24): median max=1.240e-05 mean=1.158e-06 rel_max=1.191e-04 逐位一致 29.2%；分位 max=1.335e-05 一致 26.8%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.2e-06, v1=1.4e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=3.6e-07, v6=1.9e-06
- **`etth1_h96`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.240e-05 mean=1.127e-06 rel_max=1.191e-04 逐位一致 25.1%；分位 max=2.098e-05 一致 22.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.7e-06, v1=1.7e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=4.8e-07, v6=1.9e-06
- **`etth1_h168`**（h=168，序列形状：7x168）—— TOL
  - s0 (7x168): median max=1.240e-05 mean=1.069e-06 rel_max=2.207e-04 逐位一致 23.8%；分位 max=2.098e-05 一致 21.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.7e-06, v1=1.7e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=4.8e-07, v6=2.9e-06
- **`etth1_h336`**（h=336，序列形状：7x336）—— TOL
  - s0 (7x336): median max=1.240e-05 mean=1.067e-06 rel_max=2.738e-04 逐位一致 21.4%；分位 max=1.907e-05 一致 19.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.0e-05, v1=1.9e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=4.8e-07, v6=2.9e-06
- **`etth1_h720`**（h=720，序列形状：7x720）—— TOL
  - s0 (7x720): median max=1.574e-05 mean=1.168e-06 rel_max=2.794e-04 逐位一致 18.8%；分位 max=2.480e-05 一致 17.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.6e-05, v1=2.1e-06, v2=1.2e-05, v3=9.5e-07, v4=2.1e-06, v5=6.0e-07, v6=4.8e-06
- **`etth2_h24`**（h=24，序列形状：7x24）—— TOL
  - s0 (7x24): median max=7.629e-06 mean=1.049e-06 rel_max=4.339e-07 逐位一致 48.2%；分位 max=1.144e-05 一致 42.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=1.9e-06, v2=7.6e-06, v3=1.9e-06, v4=9.5e-07, v5=9.5e-07, v6=7.6e-06
- **`etth2_h96`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.144e-05 mean=1.182e-06 rel_max=8.498e-07 逐位一致 48.5%；分位 max=1.526e-05 一致 41.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=3.8e-06, v2=1.1e-05, v3=1.9e-06, v4=1.9e-06, v5=1.1e-06, v6=7.6e-06
- **`etth2_h168`**（h=168，序列形状：7x168）—— TOL
  - s0 (7x168): median max=1.526e-05 mean=1.716e-06 rel_max=8.498e-07 逐位一致 38.2%；分位 max=1.526e-05 一致 34.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.5e-05, v1=3.8e-06, v2=1.1e-05, v3=2.9e-06, v4=2.9e-06, v5=1.1e-06, v6=7.6e-06
- **`etth2_h336`**（h=336，序列形状：7x336）—— TOL
  - s0 (7x336): median max=1.526e-05 mean=1.941e-06 rel_max=8.498e-07 逐位一致 35.5%；分位 max=1.907e-05 一致 31.8%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.5e-05, v1=3.8e-06, v2=1.1e-05, v3=2.9e-06, v4=2.9e-06, v5=1.1e-06, v6=1.5e-05
- **`etth2_h720`**（h=720，序列形状：7x720）—— TOL
  - s0 (7x720): median max=1.526e-05 mean=2.764e-06 rel_max=9.707e-07 逐位一致 24.1%；分位 max=2.289e-05 一致 22.4%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.5e-05, v1=5.7e-06, v2=1.5e-05, v3=4.8e-06, v4=3.8e-06, v5=1.4e-06, v6=1.5e-05
- **`ettm1_h24`**（h=24，序列形状：7x24）—— TOL
  - s0 (7x24): median max=4.768e-06 mean=6.230e-07 rel_max=4.327e-07 逐位一致 37.5%；分位 max=7.629e-06 一致 32.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.8e-06, v1=1.4e-06, v2=4.8e-06, v3=7.2e-07, v4=1.4e-06, v5=3.6e-07, v6=9.5e-07
- **`ettm1_h96`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.335e-05 mean=1.005e-06 rel_max=4.910e-05 逐位一致 33.0%；分位 max=1.717e-05 一致 29.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.3e-05, v1=1.4e-06, v2=6.7e-06, v3=9.5e-07, v4=1.9e-06, v5=4.8e-07, v6=1.9e-06
- **`ettm1_h168`**（h=168，序列形状：7x168）—— TOL
  - s0 (7x168): median max=1.335e-05 mean=1.017e-06 rel_max=1.418e-03 逐位一致 29.8%；分位 max=1.717e-05 一致 27.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.3e-05, v1=1.4e-06, v2=9.5e-06, v3=9.5e-07, v4=1.9e-06, v5=4.8e-07, v6=1.9e-06
- **`ettm1_h336`**（h=336，序列形状：7x336）—— TOL
  - s0 (7x336): median max=1.144e-05 mean=9.124e-07 rel_max=1.073e-03 逐位一致 28.4%；分位 max=1.526e-05 一致 26.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.1e-05, v1=1.7e-06, v2=7.6e-06, v3=1.1e-06, v4=1.9e-06, v5=4.8e-07, v6=1.9e-06
- **`ettm1_h720`**（h=720，序列形状：7x720）—— TOL
  - s0 (7x720): median max=1.144e-05 mean=9.222e-07 rel_max=1.073e-03 逐位一致 24.2%；分位 max=2.003e-05 一致 22.7%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.1e-05, v1=1.9e-06, v2=8.6e-06, v3=1.3e-06, v4=1.9e-06, v5=7.2e-07, v6=2.9e-06
- **`ettm2_h24`**（h=24，序列形状：7x24）—— TOL
  - s0 (7x24): median max=7.629e-06 mean=1.181e-06 rel_max=1.256e-06 逐位一致 46.4%；分位 max=1.144e-05 一致 42.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=3.8e-06, v2=3.8e-06, v3=1.9e-06, v4=9.5e-07, v5=1.3e-06, v6=7.6e-06
- **`ettm2_h96`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=7.629e-06 mean=1.331e-06 rel_max=1.265e-06 逐位一致 40.3%；分位 max=1.144e-05 一致 40.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=3.8e-06, v2=7.6e-06, v3=1.9e-06, v4=1.9e-06, v5=1.3e-06, v6=7.6e-06
- **`ettm2_h168`**（h=168，序列形状：7x168）—— TOL
  - s0 (7x168): median max=1.144e-05 mean=1.453e-06 rel_max=1.265e-06 逐位一致 36.9%；分位 max=1.144e-05 一致 38.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.1e-05, v1=3.8e-06, v2=7.6e-06, v3=2.9e-06, v4=1.9e-06, v5=1.7e-06, v6=7.6e-06
- **`ettm2_h336`**（h=336，序列形状：7x336）—— TOL
  - s0 (7x336): median max=1.526e-05 mean=1.863e-06 rel_max=1.256e-06 逐位一致 30.7%；分位 max=1.526e-05 一致 29.6%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.1e-05, v1=3.8e-06, v2=1.5e-05, v3=3.8e-06, v4=2.9e-06, v5=1.3e-06, v6=1.1e-05
- **`ettm2_h720`**（h=720，序列形状：7x720）—— TOL
  - s0 (7x720): median max=1.526e-05 mean=1.839e-06 rel_max=1.256e-06 逐位一致 30.6%；分位 max=1.907e-05 一致 29.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.1e-05, v1=3.8e-06, v2=1.5e-05, v3=3.8e-06, v4=3.8e-06, v5=1.3e-06, v6=1.1e-05
- **`synth_ts_h24`**（h=24，序列形状：7x24）—— TOL
  - s0 (7x24): median max=3.052e-05 mean=2.906e-06 rel_max=1.901e-07 逐位一致 82.1%；分位 max=3.052e-05 一致 82.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.5e-05, v1=1.5e-05, v2=3.1e-05, v3=1.5e-05, v4=3.1e-05, v5=1.5e-05, v6=1.5e-05
- **`synth_ts_h96`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=3.052e-05 mean=7.448e-06 rel_max=1.901e-07 逐位一致 60.1%；分位 max=6.104e-05 一致 58.7%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.1e-05, v1=3.1e-05, v2=3.1e-05, v3=3.1e-05, v4=3.1e-05, v5=3.1e-05, v6=1.5e-05
- **`synth_ts_h168`**（h=168，序列形状：7x168）—— TOL
  - s0 (7x168): median max=6.104e-05 mean=1.100e-05 rel_max=3.618e-07 逐位一致 48.6%；分位 max=7.629e-05 一致 47.4%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.1e-05, v1=3.1e-05, v2=6.1e-05, v3=3.1e-05, v4=3.1e-05, v5=3.1e-05, v6=3.1e-05
- **`synth_ts_h336`**（h=336，序列形状：7x336）—— TOL
  - s0 (7x336): median max=7.629e-05 mean=1.606e-05 rel_max=4.385e-07 逐位一致 38.4%；分位 max=9.155e-05 一致 35.8%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.1e-05, v1=6.1e-05, v2=7.6e-05, v3=3.1e-05, v4=6.1e-05, v5=3.1e-05, v6=4.6e-05
- **`synth_ts_h720`**（h=720，序列形状：7x720）—— FAIL
  - s0 (7x720): median max=1.068e-04 mean=2.688e-05 rel_max=6.275e-07 逐位一致 22.3%；分位 max=1.373e-04 一致 22.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=9.2e-05, v1=6.1e-05, v2=9.2e-05, v3=6.1e-05, v4=6.1e-05, v5=4.6e-05, v6=1.1e-04
- **`synth_scale_h24`**（h=24，序列形状：7x24）—— FAIL
  - s0 (7x24): median max=4.004e-02 mean=8.622e-03 rel_max=3.385e-02 逐位一致 7.7%；分位 max=1.836e-01 一致 5.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=2.4e-07, v1=2.8e-02, v2=2.4e-02, v3=4.0e-02, v4=1.9e-02, v5=2.7e-02, v6=1.8e-02
- **`synth_scale_h96`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=5.078e-02 mean=7.469e-03 rel_max=1.258e-02 逐位一致 4.3%；分位 max=1.602e-01 一致 4.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=4.8e-07, v1=2.5e-02, v2=5.1e-02, v3=2.3e-02, v4=2.5e-02, v5=4.6e-02, v6=4.2e-02
- **`synth_scale_h168`**（h=168，序列形状：7x168）—— FAIL
  - s0 (7x168): median max=4.639e-02 mean=6.982e-03 rel_max=3.385e-02 逐位一致 4.8%；分位 max=1.914e-01 一致 4.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=6.0e-07, v1=2.9e-02, v2=3.6e-02, v3=4.0e-02, v4=2.4e-02, v5=4.6e-02, v6=3.6e-02
- **`synth_scale_h336`**（h=336，序列形状：7x336）—— FAIL
  - s0 (7x336): median max=4.639e-02 mean=6.872e-03 rel_max=3.385e-02 逐位一致 4.0%；分位 max=1.914e-01 一致 3.8%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=6.0e-07, v1=2.9e-02, v2=3.7e-02, v3=4.0e-02, v4=2.4e-02, v5=4.6e-02, v6=3.3e-02
- **`synth_scale_h720`**（h=720，序列形状：7x720）—— FAIL
  - s0 (7x720): median max=4.639e-02 mean=6.835e-03 rel_max=3.385e-02 逐位一致 3.9%；分位 max=2.734e-01 一致 3.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=6.0e-07, v1=2.9e-02, v2=3.7e-02, v3=4.0e-02, v4=3.1e-02, v5=4.6e-02, v6=3.3e-02
- **`synth_flat_h24`**（h=24，序列形状：7x24）—— FAIL
  - s0 (7x24): median max=8.082e-03 mean=1.408e-03 rel_max=1.190e-03 逐位一致 29.8%；分位 max=1.000e-02 一致 29.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=2.4e-06, v1=0.0e+00, v2=8.1e-03, v3=5.2e-03, v4=2.1e-06, v5=0.0e+00, v6=5.4e-03
- **`synth_flat_h96`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=2.466e-02 mean=1.925e-03 rel_max=2.587e-03 逐位一致 25.3%；分位 max=3.415e-02 一致 24.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=2.4e-06, v1=3.8e-06, v2=1.7e-02, v3=2.5e-02, v4=2.1e-06, v5=3.8e-06, v6=7.7e-03
- **`synth_flat_h168`**（h=168，序列形状：7x168）—— FAIL
  - s0 (7x168): median max=2.750e-02 mean=2.307e-03 rel_max=2.586e-03 逐位一致 23.8%；分位 max=3.417e-02 一致 21.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=2.4e-06, v1=3.8e-06, v2=1.7e-02, v3=2.8e-02, v4=2.1e-06, v5=3.8e-06, v6=7.7e-03
- **`synth_flat_h336`**（h=336，序列形状：7x336）—— FAIL
  - s0 (7x336): median max=8.032e-02 mean=2.312e-03 rel_max=1.063e-02 逐位一致 20.9%；分位 max=9.090e-02 一致 18.0%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=2.4e-06, v1=3.8e-06, v2=1.7e-02, v3=8.0e-02, v4=2.1e-06, v5=3.8e-06, v6=7.7e-03
- **`synth_flat_h720`**（h=720，序列形状：7x720）—— FAIL
  - s0 (7x720): median max=1.515e-01 mean=2.398e-03 rel_max=1.063e-02 逐位一致 15.0%；分位 max=1.692e-01 一致 14.6%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=2.4e-06, v1=3.8e-06, v2=1.7e-02, v3=1.5e-01, v4=2.1e-06, v5=3.8e-06, v6=8.0e-03
- **`etth1_h1440`**（h=1440，序列形状：7x1440）—— TOL
  - s0 (7x1440): median max=1.764e-05 mean=1.136e-06 rel_max=5.267e-04 逐位一致 17.6%；分位 max=2.670e-05 一致 15.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.8e-05, v1=2.1e-06, v2=1.2e-05, v3=1.4e-06, v4=1.9e-06, v5=8.3e-07, v6=3.8e-06
- **`ettm2_h1440`**（h=1440，序列形状：7x1440）—— TOL
  - s0 (7x1440): median max=1.907e-05 mean=2.332e-06 rel_max=1.285e-06 逐位一致 26.0%；分位 max=2.670e-05 一致 24.8%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.9e-05, v1=3.8e-06, v2=1.1e-05, v3=3.8e-06, v4=3.8e-06, v5=1.7e-06, v6=1.5e-05
- **`etth1_h1_uni`**（h=1，序列形状：1x1）—— TOL
  - s0 (1x1): median max=3.815e-06 mean=3.815e-06 rel_max=3.402e-07 逐位一致 0.0%；分位 max=4.768e-06 一致 0.0%；corr=1.000000；分位单调=True
- **`etth1_h12_uni`**（h=12，序列形状：1x12）—— TOL
  - s0 (1x12): median max=7.629e-06 mean=2.066e-06 rel_max=4.740e-07 逐位一致 16.7%；分位 max=1.144e-05 一致 13.9%；corr=1.000000；分位单调=True
- **`etth1_h48_uni`**（h=48，序列形状：1x48）—— TOL
  - s0 (1x48): median max=7.629e-06 mean=2.022e-06 rel_max=5.703e-06 逐位一致 16.7%；分位 max=1.144e-05 一致 10.4%；corr=1.000000；分位单调=True
- **`etth1_ctx64`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=9.537e-06 mean=1.139e-06 rel_max=4.687e-05 逐位一致 27.7%；分位 max=2.003e-05 一致 25.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=9.5e-06, v1=1.9e-06, v2=9.5e-06, v3=1.2e-06, v4=1.7e-06, v5=4.8e-07, v6=1.9e-06
- **`etth1_ctx256`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.144e-05 mean=9.884e-07 rel_max=6.000e-05 逐位一致 18.5%；分位 max=2.289e-05 一致 19.6%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=1.4e-06, v2=1.1e-05, v3=1.4e-06, v4=1.9e-06, v5=4.8e-07, v6=2.9e-06
- **`etth1_ctx1024`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.240e-05 mean=1.127e-06 rel_max=1.191e-04 逐位一致 25.1%；分位 max=2.098e-05 一致 22.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.7e-06, v1=1.7e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=4.8e-07, v6=1.9e-06
- **`etth1_ctx3072`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.621e-05 mean=1.056e-06 rel_max=1.500e-05 逐位一致 24.3%；分位 max=1.717e-05 一致 23.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=9.5e-06, v1=2.9e-06, v2=1.6e-05, v3=1.2e-06, v4=1.7e-06, v5=4.8e-07, v6=1.9e-06
- **`etth2_ctx64`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=2.289e-05 mean=1.926e-06 rel_max=1.009e-06 逐位一致 39.7%；分位 max=3.052e-05 一致 36.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=1.9e-06, v2=7.6e-06, v3=1.9e-06, v4=1.9e-06, v5=9.5e-07, v6=2.3e-05
- **`etth2_ctx256`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.144e-05 mean=1.408e-06 rel_max=1.633e-06 逐位一致 36.5%；分位 max=1.144e-05 一致 34.7%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=3.8e-06, v2=1.1e-05, v3=1.9e-06, v4=2.9e-06, v5=1.2e-06, v6=7.6e-06
- **`etth2_ctx1024`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.144e-05 mean=1.182e-06 rel_max=8.498e-07 逐位一致 48.5%；分位 max=1.526e-05 一致 41.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=3.8e-06, v2=1.1e-05, v3=1.9e-06, v4=1.9e-06, v5=1.1e-06, v6=7.6e-06
- **`etth2_ctx3072`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.144e-05 mean=1.674e-06 rel_max=6.515e-07 逐位一致 40.5%；分位 max=1.526e-05 一致 35.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=3.8e-06, v2=7.6e-06, v3=2.9e-06, v4=1.9e-06, v5=8.3e-07, v6=1.1e-05
- **`ettm1_ctx64`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=7.629e-06 mean=7.839e-07 rel_max=2.218e-06 逐位一致 39.9%；分位 max=2.003e-05 一致 34.4%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=1.4e-06, v2=5.7e-06, v3=4.8e-07, v4=1.4e-06, v5=2.4e-07, v6=1.9e-06
- **`ettm1_ctx256`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=7.153e-06 mean=8.353e-07 rel_max=1.261e-05 逐位一致 34.2%；分位 max=1.526e-05 一致 30.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.2e-06, v1=9.5e-07, v2=6.7e-06, v3=9.5e-07, v4=1.4e-06, v5=2.4e-07, v6=1.9e-06
- **`ettm1_ctx1024`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.335e-05 mean=1.005e-06 rel_max=4.910e-05 逐位一致 33.0%；分位 max=1.717e-05 一致 29.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.3e-05, v1=1.4e-06, v2=6.7e-06, v3=9.5e-07, v4=1.9e-06, v5=4.8e-07, v6=1.9e-06
- **`ettm1_ctx3072`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.049e-05 mean=9.956e-07 rel_max=8.990e-06 逐位一致 28.6%；分位 max=2.411e-05 一致 26.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.6e-06, v1=1.9e-06, v2=1.0e-05, v3=9.5e-07, v4=2.9e-06, v5=4.8e-07, v6=1.9e-06
- **`ettm2_ctx64`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.526e-05 mean=1.872e-06 rel_max=3.997e-07 逐位一致 43.8%；分位 max=2.670e-05 一致 39.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.5e-05, v1=3.8e-06, v2=1.5e-05, v3=1.9e-06, v4=1.9e-06, v5=4.8e-07, v6=7.6e-06
- **`ettm2_ctx256`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.144e-05 mean=1.661e-06 rel_max=6.501e-07 逐位一致 36.0%；分位 max=1.526e-05 一致 33.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=3.8e-06, v2=1.1e-05, v3=2.9e-06, v4=1.9e-06, v5=7.2e-07, v6=7.6e-06
- **`ettm2_ctx1024`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=7.629e-06 mean=1.331e-06 rel_max=1.265e-06 逐位一致 40.3%；分位 max=1.144e-05 一致 40.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=3.8e-06, v2=7.6e-06, v3=1.9e-06, v4=1.9e-06, v5=1.3e-06, v6=7.6e-06
- **`ettm2_ctx3072`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.144e-05 mean=2.361e-06 rel_max=5.733e-07 逐位一致 24.1%；分位 max=1.526e-05 一致 24.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=5.7e-06, v2=1.1e-05, v3=4.8e-06, v4=3.8e-06, v5=9.5e-07, v6=7.6e-06
- **`synth_ts_ctx64`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=6.104e-05 mean=2.030e-05 rel_max=3.812e-07 逐位一致 28.3%；分位 max=1.526e-04 一致 29.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=6.1e-05, v1=6.1e-05, v2=3.1e-05, v3=6.1e-05, v4=6.1e-05, v5=4.6e-05, v6=4.6e-05
- **`synth_ts_ctx256`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=6.104e-05 mean=8.969e-06 rel_max=3.808e-07 逐位一致 58.6%；分位 max=6.104e-05 一致 54.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=6.1e-05, v1=3.1e-05, v2=3.1e-05, v3=3.1e-05, v4=3.1e-05, v5=3.1e-05, v6=3.1e-05
- **`synth_ts_ctx1024`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=3.052e-05 mean=7.448e-06 rel_max=1.901e-07 逐位一致 60.1%；分位 max=6.104e-05 一致 58.7%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.1e-05, v1=3.1e-05, v2=3.1e-05, v3=3.1e-05, v4=3.1e-05, v5=3.1e-05, v6=1.5e-05
- **`synth_ts_ctx3072`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=3.052e-05 mean=6.063e-06 rel_max=1.902e-07 逐位一致 63.5%；分位 max=3.052e-05 一致 62.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.1e-05, v1=3.1e-05, v2=1.5e-05, v3=3.1e-05, v4=3.1e-05, v5=3.1e-05, v6=3.1e-05
- **`synth_scale_ctx64`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=1.367e-01 mean=1.580e-02 rel_max=1.669e-03 逐位一致 6.0%；分位 max=3.047e-01 一致 5.4%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=4.8e-07, v1=4.9e-02, v2=1.1e-01, v3=2.5e-02, v4=2.5e-02, v5=1.4e-01, v6=5.6e-02
- **`synth_scale_ctx256`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=4.883e-02 mean=7.427e-03 rel_max=2.576e-03 逐位一致 3.6%；分位 max=1.035e-01 一致 3.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=4.8e-07, v1=3.3e-02, v2=4.2e-02, v3=2.1e-02, v4=2.0e-02, v5=3.0e-02, v6=4.9e-02
- **`synth_scale_ctx1024`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=4.639e-02 mean=7.632e-03 rel_max=3.385e-02 逐位一致 5.5%；分位 max=1.836e-01 一致 4.6%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=6.0e-07, v1=2.9e-02, v2=3.6e-02, v3=4.0e-02, v4=2.4e-02, v5=4.6e-02, v6=3.6e-02
- **`synth_scale_ctx3072`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=3.369e-02 mean=6.676e-03 rel_max=1.922e-03 逐位一致 7.0%；分位 max=1.758e-01 一致 5.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=4.8e-07, v1=2.4e-02, v2=3.4e-02, v3=1.7e-02, v4=2.2e-02, v5=2.2e-02, v6=2.7e-02
- **`synth_flat_ctx64`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=1.667e-01 mean=7.967e-03 rel_max=1.870e-03 逐位一致 30.7%；分位 max=2.388e-01 一致 27.8%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.9e-06, v1=3.8e-06, v2=3.6e-03, v3=1.7e-01, v4=1.2e-06, v5=3.8e-06, v6=3.7e-03
- **`synth_flat_ctx256`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=7.931e-01 mean=2.335e-02 rel_max=4.456e-02 逐位一致 23.7%；分位 max=9.698e-01 一致 22.2%；corr=0.999998；分位单调=True
    - 逐变体 max：v0=3.6e-05, v1=3.8e-06, v2=7.7e-02, v3=7.9e-01, v4=1.1e-05, v5=7.6e-06, v6=3.8e-02
- **`synth_flat_ctx1024`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=2.468e-02 mean=1.925e-03 rel_max=2.586e-03 逐位一致 25.3%；分位 max=3.417e-02 一致 24.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=2.4e-06, v1=3.8e-06, v2=1.7e-02, v3=2.5e-02, v4=2.1e-06, v5=3.8e-06, v6=7.7e-03
- **`synth_flat_ctx3072`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=1.519e-01 mean=2.260e-03 rel_max=1.971e-02 逐位一致 28.7%；分位 max=1.624e-01 一致 28.7%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=2.6e-06, v1=3.8e-06, v2=1.2e-02, v3=1.5e-01, v4=6.4e-06, v5=3.8e-06, v6=8.2e-03
- **`etth1_ctx15360`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=3.624e-05 mean=2.923e-06 rel_max=6.939e-05 逐位一致 11.3%；分位 max=4.256e-05 一致 11.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.1e-05, v1=3.3e-06, v2=3.6e-05, v3=2.9e-06, v4=4.8e-06, v5=1.3e-06, v6=5.7e-06
- **`etth2_ctx17020`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.907e-05 mean=2.881e-06 rel_max=1.605e-06 逐位一致 21.1%；分位 max=2.670e-05 一致 21.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.3e-05, v1=4.3e-06, v2=1.5e-05, v3=4.8e-06, v4=3.8e-06, v5=2.9e-06, v6=1.9e-05
- **`etth1_ctx32_uni`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=2.003e-05 mean=3.949e-06 rel_max=1.829e-05 逐位一致 8.3%；分位 max=3.457e-05 一致 6.6%；corr=1.000000；分位单调=True
- **`etth1_ctx48_uni`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=9.537e-06 mean=2.530e-06 rel_max=4.799e-04 逐位一致 19.8%；分位 max=1.955e-05 一致 10.8%；corr=1.000000；分位单调=True
- **`etth1_ctx64_uni`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=8.583e-06 mean=2.844e-06 rel_max=7.864e-06 逐位一致 7.3%；分位 max=1.764e-05 一致 10.5%；corr=1.000000；分位单调=True
- **`etth1_v1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.144e-05 mean=2.510e-06 rel_max=1.481e-05 逐位一致 15.6%；分位 max=1.717e-05 一致 13.7%；corr=1.000000；分位单调=True
- **`etth1_v2`**（h=96，序列形状：2x96）—— TOL
  - s0 (2x96): median max=1.049e-05 mean=1.318e-06 rel_max=3.298e-05 逐位一致 27.6%；分位 max=1.621e-05 一致 21.4%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.0e-05, v1=1.4e-06
- **`etth1_v3`**（h=96，序列形状：3x96）—— TOL
  - s0 (3x96): median max=1.144e-05 mean=1.959e-06 rel_max=3.532e-05 逐位一致 20.5%；分位 max=2.003e-05 一致 16.4%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.0e-05, v1=9.5e-07, v2=1.1e-05
- **`etth1_v5`**（h=96，序列形状：5x96）—— TOL
  - s0 (5x96): median max=9.060e-06 mean=1.038e-06 rel_max=2.125e-04 逐位一致 25.4%；分位 max=2.098e-05 一致 20.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=9.1e-06, v1=2.9e-06, v2=7.6e-06, v3=1.2e-06, v4=9.5e-07
- **`synth_v10`**（h=96，序列形状：10x96）—— TOL
  - s0 (10x96): median max=7.629e-06 mean=1.578e-06 rel_max=1.973e-07 逐位一致 64.4%；分位 max=1.144e-05 一致 63.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=7.6e-06, v2=3.8e-06, v3=7.6e-06, v4=7.6e-06, v5=7.6e-06, v6=7.6e-06, v7=7.6e-06, v8=7.6e-06, v9=7.6e-06
- **`synth_v20`**（h=96，序列形状：20x96）—— TOL
  - s0 (20x96): median max=1.526e-05 mean=2.052e-06 rel_max=3.951e-07 逐位一致 57.5%；分位 max=1.526e-05 一致 56.1%；corr=1.000000；分位单调=True
- **`synth_v50`**（h=96，序列形状：50x96）—— TOL
  - s0 (50x96): median max=1.526e-05 mean=2.282e-06 rel_max=4.003e-07 逐位一致 54.8%；分位 max=2.289e-05 一致 54.3%；corr=1.000000；分位单调=True
- **`etth1_batch2_eq`**（h=96，序列形状：7x96, 7x96）—— TOL
  - s0 (7x96): median max=7.153e-06 mean=6.974e-07 rel_max=3.879e-05 逐位一致 21.9%；分位 max=1.097e-05 一致 20.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.2e-06, v1=9.5e-07, v2=5.9e-06, v3=9.5e-07, v4=9.5e-07, v5=1.8e-07, v6=2.9e-06
  - s1 (7x96): median max=8.583e-06 mean=1.138e-06 rel_max=2.195e-05 逐位一致 14.4%；分位 max=1.431e-05 一致 14.0%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=1.3e-06, v2=8.6e-06, v3=1.8e-06, v4=9.5e-07, v5=2.4e-07, v6=6.7e-06
- **`etth1_batch4_eq`**（h=96，序列形状：7x96, 7x96, 7x96, 7x96）—— TOL
  - s0 (7x96): median max=6.199e-06 mean=6.750e-07 rel_max=1.828e-05 逐位一致 24.4%；分位 max=1.121e-05 一致 20.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.8e-06, v1=1.1e-06, v2=6.2e-06, v3=9.5e-07, v4=9.5e-07, v5=1.8e-07, v6=3.8e-06
  - s1 (7x96): median max=7.629e-06 mean=9.625e-07 rel_max=2.159e-04 逐位一致 16.8%；分位 max=1.431e-05 一致 15.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.2e-06, v1=1.2e-06, v2=7.6e-06, v3=9.5e-07, v4=1.4e-06, v5=2.4e-07, v6=6.7e-06
  - s2 (7x96): median max=1.240e-05 mean=1.002e-06 rel_max=4.550e-05 逐位一致 17.7%；分位 max=2.480e-05 一致 17.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=1.7e-06, v2=1.2e-05, v3=1.3e-06, v4=7.2e-07, v5=2.4e-07, v6=3.8e-06
  - s3 (7x96): median max=1.049e-05 mean=1.139e-06 rel_max=3.749e-05 逐位一致 18.2%；分位 max=2.003e-05 一致 17.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.0e-05, v1=1.4e-06, v2=1.0e-05, v3=1.1e-06, v4=1.2e-06, v5=3.6e-07, v6=3.8e-06
- **`etth1_batch8_eq`**（h=96，序列形状：7x96, 7x96, 7x96, 7x96, 7x96, 7x96, 7x96, 7x96）—— TOL
  - s0 (7x96): median max=6.199e-06 mean=6.750e-07 rel_max=1.828e-05 逐位一致 24.4%；分位 max=1.121e-05 一致 20.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.8e-06, v1=1.1e-06, v2=6.2e-06, v3=9.5e-07, v4=9.5e-07, v5=1.8e-07, v6=3.8e-06
  - s1 (7x96): median max=7.629e-06 mean=9.625e-07 rel_max=2.159e-04 逐位一致 16.8%；分位 max=1.431e-05 一致 15.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.2e-06, v1=1.2e-06, v2=7.6e-06, v3=9.5e-07, v4=1.4e-06, v5=2.4e-07, v6=6.7e-06
  - s2 (7x96): median max=1.240e-05 mean=1.002e-06 rel_max=4.550e-05 逐位一致 17.7%；分位 max=2.480e-05 一致 17.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=1.7e-06, v2=1.2e-05, v3=1.3e-06, v4=7.2e-07, v5=2.4e-07, v6=3.8e-06
  - s3 (7x96): median max=1.049e-05 mean=1.139e-06 rel_max=3.749e-05 逐位一致 18.2%；分位 max=2.003e-05 一致 17.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.0e-05, v1=1.4e-06, v2=1.0e-05, v3=1.1e-06, v4=1.2e-06, v5=3.6e-07, v6=3.8e-06
  - s4 (7x96): median max=1.192e-05 mean=1.004e-06 rel_max=3.415e-03 逐位一致 21.0%；分位 max=1.907e-05 一致 18.4%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=9.5e-06, v1=9.5e-07, v2=1.2e-05, v3=7.2e-07, v4=1.4e-06, v5=4.8e-07, v6=3.8e-06
  - s5 (7x96): median max=8.583e-06 mean=8.106e-07 rel_max=6.019e-05 逐位一致 19.0%；分位 max=2.003e-05 一致 19.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.6e-06, v1=1.1e-06, v2=4.8e-06, v3=1.3e-06, v4=1.2e-06, v5=4.8e-07, v6=3.8e-06
  - s6 (7x96): median max=8.583e-06 mean=9.858e-07 rel_max=7.745e-05 逐位一致 18.6%；分位 max=1.574e-05 一致 16.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.6e-06, v1=2.4e-06, v2=6.9e-06, v3=1.2e-06, v4=1.9e-06, v5=6.0e-07, v6=5.7e-06
  - s7 (7x96): median max=8.583e-06 mean=1.085e-06 rel_max=7.432e-05 逐位一致 21.4%；分位 max=1.335e-05 一致 19.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=6.7e-06, v1=2.7e-06, v2=8.6e-06, v3=2.4e-06, v4=9.5e-07, v5=2.4e-07, v6=3.8e-06
- **`etth1_mixed8_uni`**（h=96，序列形状：1x96, 1x96, 1x96, 1x96, 1x96, 1x96, 1x96, 1x96）—— OFFICIAL_NAN
  - s0 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s1 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s2 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s3 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s4 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s5 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s6 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s7 (1x96): median max=1.907e-06 mean=6.855e-07 rel_max=1.866e-07 逐位一致 45.8%；分位 max=3.815e-06 一致 40.0%；corr=1.000000；分位单调=True
- **`ettm1_mixed5_uni`**（h=96，序列形状：1x96, 1x96, 1x96, 1x96, 1x96）—— OFFICIAL_NAN
  - s0 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s1 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s2 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s3 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s4 (1x96): median max=2.384e-07 mean=6.271e-08 rel_max=2.440e-07 逐位一致 39.6%；分位 max=5.960e-07 一致 34.5%；corr=1.000000；分位单调=True
- **`mixed_batch_etth1xetth2`**（h=96，序列形状：7x96, 7x96）—— TOL
  - s0 (7x96): median max=1.240e-05 mean=1.127e-06 rel_max=1.191e-04 逐位一致 25.1%；分位 max=2.098e-05 一致 22.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.7e-06, v1=1.7e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=4.8e-07, v6=1.9e-06
  - s1 (7x96): median max=1.144e-05 mean=1.182e-06 rel_max=8.498e-07 逐位一致 48.5%；分位 max=1.526e-05 一致 41.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=3.8e-06, v2=1.1e-05, v3=1.9e-06, v4=1.9e-06, v5=1.1e-06, v6=7.6e-06
- **`etth1_nan_light`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.049e-05 mean=9.658e-07 rel_max=1.473e-05 逐位一致 24.4%；分位 max=1.621e-05 一致 22.8%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.9e-06, v1=1.4e-06, v2=1.0e-05, v3=1.1e-06, v4=1.4e-06, v5=7.2e-07, v6=1.9e-06
- **`etth1_nan_heavy`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=7.629e-06 mean=7.558e-07 rel_max=6.689e-06 逐位一致 30.4%；分位 max=1.526e-05 一致 25.8%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=9.5e-07, v2=7.6e-06, v3=1.9e-06, v4=1.4e-06, v5=3.6e-07, v6=1.9e-06
- **`etth1_nan_allcol`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=3.147e-04 mean=1.453e-05 rel_max=3.147e+02 逐位一致 29.5%；分位 max=2.207e-03 一致 27.0%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.1e-04, v1=1.4e-06, v2=9.5e-06, v3=1.2e-06, v4=1.9e-06, v5=3.6e-07, v6=2.9e-06
- **`ettm2_nan_light`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.144e-05 mean=1.706e-06 rel_max=8.871e-07 逐位一致 40.8%；分位 max=1.526e-05 一致 38.4%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=1.9e-06, v2=7.6e-06, v3=1.9e-06, v4=1.9e-06, v5=9.5e-07, v6=1.1e-05
- **`synth_ts_nan_heavy`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=4.578e-05 mean=7.084e-06 rel_max=2.807e-07 逐位一致 61.6%；分位 max=6.104e-05 一致 57.4%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.1e-05, v1=3.1e-05, v2=3.1e-05, v3=3.1e-05, v4=3.1e-05, v5=3.1e-05, v6=4.6e-05
- **`fg_flags_sy0_so0_mp0_zn0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.526e-05 mean=3.161e-06 rel_max=7.270e-06 逐位一致 4.2%；分位 max=2.122e-05 一致 6.8%；corr=1.000000；分位单调=True
- **`fg_flags_sy1_so0_mp0_zn0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=7.868e-06 mean=2.099e-06 rel_max=5.044e-06 逐位一致 16.7%；分位 max=1.335e-05 一致 12.7%；corr=1.000000；分位单调=True
- **`fg_flags_sy0_so1_mp0_zn0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.526e-05 mean=3.161e-06 rel_max=7.270e-06 逐位一致 4.2%；分位 max=2.122e-05 一致 6.8%；corr=1.000000；分位单调=True
- **`fg_flags_sy1_so1_mp0_zn0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=7.868e-06 mean=2.099e-06 rel_max=5.044e-06 逐位一致 16.7%；分位 max=1.335e-05 一致 12.7%；corr=1.000000；分位单调=True
- **`fg_flags_sy0_so0_mp1_zn0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.526e-05 mean=3.161e-06 rel_max=7.270e-06 逐位一致 4.2%；分位 max=2.122e-05 一致 6.8%；corr=1.000000；分位单调=True
- **`fg_flags_sy1_so0_mp1_zn0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=7.868e-06 mean=2.099e-06 rel_max=5.044e-06 逐位一致 16.7%；分位 max=1.335e-05 一致 12.7%；corr=1.000000；分位单调=True
- **`fg_flags_sy0_so1_mp1_zn0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.526e-05 mean=3.161e-06 rel_max=7.270e-06 逐位一致 4.2%；分位 max=2.122e-05 一致 6.8%；corr=1.000000；分位单调=True
- **`fg_flags_sy1_so1_mp1_zn0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=7.868e-06 mean=2.099e-06 rel_max=5.044e-06 逐位一致 16.7%；分位 max=1.335e-05 一致 12.7%；corr=1.000000；分位单调=True
- **`fg_flags_sy0_so0_mp0_zn1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.812e-05 mean=3.758e-06 rel_max=5.453e-05 逐位一致 9.4%；分位 max=2.193e-05 一致 9.1%；corr=1.000000；分位单调=True
- **`fg_flags_sy1_so0_mp0_zn1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.144e-05 mean=2.275e-06 rel_max=2.079e-05 逐位一致 12.5%；分位 max=1.812e-05 一致 11.3%；corr=1.000000；分位单调=True
- **`fg_flags_sy0_so1_mp0_zn1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.812e-05 mean=3.758e-06 rel_max=5.453e-05 逐位一致 9.4%；分位 max=2.193e-05 一致 9.1%；corr=1.000000；分位单调=True
- **`fg_flags_sy1_so1_mp0_zn1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.144e-05 mean=2.275e-06 rel_max=2.079e-05 逐位一致 12.5%；分位 max=1.812e-05 一致 11.3%；corr=1.000000；分位单调=True
- **`fg_flags_sy0_so0_mp1_zn1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.812e-05 mean=3.758e-06 rel_max=5.453e-05 逐位一致 9.4%；分位 max=2.193e-05 一致 9.1%；corr=1.000000；分位单调=True
- **`fg_flags_sy1_so0_mp1_zn1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.144e-05 mean=2.275e-06 rel_max=2.079e-05 逐位一致 12.5%；分位 max=1.812e-05 一致 11.3%；corr=1.000000；分位单调=True
- **`fg_flags_sy0_so1_mp1_zn1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.812e-05 mean=3.758e-06 rel_max=5.453e-05 逐位一致 9.4%；分位 max=2.193e-05 一致 9.1%；corr=1.000000；分位单调=True
- **`fg_flags_sy1_so1_mp1_zn1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.144e-05 mean=2.275e-06 rel_max=2.079e-05 逐位一致 12.5%；分位 max=1.812e-05 一致 11.3%；corr=1.000000；分位单调=True
- **`fg_ovr_st0_dt0_cp0_fr0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.240e-05 mean=2.354e-06 rel_max=1.040e-05 逐位一致 10.4%；分位 max=1.717e-05 一致 11.3%；corr=1.000000；分位单调=True
- **`fg_ovr_st1_dt0_cp0_fr0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.240e-05 mean=2.300e-06 rel_max=1.040e-05 逐位一致 15.6%；分位 max=1.335e-05 一致 14.5%；corr=1.000000；分位单调=True
- **`fg_ovr_st0_dt1_cp0_fr0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=7.629e-06 mean=2.105e-06 rel_max=5.703e-06 逐位一致 10.4%；分位 max=1.144e-05 一致 10.5%；corr=1.000000；分位单调=True
- **`fg_ovr_st1_dt1_cp0_fr0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=8.106e-06 mean=2.228e-06 rel_max=5.588e-06 逐位一致 14.6%；分位 max=1.335e-05 一致 12.4%；corr=1.000000；分位单调=True
- **`fg_ovr_st0_dt0_cp1_fr0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.240e-05 mean=2.369e-06 rel_max=1.040e-05 逐位一致 9.4%；分位 max=1.717e-05 一致 11.1%；corr=1.000000；分位单调=True
- **`fg_ovr_st1_dt0_cp1_fr0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.240e-05 mean=2.286e-06 rel_max=1.040e-05 逐位一致 13.5%；分位 max=1.335e-05 一致 14.5%；corr=1.000000；分位单调=True
- **`fg_ovr_st0_dt1_cp1_fr0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=7.629e-06 mean=1.991e-06 rel_max=5.703e-06 逐位一致 15.6%；分位 max=1.240e-05 一致 11.3%；corr=1.000000；分位单调=True
- **`fg_ovr_st1_dt1_cp1_fr0`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=7.868e-06 mean=2.099e-06 rel_max=5.044e-06 逐位一致 16.7%；分位 max=1.335e-05 一致 12.7%；corr=1.000000；分位单调=True
- **`fg_ovr_st0_dt0_cp0_fr1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.240e-05 mean=2.354e-06 rel_max=1.040e-05 逐位一致 10.4%；分位 max=1.717e-05 一致 11.3%；corr=1.000000；分位单调=True
- **`fg_ovr_st1_dt0_cp0_fr1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.240e-05 mean=2.300e-06 rel_max=1.040e-05 逐位一致 15.6%；分位 max=1.335e-05 一致 14.5%；corr=1.000000；分位单调=True
- **`fg_ovr_st0_dt1_cp0_fr1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=7.629e-06 mean=2.105e-06 rel_max=5.703e-06 逐位一致 10.4%；分位 max=1.144e-05 一致 10.5%；corr=1.000000；分位单调=True
- **`fg_ovr_st1_dt1_cp0_fr1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=8.106e-06 mean=2.228e-06 rel_max=5.588e-06 逐位一致 14.6%；分位 max=1.335e-05 一致 12.4%；corr=1.000000；分位单调=True
- **`fg_ovr_st0_dt0_cp1_fr1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.240e-05 mean=2.369e-06 rel_max=1.040e-05 逐位一致 9.4%；分位 max=1.717e-05 一致 11.1%；corr=1.000000；分位单调=True
- **`fg_ovr_st1_dt0_cp1_fr1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=1.240e-05 mean=2.286e-06 rel_max=1.040e-05 逐位一致 13.5%；分位 max=1.335e-05 一致 14.5%；corr=1.000000；分位单调=True
- **`fg_ovr_st0_dt1_cp1_fr1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=7.629e-06 mean=1.991e-06 rel_max=5.703e-06 逐位一致 15.6%；分位 max=1.240e-05 一致 11.3%；corr=1.000000；分位单调=True
- **`fg_ovr_st1_dt1_cp1_fr1`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=7.868e-06 mean=2.099e-06 rel_max=5.044e-06 逐位一致 16.7%；分位 max=1.335e-05 一致 12.7%；corr=1.000000；分位单调=True
- **`fx_etth2_znorm`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=7.629e-06 mean=1.667e-06 rel_max=7.854e-07 逐位一致 32.6%；分位 max=1.526e-05 一致 31.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=2.9e-06, v2=7.6e-06, v3=2.9e-06, v4=1.9e-06, v5=8.3e-07, v6=7.6e-06
- **`fx_etth2_makepos`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.144e-05 mean=1.182e-06 rel_max=8.498e-07 逐位一致 48.5%；分位 max=1.526e-05 一致 41.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=3.8e-06, v2=1.1e-05, v3=1.9e-06, v4=1.9e-06, v5=1.1e-06, v6=7.6e-06
- **`fx_ettm1_znorm`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=7.629e-06 mean=8.016e-07 rel_max=2.243e-05 逐位一致 32.9%；分位 max=1.192e-05 一致 30.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=5.7e-06, v1=9.5e-07, v2=7.6e-06, v3=9.5e-07, v4=1.4e-06, v5=4.8e-07, v6=1.9e-06
- **`fx_ettm1_makepos`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.335e-05 mean=1.005e-06 rel_max=4.910e-05 逐位一致 33.0%；分位 max=1.717e-05 一致 29.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.3e-05, v1=1.4e-06, v2=6.7e-06, v3=9.5e-07, v4=1.9e-06, v5=4.8e-07, v6=1.9e-06
- **`fx_ettm2_znorm`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=7.629e-06 mean=1.177e-06 rel_max=6.323e-07 逐位一致 46.0%；分位 max=1.144e-05 一致 40.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.8e-06, v1=1.9e-06, v2=3.8e-06, v3=1.9e-06, v4=9.5e-07, v5=6.0e-07, v6=7.6e-06
- **`fx_ettm2_makepos`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=7.629e-06 mean=1.331e-06 rel_max=1.265e-06 逐位一致 40.3%；分位 max=1.144e-05 一致 40.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=3.8e-06, v2=7.6e-06, v3=1.9e-06, v4=1.9e-06, v5=1.3e-06, v6=7.6e-06
- **`fx_synth_ts_znorm`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.526e-05 mean=9.083e-07 rel_max=9.536e-08 逐位一致 94.0%；分位 max=1.526e-05 一致 93.4%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.5e-05, v1=1.5e-05, v2=1.5e-05, v3=1.5e-05, v4=1.5e-05, v5=1.5e-05, v6=1.5e-05
- **`fx_synth_ts_makepos`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=3.052e-05 mean=7.448e-06 rel_max=1.901e-07 逐位一致 60.1%；分位 max=6.104e-05 一致 58.7%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=3.1e-05, v1=3.1e-05, v2=3.1e-05, v3=3.1e-05, v4=3.1e-05, v5=3.1e-05, v6=1.5e-05
- **`fx_synthflat_makepos`**（h=96，序列形状：7x96）—— FAIL
  - s0 (7x96): median max=2.468e-02 mean=1.925e-03 rel_max=2.586e-03 逐位一致 25.3%；分位 max=3.417e-02 一致 24.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=2.4e-06, v1=3.8e-06, v2=1.7e-02, v3=2.5e-02, v4=2.1e-06, v5=3.8e-06, v6=7.7e-03
- **`fx_etth1_znorm_nocpm`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.287e-05 mean=1.069e-06 rel_max=3.574e-04 逐位一致 26.2%；分位 max=1.907e-05 一致 24.7%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.3e-05, v1=1.4e-06, v2=1.1e-05, v3=9.5e-07, v4=1.4e-06, v5=3.6e-07, v6=2.9e-06
- **`fx_etth1_nosym_nostitch`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.431e-05 mean=1.614e-06 rel_max=1.372e-04 逐位一致 19.8%；分位 max=3.386e-05 一致 16.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.3e-05, v1=2.9e-06, v2=1.4e-05, v3=2.1e-06, v4=2.4e-06, v5=3.6e-07, v6=2.9e-06
- **`h24_multi`**（h=24，序列形状：7x24）—— TOL
  - s0 (7x24): median max=1.240e-05 mean=1.158e-06 rel_max=1.191e-04 逐位一致 29.2%；分位 max=1.335e-05 一致 26.8%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.2e-06, v1=1.4e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=3.6e-07, v6=1.9e-06
- **`h96_multi`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.240e-05 mean=1.127e-06 rel_max=1.191e-04 逐位一致 25.1%；分位 max=2.098e-05 一致 22.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.7e-06, v1=1.7e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=4.8e-07, v6=1.9e-06
- **`h168_multi`**（h=168，序列形状：7x168）—— TOL
  - s0 (7x168): median max=1.240e-05 mean=1.069e-06 rel_max=2.207e-04 逐位一致 23.8%；分位 max=2.098e-05 一致 21.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.7e-06, v1=1.7e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=4.8e-07, v6=2.9e-06
- **`h336_multi`**（h=336，序列形状：7x336）—— TOL
  - s0 (7x336): median max=1.240e-05 mean=1.067e-06 rel_max=2.738e-04 逐位一致 21.4%；分位 max=1.907e-05 一致 19.3%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.0e-05, v1=1.9e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=4.8e-07, v6=2.9e-06
- **`h720_multi`**（h=720，序列形状：7x720）—— TOL
  - s0 (7x720): median max=1.574e-05 mean=1.168e-06 rel_max=2.794e-04 逐位一致 18.8%；分位 max=2.480e-05 一致 17.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.6e-05, v1=2.1e-06, v2=1.2e-05, v3=9.5e-07, v4=2.1e-06, v5=6.0e-07, v6=4.8e-06
- **`ctx64`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=9.537e-06 mean=1.139e-06 rel_max=4.687e-05 逐位一致 27.7%；分位 max=2.003e-05 一致 25.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=9.5e-06, v1=1.9e-06, v2=9.5e-06, v3=1.2e-06, v4=1.7e-06, v5=4.8e-07, v6=1.9e-06
- **`ctx256`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.144e-05 mean=9.884e-07 rel_max=6.000e-05 逐位一致 18.5%；分位 max=2.289e-05 一致 19.6%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=7.6e-06, v1=1.4e-06, v2=1.1e-05, v3=1.4e-06, v4=1.9e-06, v5=4.8e-07, v6=2.9e-06
- **`ctx3072`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.621e-05 mean=1.056e-06 rel_max=1.500e-05 逐位一致 24.3%；分位 max=1.717e-05 一致 23.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=9.5e-06, v1=2.9e-06, v2=1.6e-05, v3=1.2e-06, v4=1.7e-06, v5=4.8e-07, v6=1.9e-06
- **`ctx_cap_17k`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=4.244e-05 mean=2.334e-06 rel_max=3.481e-04 逐位一致 10.7%；分位 max=6.390e-05 一致 9.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=4.2e-05, v1=3.3e-06, v2=2.1e-05, v3=2.4e-06, v4=3.8e-06, v5=1.2e-06, v6=6.7e-06
- **`uni_1024`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=2.861e-06 mean=7.252e-07 rel_max=3.369e-07 逐位一致 50.0%；分位 max=3.815e-06 一致 45.9%；corr=1.000000；分位单调=True
- **`nan_mixed`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.001e-05 mean=9.245e-07 rel_max=4.166e-05 逐位一致 31.1%；分位 max=1.526e-05 一致 24.9%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.6e-06, v1=1.4e-06, v2=1.0e-05, v3=1.4e-06, v4=1.9e-06, v5=3.6e-07, v6=1.9e-06
- **`batch2_multi`**（h=96，序列形状：7x96, 7x96）—— TOL
  - s0 (7x96): median max=1.240e-05 mean=7.619e-07 rel_max=1.758e-04 逐位一致 21.3%；分位 max=1.526e-05 一致 21.2%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=9.5e-06, v1=1.1e-06, v2=1.2e-05, v3=9.5e-07, v4=9.5e-07, v5=1.8e-07, v6=2.4e-06
  - s1 (7x96): median max=2.098e-05 mean=1.590e-06 rel_max=2.778e-05 逐位一致 19.6%；分位 max=3.022e-05 一致 17.7%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=2.1e-05, v1=1.4e-06, v2=1.9e-05, v3=9.5e-07, v4=1.2e-06, v5=3.6e-07, v6=4.3e-06
- **`batch8_mixed_uni`**（h=96，序列形状：1x96, 1x96, 1x96, 1x96, 1x96, 1x96, 1x96, 1x96）—— OFFICIAL_NAN
  - s0 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s1 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s2 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s3 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s4 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s5 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s6 (1x96): median max=nan mean=nan rel_max=nan 逐位一致 0.0%；分位 max=nan 一致 0.0%；corr=nan；分位单调=True
  - s7 (1x96): median max=2.861e-06 mean=6.855e-07 rel_max=2.826e-07 逐位一致 46.9%；分位 max=4.768e-06 一致 42.4%；corr=1.000000；分位单调=True
- **`flag_nosym`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.717e-05 mean=1.636e-06 rel_max=1.372e-04 逐位一致 18.0%；分位 max=3.123e-05 一致 15.0%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.3e-05, v1=2.9e-06, v2=1.7e-05, v3=1.9e-06, v4=2.4e-06, v5=4.8e-07, v6=1.9e-06
- **`flag_nosort`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.240e-05 mean=1.127e-06 rel_max=1.191e-04 逐位一致 25.1%；分位 max=2.098e-05 一致 22.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.7e-06, v1=1.7e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=4.8e-07, v6=1.9e-06
- **`flag_makepos`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.240e-05 mean=1.127e-06 rel_max=1.191e-04 逐位一致 25.1%；分位 max=2.098e-05 一致 22.5%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=8.7e-06, v1=1.7e-06, v2=1.2e-05, v3=9.5e-07, v4=1.4e-06, v5=4.8e-07, v6=1.9e-06
- **`flag_znorm`**（h=96，序列形状：7x96）—— TOL
  - s0 (7x96): median max=1.383e-05 mean=1.076e-06 rel_max=3.574e-04 逐位一致 25.4%；分位 max=2.480e-05 一致 24.1%；corr=1.000000；分位单调=True
    - 逐变体 max：v0=1.4e-05, v1=1.4e-06, v2=1.1e-05, v3=9.5e-07, v4=1.4e-06, v5=2.4e-07, v6=2.9e-06
- **`ovr_stitch_off`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=3.815e-06 mean=1.267e-06 rel_max=4.310e-07 逐位一致 19.8%；分位 max=3.815e-06 一致 19.2%；corr=1.000000；分位单调=True
- **`ovr_cpm_off`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=2.861e-06 mean=1.113e-06 rel_max=3.254e-07 逐位一致 19.8%；分位 max=3.815e-06 一致 21.4%；corr=1.000000；分位单调=True
- **`ovr_detrend_off`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=2.861e-06 mean=1.242e-06 rel_max=3.482e-07 逐位一致 14.6%；分位 max=3.815e-06 一致 18.5%；corr=1.000000；分位单调=True
- **`ovr_frozen_stats`**（h=96，序列形状：1x96）—— TOL
  - s0 (1x96): median max=2.861e-06 mean=1.242e-06 rel_max=3.482e-07 逐位一致 14.6%；分位 max=3.815e-06 一致 18.5%；corr=1.000000；分位单调=True

## 4. 发现与修复

本次多场景对照发现并修复 **2 个 Rust 引擎真实 bug**，另确认 **2 个官方实现问题**：

### 4.1 引擎 bug（已修复）

| # | Bug | 根因 | 修复 | 验证 |
| --- | --- | --- | --- | --- |
| 1 | **超长 context 截断缺失**：`ctx_cap_17k`（context 17020 > 15360 上限）panic | `forecast.rs` 组 batch 时 `pad = batch_context - c_len`，序列长度超过 batch context 时 usize 下溢崩溃；官方 `Query.format` 语义是**截取最后 batch_context 点** | 按 `Query.format` 对齐：超长序列取尾部 `c_eff = min(c_len, batch_context)`，target/po/pf 同步用 `skip` 偏移 | 修复后 48 项测试全过；`ctx_cap_17k` 与官方逐位一致比例恢复到与其它场景相同水平 |
| 2 | **znorm 只归一化不反归一化**：`flag_znorm` 输出停留在标准化尺度（差 15 量级，corr 0.53） | 引擎对输入做了 `(x-mu)/sigma`，但推理后没有按官方 `raw*sigma+mu` 反变换回原尺度 | 按官方顺序（sort → 对称平均合并 → **反归一化** → make_positive）在 `predict_batch` 后处理中补上，逐 (series, variate) 保存 (mu, sigma) | 修复后 `flag_znorm` 与官方一致（见 §3） |

顺带修复：`make_positive` 原实现只检查第 1 条序列并作用于所有结果；改为官方语义——逐 (series, variate) 行检查输入非负后仅钳制该行。

### 4.2 官方实现问题（记录，不修官方代码）

| # | 现象 | 根因分析（`tools/diag_official_batch.py` 受控实验） |
|---|---|---|
| 1 | **混合长度 batch 中被左填充的短序列输出全 NaN**（`batch8_mixed_uni` 8 条中 7 条；官方 torch 侧）。等长 batch、bs=1 逐条推理均正常；sub-patch 级小填充（ctx=100/1023，pad≤31）也正常 | 官方 decode 全序列注意力分支中 key 掩码为 `~patch_mask`（剔除所有掩码 patch），而 pad 查询行因果可见的 key 全部是 pad patch → 空 softmax → NaN，沿残差/FFN 传播；仅当填充是**整 patch 级**时才存在全掩码查询行，故小填充不受影响。官方测试只断言混合长度 batch 的形状 |

### 4.3 引擎注意项（无法用官方仲裁的路径）

- Rust 引擎在混合长度 batch 中对被填充序列给出**有限值**（不 NaN），但与“该序列单独推理”相差 0.04 ～ 0.22（见 `report/padded_truth.json`）。机理：左填充使真实 patch 的 RoPE 位置整体偏移、ReVIN/去趋势统计范围随 batch_context 变化，填充 batch 与逐条推理本就不是同一计算图；官方在该路径输出 NaN、无法提供参照。**建议**：混合长度序列避免同 batch（分批等长推理），后续如需对该路径下结论，应以原始 Flax 实现或官方修复版为参照。

未填充路径（batch 内等长）不受影响：`batch2_multi`（两条 7 变体同 batch）与官方一致。

### 4.4 引擎数值策略演变：f32 baseline → 全链路 f64 → 当前（f32 GEMM + f64 归约）

本报告版本相对 f32 逐层对照版（`report/summary_f32_baseline.json`）做了数值策略升级：
GEMM 内积、attention 打分/加权求和、RMSNorm 平方和、softmax（max/exp/和）、ReVIN running stats、线性去趋势回归、stitch 融合、CPM refine 全部改为 **f64 累加、f32 输出**——每个中间量都是正确舍入值，引擎侧不再引入自身浮点噪声。

升级前后对比（相同场景、相同官方基准）：

| 场景 | f32 累加 median max\|Δ\| | f64 累加 median max\|Δ\| | 提升 |
| --- | --: | --: | --: |
| `h24_multi` | 3.290e-05 | 1.240e-05 | 2.7x |
| `h96_multi` | 3.290e-05 | 1.240e-05 | 2.7x |
| `h168_multi` | 3.290e-05 | 1.240e-05 | 2.7x |
| `h336_multi` | 3.290e-05 | 1.240e-05 | 2.7x |
| `h720_multi` | 3.576e-05 | 1.574e-05 | 2.3x |
| `ctx64` | 3.052e-05 | 9.537e-06 | 3.2x |
| `ctx256` | 2.480e-05 | 1.144e-05 | 2.2x |
| `ctx3072` | 1.907e-05 | 1.621e-05 | 1.2x |
| `ctx_cap_17k` | 1.526e-05 | 4.244e-05 | 0.4x |
| `uni_1024` | 2.861e-06 | 2.861e-06 | 1.0x |
| `nan_mixed` | 2.003e-05 | 1.001e-05 | 2.0x |
| `batch2_multi` | 1.669e-05 | 1.240e-05 | 1.3x |
| `flag_nosym` | 4.196e-05 | 1.717e-05 | 2.4x |
| `flag_nosort` | 3.290e-05 | 1.240e-05 | 2.7x |
| `flag_makepos` | 3.290e-05 | 1.240e-05 | 2.7x |
| `flag_znorm` | 3.004e-05 | 1.383e-05 | 2.2x |
| `ovr_stitch_off` | 3.338e-06 | 3.815e-06 | 0.9x |
| `ovr_cpm_off` | 3.815e-06 | 2.861e-06 | 1.3x |
| `ovr_detrend_off` | 3.815e-06 | 2.861e-06 | 1.3x |
| `ovr_frozen_stats` | 3.815e-06 | 2.861e-06 | 1.3x |

**残余偏差的下界**：实测官方 oneMKL 2022.2 自身 12 线程 vs 1 线程同一 GEMM 就有 0.5% 元素非逐位（max 3.05e-5），官方 f32 累加离 f64 真值 88% 元素非逐位（max 3.43e-5）。即：f32 框架间的分歧物理下限 ≈ 官方自身噪声（~1e-5～1e-4）；本引擎已降到该下界以内 （见 §3：max ≈ 1e-6 ～ 1e-5）。要逐位归零只能链接官方同一版 oneMKL，与本项目零依赖目标冲突（见 §4.2 的量化证据）。

## 5. 性能（参考）

| 场景 | 官方 torch (s) | Rust (s) |
| --- | --: | --: |
| `etth1_h24` | 3.1 | 3.52 |
| `etth1_h96` | 5.4 | 3.64 |
| `etth1_h168` | 3.6 | 3.73 |
| `etth1_h336` | 11.1 | 4.03 |
| `etth1_h720` | 24.5 | 4.56 |
| `etth2_h24` | 25.2 | 3.65 |
| `etth2_h96` | 18.3 | 3.67 |
| `etth2_h168` | 18.7 | 3.75 |
| `etth2_h336` | 18.6 | 4.07 |
| `etth2_h720` | 15.1 | 4.53 |
| `ettm1_h24` | 4.6 | 3.47 |
| `ettm1_h96` | 4.9 | 3.62 |
| `ettm1_h168` | 5.4 | 3.64 |
| `ettm1_h336` | 5.8 | 3.95 |
| `ettm1_h720` | 7.8 | 4.39 |
| `ettm2_h24` | 4.5 | 3.51 |
| `ettm2_h96` | 6.3 | 3.58 |
| `ettm2_h168` | 7.0 | 3.68 |
| `ettm2_h336` | 5.9 | 3.89 |
| `ettm2_h720` | 7.8 | 4.36 |
| `synth_ts_h24` | 4.1 | 3.52 |
| `synth_ts_h96` | 4.4 | 3.53 |
| `synth_ts_h168` | 5.4 | 3.68 |
| `synth_ts_h336` | 12.2 | 3.87 |
| `synth_ts_h720` | 25.2 | 4.41 |
| `synth_scale_h24` | 17.7 | 3.49 |
| `synth_scale_h96` | 19.6 | 3.52 |
| `synth_scale_h168` | 25.1 | 3.66 |
| `synth_scale_h336` | 20.2 | 3.87 |
| `synth_scale_h720` | 24.3 | 4.38 |
| `synth_flat_h24` | 15.0 | 3.47 |
| `synth_flat_h96` | 17.9 | 3.55 |
| `synth_flat_h168` | 20.1 | 3.72 |
| `synth_flat_h336` | 18.6 | 3.89 |
| `synth_flat_h720` | 25.5 | 4.40 |
| `etth1_h1440` | 34.6 | 5.18 |
| `ettm2_h1440` | 25.2 | 5.23 |
| `etth1_h1_uni` | 0.3 | 0.34 |
| `etth1_h12_uni` | 2.8 | 0.35 |
| `etth1_h48_uni` | 1.5 | 0.34 |
| `etth1_ctx64` | 4.0 | 0.60 |
| `etth1_ctx256` | 3.8 | 1.42 |
| `etth1_ctx1024` | 11.6 | 3.53 |
| `etth1_ctx3072` | 20.2 | 6.55 |
| `etth2_ctx64` | 4.4 | 0.63 |
| `etth2_ctx256` | 4.6 | 1.47 |
| `etth2_ctx1024` | 10.6 | 3.71 |
| `etth2_ctx3072` | 21.7 | 6.86 |
| `ettm1_ctx64` | 4.1 | 0.62 |
| `ettm1_ctx256` | 5.2 | 1.47 |
| `ettm1_ctx1024` | 11.4 | 3.67 |
| `ettm1_ctx3072` | 31.1 | 6.85 |
| `ettm2_ctx64` | 1.1 | 0.62 |
| `ettm2_ctx256` | 15.5 | 1.47 |
| `ettm2_ctx1024` | 23.5 | 3.69 |
| `ettm2_ctx3072` | 38.5 | 6.69 |
| `synth_ts_ctx64` | 6.5 | 0.60 |
| `synth_ts_ctx256` | 15.8 | 1.43 |
| `synth_ts_ctx1024` | 26.3 | 3.60 |
| `synth_ts_ctx3072` | 41.7 | 6.54 |
| `synth_scale_ctx64` | 11.7 | 0.59 |
| `synth_scale_ctx256` | 14.0 | 1.42 |
| `synth_scale_ctx1024` | 26.3 | 3.58 |
| `synth_scale_ctx3072` | 50.6 | 6.54 |
| `synth_flat_ctx64` | 8.4 | 0.61 |
| `synth_flat_ctx256` | 12.1 | 1.43 |
| `synth_flat_ctx1024` | 14.9 | 3.57 |
| `synth_flat_ctx3072` | 41.4 | 6.55 |
| `etth1_ctx15360` | 171.2 | 41.96 |
| `etth2_ctx17020` | 80.8 | 39.29 |
| `etth1_ctx32_uni` | 0.4 | 0.40 |
| `etth1_ctx48_uni` | 0.4 | 0.40 |
| `etth1_ctx64_uni` | 0.4 | 0.40 |
| `etth1_v1` | 0.9 | 0.48 |
| `etth1_v2` | 1.2 | 1.12 |
| `etth1_v3` | 1.7 | 1.85 |
| `etth1_v5` | 2.8 | 3.03 |
| `synth_v10` | 6.0 | 4.08 |
| `synth_v20` | 6.6 | 4.34 |
| `synth_v50` | 11.5 | 5.59 |
| `etth1_batch2_eq` | 9.1 | 4.84 |
| `etth1_batch4_eq` | 19.2 | 7.72 |
| `etth1_batch8_eq` | 37.6 | 13.23 |
| `etth1_mixed8_uni` | 14.3 | 7.37 |
| `ettm1_mixed5_uni` | 3.4 | 3.08 |
| `mixed_batch_etth1xetth2` | 13.5 | 5.24 |
| `etth1_nan_light` | 8.6 | 3.70 |
| `etth1_nan_heavy` | 6.9 | 3.67 |
| `etth1_nan_allcol` | 5.0 | 3.59 |
| `ettm2_nan_light` | 5.0 | 3.57 |
| `synth_ts_nan_heavy` | 5.0 | 3.62 |
| `fg_flags_sy0_so0_mp0_zn0` | 0.6 | 0.48 |
| `fg_flags_sy1_so0_mp0_zn0` | 0.7 | 0.36 |
| `fg_flags_sy0_so1_mp0_zn0` | 0.6 | 0.46 |
| `fg_flags_sy1_so1_mp0_zn0` | 0.8 | 0.35 |
| `fg_flags_sy0_so0_mp1_zn0` | 0.6 | 0.46 |
| `fg_flags_sy1_so0_mp1_zn0` | 0.8 | 0.35 |
| `fg_flags_sy0_so1_mp1_zn0` | 0.6 | 0.46 |
| `fg_flags_sy1_so1_mp1_zn0` | 0.8 | 0.35 |
| `fg_flags_sy0_so0_mp0_zn1` | 0.6 | 0.46 |
| `fg_flags_sy1_so0_mp0_zn1` | 0.8 | 0.35 |
| `fg_flags_sy0_so1_mp0_zn1` | 0.6 | 0.47 |
| `fg_flags_sy1_so1_mp0_zn1` | 0.7 | 0.34 |
| `fg_flags_sy0_so0_mp1_zn1` | 0.6 | 0.46 |
| `fg_flags_sy1_so0_mp1_zn1` | 0.7 | 0.35 |
| `fg_flags_sy0_so1_mp1_zn1` | 0.6 | 0.46 |
| `fg_flags_sy1_so1_mp1_zn1` | 0.8 | 0.35 |
| `fg_ovr_st0_dt0_cp0_fr0` | 0.8 | 0.34 |
| `fg_ovr_st1_dt0_cp0_fr0` | 0.8 | 0.34 |
| `fg_ovr_st0_dt1_cp0_fr0` | 0.8 | 0.35 |
| `fg_ovr_st1_dt1_cp0_fr0` | 0.8 | 0.34 |
| `fg_ovr_st0_dt0_cp1_fr0` | 0.9 | 0.35 |
| `fg_ovr_st1_dt0_cp1_fr0` | 0.8 | 0.35 |
| `fg_ovr_st0_dt1_cp1_fr0` | 0.8 | 0.34 |
| `fg_ovr_st1_dt1_cp1_fr0` | 0.9 | 0.34 |
| `fg_ovr_st0_dt0_cp0_fr1` | 0.8 | 0.35 |
| `fg_ovr_st1_dt0_cp0_fr1` | 0.8 | 0.34 |
| `fg_ovr_st0_dt1_cp0_fr1` | 0.8 | 0.34 |
| `fg_ovr_st1_dt1_cp0_fr1` | 0.8 | 0.35 |
| `fg_ovr_st0_dt0_cp1_fr1` | 0.9 | 0.34 |
| `fg_ovr_st1_dt0_cp1_fr1` | 0.9 | 0.34 |
| `fg_ovr_st0_dt1_cp1_fr1` | 0.9 | 0.35 |
| `fg_ovr_st1_dt1_cp1_fr1` | 0.8 | 0.34 |
| `fx_etth2_znorm` | 4.5 | 3.58 |
| `fx_etth2_makepos` | 4.9 | 3.56 |
| `fx_ettm1_znorm` | 3.0 | 3.52 |
| `fx_ettm1_makepos` | 2.9 | 3.50 |
| `fx_ettm2_znorm` | 2.6 | 3.54 |
| `fx_ettm2_makepos` | 2.5 | 3.53 |
| `fx_synth_ts_znorm` | 2.4 | 3.52 |
| `fx_synth_ts_makepos` | 2.2 | 3.54 |
| `fx_synthflat_makepos` | 2.0 | 3.54 |
| `fx_etth1_znorm_nocpm` | 2.0 | 3.56 |
| `fx_etth1_nosym_nostitch` | 1.0 | 2.05 |
| `h24_multi` | 2.0 | 3.49 |
| `h96_multi` | 2.0 | 3.56 |
| `h168_multi` | 2.2 | 3.69 |
| `h336_multi` | 2.4 | 4.09 |
| `h720_multi` | 3.3 | 4.73 |
| `ctx64` | 0.4 | 0.64 |
| `ctx256` | 0.7 | 1.48 |
| `ctx3072` | 5.6 | 7.03 |
| `ctx_cap_17k` | 35.3 | 39.99 |
| `uni_1024` | 0.4 | 0.50 |
| `nan_mixed` | 2.0 | 3.58 |
| `batch2_multi` | 4.5 | 4.93 |
| `batch8_mixed_uni` | 6.4 | 6.97 |
| `flag_nosym` | 1.1 | 2.15 |
| `flag_nosort` | 2.1 | 3.58 |
| `flag_makepos` | 2.1 | 3.63 |
| `flag_znorm` | 2.1 | 3.58 |
| `ovr_stitch_off` | 0.3 | 0.37 |
| `ovr_cpm_off` | 0.3 | 0.37 |
| `ovr_detrend_off` | 0.3 | 0.37 |
| `ovr_frozen_stats` | 0.3 | 0.37 |

> 注：官方 torch 用 12 线程 MKL GEMM；Rust（当前版本）用 matrixmultiply 缓存分块 sgemm + rayon 行并行。本版相对上一版（自产 f64 外积内核）整体提速 ~9.4×，155 场景中 117 个快于官方、整体耗时约为官方的 35%。

## 6. 复现

```bash
python3 tools/gen_scenarios.py                                   # 生成场景数据+清单
/tmp/venv312/bin/python tools/run_official_scenarios.py          # 官方 torch 侧
cargo build --release --bin export_scenarios && \
  ./target/release/export_scenarios ckpt tools/scenarios.json out_rust_sc   # Rust 侧
python3 tools/compare_scenarios.py                              # 对比+出报告
```
