use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::State;
use timesfm3::forecast::ForecastOptions;

use crate::csv_helper::{self, CsvInspectionResult};
use crate::state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub name: String,
    pub path: String,
    pub precision: String,
    pub size_desc: String,
    pub is_recommended: bool,
    pub exists: bool,
}

#[tauri::command]
pub fn list_checkpoints() -> Result<Vec<CheckpointMeta>, String> {
    let candidates = [
        ("均衡模式 (Balanced · 推荐)", "publish/timesfm3.0-balanced", "均衡高精", "约 730 MB · 推荐首选，在精度与运行速度间达到最佳平衡", true),
        ("极速轻量模式", "publish/timesfm3.0-f16", "极速轻量", "约 631 MB · 内存占用低，适合轻量设备或快速批量推断", false),
        ("高保真原版模式", "ckpt", "完整原版", "约 1.32 GB · 官方原版全量精度", false),
    ];

    let root = find_project_root();
    let mut results = Vec::new();

    for (name, rel_path, precision, size_desc, is_rec) in candidates {
        let abs_path = root.join(rel_path);
        let exists = abs_path.join("model.safetensors").exists();
        results.push(CheckpointMeta {
            name: name.into(),
            path: abs_path.to_string_lossy().to_string(),
            precision: precision.into(),
            size_desc: size_desc.into(),
            is_recommended: is_rec,
            exists,
        });
    }

    // Also check standard user cache directory: ~/.cache/timesfm3
    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        let home_cache = PathBuf::from(home).join(".cache").join("timesfm3");
        if home_cache.join("model.safetensors").exists() {
            let meta = std::fs::metadata(home_cache.join("model.safetensors")).ok();
            let size_str = meta.map(|m| format!("{:.1} MB", m.len() as f64 / 1048576.0)).unwrap_or_else(|| "本地缓存".into());
            results.push(CheckpointMeta {
                name: "系统缓存模型 (~/.cache/timesfm3)".into(),
                path: home_cache.to_string_lossy().to_string(),
                precision: "自动识别".into(),
                size_desc: format!("{size_str} · 系统缓存目录"),
                is_recommended: false,
                exists: true,
            });
        }
    }

    Ok(results)
}

#[tauri::command]
pub fn inspect_model_path(raw_path: String) -> Result<CheckpointMeta, String> {
    let raw = raw_path.trim();
    if raw.is_empty() {
        return Err("模型路径不能为空".to_string());
    }

    let p = PathBuf::from(raw);
    let dir = if p.is_file() {
        p.parent().unwrap_or(&p).to_path_buf()
    } else {
        p
    };

    let st_path = dir.join("model.safetensors");
    if !st_path.exists() {
        return Err(format!(
            "在指定路径 [{}] 下未找到 model.safetensors 权重文件，请确认所选目录完整",
            dir.display()
        ));
    }

    let meta = std::fs::metadata(&st_path)
        .map_err(|e| format!("读取模型文件元数据失败: {e}"))?;
    let size_bytes = meta.len();
    let size_mb = size_bytes as f64 / (1024.0 * 1024.0);
    let size_str = if size_mb >= 1024.0 {
        format!("{:.2} GB", size_mb / 1024.0)
    } else {
        format!("{:.1} MB", size_mb)
    };

    let cfg_path = dir.join("config.json");
    let mut precision_name = "自动推断精度".to_string();
    let mut desc = format!("权重大小: {size_str}");

    if cfg_path.exists() {
        if let Ok(cfg_bytes) = std::fs::read(&cfg_path) {
            if let Ok(json_str) = String::from_utf8(cfg_bytes) {
                let lower = json_str.to_lowercase();
                if lower.contains("balanced") {
                    precision_name = "均衡高精 (Balanced)".into();
                    desc = format!("{size_str} · 推荐模式");
                } else if lower.contains("f16") || lower.contains("fp16") {
                    precision_name = "FP16 半精度".into();
                    desc = format!("{size_str} · 极速轻量");
                } else if lower.contains("f32") || lower.contains("fp32") {
                    precision_name = "FP32 全精度".into();
                    desc = format!("{size_str} · 高保真原版");
                }
            }
        }
    } else {
        if size_mb < 700.0 {
            precision_name = "FP16 轻量版".into();
        } else if size_mb < 900.0 {
            precision_name = "Balanced 均衡版".into();
        } else {
            precision_name = "FP32 原版完整权重".into();
        }
    }

    let folder_name = dir.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "自定义模型".to_string());

    Ok(CheckpointMeta {
        name: format!("自定义模型 ({folder_name})"),
        path: dir.to_string_lossy().to_string(),
        precision: precision_name,
        size_desc: desc,
        is_recommended: true,
        exists: true,
    })
}

#[tauri::command]
pub async fn pick_model_directory() -> Result<Option<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        #[cfg(target_os = "macos")]
        {
            let script = r#"
                try
                    set chosenItem to choose folder with prompt "请选择包含 model.safetensors 的 TimesFM 3.0 模型目录"
                    return POSIX path of chosenItem
                on error
                    try
                        set chosenFile to choose file with prompt "或直接选择 model.safetensors 权重文件" of type {"safetensors", "bin"}
                        return POSIX path of chosenFile
                    on error
                        return ""
                    end try
                end try
            "#;
            let output = std::process::Command::new("osascript")
                .arg("-e")
                .arg(script)
                .output()
                .map_err(|e| format!("调用系统目录选择器失败: {e}"))?;

            if output.status.success() {
                let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path_str.is_empty() {
                    let p = Path::new(&path_str);
                    if p.is_file() {
                        if let Some(parent) = p.parent() {
                            return Ok(Some(parent.to_string_lossy().to_string()));
                        }
                    }
                    return Ok(Some(path_str));
                }
            }
            Ok(None)
        }
        #[cfg(target_os = "windows")]
        {
            let script = r#"
                Add-Type -AssemblyName System.Windows.Forms
                $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
                $dialog.Description = "请选择 TimesFM 3.0 模型权重目录 (包含 model.safetensors)"
                if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
                    Write-Output $dialog.SelectedPath
                }
            "#;
            let output = std::process::Command::new("powershell")
                .args(["-NoProfile", "-Command", script])
                .output()
                .map_err(|e| format!("调用 Windows 目录选择器失败: {e}"))?;
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(Some(path));
                }
            }
            Ok(None)
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            Ok(None)
        }
    })
    .await
    .map_err(|e| format!("选择模型目录任务异常: {e}"))?
}

#[tauri::command]
pub async fn inspect_csv(file_path: String) -> Result<CsvInspectionResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        csv_helper::inspect_csv(&file_path)
    })
    .await
    .map_err(|e| format!("解析 CSV 任务异常: {e}"))?
}

#[tauri::command]
pub async fn read_file_binary(file_path: String) -> Result<Vec<u8>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let p = PathBuf::from(&file_path);
        if !p.exists() {
            return Err(format!("文件不存在: {file_path}"));
        }
        std::fs::read(&p).map_err(|e| format!("读取文件字节失败: {e}"))
    })
    .await
    .map_err(|e| format!("读取文件任务异常: {e}"))?
}

#[tauri::command]
pub async fn read_file_text(file_path: String) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let p = PathBuf::from(&file_path);
        if !p.exists() {
            return Err(format!("文件不存在: {file_path}"));
        }
        std::fs::read_to_string(&p).map_err(|e| format!("读取文件文本内容失败: {e}"))
    })
    .await
    .map_err(|e| format!("读取文件任务异常: {e}"))?
}

#[tauri::command]
pub async fn pick_csv_file() -> Result<Option<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        #[cfg(target_os = "macos")]
        {
            let script = r#"
                try
                    set chosenFile to choose file with prompt "请选择要预测的时序数据文件 (CSV/Excel/TSV/TXT/JSON)" of type {"csv", "tsv", "txt", "xlsx", "xls", "json", "public.comma-separated-values-text", "public.plain-text", "public.json", "org.openxmlformats.spreadsheetml.sheet", "com.microsoft.excel.xls"}
                    return POSIX path of chosenFile
                on error
                    return ""
                end try
            "#;
            let output = std::process::Command::new("osascript")
                .arg("-e")
                .arg(script)
                .output()
                .map_err(|e| format!("调用系统文件选择器失败: {e}"))?;

            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(Some(path));
                }
            }
            Ok(None)
        }
        #[cfg(target_os = "windows")]
        {
            let script = r#"
                Add-Type -AssemblyName System.Windows.Forms
                $dialog = New-Object System.Windows.Forms.OpenFileDialog
                $dialog.Title = "请选择要预测的时序数据文件"
                $dialog.Filter = "时序数据文件 (*.csv;*.xlsx;*.xls;*.tsv;*.txt;*.json)|*.csv;*.xlsx;*.xls;*.tsv;*.txt;*.json|所有文件 (*.*)|*.*"
                if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
                    Write-Output $dialog.FileName
                }
            "#;
            let output = std::process::Command::new("powershell")
                .args(["-NoProfile", "-Command", script])
                .output()
                .map_err(|e| format!("调用 Windows 文件选择器失败: {e}"))?;
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(Some(path));
                }
            }
            Ok(None)
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            Ok(None)
        }
    })
    .await
    .map_err(|e| format!("选择文件任务异常: {e}"))?
}

#[tauri::command]
pub async fn upload_csv_content(file_name: String, content: String) -> Result<CsvInspectionResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let temp_dir = std::env::temp_dir().join("timesfm3_uploads");
        std::fs::create_dir_all(&temp_dir).map_err(|e| format!("创建上传临时目录失败: {e}"))?;

        let sanitized_name = std::path::Path::new(&file_name)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("uploaded_timeseries.csv");

        let temp_path = temp_dir.join(sanitized_name);
        std::fs::write(&temp_path, content.as_bytes())
            .map_err(|e| format!("保存上传数据文件失败: {e}"))?;

        csv_helper::inspect_csv(&temp_path)
    })
    .await
    .map_err(|e| format!("缓存上传文件任务异常: {e}"))?
}

#[tauri::command]
pub async fn load_sample_dataset(sample_key: String) -> Result<CsvInspectionResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = find_project_root();
        let rel = match sample_key.as_str() {
            "etth1" => "data/etth1.csv",
            "ettm1" => "data/ETTm1.csv",
            "etth2" => "data/ETTh2.csv",
            _ => "data/etth1.csv",
        };
        let path = root.join(rel);
        if !path.exists() {
            return Err(format!("示例数据文件不存在: {}", path.display()));
        }
        csv_helper::inspect_csv(&path)
    })
    .await
    .map_err(|e| format!("载入示例数据任务异常: {e}"))?
}

#[derive(Debug, Clone, Deserialize)]
pub struct ForecastRequest {
    pub file_path: String,
    pub target_cols: Vec<String>,
    pub date_col: Option<String>,
    pub horizon: usize,
    pub seasonal_period: Option<usize>,
    pub checkpoint_path: String,
    pub make_positive: bool,
    pub use_symmetric_averaging: bool,
    pub use_boundary_smoothing: bool,
    pub use_robust_znorm: bool,
    pub is_transposed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetForecastResult {
    pub column_name: String,
    pub history_timestamps: Vec<String>,
    pub history_values: Vec<f32>,
    pub forecast_timestamps: Vec<String>,
    pub median: Vec<f32>,
    pub q10: Vec<f32>,
    pub q20: Vec<f32>,
    pub q80: Vec<f32>,
    pub q90: Vec<f32>,
    pub mean_forecast: f32,
    pub min_forecast: f32,
    pub max_forecast: f32,
    pub trend: String, // "上升" / "平稳" / "下降"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResponse {
    pub series: Vec<TargetForecastResult>,
    pub horizon: usize,
    pub elapsed_ms: u128,
    pub checkpoint_used: String,
}

#[tauri::command]
pub async fn run_forecast(
    state: State<'_, AppState>,
    request: ForecastRequest,
) -> Result<ForecastResponse, String> {
    let state_inner = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        execute_forecast(&state_inner, request)
    })
    .await
    .map_err(|e| format!("异步执行预测失败: {e}"))?
}

pub fn execute_forecast(
    state: &AppState,
    request: ForecastRequest,
) -> Result<ForecastResponse, String> {
    let t0 = Instant::now();

    // 1. Prepare data
    let (flat_context, (num_vars, ctx_len), timestamps, step_duration) =
        csv_helper::load_series_for_inference(
            &request.file_path,
            request.date_col.as_deref(),
            &request.target_cols,
            request.is_transposed,
        )?;

    // 2. Load model from state (cached if already loaded)
    let model = state.get_or_load_model(&request.checkpoint_path, None)?;

    // 3. Configure ForecastOptions
    let opts = ForecastOptions {
        return_quantiles: true,
        use_symmetric_averaging: request.use_symmetric_averaging,
        make_positive: request.make_positive,
        sort_quantiles: true,
        use_znorm: false,
        per_core_batch_size: 32,
        use_bucket_batching: true,
        use_robust_znorm: request.use_robust_znorm,
        num_threads: None,
        use_isotonic_quantiles: true,
        seasonal_period: request.seasonal_period,
        use_boundary_smoothing: request.use_boundary_smoothing,
        ..Default::default()
    };

    let empty_covariates = vec![None; 1];
    let empty_dims = vec![None; 1];

    // 4. Run predict_batch
    let raw_results = model.predict_batch(
        &[flat_context.clone()],
        &[(num_vars, ctx_len)],
        request.horizon,
        &empty_covariates,
        &empty_dims,
        &empty_covariates,
        &empty_dims,
        &opts,
    ).map_err(|e| format!("推理预测执行失败: {e}"))?;

    let result = raw_results
        .first()
        .ok_or_else(|| "模型未返回预测结果".to_string())?;

    // 5. Build future projected timestamps
    let last_ts_str = timestamps.last().cloned().unwrap_or_default();
    let last_dt = csv_helper::try_parse_datetime(&last_ts_str);

    let future_timestamps: Vec<String> = (1..=request.horizon)
        .map(|step| {
            if let (Some(dt), Some(dur)) = (last_dt, step_duration) {
                let future_dt = dt + dur * (step as i32);
                future_dt.format("%Y-%m-%d %H:%M:%S").to_string()
            } else if !last_ts_str.is_empty() {
                format!("+{}步", step)
            } else {
                format!("T+{}", step)
            }
        })
        .collect();

    // 6. Slice history context (keep last 120 points for clear visual rendering in chart)
    let display_history_len = ctx_len.min(120);
    let hist_start = ctx_len - display_history_len;
    let display_hist_timestamps = timestamps[hist_start..].to_vec();

    let mut series_outputs = Vec::new();

    for (v_idx, col_name) in request.target_cols.iter().enumerate() {
        let hist_base = v_idx * ctx_len;
        let hist_vals = flat_context[hist_base + hist_start..hist_base + ctx_len].to_vec();

        let pred_base = v_idx * request.horizon;
        let median = result.forecast[pred_base..pred_base + request.horizon].to_vec();

        // Quantiles: 9 quantiles [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        let mut q10 = Vec::with_capacity(request.horizon);
        let mut q20 = Vec::with_capacity(request.horizon);
        let mut q80 = Vec::with_capacity(request.horizon);
        let mut q90 = Vec::with_capacity(request.horizon);

        if let Some(ref q) = result.quantiles {
            let q_stride = result.num_quantiles; // 9
            let q_base = v_idx * request.horizon * q_stride;
            for h in 0..request.horizon {
                let offset = q_base + h * q_stride;
                q10.push(q[offset + 0]); // 0.1
                q20.push(q[offset + 1]); // 0.2
                q80.push(q[offset + 7]); // 0.8
                q90.push(q[offset + 8]); // 0.9
            }
        } else {
            q10 = median.clone();
            q20 = median.clone();
            q80 = median.clone();
            q90 = median.clone();
        }

        let mean_forecast = median.iter().copied().sum::<f32>() / (median.len() as f32);
        let min_forecast = median.iter().copied().fold(f32::INFINITY, f32::min);
        let max_forecast = median.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let hist_last = hist_vals.last().copied().unwrap_or(mean_forecast);
        let trend = if mean_forecast > hist_last * 1.02 {
            "上升".into()
        } else if mean_forecast < hist_last * 0.98 {
            "下降".into()
        } else {
            "平稳".into()
        };

        series_outputs.push(TargetForecastResult {
            column_name: col_name.clone(),
            history_timestamps: display_hist_timestamps.clone(),
            history_values: hist_vals,
            forecast_timestamps: future_timestamps.clone(),
            median,
            q10,
            q20,
            q80,
            q90,
            mean_forecast,
            min_forecast,
            max_forecast,
            trend,
        });
    }

    Ok(ForecastResponse {
        series: series_outputs,
        horizon: request.horizon,
        elapsed_ms: t0.elapsed().as_millis(),
        checkpoint_used: request.checkpoint_path,
    })
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExportPayload {
    pub series_name: String,
    pub timestamps: Vec<String>,
    pub median: Vec<f32>,
    pub q10: Vec<f32>,
    pub q90: Vec<f32>,
}

#[tauri::command]
pub fn export_forecast_csv(file_path: String, payload: ExportPayload) -> Result<(), String> {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(true)
        .from_path(&file_path)
        .map_err(|e| format!("创建输出文件失败: {e}"))?;

    writer.write_record(&["时间戳", &format!("{}_预测中位数", payload.series_name), "10%分位数(下界)", "90%分位数(上界)"])
        .map_err(|e| e.to_string())?;

    for i in 0..payload.timestamps.len() {
        let ts = payload.timestamps.get(i).map(|s| s.as_str()).unwrap_or("");
        let m = payload.median.get(i).copied().unwrap_or(0.0);
        let q_low = payload.q10.get(i).copied().unwrap_or(0.0);
        let q_high = payload.q90.get(i).copied().unwrap_or(0.0);

        writer.write_record(&[
            ts,
            &format!("{:.4}", m),
            &format!("{:.4}", q_low),
            &format!("{:.4}", q_high),
        ]).map_err(|e| e.to_string())?;
    }

    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn find_project_root() -> PathBuf {
    // Check current dir or traverse up until Cargo.lock and ckpt are found
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("ckpt").exists() || cwd.join("publish").exists() {
            return cwd;
        }
        if let Some(parent) = cwd.parent() {
            if parent.join("ckpt").exists() || parent.join("publish").exists() {
                return parent.to_path_buf();
            }
            if let Some(grand) = parent.parent() {
                if grand.join("ckpt").exists() || grand.join("publish").exists() {
                    return grand.to_path_buf();
                }
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            if parent.join("ckpt").exists() || parent.join("publish").exists() {
                return parent.to_path_buf();
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inspect_model_path() {
        let root = find_project_root();
        let valid_path = root.join("publish/timesfm3.0-balanced");
        if valid_path.exists() {
            let meta = inspect_model_path(valid_path.to_string_lossy().to_string()).expect("inspect failed");
            assert!(meta.exists);
            assert!(meta.size_desc.contains("MB") || meta.size_desc.contains("GB"));
        }
        assert!(inspect_model_path("non_existent_dir_12345".into()).is_err());
    }

    #[test]
    fn test_list_checkpoints() {
        let ckpts = list_checkpoints().expect("list failed");
        assert!(!ckpts.is_empty());
        let balanced = ckpts.iter().find(|c| c.name.contains("Balanced"));
        assert!(balanced.is_some());
        assert!(balanced.unwrap().exists, "Balanced checkpoint should exist in repo");
    }

    #[test]
    fn test_upload_csv_content() {
        tauri::async_runtime::block_on(async {
            let sample_csv = "date,value\n2026-01-01,10.5\n2026-01-02,12.3\n2026-01-03,15.1\n";
            let res = upload_csv_content("test_unit.csv".into(), sample_csv.into()).await.expect("upload failed");
            assert_eq!(res.total_rows, 3);
            assert_eq!(res.proposed_date_col.as_deref(), Some("date"));
            assert!(res.proposed_target_cols.contains(&"value".to_string()));
            assert!(std::path::Path::new(&res.file_path).exists());
        });
    }

    #[test]
    fn test_run_forecast_end_to_end() {
        timesfm3::init_default_thread_pool();
        let root = find_project_root();
        let ckpt = root.join("publish/timesfm3.0-balanced");
        let data_csv = root.join("data/etth1.csv");

        if !ckpt.exists() || !data_csv.exists() {
            eprintln!("Checkpoint or data not found, skipping e2e test");
            return;
        }

        let state = AppState::default();
        let req = ForecastRequest {
            file_path: data_csv.to_string_lossy().to_string(),
            target_cols: vec!["OT".into()],
            date_col: Some("date".into()),
            horizon: 24,
            seasonal_period: Some(24),
            checkpoint_path: ckpt.to_string_lossy().to_string(),
            make_positive: false,
            use_symmetric_averaging: true,
            use_boundary_smoothing: true,
            use_robust_znorm: false,
            is_transposed: false,
        };

        let resp = execute_forecast(&state, req).expect("forecast failed");

        assert_eq!(resp.series.len(), 1);
        assert_eq!(resp.series[0].column_name, "OT");
        assert_eq!(resp.series[0].median.len(), 24);
        assert_eq!(resp.series[0].q10.len(), 24);
        assert_eq!(resp.series[0].q90.len(), 24);
        assert_eq!(resp.series[0].forecast_timestamps.len(), 24);
        println!("E2E forecast completed in {} ms, series: OT", resp.elapsed_ms);
    }
}
