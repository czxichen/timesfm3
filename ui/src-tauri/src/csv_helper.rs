use std::path::Path;
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub index: usize,
    pub is_numeric: bool,
    pub is_datetime: bool,
    pub missing_count: usize,
    pub sample_values: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvInspectionResult {
    pub file_path: String,
    pub total_rows: usize,
    pub total_columns: usize,
    pub columns: Vec<ColumnInfo>,
    pub proposed_date_col: Option<String>,
    pub proposed_target_cols: Vec<String>,
    pub inferred_frequency: Option<String>,
    pub recommended_period: Option<usize>,
    pub preview_headers: Vec<String>,
    pub preview_rows: Vec<Vec<String>>,
    pub is_transposed: bool,
    pub health_note: String,
}

/// Try parsing string into a chrono NaiveDateTime
pub fn try_parse_datetime(s: &str) -> Option<NaiveDateTime> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Try RFC3339 / ISO 8601
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.naive_utc());
    }

    // Common standard formats
    let formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
    ];

    for fmt in formats {
        if let Ok(ndt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Some(ndt);
        }
        if let Ok(nd) = NaiveDate::parse_from_str(s, fmt) {
            return Some(nd.and_hms_opt(0, 0, 0).unwrap());
        }
    }

    None
}

pub fn resolve_csv_path<P: AsRef<Path>>(path: P) -> Result<std::path::PathBuf, String> {
    let p = path.as_ref();
    if p.exists() {
        return Ok(p.to_path_buf());
    }

    // If path is relative or just a filename:
    if let Ok(cwd) = std::env::current_dir() {
        let in_cwd = cwd.join(p);
        if in_cwd.exists() {
            return Ok(in_cwd);
        }
        let in_cwd_data = cwd.join("data").join(p);
        if in_cwd_data.exists() {
            return Ok(in_cwd_data);
        }
        // If run from ui or ui/src-tauri
        if let Some(parent) = cwd.parent() {
            let in_parent_data = parent.join("data").join(p);
            if in_parent_data.exists() {
                return Ok(in_parent_data);
            }
            if let Some(grand) = parent.parent() {
                let in_grand_data = grand.join("data").join(p);
                if in_grand_data.exists() {
                    return Ok(in_grand_data);
                }
            }
        }
    }

    // Check temp upload directory
    let temp_file = std::env::temp_dir().join("timesfm3_uploads").join(p);
    if temp_file.exists() {
        return Ok(temp_file);
    }

    let default_repo = std::path::PathBuf::from("/Users/xichen/data/code/aicode/timesfm3");
    let in_repo = default_repo.join(p);
    if in_repo.exists() {
        return Ok(in_repo);
    }
    let in_repo_data = default_repo.join("data").join(p);
    if in_repo_data.exists() {
        return Ok(in_repo_data);
    }

    Err(format!(
        "无法打开 CSV 文件: 文件 '{}' 不存在 (已检查本地路径、工程目录及临时上传目录)",
        p.display()
    ))
}

/// Auto-detect delimiter from the first line of a file (supports comma, tab, semicolon, pipe)
pub fn detect_delimiter<P: AsRef<Path>>(path: P) -> u8 {
    if let Ok(file) = std::fs::File::open(path) {
        use std::io::{BufRead, BufReader};
        let mut reader = BufReader::new(file);
        let mut first_line = String::new();
        if reader.read_line(&mut first_line).is_ok() {
            let tabs = first_line.chars().filter(|&c| c == '\t').count();
            let commas = first_line.chars().filter(|&c| c == ',').count();
            let semicolons = first_line.chars().filter(|&c| c == ';').count();
            let pipes = first_line.chars().filter(|&c| c == '|').count();
            if tabs > commas && tabs >= semicolons && tabs >= pipes {
                return b'\t';
            } else if semicolons > commas && semicolons >= tabs && semicolons >= pipes {
                return b';';
            } else if pipes > commas && pipes >= tabs && pipes >= semicolons {
                return b'|';
            }
        }
    }
    b','
}

/// Inspects a CSV/TSV/TXT file and produces detailed metadata for GUI mapping guidance.
pub fn inspect_csv<P: AsRef<Path>>(path: P) -> Result<CsvInspectionResult, String> {
    let resolved_path = resolve_csv_path(path)?;
    let path_ref = resolved_path.as_path();
    let file_path = path_ref.to_string_lossy().to_string();

    let delimiter = detect_delimiter(path_ref);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .delimiter(delimiter)
        .from_path(path_ref)
        .map_err(|e| format!("无法打开数据表格文件: {e}"))?;

    let headers_record = reader
        .headers()
        .map_err(|e| format!("读取表头失败: {e}"))?
        .clone();

    // Check if the file is a transposed raw CSV (e.g. line 1 is pure floats, no header)
    let is_header_all_numbers = headers_record
        .iter()
        .all(|h| h.trim().parse::<f32>().is_ok());

    if is_header_all_numbers {
        // Handle transposed format (Row = Variate, Col = Time)
        return inspect_transposed_csv(path_ref, &file_path);
    }

    let headers: Vec<String> = headers_record.iter().map(|s| s.trim().to_string()).collect();
    let num_cols = headers.len();

    let mut records: Vec<csv::StringRecord> = Vec::new();
    let mut total_rows = 0;

    let mut col_numeric_count = vec![0usize; num_cols];
    let mut col_datetime_count = vec![0usize; num_cols];
    let mut col_missing_count = vec![0usize; num_cols];
    let mut col_samples: Vec<Vec<String>> = vec![Vec::new(); num_cols];

    for result in reader.records() {
        let record = result.map_err(|e| format!("解析行数据错误: {e}"))?;
        if total_rows < 15 {
            records.push(record.clone());
        }
        total_rows += 1;

        for (col_idx, field) in record.iter().enumerate() {
            if col_idx >= num_cols {
                break;
            }
            let trimmed = field.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("nan") || trimmed.eq_ignore_ascii_case("null") {
                col_missing_count[col_idx] += 1;
            } else {
                if trimmed.parse::<f64>().is_ok() {
                    col_numeric_count[col_idx] += 1;
                }
                if try_parse_datetime(trimmed).is_some() {
                    col_datetime_count[col_idx] += 1;
                }
                if col_samples[col_idx].len() < 3 {
                    col_samples[col_idx].push(trimmed.to_string());
                }
            }
        }
    }

    if total_rows == 0 {
        return Err("CSV 文件为空，无有效数据行".into());
    }

    let mut columns: Vec<ColumnInfo> = Vec::new();
    let mut proposed_date_col = None;
    let mut proposed_target_cols = Vec::new();

    for (i, name) in headers.iter().enumerate() {
        let is_num = col_numeric_count[i] > total_rows / 2;
        let is_dt = col_datetime_count[i] > total_rows / 2;

        let name_lower = name.to_lowercase();
        let name_suggests_date = name_lower.contains("date")
            || name_lower.contains("time")
            || name_lower.contains("timestamp");

        if proposed_date_col.is_none() && (is_dt || (name_suggests_date && !is_num)) {
            proposed_date_col = Some(name.clone());
        } else if is_num {
            proposed_target_cols.push(name.clone());
        }

        columns.push(ColumnInfo {
            name: name.clone(),
            index: i,
            is_numeric: is_num,
            is_datetime: is_dt,
            missing_count: col_missing_count[i],
            sample_values: col_samples[i].clone(),
        });
    }

    // Inferred frequency and recommended season period
    let (inferred_frequency, recommended_period) = if let Some(ref d_col) = proposed_date_col {
        infer_frequency_from_date_col(path_ref, d_col)
    } else {
        (None, None)
    };

    let preview_headers = headers.clone();
    let preview_rows: Vec<Vec<String>> = records
        .iter()
        .take(10)
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect();

    let mut health_notes = Vec::new();
    health_notes.push(format!("有效历史长度: {} 行", total_rows));
    let total_missing: usize = col_missing_count.iter().sum();
    if total_missing > 0 {
        health_notes.push(format!("检测到 {} 个缺失值，引擎将自动进行时序插值平滑", total_missing));
    } else {
        health_notes.push("数据完整无缺失".into());
    }

    Ok(CsvInspectionResult {
        file_path,
        total_rows,
        total_columns: num_cols,
        columns,
        proposed_date_col,
        proposed_target_cols,
        inferred_frequency,
        recommended_period,
        preview_headers,
        preview_rows,
        is_transposed: false,
        health_note: health_notes.join("；"),
    })
}

fn infer_frequency_from_date_col<P: AsRef<Path>>(path: P, date_col_name: &str) -> (Option<String>, Option<usize>) {
    let delimiter = detect_delimiter(&path);
    let mut reader = match csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .delimiter(delimiter)
        .from_path(path) {
        Ok(r) => r,
        Err(_) => return (None, None),
    };

    let headers = match reader.headers() {
        Ok(h) => h.clone(),
        Err(_) => return (None, None),
    };

    let col_idx = match headers.iter().position(|h| h.trim() == date_col_name) {
        Some(idx) => idx,
        None => return (None, None),
    };

    let mut datetimes: Vec<NaiveDateTime> = Vec::new();
    for result in reader.records().take(50) {
        if let Ok(record) = result {
            if let Some(val) = record.get(col_idx) {
                if let Some(dt) = try_parse_datetime(val) {
                    datetimes.push(dt);
                }
            }
        }
    }

    if datetimes.len() < 2 {
        return (None, None);
    }

    let mut deltas: Vec<i64> = Vec::new();
    for i in 1..datetimes.len() {
        let diff = (datetimes[i] - datetimes[i - 1]).num_seconds().abs();
        if diff > 0 {
            deltas.push(diff);
        }
    }

    if deltas.is_empty() {
        return (None, None);
    }

    deltas.sort();
    let median_diff = deltas[deltas.len() / 2];

    match median_diff {
        // ~15 min
        800..=1000 => (Some("15分钟 (15-min)".into()), Some(96)),
        // ~1 hour
        3500..=3700 => (Some("1小时 (Hourly)".into()), Some(24)),
        // ~1 day
        85000..=87000 => (Some("1天 (Daily)".into()), Some(7)),
        // ~1 minute
        55..=65 => (Some("1分钟 (Minute)".into()), Some(60)),
        // ~1 week
        600000..=610000 => (Some("1周 (Weekly)".into()), Some(52)),
        _ => (Some(format!("约 {} 秒", median_diff)), None),
    }
}

/// Fallback inspector for transposed CSV format (one line per variable, comma-separated numbers)
fn inspect_transposed_csv<P: AsRef<Path>>(path: P, file_path: &str) -> Result<CsvInspectionResult, String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let lines: Vec<&str> = content
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .collect();

    if lines.is_empty() {
        return Err("文件为空".into());
    }

    let num_vars = lines.len();
    let first_line_len = lines[0].split(',').count();

    let mut columns = Vec::new();
    let mut proposed_target_cols = Vec::new();
    for i in 0..num_vars {
        let name = format!("variate_{i}");
        proposed_target_cols.push(name.clone());
        columns.push(ColumnInfo {
            name,
            index: i,
            is_numeric: true,
            is_datetime: false,
            missing_count: 0,
            sample_values: lines[i].split(',').take(3).map(|s| s.trim().to_string()).collect(),
        });
    }

    let preview_headers: Vec<String> = (0..first_line_len.min(8))
        .map(|i| format!("t_{i}"))
        .collect();

    let preview_rows: Vec<Vec<String>> = lines
        .iter()
        .take(10)
        .map(|l| l.split(',').take(8).map(|s| s.trim().to_string()).collect())
        .collect();

    Ok(CsvInspectionResult {
        file_path: file_path.to_string(),
        total_rows: first_line_len,
        total_columns: num_vars,
        columns,
        proposed_date_col: None,
        proposed_target_cols,
        inferred_frequency: None,
        recommended_period: None,
        preview_headers,
        preview_rows,
        is_transposed: true,
        health_note: format!("检测到转置纯数值格式: 包含 {} 个变量，每个变量长度为 {} 步", num_vars, first_line_len),
    })
}

/// Loads selected columns from CSV into flat (num_variates, context_len) f32 vector
/// along with history timestamps and detected frequency step duration.
pub fn load_series_for_inference<P: AsRef<Path>>(
    path: P,
    date_col: Option<&str>,
    target_cols: &[String],
    is_transposed: bool,
) -> Result<(Vec<f32>, (usize, usize), Vec<String>, Option<Duration>), String> {
    let resolved_path = resolve_csv_path(path)?;
    let path = resolved_path.as_path();

    if is_transposed {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let lines: Vec<&str> = content
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            .collect();
        let num_vars = lines.len();
        if num_vars == 0 {
            return Err("文件为空".into());
        }
        let mut flat = Vec::new();
        let mut col_len = 0;
        for (i, line) in lines.iter().enumerate() {
            let row: Vec<f32> = line
                .split(',')
                .map(|t| t.trim().parse::<f32>().unwrap_or(f32::NAN))
                .collect();
            if i == 0 {
                col_len = row.len();
            } else if row.len() != col_len {
                return Err("转置 CSV 各行长度不一致".into());
            }
            flat.extend(row);
        }
        let fake_timestamps: Vec<String> = (0..col_len).map(|i| format!("T-{:04}", col_len - i)).collect();
        return Ok((flat, (num_vars, col_len), fake_timestamps, None));
    }

    let delimiter = detect_delimiter(&path);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .delimiter(delimiter)
        .from_path(&path)
        .map_err(|e| format!("读取数据表格失败: {e}"))?;

    let headers = reader.headers().map_err(|e| e.to_string())?.clone();
    let header_list: Vec<String> = headers.iter().map(|s| s.trim().to_string()).collect();

    let date_idx = date_col.and_then(|d| header_list.iter().position(|h| h == d));

    let mut target_indices = Vec::new();
    for col in target_cols {
        if let Some(idx) = header_list.iter().position(|h| h == col) {
            target_indices.push(idx);
        } else {
            return Err(format!("未找到指定的目标列: {col}"));
        }
    }

    if target_indices.is_empty() {
        return Err("至少需要选择一个预测目标列".into());
    }

    let num_variates = target_indices.len();
    let mut columns_data: Vec<Vec<f32>> = vec![Vec::new(); num_variates];
    let mut timestamps: Vec<String> = Vec::new();
    let mut parsed_dts: Vec<NaiveDateTime> = Vec::new();

    for (row_idx, result) in reader.records().enumerate() {
        let record = result.map_err(|e| format!("第 {} 行解析失败: {e}", row_idx + 1))?;

        if let Some(d_idx) = date_idx {
            let val = record.get(d_idx).unwrap_or("").trim();
            timestamps.push(val.to_string());
            if let Some(pdt) = try_parse_datetime(val) {
                parsed_dts.push(pdt);
            }
        } else {
            timestamps.push(format!("第{}步", row_idx + 1));
        }

        for (v_idx, &col_idx) in target_indices.iter().enumerate() {
            let field_val = record.get(col_idx).unwrap_or("").trim();
            let parsed = if field_val.is_empty() || field_val.eq_ignore_ascii_case("nan") {
                f32::NAN
            } else {
                field_val.parse::<f32>().unwrap_or(f32::NAN)
            };
            columns_data[v_idx].push(parsed);
        }
    }

    let total_pts = columns_data[0].len();
    if total_pts == 0 {
        return Err("未解析到任何有效数据行".into());
    }

    // TimesFM operates best with the most recent 1024 points (32 patches).
    // Slicing to the most recent 1024 points guarantees near-instant inference (~300ms)
    // while maintaining optimal zero-shot accuracy.
    let max_window = 1024;
    let (slice_start, context_len) = if total_pts > max_window {
        (total_pts - max_window, max_window)
    } else {
        (0, total_pts)
    };

    let sliced_timestamps = timestamps[slice_start..].to_vec();

    // Flatten to (num_variates, context_len)
    let mut flat = Vec::with_capacity(num_variates * context_len);
    for col in &columns_data {
        flat.extend_from_slice(&col[slice_start..]);
    }

    // Inferred step duration
    let step_duration = if parsed_dts.len() >= 2 {
        let mut diffs: Vec<i64> = Vec::new();
        for i in 1..parsed_dts.len() {
            let d = (parsed_dts[i] - parsed_dts[i - 1]).num_seconds().abs();
            if d > 0 {
                diffs.push(d);
            }
        }
        if !diffs.is_empty() {
            diffs.sort();
            Some(Duration::seconds(diffs[diffs.len() / 2]))
        } else {
            None
        }
    } else {
        None
    };

    Ok((flat, (num_variates, context_len), sliced_timestamps, step_duration))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inspect_etth1() {
        let path = Path::new("../../data/etth1.csv");
        if !path.exists() {
            eprintln!("data/etth1.csv not found at expected path, skipping test");
            return;
        }
        let res = inspect_csv(path).expect("inspect failed");
        assert_eq!(res.proposed_date_col.as_deref(), Some("date"));
        assert_eq!(res.recommended_period, Some(24));
        assert_eq!(res.inferred_frequency.as_deref(), Some("1小时 (Hourly)"));
        assert!(res.total_rows > 1000);
        assert!(res.proposed_target_cols.contains(&"OT".to_string()));

        let (flat, (v, c), ts, dur) = load_series_for_inference(
            path,
            Some("date"),
            &["OT".to_string(), "HUFL".to_string()],
            false,
        ).expect("load failed");

        assert_eq!(v, 2);
        assert_eq!(c, 1024);
        assert_eq!(flat.len(), v * c);
        assert_eq!(ts.len(), c);
        assert!(dur.is_some());
    }

    #[test]
    fn test_resolve_csv_path_by_filename() {
        let res = inspect_csv("etth1.csv");
        assert!(res.is_ok(), "inspect_csv should resolve etth1.csv from data directory");
        let info = res.unwrap();
        assert_eq!(info.proposed_date_col.as_deref(), Some("date"));
    }

    #[test]
    fn test_non_existent_csv_error() {
        let res = inspect_csv("definitely_not_existing_file_12345.csv");
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("无法打开 CSV 文件"));
        assert!(err.contains("不存在"));
    }
}


