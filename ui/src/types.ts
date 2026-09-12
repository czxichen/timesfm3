export interface ColumnInfo {
  name: string;
  index: number;
  is_numeric: boolean;
  is_datetime: boolean;
  missing_count: number;
  sample_values: string[];
}

export interface CsvInspectionResult {
  file_path: string;
  total_rows: number;
  total_columns: number;
  columns: ColumnInfo[];
  proposed_date_col: string | null;
  proposed_target_cols: string[];
  inferred_frequency: string | null;
  recommended_period: number | null;
  preview_headers: string[];
  preview_rows: string[][];
  is_transposed: boolean;
  health_note: string;
}

export interface CheckpointMeta {
  name: string;
  path: string;
  precision: string;
  size_desc: string;
  is_recommended: boolean;
  exists: boolean;
}

export interface ForecastRequest {
  file_path: string;
  target_cols: string[];
  date_col: string | null;
  horizon: number;
  seasonal_period: number | null;
  checkpoint_path: string;
  make_positive: boolean;
  use_symmetric_averaging: boolean;
  use_boundary_smoothing: boolean;
  use_robust_znorm: boolean;
  is_transposed: boolean;
}

export interface TargetForecastResult {
  column_name: string;
  history_timestamps: string[];
  history_values: number[];
  forecast_timestamps: string[];
  median: number[];
  q10: number[];
  q20: number[];
  q80: number[];
  q90: number[];
  mean_forecast: number;
  min_forecast: number;
  max_forecast: number;
  trend: string;
}

export interface ForecastResponse {
  series: TargetForecastResult[];
  horizon: number;
  elapsed_ms: number;
  checkpoint_used: string;
}
