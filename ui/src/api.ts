import { invoke as tauriInvoke } from '@tauri-apps/api/core';
import type {
  CsvInspectionResult,
  CheckpointMeta,
  ForecastRequest,
  ForecastResponse,
} from './types';

export const isTauri = () =>
  typeof window !== 'undefined' &&
  ('__TAURI_INTERNALS__' in window || '__TAURI__' in window);

export async function listCheckpoints(): Promise<CheckpointMeta[]> {
  if (isTauri()) {
    return tauriInvoke<CheckpointMeta[]>('list_checkpoints');
  }
  return [
    {
      name: '均衡模式 (Balanced · 推荐)',
      path: 'publish/timesfm3.0-balanced',
      precision: '均衡高精',
      size_desc: '约 730 MB · 推荐首选，在精度与运行速度间达到最佳平衡',
      is_recommended: true,
      exists: true,
    },
    {
      name: '极速轻量模式',
      path: 'publish/timesfm3.0-f16',
      precision: '极速轻量',
      size_desc: '约 631 MB · 内存占用低，适合轻量设备或快速批量推断',
      is_recommended: false,
      exists: true,
    },
    {
      name: '高保真原版模式',
      path: 'ckpt',
      precision: '完整原版',
      size_desc: '约 1.32 GB · 官方原版全量精度',
      is_recommended: false,
      exists: true,
    },
  ];
}

export async function pickModelDirectory(): Promise<string | null> {
  if (isTauri()) {
    return tauriInvoke<string | null>('pick_model_directory');
  }
  return null;
}

export async function inspectModelPath(rawPath: string): Promise<CheckpointMeta> {
  if (isTauri()) {
    return tauriInvoke<CheckpointMeta>('inspect_model_path', { rawPath });
  }
  return {
    name: '自定义本地模型',
    path: rawPath,
    precision: '均衡高精 (Balanced)',
    size_desc: '约 730 MB · 本地指定目录',
    is_recommended: true,
    exists: true,
  };
}

export async function inspectCsv(filePath: string): Promise<CsvInspectionResult> {
  if (isTauri()) {
    return tauriInvoke<CsvInspectionResult>('inspect_csv', { filePath });
  }
  return getMockInspection(filePath);
}

export async function pickCsvFile(): Promise<string | null> {
  if (isTauri()) {
    return tauriInvoke<string | null>('pick_csv_file');
  }
  return null;
}

export async function uploadCsvContent(
  fileName: string,
  content: string
): Promise<CsvInspectionResult> {
  if (isTauri()) {
    return tauriInvoke<CsvInspectionResult>('upload_csv_content', {
      fileName,
      content,
    });
  }
  return getMockInspection(fileName);
}

export async function loadSampleDataset(sampleKey: string): Promise<CsvInspectionResult> {
  if (isTauri()) {
    return tauriInvoke<CsvInspectionResult>('load_sample_dataset', { sampleKey });
  }
  return getMockInspection(`data/${sampleKey}.csv`);
}

export async function runForecast(request: ForecastRequest): Promise<ForecastResponse> {
  if (isTauri()) {
    return tauriInvoke<ForecastResponse>('run_forecast', { request });
  }

  // Simulate fast realistic forecast for browser preview
  await new Promise((r) => setTimeout(r, 600));

  const horizon = request.horizon || 48;
  const targetCols = request.target_cols.length > 0 ? request.target_cols : ['OT'];

  const series = targetCols.map((col) => {
    // Generate synthetic realistic electricity load trajectory
    const histLen = 120;
    const historyValues: number[] = [];
    const historyTimestamps: number[] = [];

    const baseVal = col === 'OT' ? 30.5 : col === 'HUFL' ? 14.2 : 8.5;
    const now = new Date('2026-09-10T12:00:00Z').getTime();

    for (let i = 0; i < histLen; i++) {
      const t = now - (histLen - i) * 3600 * 1000;
      historyTimestamps.push(t);
      const hour = (i % 24) * (Math.PI / 12);
      const val = baseVal + 6 * Math.sin(hour) + 2.5 * Math.sin(hour * 2) + (Math.random() - 0.5) * 1.2;
      historyValues.push(val);
    }

    const lastHist = historyValues[historyValues.length - 1];
    const median: number[] = [];
    const q10: number[] = [];
    const q20: number[] = [];
    const q80: number[] = [];
    const q90: number[] = [];
    const forecastTimestamps: string[] = [];

    for (let h = 1; h <= horizon; h++) {
      const ft = new Date(now + h * 3600 * 1000).toISOString().replace('T', ' ').substring(0, 19);
      forecastTimestamps.push(ft);

      const hour = ((histLen + h) % 24) * (Math.PI / 12);
      const trendComponent = 0.02 * h;
      const m = lastHist * 0.95 + 6 * Math.sin(hour) + 2.2 * Math.sin(hour * 2) + trendComponent;
      median.push(m);

      const spread = 1.2 + 0.08 * h; // Fan uncertainty increases over horizon
      q20.push(m - spread * 0.85);
      q80.push(m + spread * 0.85);
      q10.push(m - spread * 1.5);
      q90.push(m + spread * 1.5);
    }

    const meanForecast = median.reduce((a, b) => a + b, 0) / median.length;
    const minForecast = Math.min(...median);
    const maxForecast = Math.max(...median);

    return {
      column_name: col,
      history_timestamps: historyTimestamps.map((t) =>
        new Date(t).toISOString().replace('T', ' ').substring(0, 19)
      ),
      history_values: historyValues,
      forecast_timestamps: forecastTimestamps,
      median,
      q10,
      q20,
      q80,
      q90,
      mean_forecast: meanForecast,
      min_forecast: minForecast,
      max_forecast: maxForecast,
      trend: meanForecast > lastHist * 1.02 ? '上升' : meanForecast < lastHist * 0.98 ? '下降' : '平稳',
    };
  });

  return {
    series,
    horizon,
    elapsed_ms: 342,
    checkpoint_used: request.checkpoint_path,
  };
}

export async function exportForecastCsv(filePath: string, payload: any): Promise<void> {
  if (isTauri()) {
    return tauriInvoke('export_forecast_csv', { filePath, payload });
  }
  // Web browser fallback: trigger download
  const headers = ['时间戳', `${payload.series_name}_预测中位数`, '10%分位数(下界)', '90%分位数(上界)'];
  const rows = payload.timestamps.map((ts: string, i: number) => [
    ts,
    payload.median[i]?.toFixed(4),
    payload.q10[i]?.toFixed(4),
    payload.q90[i]?.toFixed(4),
  ]);
  const csvContent = [headers.join(','), ...rows.map((r: string[]) => r.join(','))].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filePath;
  a.click();
}

function getMockInspection(filePath: string): CsvInspectionResult {
  const cols = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'];
  return {
    file_path: filePath,
    total_rows: 17420,
    total_columns: 8,
    columns: cols.map((name, i) => ({
      name,
      index: i,
      is_numeric: name !== 'date',
      is_datetime: name === 'date',
      missing_count: 0,
      sample_values: name === 'date' ? ['2016-07-01 00:00:00', '2016-07-01 01:00:00'] : ['5.827', '2.009', '30.531'],
    })),
    proposed_date_col: 'date',
    proposed_target_cols: ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
    inferred_frequency: '1小时 (Hourly)',
    recommended_period: 24,
    preview_headers: cols,
    preview_rows: [
      ['2016-07-01 00:00:00', '5.827', '2.009', '1.599', '0.462', '4.203', '1.340', '30.531'],
      ['2016-07-01 01:00:00', '5.693', '2.076', '1.492', '0.426', '4.142', '1.371', '27.787'],
      ['2016-07-01 02:00:00', '5.157', '1.741', '1.279', '0.355', '3.777', '1.218', '27.787'],
      ['2016-07-01 03:00:00', '5.090', '1.942', '1.279', '0.391', '3.807', '1.279', '25.044'],
      ['2016-07-01 04:00:00', '5.358', '1.942', '1.492', '0.462', '3.868', '1.279', '21.948'],
    ],
    is_transposed: false,
    health_note: '有效历史长度: 17420 行；数据完整无缺失',
  };
}
