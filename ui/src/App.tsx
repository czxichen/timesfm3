import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  FileSpreadsheet,
  CheckCircle2,
  TrendingUp,
  Settings2,
  Calendar,
  Layers,
  ArrowRight,
  ArrowLeft,
  Play,
  Download,
  AlertCircle,
  Clock,
  Database,
  RefreshCw,
  BarChart3,
  Zap,
  Sparkles,
  ShieldCheck,
  Activity,
  Check,
  FolderOpen,
  HelpCircle,
  AlertTriangle,
} from 'lucide-react';
import type {
  CsvInspectionResult,
  CheckpointMeta,
  ForecastResponse,
} from './types';
import {
  listCheckpoints,
  inspectCsv,
  loadSampleDataset,
  runForecast,
  exportForecastCsv,
  pickCsvFile,
  uploadCsvContent,
  isTauri,
  pickModelDirectory,
  inspectModelPath,
} from './api';
import { FanChart } from './components/FanChart';

export function App() {
  // Wizard current step: 1 = Upload, 2 = Mapping, 3 = Config, 4 = Results
  const [currentStep, setCurrentStep] = useState<number>(1);

  // Data inspection & mapping state
  const [inspection, setInspection] = useState<CsvInspectionResult | null>(null);
  const [selectedDateCol, setSelectedDateCol] = useState<string | null>(null);
  const [selectedTargets, setSelectedTargets] = useState<string[]>([]);
  const [seasonalPeriod, setSeasonalPeriod] = useState<number | null>(null);

  // Forecast settings
  const [horizon, setHorizon] = useState<number>(48);
  const [checkpoints, setCheckpoints] = useState<CheckpointMeta[]>([]);
  const [selectedCkptPath, setSelectedCkptPath] = useState<string>(() => {
    return localStorage.getItem('timesfm3_model_path') || '';
  });
  const [customModelPathInput, setCustomModelPathInput] = useState<string>('');
  const [modelError, setModelError] = useState<string | null>(null);
  const [isVerifyingModel, setIsVerifyingModel] = useState<boolean>(false);
  const [showModelDownloadGuide, setShowModelDownloadGuide] = useState<boolean>(false);
  const [makePositive, setMakePositive] = useState<boolean>(false);
  const [useSymmetric, setUseSymmetric] = useState<boolean>(true);
  const [useSmoothing, setUseSmoothing] = useState<boolean>(true);

  // Running & results state
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isInspecting, setIsInspecting] = useState<boolean>(false);
  const [inspectingFileName, setInspectingFileName] = useState<string>('');
  const [inspectingPhase, setInspectingPhase] = useState<string>('正在解析表格数据...');
  const [inspectElapsedSec, setInspectElapsedSec] = useState<number>(0);
  const [isForecasting, setIsForecasting] = useState<boolean>(false);
  const [forecastProgress, setForecastProgress] = useState<number>(0);
  const [forecastPhase, setForecastPhase] = useState<number>(1);
  const [forecastElapsedSec, setForecastElapsedSec] = useState<number>(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [forecastResult, setForecastResult] = useState<ForecastResponse | null>(null);
  const [activeSeriesTab, setActiveSeriesTab] = useState<number>(0);

  // Apply inspection result into mapping defaults
  const applyInspection = (res: CsvInspectionResult) => {
    setInspection(res);
    setSelectedDateCol(res.proposed_date_col);
    setSelectedTargets(
      res.proposed_target_cols.length > 0
        ? [res.proposed_target_cols[res.proposed_target_cols.length - 1]]
        : []
    );
    setSeasonalPeriod(res.recommended_period || 24);

    // Auto-set sensible default horizon according to detected sampling frequency
    const freq = (res.inferred_frequency || '').toLowerCase();
    if (freq.includes('天') || freq.includes('daily')) {
      setHorizon(30);
    } else if (freq.includes('周') || freq.includes('weekly')) {
      setHorizon(12);
    } else if (freq.includes('15分') || freq.includes('15-min')) {
      setHorizon(96);
    } else if (freq.includes('分') || freq.includes('minute')) {
      setHorizon(120);
    } else {
      setHorizon(48);
    }
  };

  // Helper to translate frequency into human-friendly duration configurations
  const getFrequencyConfig = () => {
    const freq = (inspection?.inferred_frequency || '').toLowerCase();
    if (freq.includes('天') || freq.includes('daily')) {
      return {
        unitName: '天',
        stepDesc: '日级采样：每预测 1 条数据 = 往后预测 1 天',
        min: 3,
        max: 180,
        step: 1,
        presets: [
          { steps: 7, label: '7 条 (未来 1 周)' },
          { steps: 14, label: '14 条 (未来 2 周)' },
          { steps: 30, label: '30 条 (未来 1 个月 · 推荐)' },
          { steps: 60, label: '60 条 (未来 2 个月)' },
          { steps: 90, label: '90 条 (未来 1 季度)' },
        ],
        formatDuration: (steps: number) => {
          if (steps === 7) return '相当于未来 1 周';
          if (steps === 14) return '相当于未来 2 周';
          if (steps === 30) return '相当于未来 1 个月';
          if (steps === 60) return '相当于未来 2 个月';
          if (steps === 90) return '相当于未来 1 季度';
          if (steps > 30) return `相当于未来 ${(steps / 30).toFixed(1)} 个月`;
          return `相当于未来 ${steps} 天`;
        },
      };
    }

    if (freq.includes('15分') || freq.includes('15-min')) {
      return {
        unitName: '15分钟',
        stepDesc: '高频采样：每预测 1 条数据 = 往后预测 15 分钟',
        min: 16,
        max: 384,
        step: 16,
        presets: [
          { steps: 24, label: '24 条 (未来 6 小时)' },
          { steps: 48, label: '48 条 (未来 12 小时)' },
          { steps: 96, label: '96 条 (未来 24 小时 · 推荐)' },
          { steps: 192, label: '192 条 (未来 48 小时)' },
          { steps: 288, label: '288 条 (未来 72 小时)' },
        ],
        formatDuration: (steps: number) => {
          const hours = steps / 4;
          if (hours < 24) return `相当于未来 ${hours.toFixed(hours % 1 === 0 ? 0 : 1)} 小时`;
          const days = hours / 24;
          return `相当于未来 ${days.toFixed(days % 1 === 0 ? 0 : 1)} 天 (${hours}小时)`;
        },
      };
    }

    if (freq.includes('周') || freq.includes('weekly')) {
      return {
        unitName: '周',
        stepDesc: '周级采样：每预测 1 条数据 = 往后预测 1 周 (7天)',
        min: 2,
        max: 52,
        step: 1,
        presets: [
          { steps: 4, label: '4 条 (未来 1 个月)' },
          { steps: 8, label: '8 条 (未来 2 个月)' },
          { steps: 12, label: '12 条 (未来 1 季度 · 推荐)' },
          { steps: 26, label: '26 条 (未来 半年)' },
          { steps: 52, label: '52 条 (未来 1 年)' },
        ],
        formatDuration: (steps: number) => {
          if (steps === 4) return '相当于未来 1 个月';
          if (steps === 12) return '相当于未来 1 季度';
          if (steps === 26) return '相当于未来 半年';
          if (steps === 52) return '相当于未来 1 年';
          return `相当于未来 ${steps} 周`;
        },
      };
    }

    if (freq.includes('分') || freq.includes('minute')) {
      return {
        unitName: '分钟',
        stepDesc: '分级采样：每预测 1 条数据 = 往后预测 1 分钟',
        min: 30,
        max: 720,
        step: 30,
        presets: [
          { steps: 60, label: '60 条 (未来 1 小时)' },
          { steps: 120, label: '120 条 (未来 2 小时 · 推荐)' },
          { steps: 240, label: '240 条 (未来 4 小时)' },
          { steps: 360, label: '360 条 (未来 6 小时)' },
          { steps: 720, label: '720 条 (未来 12 小时)' },
        ],
        formatDuration: (steps: number) => {
          if (steps < 60) return `相当于未来 ${steps} 分钟`;
          const h = steps / 60;
          return `相当于未来 ${h.toFixed(h % 1 === 0 ? 0 : 1)} 小时`;
        },
      };
    }

    // Default: Hourly / Regular timeseries
    return {
      unitName: '小时',
      stepDesc: '小时采样：每预测 1 条数据 = 往后预测 1 小时',
      min: 12,
      max: 336,
      step: 12,
      presets: [
        { steps: 24, label: '24 条 (未来 1 天)' },
        { steps: 48, label: '48 条 (未来 2 天 · 推荐)' },
        { steps: 72, label: '72 条 (未来 3 天)' },
        { steps: 96, label: '96 条 (未来 4 天)' },
        { steps: 168, label: '168 条 (未来 1 周)' },
      ],
      formatDuration: (steps: number) => {
        if (steps === 24) return '相当于未来 1 天';
        if (steps === 48) return '相当于未来 2 天';
        if (steps === 72) return '相当于未来 3 天';
        if (steps === 96) return '相当于未来 4 天';
        if (steps === 168) return '相当于未来 1 周 (7天)';
        const d = steps / 24;
        return `相当于未来 ${d.toFixed(d % 1 === 0 ? 0 : 1)} 天 (${steps}小时)`;
      },
    };
  };

  // Load available checkpoints on mount, restoring custom saved path if any
  useEffect(() => {
    const initCheckpoints = async () => {
      try {
        const ckpts = await listCheckpoints();
        const savedPath = localStorage.getItem('timesfm3_model_path');
        let mergedList = [...ckpts];

        if (savedPath) {
          try {
            const inspected = await inspectModelPath(savedPath);
            const existingIdx = mergedList.findIndex((c) => c.path === inspected.path);
            if (existingIdx >= 0) {
              mergedList[existingIdx] = inspected;
            } else {
              mergedList.unshift(inspected);
            }
            setSelectedCkptPath(inspected.path);
          } catch {
            // If saved path is no longer valid, keep default list
          }
        }

        setCheckpoints(mergedList);

        const currentChosen = mergedList.find((c) => c.path === selectedCkptPath && c.exists);
        if (!currentChosen) {
          const firstValid = mergedList.find((c) => c.is_recommended && c.exists) || mergedList.find((c) => c.exists);
          if (firstValid) {
            setSelectedCkptPath(firstValid.path);
            localStorage.setItem('timesfm3_model_path', firstValid.path);
          }
        }
      } catch (e) {
        console.error('获取模型列表失败:', e);
      }
    };

    initCheckpoints();
  }, []);

  const handlePickModelDirectory = async () => {
    setModelError(null);
    setIsVerifyingModel(true);
    try {
      const picked = await pickModelDirectory();
      if (!picked) {
        setIsVerifyingModel(false);
        return;
      }
      const meta = await inspectModelPath(picked);
      localStorage.setItem('timesfm3_model_path', meta.path);
      setCheckpoints((prev) => {
        const next = prev.filter((c) => c.path !== meta.path);
        return [meta, ...next];
      });
      setSelectedCkptPath(meta.path);
      setCustomModelPathInput('');
    } catch (err: any) {
      setModelError(typeof err === 'string' ? err : err.message || '模型目录验证失败');
    } finally {
      setIsVerifyingModel(false);
    }
  };

  const handleApplyCustomPathInput = async () => {
    const raw = customModelPathInput.trim();
    if (!raw) return;
    setModelError(null);
    setIsVerifyingModel(true);
    try {
      const meta = await inspectModelPath(raw);
      localStorage.setItem('timesfm3_model_path', meta.path);
      setCheckpoints((prev) => {
        const next = prev.filter((c) => c.path !== meta.path);
        return [meta, ...next];
      });
      setSelectedCkptPath(meta.path);
      setCustomModelPathInput('');
    } catch (err: any) {
      setModelError(typeof err === 'string' ? err : err.message || '该路径下未找到有效的 model.safetensors 权重');
    } finally {
      setIsVerifyingModel(false);
    }
  };

  useEffect(() => {
    // Support URL query param for direct preview: ?step=2 or ?step=3 or ?step=4
    const params = new URLSearchParams(window.location.search);
    const stepParam = parseInt(params.get('step') || '0', 10);
    if (stepParam >= 2) {
      loadSampleDataset('etth1').then(async (res) => {
        applyInspection(res);
        if (stepParam === 2) {
          setCurrentStep(2);
        } else if (stepParam === 3) {
          setCurrentStep(3);
        } else if (stepParam === 4) {
          setCurrentStep(4);
          const targetCols = ['OT'];
          const resp = await runForecast({
            file_path: res.file_path,
            target_cols: targetCols,
            date_col: res.proposed_date_col,
            horizon: 48,
            seasonal_period: 24,
            checkpoint_path: 'publish/timesfm3.0-balanced',
            make_positive: false,
            use_symmetric_averaging: true,
            use_boundary_smoothing: true,
            use_robust_znorm: false,
            is_transposed: false,
          });
          setForecastResult(resp);
        }
      });
    }
  }, []);

  // Progress tracker timer during inference
  useEffect(() => {
    if (!isForecasting) {
      setForecastProgress(0);
      setForecastPhase(1);
      setForecastElapsedSec(0);
      return;
    }

    const startTime = Date.now();
    const timer = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      setForecastElapsedSec(elapsed);

      // Smooth multi-stage progress progression
      if (elapsed < 0.4) {
        setForecastPhase(1);
        setForecastProgress(Math.min(25, elapsed * 60));
      } else if (elapsed < 1.2) {
        setForecastPhase(2);
        setForecastProgress(Math.min(55, 25 + (elapsed - 0.4) * 38));
      } else if (elapsed < 2.5) {
        setForecastPhase(3);
        setForecastProgress(Math.min(85, 55 + (elapsed - 1.2) * 23));
      } else {
        setForecastPhase(4);
        setForecastProgress(Math.min(96, 85 + (elapsed - 2.5) * 3));
      }
    }, 100);

    return () => clearInterval(timer);
  }, [isForecasting]);

  // Timer for inspection progress
  useEffect(() => {
    if (!isInspecting) {
      setInspectElapsedSec(0);
      return;
    }
    const startTime = Date.now();
    const timer = setInterval(() => {
      setInspectElapsedSec((Date.now() - startTime) / 1000);
    }, 100);
    return () => clearInterval(timer);
  }, [isInspecting]);

  // Listen to native Tauri window drag-and-drop
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    if (isTauri()) {
      import('@tauri-apps/api/webview')
        .then(({ getCurrentWebview }) => {
          return getCurrentWebview().onDragDropEvent(async (event) => {
            if (event.payload.type === 'drop' && event.payload.paths && event.payload.paths.length > 0) {
              const droppedPath = event.payload.paths[0];
              const name = droppedPath.split('/').pop() || droppedPath.split('\\').pop() || '时序数据.csv';
              setInspectingFileName(name);
              setInspectingPhase('正在读取并扫描时序表格数据...');
              setIsInspecting(true);
              setIsLoading(true);
              setErrorMsg(null);
              try {
                const res = await inspectCsv(droppedPath);
                applyInspection(res);
                setCurrentStep(2);
              } catch (err: any) {
                setErrorMsg(`读取 CSV 失败: ${err.toString()}`);
              } finally {
                setIsLoading(false);
                setIsInspecting(false);
              }
            }
          });
        })
        .then((unlistenFn) => {
          unlisten = unlistenFn;
        })
        .catch((err) => {
          console.warn('Native drag-drop listener registration failed:', err);
        });
    }
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);

  // Process uploaded or dropped File object
  const processUploadedFile = async (file: File) => {
    setInspectingFileName(file.name);
    setInspectingPhase('正在读取本地时序表格文件...');
    setIsInspecting(true);
    setIsLoading(true);
    setErrorMsg(null);
    try {
      if (isTauri()) {
        const possiblePath = (file as any).path;
        if (possiblePath && typeof possiblePath === 'string' && (possiblePath.startsWith('/') || possiblePath.includes(':\\'))) {
          setInspectingPhase('正在解析表格字段与时间周期...');
          const res = await inspectCsv(possiblePath);
          applyInspection(res);
          setCurrentStep(2);
          return;
        }
        // Save content to local temp file to ensure downstream inference can access it
        setInspectingPhase('正在缓存数据并分析时间序列结构...');
        const content = await file.text();
        const res = await uploadCsvContent(file.name, content);
        applyInspection(res);
        setCurrentStep(2);
      } else {
        const res = await inspectCsv(file.name);
        applyInspection(res);
        setCurrentStep(2);
      }
    } catch (err: any) {
      setErrorMsg(`读取 CSV 失败: ${err.toString()}`);
    } finally {
      setIsLoading(false);
      setIsInspecting(false);
    }
  };

  // Open native OS file dialog or trigger file input
  const handlePickFile = async (e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    if (isTauri()) {
      try {
        setErrorMsg(null);
        const path = await pickCsvFile();
        if (path) {
          const name = path.split('/').pop() || path.split('\\').pop() || '时序数据.csv';
          setInspectingFileName(name);
          setInspectingPhase('正在扫描表格字段、识别时间列与采样周期...');
          setIsInspecting(true);
          setIsLoading(true);
          const res = await inspectCsv(path);
          applyInspection(res);
          setCurrentStep(2);
        }
      } catch (err: any) {
        setErrorMsg(`读取 CSV 失败: ${err.toString()}`);
      } finally {
        setIsLoading(false);
        setIsInspecting(false);
      }
      return;
    }

    fileInputRef.current?.click();
  };

  // Handle local CSV file input change
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await processUploadedFile(file);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // HTML5 Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      await processUploadedFile(file);
    }
  };

  // Toggle column selection in targets
  const toggleTarget = (colName: string) => {
    if (selectedTargets.includes(colName)) {
      if (selectedTargets.length === 1) return;
      setSelectedTargets(selectedTargets.filter((c) => c !== colName));
    } else {
      setSelectedTargets([...selectedTargets, colName]);
    }
  };

  // Select all numeric targets
  const selectAllTargets = () => {
    if (!inspection) return;
    const allNumeric = inspection.columns
      .filter((c) => c.is_numeric && c.name !== selectedDateCol)
      .map((c) => c.name);
    setSelectedTargets(allNumeric);
  };

  // Run forecast invocation
  const handleRunForecast = async () => {
    if (!inspection) {
      setErrorMsg('请先导入时序数据表格');
      return;
    }
    if (selectedTargets.length === 0) {
      setErrorMsg('请至少选择一个待预测目标列');
      return;
    }

    const activeCkpt = checkpoints.find((c) => c.path === selectedCkptPath);
    const hasValidModel = Boolean(activeCkpt && activeCkpt.exists);

    if (!hasValidModel) {
      setErrorMsg('尚未指定或配置有效的 TimesFM 3.0 模型权重，请先在下方点击【选择本地模型目录】');
      return;
    }
    const ckptPath = activeCkpt!.path;

    setIsLoading(true);
    setIsForecasting(true);
    setErrorMsg(null);

    try {
      const resp = await runForecast({
        file_path: inspection.file_path,
        target_cols: selectedTargets,
        date_col: selectedDateCol,
        horizon,
        seasonal_period: seasonalPeriod && seasonalPeriod >= 2 ? seasonalPeriod : null,
        checkpoint_path: ckptPath,
        make_positive: makePositive,
        use_symmetric_averaging: useSymmetric,
        use_boundary_smoothing: useSmoothing,
        use_robust_znorm: false,
        is_transposed: inspection.is_transposed,
      });

      setForecastProgress(100);
      setForecastPhase(4);
      // Brief pause to let user see 100% completion before smooth transition
      await new Promise((r) => setTimeout(r, 250));

      setForecastResult(resp);
      setActiveSeriesTab(0);
      setCurrentStep(4);
    } catch (err: any) {
      setErrorMsg(`预测推理执行失败: ${err.toString()}`);
    } finally {
      setIsLoading(false);
      setIsForecasting(false);
    }
  };

  // Export current active series to CSV
  const handleExportCsv = async () => {
    if (!forecastResult) return;
    const activeData = forecastResult.series[activeSeriesTab];
    if (!activeData) return;

    try {
      const defaultName = `${activeData.column_name}_forecast_${forecastResult.horizon}.csv`;
      await exportForecastCsv(defaultName, {
        series_name: activeData.column_name,
        timestamps: activeData.forecast_timestamps,
        median: activeData.median,
        q10: activeData.q10,
        q90: activeData.q90,
      });
      alert(`已成功导出到当前目录文件: ${defaultName}`);
    } catch (err: any) {
      alert(`导出失败: ${err.toString()}`);
    }
  };

  const steps = [
    { num: 1, title: '导入数据', desc: '上传本地表格或选择示例' },
    { num: 2, title: '指标与时间', desc: '智能识别时间与待预测指标' },
    { num: 3, title: '预测设置', desc: '设定推断长度与精度档位' },
    { num: 4, title: '结果看板', desc: '交互式趋势图与数据导出' },
  ];

  return (
    <div className="flex flex-col min-h-screen bg-slate-50 text-slate-900 selection:bg-indigo-100 selection:text-indigo-900">
      {/* Top Header */}
      <header className="bg-white border-b border-slate-200/80 sticky top-0 z-30 px-6 sm:px-8 py-3.5 shadow-xs">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Brand Logo */}
            <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center shadow-xs text-white select-none">
              <Activity className="w-5 h-5" />
            </div>
            <div className="flex flex-wrap items-baseline gap-2.5">
              <div className="flex items-center gap-1.5">
                <span className="font-extrabold text-slate-950 tracking-tight text-lg">TimesFM</span>
                <span className="text-[11px] font-bold font-mono text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded-full border border-indigo-200 shadow-2xs">
                  v3.0
                </span>
              </div>
              <span className="hidden md:inline text-xs text-slate-500 pl-2 border-l border-slate-200">
                智能时间序列预测工作台
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 text-slate-600 border border-slate-200 text-[11px] font-medium">
              <ShieldCheck className="w-3.5 h-3.5 text-slate-500" />
              <span>本地离线运行 · 数据安全</span>
            </span>
            {(() => {
              const activeCkpt = checkpoints.find((c) => c.path === selectedCkptPath);
              const hasValidModel = Boolean(activeCkpt && activeCkpt.exists);
              return hasValidModel ? (
                <div
                  className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs font-medium cursor-pointer hover:bg-emerald-100 transition-colors"
                  onClick={() => setCurrentStep(3)}
                  title={`当前已加载: ${activeCkpt?.path}`}
                >
                  <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                  <span className="text-[11px] font-medium">模型已就绪</span>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => setCurrentStep(3)}
                  className="flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 hover:bg-amber-100 text-amber-800 border border-amber-200 text-xs font-medium cursor-pointer transition-colors"
                >
                  <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
                  <span className="text-[11px] font-medium">未配置模型 (点击配置)</span>
                </button>
              );
            })()}
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-6 sm:px-8 py-8 space-y-8">
        {/* Step Progress Bar */}
        <nav className="bg-white p-1.5 rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-1.5">
            {steps.map((s) => {
              const isActive = currentStep === s.num;
              const isDone = currentStep > s.num;
              return (
                <button
                  key={s.num}
                  onClick={() => {
                    if (isDone || (s.num === 3 && inspection) || (s.num === 2 && inspection)) {
                      setCurrentStep(s.num);
                    }
                  }}
                  className={`flex items-center gap-3 px-4 py-2.5 rounded-xl text-left transition-all duration-150 ${
                    isActive
                      ? 'bg-indigo-50/70 border border-indigo-200 text-slate-950 shadow-xs ring-1 ring-indigo-200/50 font-medium'
                      : isDone
                      ? 'cursor-pointer hover:bg-slate-50 text-slate-700'
                      : 'opacity-45 cursor-default text-slate-400'
                  }`}
                >
                  <div
                    className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 transition-transform ${
                      isActive
                        ? 'bg-indigo-600 text-white shadow-xs scale-105'
                        : isDone
                        ? 'bg-emerald-600 text-white'
                        : 'bg-slate-100 text-slate-500 border border-slate-200'
                    }`}
                  >
                    {isDone ? <CheckCircle2 className="w-4 h-4 text-white" /> : s.num}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-semibold truncate leading-tight">{s.title}</div>
                    <div className="text-[10px] text-slate-500 truncate mt-0.5">{s.desc}</div>
                  </div>
                </button>
              );
            })}
          </div>
        </nav>

        {/* Global Error Banner */}
        {errorMsg && (
          <div className="bg-rose-50 border border-rose-200 text-rose-800 px-4 py-3 rounded-2xl flex items-center gap-3 text-xs shadow-xs">
            <AlertCircle className="w-4.5 h-4.5 text-rose-600 shrink-0" />
            <span className="flex-1">{errorMsg}</span>
            <button
              onClick={() => setErrorMsg(null)}
              className="text-[11px] font-semibold text-rose-700 hover:text-rose-900 hover:underline px-2 py-1 rounded"
            >
              忽略
            </button>
          </div>
        )}

        {/* Floating CSV Inspection Notification Pill */}
        {isInspecting && (
          <div className="fixed top-5 left-1/2 -translate-x-1/2 z-50 bg-slate-900/95 backdrop-blur-md text-white px-5 py-3 rounded-2xl shadow-2xl border border-slate-700/80 flex items-center gap-3.5 text-xs animate-in fade-in slide-in-from-top-3 duration-150 select-none">
            <RefreshCw className="w-4 h-4 animate-spin text-indigo-400 shrink-0" />
            <div className="flex items-center gap-2">
              <span className="font-bold text-slate-100">{inspectingFileName || '时序数据'}</span>
              <span className="text-slate-400">·</span>
              <span className="text-slate-300">{inspectingPhase}</span>
            </div>
            <span className="font-mono text-indigo-300 font-bold bg-indigo-950/80 px-2.5 py-0.5 rounded border border-indigo-800/50">
              {inspectElapsedSec.toFixed(1)}s
            </span>
          </div>
        )}

        {/* ========================================================================= */}
        {/* STEP 1: IMPORT DATA                                                      */}
        {/* ========================================================================= */}
        {currentStep === 1 && (
          <div className="space-y-8 max-w-4xl mx-auto">
            {/* Upload Canvas Card */}
            <div className="bg-white rounded-2xl border border-slate-200/80 p-8 sm:p-10 shadow-xs">
              <div
                onClick={isInspecting ? undefined : handlePickFile}
                onDragOver={isInspecting ? undefined : handleDragOver}
                onDragLeave={isInspecting ? undefined : handleDragLeave}
                onDrop={isInspecting ? undefined : handleDrop}
                className={`block border-2 border-dashed ${
                  isInspecting
                    ? 'border-indigo-400 bg-indigo-50/40 cursor-wait'
                    : isDragOver
                    ? 'border-indigo-500 bg-indigo-50/50 scale-[1.005] cursor-pointer'
                    : 'border-slate-300 hover:border-indigo-500 bg-slate-50/50 hover:bg-indigo-50/20 cursor-pointer'
                } rounded-2xl py-12 px-6 transition-all duration-150 text-center group relative overflow-hidden`}
              >
                {isInspecting ? (
                  <div className="space-y-4 py-2 animate-in fade-in zoom-in-95 duration-150">
                    <div className="w-14 h-14 rounded-2xl bg-indigo-100 border border-indigo-200 text-indigo-700 shadow-sm flex items-center justify-center mx-auto">
                      <RefreshCw className="w-7 h-7 animate-spin text-indigo-600" />
                    </div>
                    <div>
                      <div className="text-base font-bold text-slate-950 flex items-center justify-center gap-2">
                        <span>正在解析时序数据表格</span>
                        <span className="text-xs font-mono font-bold text-indigo-700 bg-indigo-100/80 px-2.5 py-0.5 rounded-full border border-indigo-200">
                          {inspectElapsedSec.toFixed(1)}s
                        </span>
                      </div>
                      <div className="text-xs font-semibold text-indigo-800 mt-1.5 flex items-center justify-center gap-1.5">
                        <FileSpreadsheet className="w-3.5 h-3.5" />
                        <span className="font-mono">{inspectingFileName}</span>
                      </div>
                      <div className="text-xs text-slate-500 mt-2 max-w-md mx-auto leading-relaxed">
                        {inspectingPhase}
                      </div>
                    </div>

                    <div className="max-w-xs mx-auto pt-1">
                      <div className="w-full h-1.5 bg-indigo-100 rounded-full overflow-hidden">
                        <div className="h-full bg-indigo-600 rounded-full animate-pulse w-3/4 mx-auto" />
                      </div>
                    </div>

                    <div className="inline-flex items-center gap-2 bg-slate-200 text-slate-500 text-xs font-semibold px-6 py-2.5 rounded-xl border border-slate-300 select-none">
                      <RefreshCw className="w-4 h-4 animate-spin text-slate-500" />
                      <span>正在扫描字段类型与推断时间周期...</span>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="w-14 h-14 rounded-2xl bg-indigo-50 border border-indigo-100 text-indigo-600 shadow-xs flex items-center justify-center group-hover:scale-105 group-hover:bg-indigo-100 transition-all mx-auto mb-4">
                      <Upload className="w-6 h-6" />
                    </div>
                    <div className="text-base font-bold text-slate-900 group-hover:text-indigo-900 transition-colors">
                      {isDragOver ? '松开鼠标即可导入时序数据' : '拖拽 CSV 文件至此处，或点击浏览本地文件'}
                    </div>
                    <div className="text-xs text-slate-500 mt-1.5 mb-6 max-w-md mx-auto leading-relaxed">
                      支持各类业务时序数据（销售额、库存、用电量、服务器指标等，带表头与时间列即可）
                    </div>

                    <div className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold px-6 py-2.5 rounded-xl shadow-xs transition-all duration-150 hover:shadow">
                      <FileSpreadsheet className="w-4 h-4" />
                      <span>选择 CSV 文件</span>
                    </div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".csv,.tsv,.txt"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </>
                )}
              </div>
            </div>

            {/* If data was already inspected, show preview table */}
            {inspection && (
              <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
                <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                  <div className="flex items-center gap-2.5">
                    <span className="font-bold text-slate-900 text-xs">已加载数据预览</span>
                    <span className="text-[11px] font-mono font-semibold text-indigo-800 bg-indigo-50 px-2.5 py-0.5 rounded-full border border-indigo-200">
                      {inspection.total_rows} 行 × {inspection.total_columns} 列
                    </span>
                  </div>
                  <button
                    onClick={() => setCurrentStep(2)}
                    className="inline-flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold px-4.5 py-2 rounded-xl transition-all shadow-xs"
                  >
                    <span>下一步：确认数据字段</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="overflow-x-auto border border-gray-200/80 rounded-xl">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-gray-50 text-gray-600 font-semibold border-b border-gray-200/80 text-[11px]">
                      <tr>
                        {inspection.preview_headers.slice(0, 8).map((h, i) => (
                          <th key={i} className="px-3.5 py-2.5">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 font-mono text-gray-700 text-xs">
                      {inspection.preview_rows.slice(0, 5).map((row, rIdx) => (
                        <tr key={rIdx} className="hover:bg-slate-50 transition-colors">
                          {row.slice(0, 8).map((cell, cIdx) => (
                            <td key={cIdx} className="px-3.5 py-2 truncate max-w-[140px]">
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* STEP 2: GUIDED MAPPING & HEALTH CHECK                                    */}
        {/* ========================================================================= */}
        {currentStep === 2 && inspection && (
          <div className="space-y-6">
            <div className="bg-white border border-indigo-100 rounded-2xl p-4.5 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-700 shadow-xs">
              <div className="flex items-center gap-2.5">
                <div className="w-6 h-6 rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center shrink-0">
                  <Sparkles className="w-3.5 h-3.5" />
                </div>
                <span>
                  <strong className="text-slate-900">数据智能识别完毕：</strong>已自动解析出时间列与数值列。您可以根据业务需要在右侧勾选预测指标。
                </span>
              </div>
              <span className="text-[11px] font-mono text-slate-400 bg-slate-50 px-2 py-0.5 rounded border border-slate-200">
                {inspection.file_path}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Left Column: Time & Seasonality Settings */}
              <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-6">
                <div className="flex items-center gap-2.5 pb-3.5 border-b border-slate-100">
                  <div className="w-7 h-7 rounded-lg bg-indigo-50 text-indigo-700 flex items-center justify-center">
                    <Calendar className="w-4 h-4" />
                  </div>
                  <h3 className="font-bold text-slate-900 text-xs">1. 时间与周期规律</h3>
                </div>

                {/* Date Column Selector */}
                <div className="space-y-2">
                  <label className="block text-xs font-semibold text-slate-800">时间列 (Timestamp)</label>
                  <select
                    value={selectedDateCol || ''}
                    onChange={(e) => setSelectedDateCol(e.target.value || null)}
                    className="w-full bg-slate-50/70 border border-slate-300 rounded-xl px-3.5 py-2.5 text-xs text-slate-900 focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 transition-all font-medium"
                  >
                    <option value="">🚫 无时间列 (按自然顺序排列)</option>
                    {inspection.columns.map((c) => (
                      <option key={c.name} value={c.name}>
                        {c.name} {c.is_datetime ? '📅 (识别为时间格式)' : ''}
                      </option>
                    ))}
                  </select>
                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    指定时间列后，系统将在预测图表上自动推算并标注真实的未来日期与时间。
                  </p>
                </div>

                {/* Detected Frequency */}
                {inspection.inferred_frequency && (
                  <div className="bg-blue-50/60 border border-blue-200 rounded-xl p-3.5 text-xs text-blue-900 flex items-center justify-between">
                    <div className="flex items-center gap-2 text-blue-800">
                      <Clock className="w-3.5 h-3.5 text-blue-600" />
                      <span className="font-medium">检测到采样间隔:</span>
                    </div>
                    <span className="font-bold font-mono bg-white text-blue-800 px-2.5 py-0.5 rounded-lg border border-blue-200 shadow-2xs">
                      {inspection.inferred_frequency}
                    </span>
                  </div>
                )}

                {/* Seasonality Guidance */}
                <div className="space-y-3 pt-3 border-t border-slate-100">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-semibold text-slate-800">业务循环规律</label>
                    <span className="text-[11px] font-mono text-indigo-900 font-bold bg-indigo-50 px-2.5 py-0.5 rounded-full border border-indigo-200">
                      {seasonalPeriod ? `每 ${seasonalPeriod} 条数据一循环` : '全自动识别'}
                    </span>
                  </div>

                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    若数据有固定的重复起伏节律（如每天24小时、每周7天），可指定多少条数据为一个循环周期；如果不确定，保持自动识别即可。
                  </p>

                  <div className="grid grid-cols-2 gap-2.5">
                    {/* Option A: Auto */}
                    <button
                      type="button"
                      onClick={() => setSeasonalPeriod(null)}
                      className={`p-3 text-xs rounded-xl border text-left transition-all cursor-pointer flex flex-col justify-between ${
                        seasonalPeriod === null
                          ? 'border-indigo-600 bg-indigo-50/70 text-indigo-950 font-semibold ring-1 ring-indigo-400 shadow-xs'
                          : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50 text-slate-700 bg-white'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-slate-900">自动识别 (推荐)</span>
                        {seasonalPeriod === null && <Check className="w-3.5 h-3.5 text-indigo-600" />}
                      </div>
                      <div className="text-[10px] text-slate-500 mt-1">
                        由 TimesFM 大模型注意力机制自主推断规律，适合无明显规律或不确定的数据
                      </div>
                    </button>

                    {/* Option B: Custom Period */}
                    <button
                      type="button"
                      onClick={() => {
                        if (seasonalPeriod === null) {
                          setSeasonalPeriod(inspection.recommended_period || 24);
                        }
                      }}
                      className={`p-3 text-xs rounded-xl border text-left transition-all cursor-pointer flex flex-col justify-between ${
                        seasonalPeriod !== null
                          ? 'border-indigo-600 bg-indigo-50/70 text-indigo-950 font-semibold ring-1 ring-indigo-400 shadow-xs'
                          : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50 text-slate-700 bg-white'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-slate-900">自定义循环周期</span>
                        {seasonalPeriod !== null && <Check className="w-3.5 h-3.5 text-indigo-600" />}
                      </div>
                      <div className="text-[10px] text-slate-500 mt-1">
                        手动指定多少条数据算一个周期，强制约束模型贴合业务规律
                      </div>
                    </button>
                  </div>

                  {/* If custom period is active, show the input box + quick pills */}
                  {seasonalPeriod !== null && (
                    <div className="bg-slate-50/90 border border-slate-200/80 rounded-xl p-3.5 space-y-2.5 animate-in fade-in slide-in-from-top-1 duration-150">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-700 shrink-0">每</span>
                        <div className="relative w-28">
                          <input
                            type="number"
                            min="2"
                            max="2048"
                            value={seasonalPeriod}
                            onChange={(e) => {
                              const val = parseInt(e.target.value, 10);
                              setSeasonalPeriod(isNaN(val) ? 2 : Math.max(2, val));
                            }}
                            className="w-full bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-xs font-bold text-indigo-950 font-mono text-center focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600"
                          />
                        </div>
                        <span className="text-xs font-semibold text-slate-700">条数据算一个循环周期</span>
                      </div>

                      <div className="flex items-center gap-1.5 flex-wrap pt-1 border-t border-slate-200/60">
                        <span className="text-[10px] text-slate-400 mr-1">快捷填入:</span>
                        {[
                          { val: 24, desc: '24 条 (如小时)' },
                          { val: 7, desc: '7 条 (如天)' },
                          { val: 30, desc: '30 条 (如月)' },
                          { val: 12, desc: '12 条 (如月份)' },
                          { val: 96, desc: '96 条 (15分钟)' },
                        ].map((pill) => (
                          <button
                            key={pill.val}
                            type="button"
                            onClick={() => setSeasonalPeriod(pill.val)}
                            className={`px-2 py-0.5 text-[11px] rounded-md border transition-all cursor-pointer ${
                              seasonalPeriod === pill.val
                                ? 'bg-indigo-600 text-white border-indigo-600 font-semibold shadow-2xs'
                                : 'bg-white hover:bg-slate-100 text-slate-600 border-slate-200'
                            }`}
                          >
                            {pill.desc}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Right Column: Target Columns Selector */}
              <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs flex flex-col space-y-4">
                <div className="flex items-center justify-between pb-3.5 border-b border-slate-100">
                  <div className="flex items-center gap-2.5">
                    <div className="w-7 h-7 rounded-lg bg-indigo-50 text-indigo-700 flex items-center justify-center">
                      <Layers className="w-4 h-4" />
                    </div>
                    <h3 className="font-bold text-slate-900 text-xs">2. 选择待预测指标</h3>
                  </div>
                  <button
                    onClick={selectAllTargets}
                    className="text-[11px] text-indigo-600 hover:text-indigo-800 font-bold transition-colors"
                  >
                    全选全部指标
                  </button>
                </div>

                <div className="flex-1 overflow-y-auto max-h-[340px] space-y-2.5 pr-1">
                  {inspection.columns.map((c) => {
                    const isTime = c.name === selectedDateCol;
                    const isTarget = selectedTargets.includes(c.name);

                    return (
                      <div
                        key={c.name}
                        onClick={() => {
                          if (!isTime && c.is_numeric) {
                            toggleTarget(c.name);
                          }
                        }}
                        className={`p-3 rounded-xl border text-xs flex items-center justify-between transition-all cursor-pointer ${
                          isTime
                            ? 'bg-slate-50/60 border-slate-200 opacity-60 cursor-not-allowed'
                            : isTarget
                            ? 'border-indigo-500 bg-indigo-50/40 ring-1 ring-indigo-300 shadow-2xs'
                            : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50/60 bg-white'
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <input
                            type="checkbox"
                            checked={isTarget}
                            disabled={isTime || !c.is_numeric}
                            onChange={() => {}}
                            className="w-4 h-4 rounded text-indigo-600 accent-indigo-600 focus:ring-indigo-400"
                          />
                          <div>
                            <div className="font-bold text-slate-900">{c.name}</div>
                            <div className="text-[11px] text-slate-500 font-mono mt-0.5">
                              样本值: {c.sample_values.slice(0, 3).join(', ')}
                            </div>
                          </div>
                        </div>

                        <div>
                          {isTime ? (
                            <span className="text-[10px] font-medium text-slate-600 bg-slate-100 px-2 py-0.5 rounded-full border border-slate-200">
                              时间列
                            </span>
                          ) : isTarget ? (
                            <span className="text-[10px] font-bold bg-indigo-600 text-white px-2.5 py-0.5 rounded-full shadow-2xs">
                              待预测
                            </span>
                          ) : (
                            <span className="text-[10px] text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full">
                              不预测
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="text-xs text-slate-500 pt-2 border-t border-slate-100">
                  当前已勾选 <strong className="text-slate-900 font-bold">{selectedTargets.length}</strong> 个预测指标。
                  {selectedTargets.length > 1
                    ? '将进行多指标联合推断。'
                    : '将进行单指标独立推断。'}
                </div>
              </div>
            </div>

            {/* Bottom: Data Health Check */}
            <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-xs flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center shrink-0">
                  <CheckCircle2 className="w-5 h-5" />
                </div>
                <div className="text-xs">
                  <div className="font-bold text-slate-900 flex items-center gap-1.5">
                    <span>数据健康校验</span>
                    <span className="text-[10px] bg-emerald-100 text-emerald-800 font-semibold px-2 py-0.2 rounded-full border border-emerald-200">
                      通过
                    </span>
                  </div>
                  <div className="text-slate-500 mt-0.5">{inspection.health_note}</div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => setCurrentStep(1)}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-700 hover:text-slate-950 px-4 py-2 rounded-xl border border-slate-200 hover:bg-slate-50 transition-colors"
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                  <span>重新选择数据</span>
                </button>
                <button
                  onClick={() => setCurrentStep(3)}
                  disabled={selectedTargets.length === 0}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2.5 rounded-xl shadow-xs transition-all disabled:opacity-50"
                >
                  <span>下一步：预测参数设置</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* STEP 3: FORECAST CONFIGURATION                                           */}
        {/* ========================================================================= */}
        {currentStep === 3 && (
          <div className="space-y-6 max-w-4xl mx-auto">
            {/* Horizon Slider Card */}
            {(() => {
              const freqCfg = getFrequencyConfig();
              return (
                <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3 pb-3.5 border-b border-slate-100">
                    <div className="flex items-center gap-2.5">
                      <div className="w-7 h-7 rounded-lg bg-indigo-50 text-indigo-700 flex items-center justify-center">
                        <TrendingUp className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="font-bold text-slate-900 text-xs">1. 预测未来多少条数据</h3>
                        <div className="text-[11px] text-slate-400 mt-0.5">{freqCfg.stepDesc}</div>
                      </div>
                    </div>
                    <div className="text-sm font-bold text-indigo-900 bg-indigo-50 px-3.5 py-1.5 rounded-xl border border-indigo-200 shadow-2xs font-mono flex items-center gap-2">
                      <span>{horizon} 条数据</span>
                      <span className="font-sans text-xs font-semibold text-indigo-700 pl-2 border-l border-indigo-200">
                        {freqCfg.formatDuration(horizon)}
                      </span>
                    </div>
                  </div>

                  <div className="bg-indigo-50/50 rounded-xl p-3 border border-indigo-100/80 flex items-start gap-2.5 text-xs text-slate-600">
                    <span className="text-base leading-none mt-0.5">💡</span>
                    <div className="leading-relaxed">
                      选择您希望向未来预测生成的数据条数（每多预测 1 条数据，代表时间向后推移 1 个{freqCfg.unitName}）。您可以拖动滑块调整，或直接点击下方快捷预设：
                    </div>
                  </div>

                  <input
                    type="range"
                    min={freqCfg.min}
                    max={freqCfg.max}
                    step={freqCfg.step}
                    value={horizon}
                    onChange={(e) => setHorizon(parseInt(e.target.value, 10))}
                    className="w-full h-2.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                  />

                  <div className="flex flex-wrap gap-2.5 pt-1">
                    {freqCfg.presets.map((preset) => (
                      <button
                        key={preset.steps}
                        onClick={() => setHorizon(preset.steps)}
                        className={`px-3.5 py-1.5 text-xs rounded-xl border font-semibold transition-all cursor-pointer ${
                          horizon === preset.steps
                            ? 'bg-indigo-600 text-white border-indigo-600 shadow-2xs'
                            : 'bg-white hover:bg-slate-50 text-slate-700 border-slate-200 hover:border-indigo-300'
                        }`}
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* Checkpoints Card */}
            {(() => {
              const activeCkpt = checkpoints.find((c) => c.path === selectedCkptPath);
              const hasValidModel = Boolean(activeCkpt && activeCkpt.exists);

              return (
                <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3 pb-3.5 border-b border-slate-100">
                    <div className="flex items-center gap-2.5">
                      <div className="w-7 h-7 rounded-lg bg-indigo-50 text-indigo-700 flex items-center justify-center">
                        <Database className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="font-bold text-slate-900 text-xs">2. TimesFM 3.0 模型权重管理</h3>
                        <div className="text-[11px] text-slate-400 mt-0.5">
                          客户端不随安装包内置 1GB+ 权重，请指定电脑本地包含 model.safetensors 的模型目录
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handlePickModelDirectory}
                        disabled={isVerifyingModel}
                        className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-bold rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white shadow-2xs transition-all cursor-pointer disabled:opacity-50"
                      >
                        <FolderOpen className="w-3.5 h-3.5" />
                        <span>{isVerifyingModel ? '正在校验...' : '选择本地模型目录'}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => setShowModelDownloadGuide(!showModelDownloadGuide)}
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-semibold rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-600 transition-colors cursor-pointer"
                      >
                        <HelpCircle className="w-3.5 h-3.5 text-slate-500" />
                        <span>获取模型指引</span>
                      </button>
                    </div>
                  </div>

                  {/* Error Message if inspection fails */}
                  {modelError && (
                    <div className="p-3.5 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 flex items-start gap-2.5">
                      <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                      <div className="leading-relaxed flex-1">
                        <strong>模型加载失败：</strong> {modelError}
                      </div>
                      <button
                        onClick={() => setModelError(null)}
                        className="text-rose-500 hover:text-rose-700 text-xs font-bold"
                      >
                        关闭
                      </button>
                    </div>
                  )}

                  {/* Active Selected Model Display */}
                  {hasValidModel ? (
                    <div className="bg-emerald-50/50 border border-emerald-200/80 rounded-2xl p-4 space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
                          <span className="font-bold text-slate-900 text-xs">{activeCkpt?.name}</span>
                          <span className="text-[10px] bg-emerald-100 text-emerald-800 font-semibold px-2 py-0.5 rounded-full border border-emerald-300">
                            {activeCkpt?.precision}
                          </span>
                        </div>
                        <span className="text-xs font-mono font-medium text-emerald-800">
                          {activeCkpt?.size_desc}
                        </span>
                      </div>

                      <div className="flex items-center justify-between gap-3 bg-white/80 border border-emerald-200/60 rounded-xl px-3 py-2 text-xs">
                        <div className="flex items-center gap-2 min-w-0 flex-1 font-mono text-slate-600 text-[11px]">
                          <span className="text-slate-400 shrink-0">文件路径:</span>
                          <span className="truncate select-all text-slate-800" title={activeCkpt?.path}>
                            {activeCkpt?.path}
                          </span>
                        </div>
                        <button
                          type="button"
                          onClick={handlePickModelDirectory}
                          className="shrink-0 text-xs font-semibold text-indigo-700 hover:text-indigo-900 underline cursor-pointer"
                        >
                          更换路径
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-amber-50/70 border border-amber-200 rounded-2xl p-5 space-y-4">
                      <div className="flex items-start gap-3">
                        <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                        <div className="space-y-1">
                          <h4 className="font-bold text-amber-950 text-xs">尚未配置本地 TimesFM 3.0 模型权重</h4>
                          <p className="text-xs text-amber-800/90 leading-relaxed">
                            本软件发布包体积精简，未随安装包内置 1GB+ 的模型大文件。首次使用请指定您本地的模型存放目录（须包含 <code className="bg-amber-100 px-1 py-0.5 rounded font-mono text-amber-900">model.safetensors</code>）。
                          </p>
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center gap-3 pt-1">
                        <button
                          type="button"
                          onClick={handlePickModelDirectory}
                          disabled={isVerifyingModel}
                          className="px-4 py-2.5 text-xs font-bold bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl shadow-xs transition-all flex items-center gap-2 cursor-pointer disabled:opacity-50"
                        >
                          <FolderOpen className="w-4 h-4" />
                          <span>{isVerifyingModel ? '正在读取校验...' : '点击浏览并选择模型目录'}</span>
                        </button>

                        <div className="flex items-center gap-2 flex-1 min-w-[260px]">
                          <input
                            type="text"
                            placeholder="或直接粘贴模型文件夹绝对路径..."
                            value={customModelPathInput}
                            onChange={(e) => setCustomModelPathInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleApplyCustomPathInput()}
                            className="flex-1 bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600"
                          />
                          <button
                            type="button"
                            onClick={handleApplyCustomPathInput}
                            disabled={!customModelPathInput.trim() || isVerifyingModel}
                            className="px-3.5 py-2 text-xs font-semibold bg-slate-800 hover:bg-slate-900 text-white rounded-xl transition-all cursor-pointer disabled:opacity-40 shrink-0"
                          >
                            加载
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Collapsible Download & Installation Guide */}
                  {showModelDownloadGuide && (
                    <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5 space-y-3.5 text-xs text-slate-700 animate-in fade-in duration-150">
                      <div className="flex items-center justify-between pb-2 border-b border-slate-200">
                        <span className="font-bold text-slate-900 flex items-center gap-1.5">
                          <Download className="w-4 h-4 text-indigo-600" />
                          <span>如何获取 TimesFM 3.0 模型权重文件？</span>
                        </span>
                        <button
                          onClick={() => setShowModelDownloadGuide(false)}
                          className="text-slate-400 hover:text-slate-600 text-xs"
                        >
                          收起指引
                        </button>
                      </div>

                      <p className="leading-relaxed text-slate-600">
                        模型文件包含两个核心文件：<strong className="text-slate-900 font-mono">model.safetensors</strong>（权重，约730MB）和 <strong className="text-slate-900 font-mono">config.json</strong>（配置文件）。您可以从以下任意渠道下载并保存在电脑任意位置：
                      </p>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                        <div className="bg-white p-3.5 rounded-xl border border-slate-200 space-y-1.5">
                          <div className="font-bold text-slate-900 flex items-center justify-between">
                            <span>🚀 国内高速：魔搭社区 (ModelScope)</span>
                            <span className="text-[10px] bg-red-50 text-red-600 border border-red-200 px-1.5 py-0.2 rounded font-semibold">推荐</span>
                          </div>
                          <div className="text-[11px] text-slate-500 leading-relaxed">
                            无需代理直接满速下载，可在终端执行：
                          </div>
                          <code className="block bg-slate-900 text-emerald-400 p-2 rounded-lg font-mono text-[11px] select-all overflow-x-auto">
                            modelscope download --model czxichen/timesfm3.0-balanced --local_dir ./models/timesfm3
                          </code>
                        </div>

                        <div className="bg-white p-3.5 rounded-xl border border-slate-200 space-y-1.5">
                          <div className="font-bold text-slate-900 flex items-center justify-between">
                            <span>🌐 官方开源：Hugging Face</span>
                            <span className="text-[10px] bg-slate-100 text-slate-600 px-1.5 py-0.2 rounded font-semibold">官方</span>
                          </div>
                          <div className="text-[11px] text-slate-500 leading-relaxed">
                            国际开发者首选，支持直接网页下载或使用 CLI：
                          </div>
                          <code className="block bg-slate-900 text-emerald-400 p-2 rounded-lg font-mono text-[11px] select-all overflow-x-auto">
                            huggingface-cli download czxichen/timesfm3.0-balanced --local-dir ./models/timesfm3
                          </code>
                        </div>
                      </div>

                      <div className="text-[11px] text-slate-500 pt-1">
                        💡 下载完成后，点击上方的 <strong>「选择本地模型目录」</strong> 按钮选择该文件夹，系统将自动识别并永久记住该路径。
                      </div>
                    </div>
                  )}

                  {/* Candidate Models Switcher (if multiple available) */}
                  {checkpoints.filter((c) => c.exists).length > 1 && (
                    <div className="pt-2 border-t border-slate-100 space-y-2">
                      <div className="text-[11px] font-semibold text-slate-500">已检测到的其他本地模型：</div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
                        {checkpoints
                          .filter((c) => c.exists)
                          .map((ckpt) => {
                            const isCurrent = selectedCkptPath === ckpt.path;
                            return (
                              <div
                                key={ckpt.path}
                                onClick={() => {
                                  setSelectedCkptPath(ckpt.path);
                                  localStorage.setItem('timesfm3_model_path', ckpt.path);
                                }}
                                className={`p-3 rounded-xl border text-xs cursor-pointer transition-all flex flex-col justify-between ${
                                  isCurrent
                                    ? 'border-indigo-600 bg-indigo-50/60 ring-1 ring-indigo-500/20 font-semibold'
                                    : 'border-slate-200 hover:border-indigo-300 hover:bg-slate-50/50 bg-white'
                                }`}
                              >
                                <div className="font-bold text-slate-900 truncate">{ckpt.name}</div>
                                <div className="text-[10px] text-slate-500 mt-1 truncate">{ckpt.size_desc}</div>
                              </div>
                            );
                          })}
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Advanced Switches Card */}
            <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-4">
              <div className="flex items-center gap-2.5 pb-3.5 border-b border-slate-100">
                <div className="w-7 h-7 rounded-lg bg-slate-100 text-slate-700 flex items-center justify-center">
                  <Settings2 className="w-4 h-4" />
                </div>
                <h3 className="font-bold text-slate-900 text-xs">3. 走势优化选项</h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <label className="flex items-start gap-3 p-4 rounded-xl border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50/20 cursor-pointer transition-all">
                  <input
                    type="checkbox"
                    checked={useSymmetric}
                    onChange={(e) => setUseSymmetric(e.target.checked)}
                    className="w-4 h-4 rounded text-indigo-600 accent-indigo-600 mt-0.5"
                  />
                  <div>
                    <div className="font-bold text-slate-900">抗噪稳定性增强</div>
                    <div className="text-[11px] text-slate-500 mt-0.5">抑制历史偶发噪点，提升中远期预测稳定性</div>
                  </div>
                </label>

                <label className="flex items-start gap-3 p-4 rounded-xl border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50/20 cursor-pointer transition-all">
                  <input
                    type="checkbox"
                    checked={useSmoothing}
                    onChange={(e) => setUseSmoothing(e.target.checked)}
                    className="w-4 h-4 rounded text-indigo-600 accent-indigo-600 mt-0.5"
                  />
                  <div>
                    <div className="font-bold text-slate-900">衔接平滑优化</div>
                    <div className="text-[11px] text-slate-500 mt-0.5">平滑消除历史实测终点与预测起点的断层</div>
                  </div>
                </label>

                <label className="flex items-start gap-3 p-4 rounded-xl border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50/20 cursor-pointer transition-all">
                  <input
                    type="checkbox"
                    checked={makePositive}
                    onChange={(e) => setMakePositive(e.target.checked)}
                    className="w-4 h-4 rounded text-indigo-600 accent-indigo-600 mt-0.5"
                  />
                  <div>
                    <div className="font-bold text-slate-900">物理非负约束</div>
                    <div className="text-[11px] text-slate-500 mt-0.5">强制结果 ≥ 0 (适合销量、库存、用电等)</div>
                  </div>
                </label>
              </div>
            </div>

            {/* Step 3 localized error alert if any */}
            {errorMsg && (
              <div className="bg-rose-50 border border-rose-200 text-rose-800 px-4 py-3 rounded-2xl flex items-center gap-3 text-xs shadow-xs animate-in fade-in duration-150">
                <AlertCircle className="w-4.5 h-4.5 text-rose-600 shrink-0" />
                <span className="flex-1 font-medium">{errorMsg}</span>
                <button
                  onClick={() => setErrorMsg(null)}
                  className="text-[11px] font-semibold text-rose-700 hover:text-rose-900 hover:underline px-2 py-1 rounded cursor-pointer"
                >
                  关闭
                </button>
              </div>
            )}

            {/* Bottom Actions */}
            {(() => {
              const activeCkpt = checkpoints.find((c) => c.path === selectedCkptPath);
              const hasValidModel = Boolean(activeCkpt && activeCkpt.exists);

              return (
                <div className="flex items-center justify-between pt-2">
                  <button
                    onClick={() => setCurrentStep(2)}
                    disabled={isLoading}
                    className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-700 hover:text-slate-950 px-4.5 py-2.5 rounded-xl border border-slate-200 hover:bg-slate-50 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <ArrowLeft className="w-3.5 h-3.5" />
                    <span>返回修改设置</span>
                  </button>

                  <button
                    onClick={handleRunForecast}
                    disabled={isLoading || !hasValidModel}
                    className={`inline-flex items-center gap-2 text-xs font-bold px-8 py-3 rounded-xl transition-all ${
                      isLoading || !hasValidModel
                        ? 'bg-slate-200 hover:bg-slate-200 text-slate-400 border border-slate-300 cursor-not-allowed shadow-none select-none'
                        : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm hover:shadow-md active:scale-[0.98] cursor-pointer'
                    }`}
                  >
                    {isLoading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin text-slate-400" />
                        <span>模型推断计算中 ({forecastElapsedSec.toFixed(1)}s)...</span>
                      </>
                    ) : !hasValidModel ? (
                      <>
                        <AlertTriangle className="w-4 h-4 text-slate-400" />
                        <span>请先在上方指定模型目录</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 fill-white text-white" />
                        <span>开始预测分析</span>
                      </>
                    )}
                  </button>
                </div>
              );
            })()}
          </div>
        )}

        {/* ========================================================================= */}
        {/* STEP 4: RESULTS & INTERACTIVE VISUALIZATION                              */}
        {/* ========================================================================= */}
        {currentStep === 4 && forecastResult && (
          <div className="space-y-6">
            {/* Top Success Bar */}
            <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-xs flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3.5">
                <div className="w-10 h-10 rounded-2xl bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center justify-center font-bold text-lg shadow-2xs">
                  ✓
                </div>
                <div>
                  <h3 className="font-bold text-slate-950 text-sm">
                    预测完成！未来 {forecastResult.horizon} 条数据已推断完毕
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    计算耗时 <strong className="text-indigo-800 font-mono bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200/80">{forecastResult.elapsed_ms} ms</strong> ·
                    共预测 {forecastResult.series.length} 个指标
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => setCurrentStep(3)}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-700 hover:text-slate-950 px-4 py-2.5 rounded-xl border border-slate-200 hover:bg-slate-50 transition-colors cursor-pointer"
                >
                  <RefreshCw className="w-3.5 h-3.5 text-slate-500" />
                  <span>重新调整预测条数</span>
                </button>

                <button
                  onClick={handleExportCsv}
                  className="inline-flex items-center gap-1.5 text-xs font-bold bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2.5 rounded-xl shadow-xs hover:shadow transition-all"
                >
                  <Download className="w-4 h-4" />
                  <span>导出预测结果 CSV</span>
                </button>
              </div>
            </div>

            {/* Target Variables Tabs (Multivariate Navigation) */}
            {forecastResult.series.length > 1 && (
              <div className="flex flex-wrap items-center gap-2 border-b border-slate-200/80 pb-3">
                <span className="text-xs font-bold text-slate-500 mr-2 flex items-center gap-1">
                  <span>选择展示指标:</span>
                </span>
                {forecastResult.series.map((s, idx) => (
                  <button
                    key={s.column_name}
                    onClick={() => setActiveSeriesTab(idx)}
                    className={`px-4 py-2 text-xs rounded-xl font-bold transition-all ${
                      activeSeriesTab === idx
                        ? 'bg-indigo-600 text-white shadow-xs'
                        : 'bg-white text-slate-700 border border-slate-200 hover:border-indigo-300 hover:bg-slate-50'
                    }`}
                  >
                    {s.column_name}
                  </button>
                ))}
              </div>
            )}

            {/* Current Active Series Metrics & Fan Chart */}
            {forecastResult.series[activeSeriesTab] && (
              <div className="space-y-6">
                {/* 4 Summary Stat Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-xs hover:shadow-sm transition-shadow">
                    <div className="flex items-center justify-between text-[11px] text-slate-500 font-semibold">
                      <span>未来预测均值</span>
                      <div className="w-6 h-6 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center">
                        <BarChart3 className="w-3.5 h-3.5" />
                      </div>
                    </div>
                    <div className="text-2xl font-extrabold font-mono text-indigo-950 mt-2">
                      {forecastResult.series[activeSeriesTab].mean_forecast.toFixed(3)}
                    </div>
                  </div>

                  <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-xs hover:shadow-sm transition-shadow">
                    <div className="flex items-center justify-between text-[11px] text-slate-500 font-semibold">
                      <span>未来峰值 (最高点)</span>
                      <div className="w-6 h-6 rounded-lg bg-rose-50 text-rose-600 flex items-center justify-center">
                        <TrendingUp className="w-3.5 h-3.5" />
                      </div>
                    </div>
                    <div className="text-2xl font-extrabold font-mono text-rose-950 mt-2">
                      {forecastResult.series[activeSeriesTab].max_forecast.toFixed(3)}
                    </div>
                  </div>

                  <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-xs hover:shadow-sm transition-shadow">
                    <div className="flex items-center justify-between text-[11px] text-slate-500 font-semibold">
                      <span>未来谷值 (最低点)</span>
                      <div className="w-6 h-6 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
                        <TrendingUp className="w-3.5 h-3.5 rotate-180" />
                      </div>
                    </div>
                    <div className="text-2xl font-extrabold font-mono text-blue-950 mt-2">
                      {forecastResult.series[activeSeriesTab].min_forecast.toFixed(3)}
                    </div>
                  </div>

                  <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-xs hover:shadow-sm transition-shadow">
                    <div className="flex items-center justify-between text-[11px] text-slate-500 font-semibold">
                      <span>总体走向趋势</span>
                      <div className="w-6 h-6 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
                        <Zap className="w-3.5 h-3.5" />
                      </div>
                    </div>
                    <div className="mt-2">
                      <span className={`inline-flex items-center gap-1 text-base font-bold font-sans px-3 py-0.5 rounded-full ${
                        forecastResult.series[activeSeriesTab].trend === '上升'
                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                          : forecastResult.series[activeSeriesTab].trend === '下降'
                          ? 'bg-rose-50 text-rose-700 border border-rose-200'
                          : 'bg-slate-100 text-slate-700 border border-slate-200'
                      }`}>
                        {forecastResult.series[activeSeriesTab].trend}
                      </span>
                    </div>
                  </div>
                </div>

                {/* The Interactive Fan Chart */}
                <FanChart data={forecastResult.series[activeSeriesTab]} height={400} />

                {/* Forecast Table Preview */}
                <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-xs space-y-3.5">
                  <div className="flex items-center justify-between">
                    <h4 className="font-bold text-slate-950 text-xs">未来预测明细表 (前 10 条详情)</h4>
                    <span className="text-[11px] text-slate-500">
                      包含预测基准值与概率区间
                    </span>
                  </div>

                  <div className="overflow-x-auto border border-slate-200/80 rounded-xl">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200/80 text-[11px]">
                        <tr>
                          <th className="px-4 py-2.5">时间点 / 序号</th>
                          <th className="px-4 py-2.5 text-indigo-950 font-bold bg-indigo-50/70 border-x border-indigo-100">预测基准值 (中位数)</th>
                          <th className="px-4 py-2.5 text-slate-500">80% 极值下限</th>
                          <th className="px-4 py-2.5 text-slate-500">60% 核心下限</th>
                          <th className="px-4 py-2.5 text-slate-500">60% 核心上限</th>
                          <th className="px-4 py-2.5 text-slate-500">80% 极值上限</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 font-mono text-slate-700 text-xs">
                        {forecastResult.series[activeSeriesTab].median
                          .slice(0, 10)
                          .map((val, idx) => (
                            <tr key={idx} className="hover:bg-slate-50 transition-colors">
                              <td className="px-4 py-2.5 font-sans text-slate-600 font-medium">
                                {forecastResult.series[activeSeriesTab].forecast_timestamps[idx] || `第 ${idx + 1} 条`}
                              </td>
                              <td className="px-4 py-2.5 font-bold text-indigo-950 bg-indigo-50/40 border-x border-indigo-100/60 font-mono">
                                {val.toFixed(4)}
                              </td>
                              <td className="px-4 py-2.5 text-slate-500">
                                {forecastResult.series[activeSeriesTab].q10[idx]?.toFixed(4)}
                              </td>
                              <td className="px-4 py-2.5 text-slate-500">
                                {forecastResult.series[activeSeriesTab].q20[idx]?.toFixed(4)}
                              </td>
                              <td className="px-4 py-2.5 text-slate-500">
                                {forecastResult.series[activeSeriesTab].q80[idx]?.toFixed(4)}
                              </td>
                              <td className="px-4 py-2.5 text-slate-500">
                                {forecastResult.series[activeSeriesTab].q90[idx]?.toFixed(4)}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* ========================================================================= */}
      {/* INFERENCE PROGRESS MODAL                                                  */}
      {/* ========================================================================= */}
      {isForecasting && (
        <div className="fixed inset-0 z-50 bg-slate-950/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-3xl border border-slate-200/90 shadow-2xl p-7 max-w-lg w-full space-y-6 animate-in fade-in zoom-in-95 duration-150">
            {/* Header */}
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 shrink-0">
                <Activity className="w-6 h-6 animate-pulse" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <h3 className="text-base font-bold text-slate-900 truncate">
                    TimesFM 3.0 深度时序推断中
                  </h3>
                  <span className="text-xs font-mono font-bold text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-lg border border-indigo-100 shrink-0">
                    {forecastElapsedSec.toFixed(1)}s
                  </span>
                </div>
                <p className="text-xs text-slate-500 mt-0.5 truncate">
                  目标变量: {selectedTargets.join(', ')} · 预测条数: {horizon} 条 · 循环规律: {seasonalPeriod ? `${seasonalPeriod}点/周期` : '自动识别'}
                </p>
              </div>
            </div>

            {/* Animated Progress Bar */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold">
                <span className="text-slate-700">
                  {forecastPhase === 1 && '阶段 1/4: 提取历史时序与上下文切片'}
                  {forecastPhase === 2 && '阶段 2/4: 加载 TimesFM 3.0 深度权重与配置'}
                  {forecastPhase === 3 && '阶段 3/4: 多核并行残差注意力前向推断'}
                  {forecastPhase >= 4 && '阶段 4/4: 逆标准化与生成 10%/50%/90% 置信区间'}
                </span>
                <span className="text-indigo-600 font-mono font-bold">
                  {Math.round(forecastProgress)}%
                </span>
              </div>
              <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden p-0.5 border border-slate-200">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 via-indigo-600 to-indigo-700 rounded-full transition-all duration-150 ease-out relative"
                  style={{ width: `${Math.max(6, forecastProgress)}%` }}
                >
                  <div className="absolute inset-0 bg-white/20 animate-pulse rounded-full" />
                </div>
              </div>
            </div>

            {/* Phase Checklist */}
            <div className="space-y-2.5 bg-slate-50 rounded-2xl p-4 border border-slate-100 text-xs">
              <div className="flex items-center gap-2.5">
                <CheckCircle2
                  className={`w-4 h-4 shrink-0 ${
                    forecastPhase > 1 ? 'text-emerald-600' : 'text-indigo-600 animate-spin'
                  }`}
                />
                <span className={forecastPhase > 1 ? 'line-through text-slate-400' : 'font-medium text-slate-800'}>
                  历史数据清洗与 1024 个时间点上下文切片
                </span>
              </div>
              <div className="flex items-center gap-2.5">
                {forecastPhase > 2 ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                ) : forecastPhase === 2 ? (
                  <RefreshCw className="w-4 h-4 text-indigo-600 animate-spin shrink-0" />
                ) : (
                  <div className="w-4 h-4 rounded-full border-2 border-slate-300 shrink-0" />
                )}
                <span
                  className={
                    forecastPhase > 2
                      ? 'line-through text-slate-400'
                      : forecastPhase === 2
                      ? 'font-medium text-slate-900'
                      : 'text-slate-400'
                  }
                >
                  装配 TimesFM 3.0 Safetensors 权重与配置
                </span>
              </div>
              <div className="flex items-center gap-2.5">
                {forecastPhase > 3 ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                ) : forecastPhase === 3 ? (
                  <RefreshCw className="w-4 h-4 text-indigo-600 animate-spin shrink-0" />
                ) : (
                  <div className="w-4 h-4 rounded-full border-2 border-slate-300 shrink-0" />
                )}
                <span
                  className={
                    forecastPhase > 3
                      ? 'line-through text-slate-400'
                      : forecastPhase === 3
                      ? 'font-medium text-slate-900'
                      : 'text-slate-400'
                  }
                >
                  多核并行注意力机制与 Patch 自回归推断
                </span>
              </div>
              <div className="flex items-center gap-2.5">
                {forecastPhase >= 4 && forecastProgress >= 99 ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                ) : forecastPhase === 4 ? (
                  <RefreshCw className="w-4 h-4 text-indigo-600 animate-spin shrink-0" />
                ) : (
                  <div className="w-4 h-4 rounded-full border-2 border-slate-300 shrink-0" />
                )}
                <span
                  className={
                    forecastPhase === 4
                      ? 'font-medium text-slate-900'
                      : 'text-slate-400'
                  }
                >
                  生成 10%、50%、90% 置信区间分位数与平滑校准
                </span>
              </div>
            </div>

            {/* Bottom Hint */}
            <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
              <span>⚡ AVX2/NEON 硬件加速已启用</span>
              <span>首次运行需从磁盘加载，后续将直接命中缓存</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
