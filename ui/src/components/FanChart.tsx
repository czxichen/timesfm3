import React, { useState, useRef } from 'react';
import type { TargetForecastResult } from '../types';

interface FanChartProps {
  data: TargetForecastResult;
  height?: number;
}

export const FanChart: React.FC<FanChartProps> = ({ data, height = 400 }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const histVals = data.history_values;
  const histTimes = data.history_timestamps;
  const predVals = data.median;
  const predTimes = data.forecast_timestamps;
  const q10 = data.q10;
  const q20 = data.q20;
  const q80 = data.q80;
  const q90 = data.q90;

  const nHist = histVals.length;
  const nPred = predVals.length;
  const totalPoints = nHist + nPred;

  if (totalPoints === 0) {
    return <div className="p-8 text-center text-slate-400">暂无图表数据</div>;
  }

  // Find min and max for Y-axis scaling
  let minY = Infinity;
  let maxY = -Infinity;

  for (const v of histVals) {
    if (v < minY) minY = v;
    if (v > maxY) maxY = v;
  }
  for (let i = 0; i < nPred; i++) {
    const low = q10[i] ?? predVals[i];
    const high = q90[i] ?? predVals[i];
    if (low < minY) minY = low;
    if (high > maxY) maxY = high;
  }

  // Add 8% margin
  const ySpan = maxY - minY || 1;
  const yBottom = minY - ySpan * 0.08;
  const yTop = maxY + ySpan * 0.08;
  const totalYSpan = yTop - yBottom || 1;

  const svgWidth = 1000;
  const padLeft = 60;
  const padRight = 30;
  const padTop = 32;
  const padBottom = 42;
  const plotWidth = svgWidth - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;

  const getX = (idx: number) => {
    return padLeft + (idx / (totalPoints - 1 || 1)) * plotWidth;
  };

  const getY = (val: number) => {
    const norm = (val - yBottom) / totalYSpan;
    return padTop + plotHeight * (1 - norm);
  };

  // Build SVG Path for History
  const histPath = histVals
    .map((v, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getY(v).toFixed(1)}`)
    .join(' ');

  // Build SVG Path for Prediction (starts from last history point for visual continuity)
  const lastHistVal = histVals[nHist - 1] ?? predVals[0];
  const predPath = [
    `M ${getX(nHist - 1).toFixed(1)} ${getY(lastHistVal).toFixed(1)}`,
    ...predVals.map((v, i) => `L ${getX(nHist + i).toFixed(1)} ${getY(v).toFixed(1)}`),
  ].join(' ');

  // Fan 10% - 90% Area
  const q90Forward = [
    `M ${getX(nHist - 1).toFixed(1)} ${getY(lastHistVal).toFixed(1)}`,
    ...q90.map((v, i) => `L ${getX(nHist + i).toFixed(1)} ${getY(v).toFixed(1)}`),
  ];
  const q10Backward = [
    ...q10
      .map((v, i) => `L ${getX(nHist + i).toFixed(1)} ${getY(v).toFixed(1)}`)
      .reverse(),
    `L ${getX(nHist - 1).toFixed(1)} ${getY(lastHistVal).toFixed(1)}`,
    'Z',
  ];
  const fan90Path = `${q90Forward.join(' ')} ${q10Backward.join(' ')}`;

  // Fan 20% - 80% Area
  const q80Forward = [
    `M ${getX(nHist - 1).toFixed(1)} ${getY(lastHistVal).toFixed(1)}`,
    ...q80.map((v, i) => `L ${getX(nHist + i).toFixed(1)} ${getY(v).toFixed(1)}`),
  ];
  const q20Backward = [
    ...q20
      .map((v, i) => `L ${getX(nHist + i).toFixed(1)} ${getY(v).toFixed(1)}`)
      .reverse(),
    `L ${getX(nHist - 1).toFixed(1)} ${getY(lastHistVal).toFixed(1)}`,
    'Z',
  ];
  const fan80Path = `${q80Forward.join(' ')} ${q20Backward.join(' ')}`;

  // Split x coordinate
  const splitX = getX(nHist - 1);

  // Y-axis grid ticks (5 lines)
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const val = yBottom + ratio * totalYSpan;
    const yPos = getY(val);
    return { val, yPos };
  });

  // Handle Mouse Move
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const normX = (mouseX - (padLeft * rect.width) / svgWidth) / ((plotWidth * rect.width) / svgWidth);
    const clampedIndex = Math.round(Math.max(0, Math.min(1, normX)) * (totalPoints - 1));
    setHoverIndex(clampedIndex);
  };

  const handleMouseLeave = () => {
    setHoverIndex(null);
  };

  // Hovered Point Info
  const isHoverInPred = hoverIndex !== null && hoverIndex >= nHist;
  const hoverTime =
    hoverIndex !== null
      ? hoverIndex < nHist
        ? histTimes[hoverIndex]
        : predTimes[hoverIndex - nHist]
      : '';
  const hoverVal =
    hoverIndex !== null
      ? hoverIndex < nHist
        ? histVals[hoverIndex]
        : predVals[hoverIndex - nHist]
      : null;
  const hoverQ10 = isHoverInPred && hoverIndex !== null ? q10[hoverIndex - nHist] : null;
  const hoverQ90 = isHoverInPred && hoverIndex !== null ? q90[hoverIndex - nHist] : null;

  return (
    <div className="relative w-full bg-white rounded-2xl border border-gray-200/90 shadow-xs p-6 select-none transition-all">
      {/* Chart Header & Legend */}
      <div className="flex flex-wrap items-center justify-between pb-3.5 mb-3 border-b border-slate-100 gap-3 text-xs">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 text-indigo-900 border border-indigo-200/80 text-xs font-medium">
            <span className="w-2 h-2 rounded-full bg-indigo-600"></span>
            <span>当前指标:</span>
            <strong className="font-bold text-slate-900">{data.column_name}</strong>
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-4 sm:gap-6 text-slate-600 text-[11px]">
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-1 bg-slate-500 rounded-full"></span>
            <span className="font-medium text-slate-700">历史实际值</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-1.5 bg-blue-600 rounded-full shadow-2xs"></span>
            <span className="font-semibold text-slate-900">预测基准走势</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-2.5 bg-blue-200/80 rounded border border-blue-300"></span>
            <span>核心波动区间 (60%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-2.5 bg-blue-100/70 rounded border border-blue-200"></span>
            <span>总体概率区间 (80%)</span>
          </div>
        </div>
      </div>

      {/* SVG Container */}
      <div ref={containerRef} className="relative w-full">
        <svg
          viewBox={`0 0 ${svgWidth} ${height}`}
          className="w-full h-auto cursor-crosshair overflow-visible"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          <defs>
            {/* Soft gradient for 10%-90% fan */}
            <linearGradient id="fan90Grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#93c5fd" stopOpacity="0.32" />
              <stop offset="100%" stopColor="#bfdbfe" stopOpacity="0.12" />
            </linearGradient>
            {/* Soft gradient for 20%-80% fan */}
            <linearGradient id="fan80Grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#93c5fd" stopOpacity="0.22" />
            </linearGradient>
          </defs>

          {/* Horizontal Grid lines */}
          {yTicks.map((tick, i) => (
            <g key={i}>
              <line
                x1={padLeft}
                y1={tick.yPos}
                x2={svgWidth - padRight}
                y2={tick.yPos}
                stroke="#f1f5f9"
                strokeWidth="1"
              />
              <text
                x={padLeft - 10}
                y={tick.yPos + 4}
                textAnchor="end"
                fontSize="11"
                fill="#94a3b8"
                fontFamily="ui-monospace, monospace"
              >
                {tick.val.toFixed(2)}
              </text>
            </g>
          ))}

          {/* Shaded Forecast Fan (90% interval) */}
          <path d={fan90Path} fill="url(#fan90Grad)" />

          {/* Shaded Forecast Fan (80% interval) */}
          <path d={fan80Path} fill="url(#fan80Grad)" />

          {/* History Line: Slate / Neutral */}
          <path
            d={histPath}
            fill="none"
            stroke="#64748b"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Median Forecast Line: Electric Royal Blue */}
          <path
            d={predPath}
            fill="none"
            stroke="#2563eb"
            strokeWidth="2.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Vertical Divider line between History and Forecast */}
          <line
            x1={splitX}
            y1={padTop}
            x2={splitX}
            y2={height - padBottom}
            stroke="#cbd5e1"
            strokeWidth="1.5"
            strokeDasharray="4 3"
          />

          {/* Boundary Tag */}
          <text
            x={splitX - 8}
            y={padTop + 14}
            textAnchor="end"
            fontSize="11"
            fill="#64748b"
            fontWeight="600"
          >
            历史实际
          </text>
          <text
            x={splitX + 8}
            y={padTop + 14}
            textAnchor="start"
            fontSize="11"
            fill="#2563eb"
            fontWeight="700"
          >
            未来预测 ➔
          </text>

          {/* Interactive Crosshair & Dot */}
          {hoverIndex !== null && hoverVal !== null && (
            <g>
              <line
                x1={getX(hoverIndex)}
                y1={padTop}
                x2={getX(hoverIndex)}
                y2={height - padBottom}
                stroke={isHoverInPred ? '#2563eb' : '#64748b'}
                strokeWidth="1.4"
                strokeDasharray="3 3"
              />
              <circle
                cx={getX(hoverIndex)}
                cy={getY(hoverVal)}
                r="5"
                fill="#ffffff"
                stroke={isHoverInPred ? '#2563eb' : '#64748b'}
                strokeWidth="3"
                className="filter drop-shadow-sm"
              />
            </g>
          )}

          {/* X-axis tick labels */}
          <text
            x={padLeft}
            y={height - padBottom + 20}
            textAnchor="start"
            fontSize="10"
            fill="#64748b"
            fontFamily="ui-monospace, monospace"
          >
            {histTimes[0] || '起始'}
          </text>
          <text
            x={splitX}
            y={height - padBottom + 20}
            textAnchor="middle"
            fontSize="10"
            fill="#64748b"
            fontWeight="600"
            fontFamily="ui-monospace, monospace"
          >
            {histTimes[nHist - 1] || '接缝点'}
          </text>
          <text
            x={svgWidth - padRight}
            y={height - padBottom + 20}
            textAnchor="end"
            fontSize="10"
            fill="#2563eb"
            fontWeight="700"
            fontFamily="ui-monospace, monospace"
          >
            {predTimes[nPred - 1] || `+${nPred}步`}
          </text>
        </svg>

        {/* Hover Tooltip Card */}
        {hoverIndex !== null && hoverVal !== null && (
          <div
            className="pointer-events-none absolute top-2 bg-slate-900/95 backdrop-blur-sm text-white px-4 py-3 rounded-xl text-xs shadow-2xl border border-slate-700/80 z-20 transition-all duration-75 min-w-[200px]"
            style={{
              left: `${Math.min(
                Math.max(getX(hoverIndex) * (100 / svgWidth) - 10, 4),
                74
              )}%`,
            }}
          >
            <div className="font-semibold text-slate-200 pb-1.5 border-b border-slate-800 mb-2 flex items-center justify-between gap-3">
              <span className="font-mono text-[11px] text-slate-300 truncate max-w-[120px]">
                {hoverTime || `序号 #${hoverIndex}`}
              </span>
              <span
                className={`text-[10px] px-2 py-0.5 rounded-full font-bold tracking-wider ${
                  isHoverInPred
                    ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                    : 'bg-slate-700/50 text-slate-300 border border-slate-600/50'
                }`}
              >
                {isHoverInPred ? '未来预测' : '历史实际'}
              </span>
            </div>

            <div className="flex items-baseline justify-between gap-3">
              <span className="text-slate-400 text-[11px]">
                {isHoverInPred ? '预测基准走势:' : '实际数值:'}
              </span>
              <span
                className={`text-base font-bold font-mono ${
                  isHoverInPred ? 'text-blue-400' : 'text-slate-100'
                }`}
              >
                {hoverVal.toFixed(4)}
              </span>
            </div>

            {isHoverInPred && hoverQ10 !== null && hoverQ90 !== null && (
              <div className="mt-2 pt-2 border-t border-slate-800 text-[11px] space-y-1">
                <div className="flex justify-between gap-3 text-slate-300">
                  <span className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-sky-400"></span>
                    <span>80% 上限 (极值预测):</span>
                  </span>
                  <span className="font-mono text-sky-300 font-semibold">{hoverQ90.toFixed(4)}</span>
                </div>
                <div className="flex justify-between gap-3 text-slate-300">
                  <span className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
                    <span>80% 下限 (悲观预测):</span>
                  </span>
                  <span className="font-mono text-blue-300 font-semibold">{hoverQ10.toFixed(4)}</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
