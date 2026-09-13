import * as XLSX from 'xlsx';

export type DetectedFormat =
  | 'excel_tsv'
  | 'numeric_1d_lines'
  | 'numeric_1d_inline'
  | 'json_records'
  | 'json_cols'
  | 'json_1d'
  | 'csv'
  | 'tsv'
  | 'semicolon'
  | 'unknown';

export interface ParseResult {
  success: boolean;
  format: DetectedFormat;
  formatName: string;
  formatDescription: string;
  totalRows: number;
  totalColumns: number;
  previewHeaders: string[];
  previewRows: string[][];
  csvContent: string;
  error?: string;
}

export interface ExcelWorkbookInfo {
  sheetNames: string[];
  activeSheet: string;
  csvContent: string;
  totalRows: number;
  totalColumns: number;
  previewHeaders: string[];
  previewRows: string[][];
}

/**
 * Parses an Excel binary buffer into CSV and preview metadata.
 * Supports choosing a specific sheet.
 */
export function parseExcelBuffer(
  data: ArrayBuffer | Uint8Array,
  targetSheetName?: string
): ExcelWorkbookInfo {
  const workbook = XLSX.read(data, { type: 'array', cellDates: true });
  const sheetNames = workbook.SheetNames;
  if (!sheetNames || sheetNames.length === 0) {
    throw new Error('Excel 文件中未找到有效工作表 (Sheet)');
  }

  const activeSheet = targetSheetName && sheetNames.includes(targetSheetName)
    ? targetSheetName
    : sheetNames[0];

  const worksheet = workbook.Sheets[activeSheet];
  if (!worksheet) {
    throw new Error(`无法读取工作表: ${activeSheet}`);
  }

  // Convert worksheet to 2D array for accurate analysis
  const rawRows = XLSX.utils.sheet_to_json<string[]>(worksheet, {
    header: 1,
    defval: '',
    raw: false,
    dateNF: 'yyyy-mm-dd hh:mm:ss',
  }) as any[][];

  // Filter out completely empty trailing rows
  const cleanRows = rawRows.filter((row) =>
    row && row.some((cell) => cell !== null && cell !== undefined && String(cell).trim() !== '')
  );

  if (cleanRows.length === 0) {
    throw new Error(`工作表 [${activeSheet}] 为空，无有效数据行`);
  }

  // Generate CSV text
  const csvContent = XLSX.utils.sheet_to_csv(worksheet, {
    dateNF: 'yyyy-mm-dd hh:mm:ss',
  });

  const previewHeaders = cleanRows[0]?.map((h, i) => String(h || `col_${i + 1}`).trim()) || [];
  const previewRows = cleanRows.slice(1, 6).map((r) =>
    previewHeaders.map((_, i) => String(r[i] ?? '').trim())
  );

  return {
    sheetNames,
    activeSheet,
    csvContent,
    totalRows: Math.max(0, cleanRows.length - 1),
    totalColumns: previewHeaders.length,
    previewHeaders,
    previewRows,
  };
}

/**
 * Smartly parse any pasted text:
 * - Direct Excel cell copy (TSV)
 * - Pure 1D numeric sequence (newline, comma, or space separated)
 * - JSON data (records array, column dict, or numeric array)
 * - CSV / Semicolon-delimited tables
 */
export function parsePastedText(rawText: string): ParseResult {
  const text = (rawText || '').trim();
  if (!text) {
    return {
      success: false,
      format: 'unknown',
      formatName: '空输入',
      formatDescription: '请输入或粘贴时序数据',
      totalRows: 0,
      totalColumns: 0,
      previewHeaders: [],
      previewRows: [],
      csvContent: '',
      error: '输入内容为空，请粘贴时序数据',
    };
  }

  // 1. Check if JSON format
  if (
    (text.startsWith('{') && text.endsWith('}')) ||
    (text.startsWith('[') && text.endsWith(']'))
  ) {
    try {
      const parsed = JSON.parse(text);

      // 1.1 JSON Array of 1D numbers: [12.5, 14.2, 16.8]
      if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === 'number') {
        const numRows = parsed.length;
        const csvLines = ['time_step,value'];
        for (let i = 0; i < numRows; i++) {
          csvLines.push(`${i + 1},${parsed[i]}`);
        }
        const previewRows = parsed.slice(0, 5).map((val, idx) => [`${idx + 1}`, `${val}`]);
        return {
          success: true,
          format: 'json_1d',
          formatName: 'JSON 纯数值数组',
          formatDescription: `包含 ${numRows} 个数值数据点`,
          totalRows: numRows,
          totalColumns: 1,
          previewHeaders: ['time_step', 'value'],
          previewRows,
          csvContent: csvLines.join('\n'),
        };
      }

      // 1.2 JSON Array of Records: [{"date": "...", "sales": 100}]
      if (Array.isArray(parsed) && parsed.length > 0 && typeof parsed[0] === 'object' && parsed[0] !== null) {
        const headers: string[] = Array.from(
          new Set(parsed.flatMap((item) => Object.keys(item || {})))
        );
        if (headers.length > 0) {
          const csvLines = [headers.join(',')];
          for (const item of parsed) {
            const row = headers.map((h) => {
              const v = item[h] !== undefined && item[h] !== null ? String(item[h]) : '';
              return v.includes(',') || v.includes('"') ? `"${v.replace(/"/g, '""')}"` : v;
            });
            csvLines.push(row.join(','));
          }
          const previewRows = parsed.slice(0, 5).map((item) =>
            headers.map((h) => String(item[h] ?? ''))
          );
          return {
            success: true,
            format: 'json_records',
            formatName: 'JSON 对象数组',
            formatDescription: `检测到 ${parsed.length} 条记录，${headers.length} 个字段`,
            totalRows: parsed.length,
            totalColumns: headers.length,
            previewHeaders: headers,
            previewRows,
            csvContent: csvLines.join('\n'),
          };
        }
      }

      // 1.3 JSON Object of Column Arrays: {"date": [...], "sales": [...]}
      if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
        const keys = Object.keys(parsed);
        const arrayKeys = keys.filter((k) => Array.isArray(parsed[k]));
        if (arrayKeys.length > 0) {
          const maxLen = Math.max(...arrayKeys.map((k) => parsed[k].length));
          const csvLines = [arrayKeys.join(',')];
          for (let i = 0; i < maxLen; i++) {
            const row = arrayKeys.map((k) => {
              const v = parsed[k][i] !== undefined && parsed[k][i] !== null ? String(parsed[k][i]) : '';
              return v.includes(',') || v.includes('"') ? `"${v.replace(/"/g, '""')}"` : v;
            });
            csvLines.push(row.join(','));
          }
          const previewRows: string[][] = [];
          for (let i = 0; i < Math.min(5, maxLen); i++) {
            previewRows.push(arrayKeys.map((k) => String(parsed[k][i] ?? '')));
          }
          return {
            success: true,
            format: 'json_cols',
            formatName: 'JSON 字段列字典',
            formatDescription: `包含 ${arrayKeys.length} 个字段，${maxLen} 行数据`,
            totalRows: maxLen,
            totalColumns: arrayKeys.length,
            previewHeaders: arrayKeys,
            previewRows,
            csvContent: csvLines.join('\n'),
          };
        }
      }
    } catch {
      // Not valid JSON, continue to text patterns
    }
  }

  // 2. Check 1D numeric sequence separated by newlines
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length === 0) {
    return {
      success: false,
      format: 'unknown',
      formatName: '无效内容',
      formatDescription: '未找到有效文本行',
      totalRows: 0,
      totalColumns: 0,
      previewHeaders: [],
      previewRows: [],
      csvContent: '',
      error: '未提取到有效数据',
    };
  }

  // Check if every line is a clean float/int (pure 1D numeric series)
  const isAllNumericLines = lines.every((l) => !isNaN(Number(l)));
  if (isAllNumericLines && lines.length >= 2) {
    const csvLines = ['time_step,value'];
    lines.forEach((v, idx) => csvLines.push(`${idx + 1},${v}`));
    const previewRows = lines.slice(0, 5).map((v, idx) => [`${idx + 1}`, v]);
    return {
      success: true,
      format: 'numeric_1d_lines',
      formatName: '纯数值时序 (单列换行)',
      formatDescription: `共 ${lines.length} 个数据点，已自动生成时序步长编号`,
      totalRows: lines.length,
      totalColumns: 1,
      previewHeaders: ['time_step', 'value'],
      previewRows,
      csvContent: csvLines.join('\n'),
    };
  }

  // Check if single line comma or space separated numbers: "100.5, 102.3, 104.1" or "100.5 102.3 104.1"
  if (lines.length === 1) {
    const singleLine = lines[0];
    let sep = ',';
    if (!singleLine.includes(',') && singleLine.includes(' ')) {
      sep = ' ';
    }
    const parts = singleLine
      .split(sep)
      .map((p) => p.trim())
      .filter((p) => p.length > 0);

    if (parts.length >= 3 && parts.every((p) => !isNaN(Number(p)))) {
      const csvLines = ['time_step,value'];
      parts.forEach((v, idx) => csvLines.push(`${idx + 1},${v}`));
      const previewRows = parts.slice(0, 5).map((v, idx) => [`${idx + 1}`, v]);
      return {
        success: true,
        format: 'numeric_1d_inline',
        formatName: '纯数值时序 (单行分隔)',
        formatDescription: `共 ${parts.length} 个数据点`,
        totalRows: parts.length,
        totalColumns: 1,
        previewHeaders: ['time_step', 'value'],
        previewRows,
        csvContent: csvLines.join('\n'),
      };
    }
  }

  // 3. Multi-column Delimited Text (TSV from Excel, CSV, Semicolon, etc.)
  const firstLine = lines[0];
  let delimiter = ',';
  let format: DetectedFormat = 'csv';
  let formatName = '标准 CSV 表格';

  if (firstLine.includes('\t')) {
    delimiter = '\t';
    format = 'excel_tsv';
    formatName = 'Excel 剪贴板表格 (Tab 制表符)';
  } else if (firstLine.includes(';') && !firstLine.includes(',')) {
    delimiter = ';';
    format = 'semicolon';
    formatName = '分号分隔表格 (Semicolon)';
  } else if (!firstLine.includes(',') && firstLine.split(/\s+/).length > 1) {
    delimiter = ' ';
    format = 'tsv';
    formatName = '空格分隔时序文本';
  }

  // Split lines
  const splitRows = lines.map((l) =>
    delimiter === ' '
      ? l.split(/\s+/).map((s) => s.trim())
      : l.split(delimiter).map((s) => s.trim())
  );

  const rawHeader = splitRows[0];
  const numCols = rawHeader.length;

  if (numCols <= 1) {
    return {
      success: false,
      format: 'unknown',
      formatName: '格式未能识别',
      formatDescription: '未检测到标准表格分隔符或数值序列',
      totalRows: lines.length,
      totalColumns: 1,
      previewHeaders: ['value'],
      previewRows: lines.slice(0, 5).map((l) => [l]),
      csvContent: '',
      error: '无法识别有效表格列，请检查是否包含逗号、制表符或换行',
    };
  }

  // Check if first line looks like headers or pure numeric data
  const isFirstLineNumeric = rawHeader.every((cell) => !isNaN(Number(cell)));
  let headers: string[];
  let dataRows: string[][];

  if (isFirstLineNumeric) {
    // Generate col_1, col_2 headers
    headers = rawHeader.map((_, idx) => `col_${idx + 1}`);
    dataRows = splitRows;
  } else {
    headers = rawHeader.map((h, idx) => (h ? h : `col_${idx + 1}`));
    dataRows = splitRows.slice(1);
  }

  // Build standard CSV
  const csvLines: string[] = [headers.join(',')];
  for (const row of dataRows) {
    const padded = headers.map((_, i) => {
      const cell = row[i] ?? '';
      return cell.includes(',') || cell.includes('"') ? `"${cell.replace(/"/g, '""')}"` : cell;
    });
    csvLines.push(padded.join(','));
  }

  const previewRows = dataRows.slice(0, 5).map((row) =>
    headers.map((_, i) => row[i] ?? '')
  );

  return {
    success: true,
    format,
    formatName,
    formatDescription: `检测到 ${headers.length} 列，${dataRows.length} 行时序记录`,
    totalRows: dataRows.length,
    totalColumns: headers.length,
    previewHeaders: headers,
    previewRows,
    csvContent: csvLines.join('\n'),
  };
}

/**
 * Sample datasets for 1-click paste testing in the UI
 */
export const PASTE_DEMO_PRESETS = [
  {
    id: 'retail_sales',
    title: '📊 日销售额序列 (纯数值)',
    desc: '单列纯数值，自动按步长编号 (60 天历史)',
    text: `1240.5\n1310.0\n1280.2\n1420.6\n1510.8\n1630.0\n1590.4\n1380.1\n1410.5\n1450.0\n1580.3\n1690.0\n1780.2\n1640.5\n1420.0\n1470.6\n1530.2\n1610.0\n1740.5\n1850.2\n1790.0\n1510.4\n1560.8\n1620.1\n1710.0\n1840.6\n1980.5\n1910.2\n1620.0\n1680.5\n1750.3\n1860.0\n1990.4\n2120.0\n2050.5\n1730.0\n1790.2\n1880.5\n2010.0\n2150.3\n2280.0\n2210.5\n1890.2\n1950.0\n2040.6\n2180.0\n2320.5\n2460.0\n2390.2\n2020.0\n2110.5\n2230.0\n2380.6\n2540.2\n2690.0\n2610.5\n2250.0\n2340.2\n2480.5\n2650.0`,
  },
  {
    id: 'sensor_tsv',
    title: '📈 传感器多指标 (Excel TSV)',
    desc: '制表符分隔，包含时间与多列指标 (温度/功率)',
    text: `date\ttemperature\tpower_kw\n2024-05-01 00:00:00\t21.4\t45.2\n2024-05-01 01:00:00\t20.8\t42.1\n2024-05-01 02:00:00\t20.2\t39.5\n2024-05-01 03:00:00\t19.7\t38.0\n2024-05-01 04:00:00\t19.5\t39.8\n2024-05-01 05:00:00\t20.1\t44.6\n2024-05-01 06:00:00\t21.3\t52.3\n2024-05-01 07:00:00\t23.0\t68.4\n2024-05-01 08:00:00\t24.8\t85.1\n2024-05-01 09:00:00\t26.5\t94.2\n2024-05-01 10:00:00\t27.8\t98.5\n2024-05-01 11:00:00\t28.6\t101.2\n2024-05-01 12:00:00\t29.1\t99.8\n2024-05-01 13:00:00\t29.4\t102.4\n2024-05-01 14:00:00\t29.0\t97.6\n2024-05-01 15:00:00\t28.2\t92.3\n2024-05-01 16:00:00\t27.1\t84.0\n2024-05-01 17:00:00\t25.8\t75.5\n2024-05-01 18:00:00\t24.5\t69.2\n2024-05-01 19:00:00\t23.6\t63.1\n2024-05-01 20:00:00\t23.0\t58.7\n2024-05-01 21:00:00\t22.5\t54.2\n2024-05-01 22:00:00\t22.1\t50.1\n2024-05-01 23:00:00\t21.8\t47.0`,
  },
  {
    id: 'api_json',
    title: '🌐 API 响应结构 (JSON)',
    desc: '标准 JSON 记录数组格式，适配接口数据',
    text: `[\n  {"timestamp": "2024-06-01", "active_users": 3420, "requests": 18200},\n  {"timestamp": "2024-06-02", "active_users": 3580, "requests": 19400},\n  {"timestamp": "2024-06-03", "active_users": 3210, "requests": 17100},\n  {"timestamp": "2024-06-04", "active_users": 3390, "requests": 18050},\n  {"timestamp": "2024-06-05", "active_users": 3670, "requests": 20100},\n  {"timestamp": "2024-06-06", "active_users": 4120, "requests": 23400},\n  {"timestamp": "2024-06-07", "active_users": 4350, "requests": 24900},\n  {"timestamp": "2024-06-08", "active_users": 3510, "requests": 18900},\n  {"timestamp": "2024-06-09", "active_users": 3620, "requests": 19600},\n  {"timestamp": "2024-06-10", "active_users": 3480, "requests": 18800}\n]`,
  },
];
