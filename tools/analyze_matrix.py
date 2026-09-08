#!/usr/bin/env python3
"""Aggregate report/summary.json (v2 matrix) into grouped comparison tables
across every axis: dataset / horizon length / context length / variate count /
batch / NaN / factor grids. Prints tables for quick human consumption and
writes report/matrix_v2.md with the full grouped picture.

Usage: python3 tools/analyze_matrix.py [--md report/matrix_v2.md]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "report" / "summary.json"


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    results: list[dict] = data["results"]

    def med(r) -> float:  # max abs diff across series (NaN-safe)
        m = r["median_max_abs"]
        return float(m) if isinstance(m, (int, float)) else float("nan")

    def q(r) -> float:
        m = r["quant_max_abs"]
        return float(m) if isinstance(m, (int, float)) else float("nan")

    def corr_min(r) -> float:
        cs = [s["corr"] for s in r["series"] if isinstance(s["corr"], (int, float))]
        return float(min(cs)) if cs else float("nan")

    def ok(r) -> bool:
        return r["verdict"] in ("TOL", "EXACT")

    groups = defaultdict(list)
    for r in results:
        n = r["name"]
        # axis classifiers (ordered by priority)
        if n.startswith("fg_ovr_"):
            groups["factor-grid: overrides"].append(r)
        elif n.startswith("fg_flags_"):
            groups["factor-grid: flags"].append(r)
        elif n.startswith("fx_"):
            groups["factor x dataset/combo"].append(r)
        elif n.startswith("mixed_batch") or "_batch" in n or "_mix" in n:
            groups["batch"].append(r)
        elif "_nan_" in n:
            groups["nan"].append(r)
        elif n.endswith("_h1_uni") or n.endswith("_h12_uni") or n.endswith("_h48_uni"):
            groups["horizon: sub-window uni"].append(r)
        elif "_h1440" in n:
            groups["horizon: 1440"].append(r)
        elif "_h720" in n:
            groups["horizon: 720"].append(r)
        elif "_h336" in n:
            groups["horizon: 336"].append(r)
        elif "_h168" in n:
            groups["horizon: 168"].append(r)
        elif "_h96" in n or n == "h96_multi":
            groups["horizon: 96"].append(r)
        elif "_h24" in n:
            groups["horizon: 24"].append(r)
        elif "_ctx15360" in n or "_ctx17020" in n or "_ctx_cap_" in n:
            groups["context: 15360/17020 cap"].append(r)
        elif "_ctx3072" in n:
            groups["context: 3072"].append(r)
        elif "_ctx1024" in n:
            groups["context: 1024"].append(r)
        elif "_ctx256" in n:
            groups["context: 256"].append(r)
        elif "_ctx64" in n:
            groups["context: 64"].append(r)
        elif "_ctx32_uni" in n or "_ctx48_uni" in n:
            groups["context: tiny 32/48"].append(r)
        elif n.startswith("synth_v") or "_v1" in n or "_v2" in n or "_v3" in n or "_v5" in n:
            groups["variate count"].append(r)
        else:
            groups["other/legacy"].append(r)

    lines: list[str] = []
    add = lines.append
    add("# v2 多数据 × 多因素 × 多维度 —— 分组精度矩阵")
    add("")
    add(f"共 {len(results)} 场景;通过(TOL/EXACT) {sum(ok(r) for r in results)}, "
        f"FAIL {sum(r['verdict'] == 'FAIL' for r in results)}, "
        f"RUST_NAN {sum(r['verdict'] == 'RUST_NAN' for r in results)}, "
        f"OFFICIAL_NAN {sum(r['verdict'] == 'OFFICIAL_NAN' for r in results)}")
    add("")

    def dump_group(title, rs):
        if not rs:
            return
        add(f"## {title}  ({len(rs)} 场景)")
        add("")
        add("| 场景 | 判定 | median max\\|Δ\\| | 分位 max\\|Δ\\| | corr min |")
        add("|---|---|--:|--:|--:|")
        for r in sorted(rs, key=lambda x: (med(x) if med(x) is not None else 1e9, x["name"]),
                        reverse=True):
            add(f"| `{r['name']}` | {r['verdict']} | "
                f"{med(r):.3e}" if med(r) is not None else "-" +
                f" | {q(r):.3e}" if q(r) is not None else "-" +
                f" | {corr_min(r):.6f}" if corr_min(r) is not None else "-" + " |")
        add("")

    # axis overview tables (aggregate only)
    add("## 各轴汇总(max|Δ| 为组内最大值)")
    add("")
    add("| 轴 | 场景数 | 通过 | 最差 median \\|Δ\\| | 最差分位 \\|Δ\\| | 最差 corr |")
    add("|---|--:|--:|--:|--:|--:|")
    axis_order = [
        "horizon: 24", "horizon: 96", "horizon: 168", "horizon: 336",
        "horizon: 720", "horizon: 1440", "horizon: sub-window uni",
        "context: 64", "context: 256", "context: 1024", "context: 3072",
        "context: 15360/17020 cap", "context: tiny 32/48",
        "variate count", "batch", "nan",
        "factor-grid: flags", "factor-grid: overrides", "factor x dataset/combo",
        "other/legacy",
    ]
    for k in axis_order:
        rs = groups.get(k)
        if not rs:
            continue
        mx = max((med(r) for r in rs if med(r) is not None), default=float("nan"))
        qx = max((q(r) for r in rs if q(r) is not None), default=float("nan"))
        cx = min((corr_min(r) for r in rs if corr_min(r) is not None), default=float("nan"))
        add(f"| {k} | {len(rs)} | {sum(ok(r) for r in rs)} | "
            f"{mx:.2e} | {qx:.2e} | {cx:.3f} |")
    add("")

    # per-dataset summary (aggregate across everything with known dataset prefix)
    add("## 分数据集汇总")
    add("")
    add("| 数据集 | 场景数 | 通过 | 最差 median \\|Δ\\| | 最差分位 \\|Δ\\| |")
    add("|---|--:|--:|--:|--:|")
    ds_order = ["etth1", "etth2", "ettm1", "ettm2", "synth_ts", "synth_scale",
                "synth_flat", "synth_v20/v50", "legacy v1"]
    for ds in ds_order:
        if ds == "synth_v20/v50":
            rs = [r for r in results if r["name"].startswith("synth_v")]
        else:
            rs = [r for r in results if r["name"].startswith(ds + "_")]
        if not rs:
            continue
        mx = max((med(r) for r in rs if med(r) is not None), default=float("nan"))
        qx = max((q(r) for r in rs if q(r) is not None), default=float("nan"))
        add(f"| {ds} | {len(rs)} | {sum(ok(r) for r in rs)} | {mx:.2e} | {qx:.2e} |")
    add("")

    worst = sorted(results, key=lambda r: (med(r) if med(r) is not None else 1e9),
                   reverse=True)[:15]
    add("## 最差 15 个场景(按 median max|Δ|)")
    add("")
    add("| 场景 | 判定 | median max\\|Δ\\| | 分位 max\\|Δ\\| |")
    add("|---|---|--:|--:|")
    for r in worst:
        add(f"| `{r['name']}` | {r['verdict']} | {med(r):.3e} | {q(r):.3e} |")
    add("")

    for k in axis_order:
        dump_group(k, groups.get(k))

    md_path = ROOT / "report" / "matrix_v2.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {md_path}")
    print(f"total {len(results)} | pass {sum(ok(r) for r in results)} | "
          f"FAIL {sum(r['verdict'] == 'FAIL' for r in results)} | "
          f"RUST_NAN {sum(r['verdict'] == 'RUST_NAN' for r in results)} | "
          f"OFFICIAL_NAN {sum(r['verdict'] == 'OFFICIAL_NAN' for r in results)}")
    for k in axis_order:
        rs = groups.get(k)
        if not rs:
            continue
        mx = max((med(r) for r in rs if med(r) is not None), default=float("nan"))
        print(f"  {k:<30} n={len(rs):<4} pass={sum(ok(r) for r in rs):<4} "
              f"worst={mx:.2e}")


if __name__ == "__main__":
    main()