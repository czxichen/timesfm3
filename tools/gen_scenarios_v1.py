#!/usr/bin/env python3
"""Generate multi-scenario comparison fixtures for official-vs-Rust parity.

Builds, from data/etth1.csv (17420 x 7 variates, header skipped):
  - data/scenarios/<name>.csv  : one line per variate, comma-separated f32
                                 ('nan' allowed)
  - tools/scenarios.json       : manifest consumed by BOTH
                                 tools/run_official_scenarios.py and
                                 target/release/export_scenarios

Usage: python3 tools/gen_scenarios.py
"""

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "etth1.csv"
OUT = ROOT / "data" / "scenarios"
MANIFEST = ROOT / "tools" / "scenarios.json"

STD_FLAGS = {
    "use_symmetric_averaging": True,
    "make_positive": False,
    "sort_quantiles": True,
    "use_znorm": False,
}


def load_data() -> list[list[float]]:
    """(17420 rows, 7 variates) as columns."""
    try:
        lines = SRC.read_text().splitlines()[1:]
    except OSError as exc:
        raise SystemExit(f"cannot read {SRC}: {exc}") from exc
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            rows.append([float(t) for t in line.strip().split(",")[1:]])
        except ValueError as exc:
            raise SystemExit(f"bad numeric row in {SRC}: {exc}") from exc
    n = len(rows)
    if any(len(r) != 7 for r in rows):
        raise SystemExit(f"expected 7 variates per row in {SRC}")
    return [[rows[i][v] for i in range(n)] for v in range(7)]  # (7, n)


def write_csv(name: str, cols: list[list[float]]) -> dict:
    """cols: (v, ctx) -> one line per variate."""
    path = OUT / f"{name}.csv"
    lines = []
    for col in cols:
        lines.append(",".join(
            "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.6f}"
            for x in col
        ))
    path.write_text("\n".join(lines) + "\n")
    return {
        "csv": str(path.relative_to(ROOT)),
        "variates": len(cols),
        "context_len": len(cols[0]),
    }


def make_nan(cols: list[list[float]]) -> list[list[float]]:
    """Deterministic NaN injection over a copy of `cols` (v, ctx)."""
    rng = random.Random(42)
    nan = math.nan
    out = [list(col) for col in cols]
    v_n = len(out)
    ctx = len(out[0])
    # leading all-NaN block on every variate -> strip-leading path (96 pts)
    for v in range(v_n):
        for i in range(96):
            out[v][i] = nan
    # interior block NaN on v0 (64 pts)
    for i in range(300, 364):
        out[0][i] = nan
    # scattered ~3% NaN on every variate
    for v in range(v_n):
        for i in range(96, ctx):
            if rng.random() < 0.03:
                out[v][i] = nan
    # trailing NaN on v3 (last 16)
    for i in range(ctx - 16, ctx):
        out[3][i] = nan
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    tail = lambda n: [col[-n:] for col in data]  # noqa: E731
    win = lambda start, n: [col[start : start + n] for col in data]  # noqa: E731
    ot = [col for col in data[6:7]]  # OT univariate (list of columns)
    ot_slice = lambda start, n: [data[6][start : start + n]]  # noqa: E731

    scenarios: list[dict] = []

    def add(name, horizon, series, flags=None, overrides=None, note=""):
        scenarios.append({
            "name": name,
            "horizon": horizon,
            "flags": dict(STD_FLAGS if flags is None else flags),
            "model_overrides": overrides or {},
            "series": series,
            "note": note,
        })

    # ---- A. horizon family (7v, ctx 1024) ----
    for h in (24, 96, 168, 336, 720):
        add(f"h{h}_multi", h, [write_csv(f"h{h}_multi", tail(1024))],
            note="horizon sweep, 7 variates")

    # ---- B. context family (7v, h 96) ----
    for c in (64, 256, 3072):
        add(f"ctx{c}", 96, [write_csv(f"ctx{c}", tail(c))],
            note="context sweep, 7 variates")
    # cap > _MAX_CONTEXT_LENGTH(15360): both sides must truncate to 15360
    add("ctx_cap_17k", 96, [write_csv("ctx_cap_17k", win(0, 17020))],
        note="context 17020 > 15360 cap; expect identical truncation")

    # ---- C. variate-shape ----
    add("uni_1024", 96, [write_csv("uni_1024", ot_slice(17420 - 1024, 1024))],
        note="univariate OT, ctx 1024")

    # ---- D. NaN robustness ----
    add("nan_mixed", 96, [write_csv("nan_mixed", make_nan(tail(1024)))],
        note="leading strip 96 + interior 64-block + ~3% scattered + trailing 16")

    # ---- E. batching ----
    add("batch2_multi", 96,
        [write_csv("batch2_multi_w4096", win(4096, 1024)),
         write_csv("batch2_multi_w12000", win(12000, 1024))],
        note="two 7-variate series in ONE predict_batch call")
    mixed = []
    for i, (start, c) in enumerate([
        (17420 - 64, 64), (2000, 128), (4000, 256), (6000, 512),
        (8000, 1024), (10000, 1536), (12000, 2048), (14000, 3072),
    ]):
        mixed.append(write_csv(f"batch8_uni_c{c}", [data[6][start : start + c]]))
    add("batch8_mixed_uni", 96, mixed,
        note="8 univariate series, ctx 64..3072 -> dynamic batch context 3072 + left padding")

    # ---- F. predict-level flags (7v, ctx 1024, h 96) ----
    base_series = [write_csv("flag_base", tail(1024))]
    add("flag_nosym", 96, base_series, flags={**STD_FLAGS, "use_symmetric_averaging": False},
        note="symmetric averaging OFF")
    add("flag_nosort", 96, base_series, flags={**STD_FLAGS, "sort_quantiles": False},
        note="quantile sorting OFF (raw quantile order compared)")
    add("flag_makepos", 96, base_series, flags={**STD_FLAGS, "make_positive": True},
        note="make_positive ON (clamps negatives)")
    add("flag_znorm", 96, base_series, flags={**STD_FLAGS, "use_znorm": True},
        note="per-row z-normalization ON")

    # ---- G. model-config overrides (univariate ctx 512, cheap) ----
    uni512 = [write_csv("ovr_uni512", ot_slice(17420 - 512, 512))]
    add("ovr_stitch_off", 96, uni512, overrides={"use_stitching": False},
        note="use_stitching=False")
    add("ovr_cpm_off", 96, uni512, overrides={"use_iterative_cpm_revin": False},
        note="use_iterative_cpm_revin=False")
    add("ovr_detrend_off", 96, uni512, overrides={"use_linear_detrending": False},
        note="use_linear_detrending=False")
    add("ovr_frozen_stats", 96, uni512, overrides={"use_frozen_running_stats": True},
        note="use_frozen_running_stats=True")

    MANIFEST.write_text(json.dumps({"scenarios": scenarios}, indent=2))
    n_fwd = sum(
        (s["flags"]["use_symmetric_averaging"] or 0) and len(s["series"]) * 2
        or len(s["series"])
        for s in scenarios
    )
    print(f"wrote {MANIFEST.name}: {len(scenarios)} scenarios, "
          f"{sum(len(s['series']) for s in scenarios)} series "
          f"(~{n_fwd} forwards with sym averaging)")
    for s in scenarios:
        shapes = [f"{x['variates']}x{x['context_len']}" for x in s["series"]]
        print(f"  {s['name']:<20} h={s['horizon']:<4} series={shapes}")


if __name__ == "__main__":
    main()
