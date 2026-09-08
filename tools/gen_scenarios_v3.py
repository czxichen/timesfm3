#!/usr/bin/env python3
"""v3: FULL-COVERAGE scenario manifest for the official-vs-Rust benchmark.

= legacy 155 scenarios (tools/scenarios.json, bit-identical inputs)
+ new real data sources (exchange / electricity / solar / traffic /
  electricity64) swept across the same horizon & context axes
+ new cross-source batching, NaN and flag-interaction cells on real data

Every scenario carries a `dims` block so the report can group results by
data source / horizon / context / variates / batch / family.

Usage: python tools/gen_scenarios_v3.py
"""

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "scenarios"
LEGACY = ROOT / "tools" / "scenarios.json"
MANIFEST = ROOT / "tools" / "scenarios_v3.json"

STD_FLAGS = {
    "use_symmetric_averaging": True,
    "make_positive": False,
    "sort_quantiles": True,
    "use_znorm": False,
}


# ---- data loading (mirrors gen_scenarios.py) -------------------------------
def _load_csv(path: Path) -> list[list[float]]:
    lines = path.read_text().splitlines()[1:]
    rows = []
    for line in lines:
        if line.strip():
            rows.append([float(t) for t in line.strip().split(",")[1:]])
    n = len(rows)
    nv = min(len(r) for r in rows)
    return [[rows[i][v] for i in range(n)] for v in range(nv)]


def _synth(n_variates: int, n: int, kind: str, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    cols: list[list[float]] = [[] for _ in range(n_variates)]
    for v in range(n_variates):
        base = rng.uniform(-1.0, 1.0)
        for i in range(n):
            t = i / 32.0
            if kind == "trend_spike":
                x = (base + 0.02 * i) + 3.0 * math.sin(2 * math.pi * t / 24.0) \
                    + 0.8 * math.sin(2 * math.pi * t / 7.0) + rng.gauss(0, 0.4)
                if rng.random() < 0.004:
                    x += rng.choice([-25.0, 25.0, 60.0])
            elif kind == "mixed_scale":
                scale = 10.0 ** rng.uniform(-3, 6) if v > 0 else 1.0
                x = (base + 0.001 * scale * i / 10.0) + scale * rng.gauss(0, 1)
                if rng.random() < 0.002:
                    x *= 1.0 + rng.random() * 3.0
            elif kind == "flat_neg":
                if v % 4 == 0:
                    x = -3.5 + 0.002 * rng.gauss(0, 1)
                elif v % 4 == 1:
                    x = 42.0 if rng.random() < 0.98 else 42.0 + rng.gauss(0, 1e-3)
                elif v % 4 == 2:
                    x = -rng.uniform(1, 10) - 0.05 * (i % 50)
                else:
                    x = -(i % 97) + rng.gauss(0, 0.7)
            else:
                raise ValueError(kind)
            cols[v].append(x)
    return cols


def load_all() -> dict[str, list[list[float]]]:
    ds: dict[str, list[list[float]]] = {}

    def trim(name: str, cols: list[list[float]]) -> None:
        n = (len(cols[0]) // 32) * 32 - 32
        ds[name] = [col[-n:] for col in cols] if n > 0 else cols

    trim("etth1", _load_csv(ROOT / "data" / "etth1.csv"))
    trim("etth2", _load_csv(ROOT / "data" / "ETTh2.csv"))
    trim("ettm1", _load_csv(ROOT / "data" / "ETTm1.csv"))
    trim("ettm2", _load_csv(ROOT / "data" / "ETTm2.csv"))
    for name in ("exchange", "electricity", "solar", "traffic", "electricity64"):
        trim(name, _load_csv(ROOT / "data" / f"{name}.csv"))
    ds["synth_ts"] = _synth(7, 8192, "trend_spike", seed=1)
    ds["synth_scale"] = _synth(7, 4096, "mixed_scale", seed=2)
    ds["synth_flat"] = _synth(7, 4096, "flat_neg", seed=3)
    ds["synth_v20"] = _synth(20, 2048, "trend_spike", seed=4)
    ds["synth_v50"] = _synth(50, 2048, "trend_spike", seed=5)
    return ds


def write_csv(name: str, cols: list[list[float]]) -> dict:
    path = OUT / f"{name}.csv"
    lines = []
    for col in cols:
        lines.append(",".join(
            "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.6f}"
            for x in col))
    path.write_text("\n".join(lines) + "\n")
    return {"csv": path.relative_to(ROOT).as_posix(),
            "variates": len(cols), "context_len": len(cols[0])}


def make_nan(cols, mode: str):
    rng = random.Random(42)
    nan = math.nan
    out = [list(c) for c in cols]
    ctx = len(out[0])
    for v in range(len(out)):
        for i in range(ctx):
            p = 0.5 if mode == "heavy" else 0.02
            if mode == "allcol" and v == 0:
                out[v][i] = nan
            elif rng.random() < p:
                out[v][i] = nan
    return out


# ---- source tagging (used by the report to group legacy scenarios) ---------
def guess_source(name: str) -> str:
    for s in ("etth1", "etth2", "ettm1", "ettm2", "synth_ts", "synth_scale",
              "synth_flat", "synth_v20", "synth_v50", "exchange", "electricity64",
              "electricity", "solar", "traffic"):
        if name.startswith(s):
            return s
    if name.startswith("mb_") or name.startswith("mixed_batch"):
        return "mixed"
    return "etth1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_all()
    legacy = json.loads(LEGACY.read_text())["scenarios"]
    scenarios: list[dict] = []
    seen = {s["name"] for s in legacy}

    def add(name, horizon, series, flags=None, overrides=None, note="",
            source="", family=""):
        if name in seen:
            return
        seen.add(name)
        v = sum(x["variates"] for x in series)
        scenarios.append({
            "name": name, "horizon": horizon,
            "flags": dict(STD_FLAGS if flags is None else flags),
            "model_overrides": overrides or {},
            "series": series, "note": note,
            "dims": {
                "source": source,
                "family": family,
                "horizon": horizon,
                "context_len": max(x["context_len"] for x in series),
                "variates": v,
                "batch": len(series),
            },
        })

    def tail(dn: str, c: int, vmax: int | None = None):
        cols = ds[dn] if vmax is None else ds[dn][:vmax]
        c = min(c, len(cols[0]))
        return [cols[v][-c:] for v in range(len(cols))]

    def win(dn: str, start: int, c: int, vmax: int | None = None):
        cols = ds[dn] if vmax is None else ds[dn][:vmax]
        return [col[start:start + c] for col in cols]

    new_sources = ["exchange", "electricity", "solar", "traffic"]

    # --- A. new real sources x horizon (ctx 1024) ---------------------------
    for dn in new_sources:
        for h in (24, 96, 168, 336, 720):
            add(f"{dn}_h{h}", h, [write_csv(f"{dn}_h{h}", tail(dn, 1024))],
                source=dn, family="source-x-horizon",
                note=f"horizon {h}, {len(ds[dn])} variates, real source {dn}")

    # --- B. new real sources x context (h 96) -------------------------------
    for dn in new_sources:
        for c in (64, 256, 1024, 3072):
            add(f"{dn}_ctx{c}", 96, [write_csv(f"{dn}_ctx{c}", tail(dn, c))],
                source=dn, family="source-x-context",
                note=f"context {c}, real source {dn}")

    # --- C. wide real table (64 variates) -----------------------------------
    # h720 on 64 variates costs ~10 min per engine pass, dropped on purpose.
    for h in (96, 336):
        add(f"electricity64_h{h}", h,
            [write_csv(f"electricity64_h{h}", tail("electricity64", 1024))],
            source="electricity64", family="source-x-horizon",
            note=f"64-variate real table, horizon {h}")
    add("electricity64_ctx3072", 96,
        [write_csv("electricity64_ctx3072", tail("electricity64", 3072))],
        source="electricity64", family="source-x-context",
        note="64-variate real table, context 3072")
    # variate-count axis purely on real data
    for v in (1, 4, 12, 24):
        add(f"electricity_v{v}", 96,
            [write_csv(f"electricity_v{v}", tail("electricity", 1024, vmax=v))],
            source="electricity", family="variates",
            note=f"{v} variates of real electricity, ctx 1024")

    # --- D. cross-source batching -------------------------------------------
    # NOTE: the official forecaster requires every context in a batch to carry
    # the SAME variate count, so cross-source batches mix sources, not shapes.
    add("xsrc_batch4_eq", 96,
        [write_csv("xsrc_traffic", tail("traffic", 1024)),
         write_csv("xsrc_elec", tail("electricity", 1024)),
         write_csv("xsrc_solar", tail("solar", 1024)),
         write_csv("xsrc_elec2", win("electricity", 3000, 1024))],
        source="mixed", family="batch",
        note="equal-length batch of 4 series from DIFFERENT real sources (24v each)")
    add("xsrc_batch2_mixlen", 96,
        [write_csv("xsrc_solar_c512", tail("solar", 512)),
         write_csv("xsrc_traffic_c2048", tail("traffic", 2048))],
        source="mixed", family="batch",
        note="mixed-length batch across solar(512) + traffic(2048)")
    add("xsrc_batch8_eq", 96,
        [write_csv(f"xsrc_b8_{i}", win(d, 1000 + i * 500, 512)) for i, d in
         enumerate(["electricity", "solar", "traffic", "electricity",
                    "solar", "traffic", "electricity", "solar"])],
        source="mixed", family="batch",
        note="equal-length batch of 8 series from 3 real sources (24v, ctx512)")

    # --- E. NaN robustness on real sources ----------------------------------
    for dn, mode in [("electricity", "light"), ("traffic", "heavy"),
                     ("exchange", "allcol")]:
        add(f"{dn}_nan_{mode}", 96,
            [write_csv(f"{dn}_nan_{mode}", make_nan(tail(dn, 1024), mode))],
            source=dn, family="nan",
            note=f"NaN robustness: {mode} on real source {dn}")

    # --- F. flag x real-source interactions ---------------------------------
    for dn in ("electricity", "solar"):
        base = [write_csv(f"xf_{dn}_base", tail(dn, 1024))]
        add(f"xf_{dn}_znorm", 96, base,
            flags={**STD_FLAGS, "use_znorm": True},
            source=dn, family="flag-x-source", note=f"znorm ON, real {dn}")
        add(f"xf_{dn}_makepos", 96, base,
            flags={**STD_FLAGS, "make_positive": True},
            source=dn, family="flag-x-source", note=f"make_positive ON, real {dn}")
        add(f"xf_{dn}_nosym", 96, base,
            flags={**STD_FLAGS, "use_symmetric_averaging": False},
            source=dn, family="flag-x-source", note=f"symmetric averaging OFF, real {dn}")

    # --- G. long context on a real non-ETT source ---------------------------
    add("solar_ctx8192", 96, [write_csv("solar_ctx8192", tail("solar", 8192))],
        source="solar", family="source-x-context",
        note="long context 8192 on real 10-min solar")

    # ---- merge legacy (tag them so the report can group them) --------------
    for s in legacy:
        v = sum(x["variates"] for x in s["series"])
        s.setdefault("dims", {
            "source": guess_source(s["name"]),
            "family": "legacy",
            "horizon": int(s["horizon"]),
            "context_len": max(x["context_len"] for x in s["series"]),
            "variates": v,
            "batch": len(s["series"]),
        })
        s["dims"]["family"] = "legacy"

    allsc = legacy + scenarios
    MANIFEST.write_text(json.dumps({"scenarios": allsc}, indent=2))
    print(f"wrote {MANIFEST.name}: {len(allsc)} scenarios "
          f"({len(legacy)} legacy + {len(scenarios)} new), "
          f"{sum(len(s['series']) for s in allsc)} series")
    from collections import Counter
    c = Counter(s["dims"]["source"] for s in allsc)
    print("  by source:", dict(c.most_common()))


if __name__ == "__main__":
    sys.exit(main())
