#!/usr/bin/env python3
"""v2: comprehensive multi-dataset / multi-factor / multi-dimension scenario
generator for official-vs-Rust parity comparison.

Covers (much broader than v1's single-dataset 21 scenarios):
  A. datasets        : etth1, etth2 (hourly), ettm1, ettm2 (15-min) + 4 synthetic
                       stress families (trend/spike, mixed-scale, flat/negative,
                       wide variate count) -> total 8 data sources
  B. horizon length  : 1, 12, 48 (sub-window, uni cell) + 24, 96, 168, 336, 720,
                       1440 (multi & uni cells)
  C. context length  : 32, 48, 64 (edge, uni) + 64, 256, 1024, 3072, 15360
                       (exact cap) + 17020 (cap+over) per dataset
  D. variate count   : 1, 2, 3, 5, 7 (etth1 subsample) + 10, 20, 50 (synth)
  E. batch shape     : equal-length b=2/4/8, mixed-length b=8, cross-dataset b=2
  F. NaN robustness  : light / heavy / all-column / per-dataset variants
  G. factor grid     : FULL 2^4 flag grid (sym x sort x makepos x znorm) and
                       FULL 2^4 override grid (stitch x detrend x cpm x frozen)
                       on a cheap uni cell; flag x dataset interactions on
                       multi cells; flag+override combined cells

Every scenario writes data/scenarios/<name>.csv + one manifest entry in
tools/scenarios.json consumed by BOTH run_official_scenarios.py and
target/release/export_scenarios. Original v1 scenario names are kept
unchanged (same inputs -> bit-identical reruns).

Usage: python3 tools/gen_scenarios.py [--max-horizon H] [--max-ctx C] [--quick]
"""

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "scenarios"
MANIFEST = ROOT / "tools" / "scenarios.json"
NQ = 9

STD_FLAGS = {
    "use_symmetric_averaging": True,
    "make_positive": False,
    "sort_quantiles": True,
    "use_znorm": False,
}

# ---- dataset registry -------------------------------------------------------
def _load_csv(path: Path) -> list[list[float]]:
    """(17420.., 7) csv -> list of 7 column lists."""
    lines = path.read_text().splitlines()[1:]
    rows = []
    for line in lines:
        if not line.strip():
            continue
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
                if rng.random() < 0.004:  # spikes
                    x += rng.choice([-25.0, 25.0, 60.0])
            elif kind == "mixed_scale":
                scale = 10.0 ** rng.uniform(-3, 6) if v > 0 else 1.0
                x = (base + 0.001 * scale * i / 10.0) + scale * rng.gauss(0, 1)
                if rng.random() < 0.002:
                    x *= 1.0 + rng.random() * 3.0
            elif kind == "flat_neg":
                # near-constant / zero-variance / negative-only stress
                if v % 4 == 0:
                    x = -3.5 + 0.002 * rng.gauss(0, 1)        # negative, ~constant
                elif v % 4 == 1:
                    x = 42.0 if rng.random() < 0.98 else 42.0 + rng.gauss(0, 1e-3)  # near-zero variance
                elif v % 4 == 2:
                    x = -rng.uniform(1, 10) - 0.05 * (i % 50)  # strictly negative
                else:
                    x = -(i % 97) + rng.gauss(0, 0.7)         # negative sawtooth
            else:
                raise ValueError(kind)
            cols[v].append(x)
    return cols


def load_all() -> dict[str, list[list[float]]]:
    """dataset name -> columns. All row counts trimmed to a multiple of 32."""
    ds: dict[str, list[list[float]]] = {}

    def trim(name: str, cols: list[list[float]]) -> None:
        n = len(cols[0])
        n = (n // 32) * 32 - 32  # drop a ragged tail, keep a full 32-block margin
        ds[name] = [col[-n:] for col in cols] if n > 0 else cols

    trim("etth1", _load_csv(ROOT / "data" / "etth1.csv"))
    trim("etth2", _load_csv(ROOT / "data" / "ETTh2.csv"))
    trim("ettm1", _load_csv(ROOT / "data" / "ETTm1.csv"))
    trim("ettm2", _load_csv(ROOT / "data" / "ETTm2.csv"))
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
            for x in col
        ))
    path.write_text("\n".join(lines) + "\n")
    return {
        "csv": path.relative_to(ROOT).as_posix(),
        "variates": len(cols),
        "context_len": len(cols[0]),
    }


def make_nan(cols: list[list[float]], mode: str) -> list[list[float]]:
    """Deterministic NaN injection. mode: light / heavy / allcol / v1mix."""
    rng = random.Random(42)
    nan = math.nan
    out = [list(c) for c in cols]
    v_n = len(out)
    ctx = len(out[0])
    if mode == "allcol":
        # an entire variate NaN (v0) -> strip-leading for that column only
        for i in range(ctx):
            out[0][i] = nan
        # plus scattered noise elsewhere to keep others interesting
        for v in range(1, v_n):
            for i in range(ctx):
                if rng.random() < 0.01:
                    out[v][i] = nan
    elif mode == "heavy":
        for v in range(v_n):
            for i in range(ctx):
                if rng.random() < 0.5:
                    out[v][i] = nan
    else:  # light = small scattered tail-ish holes, no leading block
        for v in range(v_n):
            for i in range(ctx):
                if rng.random() < 0.02:
                    out[v][i] = nan
    return out


def main() -> None:
    quick = "--quick" in sys.argv
    max_h = int(_arg("--max-horizon", 720))
    max_ctx = int(_arg("--max-ctx", 4096))
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_all()
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

    def ctx_tail(dn: str, c: int, vmax: int | None = None) -> list[list[float]]:
        cols = ds[dn] if vmax is None else ds[dn][:vmax]
        return [cols[v][-c:] for v in range(len(cols))]

    def win(dn: str, start: int, c: int) -> list[list[float]]:
        return [col[start:start + c] for col in ds[dn]]

    datasets_multi = ["etth1", "etth2", "ettm1", "ettm2", "synth_ts"]
    datasets_all = datasets_multi + ["synth_scale", "synth_flat"]

    # =========================================================================
    # A. horizon family  (7v, ctx 1024, per dataset)
    # =========================================================================
    for dn in datasets_all:
        for h in (24, 96, 168, 336, 720):
            if h > max_h:
                continue
            add(f"{dn}_h{h}", h, [write_csv(f"{dn}_h{h}", ctx_tail(dn, 1024))],
                note=f"horizon {h}, 7 variates, dataset {dn}")
    # long horizons
    if 1440 <= max_h:
        for dn in ("etth1", "ettm2"):
            add(f"{dn}_h1440", 1440,
                [write_csv(f"{dn}_h1440", ctx_tail(dn, 1024))],
                note=f"long horizon 1440, dataset {dn}")
    # sub-output-window horizons on a cheap uni cell (edge: non-multiples of 64)
    for h in (1, 12, 48):
        add(f"etth1_h{h}_uni", h, [write_csv(f"etth1_h{h}_uni", ctx_tail("etth1", 512, vmax=1))],
            note=f"sub-window horizon {h}, univariate edge case")

    # =========================================================================
    # B. context family  (7v, h 96, per dataset)
    # =========================================================================
    for dn in datasets_all:
        for c in (64, 256, 1024, 3072):
            if c > max_ctx:
                continue
            add(f"{dn}_ctx{c}", 96, [write_csv(f"{dn}_ctx{c}", ctx_tail(dn, c))],
                note=f"context {c}, dataset {dn}")
    # near-exact-cap / over-cap truncation cells
    if max_ctx >= 15360:
        add("etth1_ctx15360", 96, [write_csv("etth1_ctx15360", ctx_tail("etth1", 15360))],
            note="context exactly at 15360 cap, etth1")
        add("etth2_ctx17020", 96, [write_csv("etth2_ctx17020", win("etth2", 0, 17020))],
            note="context 17020 > cap, etth2; expect truncation")
    # tiny contexts on uni cell (edge: below/between patches)
    for c in (32, 48, 64):
        add(f"etth1_ctx{c}_uni", 96, [write_csv(f"etth1_ctx{c}_uni", ctx_tail("etth1", c, vmax=1))],
            note=f"tiny context {c}, univariate edge case")

    # =========================================================================
    # C. variate count  (multi-dimension: number of variables)
    # =========================================================================
    for v in (1, 2, 3, 5):
        add(f"etth1_v{v}", 96, [write_csv(f"etth1_v{v}", ctx_tail("etth1", 1024, vmax=v))],
            note=f"{v} variates of etth1, ctx 1024")
    add("synth_v10", 96, [write_csv("synth_v10", ctx_tail("synth_v20", 1024, vmax=10))],
        note="10 variates, ctx 1024")
    add("synth_v20", 96, [write_csv("synth_v20", ctx_tail("synth_v20", 512, vmax=20))],
        note="20 variates, ctx 512")
    add("synth_v50", 96, [write_csv("synth_v50", ctx_tail("synth_v50", 256, vmax=50))],
        note="50 variates, ctx 256")

    # =========================================================================
    # D. batching
    # =========================================================================
    # equal-length batch sizes 2/4/8 (7v, ctx 1024, etth1; different windows)
    for b in (2, 4, 8):
        rows = []
        for i in range(b):
            start = 4000 + i * 800
            rows.append(write_csv(f"etth1_batch{b}_w{i}", win("etth1", start, 1024)))
        add(f"etth1_batch{b}_eq", 96, rows, note=f"equal-length batch of {b} x 7v ctx1024")
    # mixed lengths (v1 reproduction + a 15-min dataset variant)
    mixed = []
    for i, (start, c) in enumerate([
        (len(ds["etth1"][0]) - 64, 64), (2000, 128), (4000, 256), (6000, 512),
        (8000, 1024), (10000, 1536), (12000, 2048), (14000, 3072),
    ]):
        mixed.append(write_csv(f"etth1_mix_c{c}", [ds["etth1"][6][start:start + c]]))
    add("etth1_mixed8_uni", 96, mixed,
        note="8 univariate mixed lengths 64..3072, etth1 OT")
    if not quick:
        m2 = []
        for i, (start, c) in enumerate([(len(ds["ettm1"][0]) - 64, 64), (2000, 128),
                                        (4000, 256), (6000, 512), (8000, 1024)]):
            m2.append(write_csv(f"ettm1_mix_c{c}", [ds["ettm1"][5][start:start + c]]))
        add("ettm1_mixed5_uni", 96, m2,
            note="5 univariate mixed lengths from 15-min ettm1")
    # cross-dataset equal-length batch (same shape, different stats)
    add("mixed_batch_etth1xetth2", 96,
        [write_csv("mb_etth1", ctx_tail("etth1", 1024)),
         write_csv("mb_etth2", ctx_tail("etth2", 1024))],
        note="cross-dataset equal batch: etth1 + etth2, 7v each")

    # =========================================================================
    # E. NaN robustness  (per dataset where cheap)
    # =========================================================================
    for dn, mode, note in [
        ("etth1", "light", "scattered ~2% NaN"),
        ("etth1", "heavy", "heavy ~50% NaN"),
        ("etth1", "allcol", "one whole variate NaN + scattered"),
        ("ettm2", "light", "scattered ~2% NaN on 15-min ettm2"),
        ("synth_ts", "heavy", "heavy ~50% NaN on synth trend series"),
    ]:
        add(f"{dn}_nan_{mode}", 96,
            [write_csv(f"{dn}_nan_{mode}", make_nan(ctx_tail(dn, 1024), mode))],
            note=note)

    # =========================================================================
    # F. FULL factor grids on a cheap uni cell (ctx 512, h 96)
    # =========================================================================
    uni512 = [write_csv("fg_uni512", ctx_tail("etth1", 512, vmax=1))]
    # 2^4 flag grid: symmetric_averaging x sort_quantiles x make_positive x znorm
    fgrid = [
        ("sy", "use_symmetric_averaging"), ("so", "sort_quantiles"),
        ("mp", "make_positive"), ("zn", "use_znorm"),
    ]
    for bits in range(16):
        fl = dict(STD_FLAGS)
        combo = []
        for k, (tag, key) in enumerate(fgrid):
            on = bool(bits & (1 << k))
            fl[key] = on
            combo.append(f"{tag}{'1' if on else '0'}")
        flags = fl
        name = "fg_flags_" + "_".join(combo)
        add(name, 96, uni512, flags=flags, note="flag grid: " + ", ".join(combo))
    # 2^4 override grid: stitch x detrend x cpm x frozen
    ogrid = [
        ("st", "use_stitching"), ("dt", "use_linear_detrending"),
        ("cp", "use_iterative_cpm_revin"), ("fr", "use_frozen_running_stats"),
    ]
    for bits in range(16):
        ovr = {}
        combo = []
        for k, (tag, key) in enumerate(ogrid):
            on = bool(bits & (1 << k))
            ovr[key] = on
            combo.append(f"{tag}{'1' if on else '0'}")
        name = "fg_ovr_" + "_".join(combo)
        add(name, 96, uni512, overrides=ovr, note="override grid: " + ", ".join(combo))
    # flag x dataset interactions (multi v7 ctx1024, the risky knobs)
    for dn in ("etth2", "ettm1", "ettm2", "synth_ts"):
        base = [write_csv(f"fx_{dn}_base", ctx_tail(dn, 1024))]
        add(f"fx_{dn}_znorm", 96, base,
            flags={**STD_FLAGS, "use_znorm": True}, note=f"znorm ON, dataset {dn}")
        add(f"fx_{dn}_makepos", 96, base,
            flags={**STD_FLAGS, "make_positive": True}, note=f"make_positive ON, dataset {dn}")
    # make_positive meaningful on truly-negative data
    add("fx_synthflat_makepos", 96,
        [write_csv("fx_synthflat_base", ctx_tail("synth_flat", 1024))],
        flags={**STD_FLAGS, "make_positive": True},
        note="make_positive ON on strictly-negative synth data")
    # combined flag+override cells (multi)
    add("fx_etth1_znorm_nocpm", 96,
        [write_csv("fx_etth1_zn_cpm", ctx_tail("etth1", 1024))],
        flags={**STD_FLAGS, "use_znorm": True},
        overrides={"use_iterative_cpm_revin": False},
        note="znorm ON + CPM OFF combined, 7v etth1")
    add("fx_etth1_nosym_nostitch", 96,
        [write_csv("fx_etth1_ns_stitch", ctx_tail("etth1", 1024))],
        flags={**STD_FLAGS, "use_symmetric_averaging": False},
        overrides={"use_stitching": False},
        note="symmetric OFF + stitching OFF combined, 7v etth1")

    # ---- keep v1 names (bit-identical reruns for continuity) ----------------
    v1 = json.loads((ROOT / "tools" / "scenarios_v1.json").read_text())["scenarios"]
    v1_kept = [s for s in v1 if s["name"] not in {x["name"] for x in scenarios}]
    # v1 names already regenerated above with identical csv layout except the
    # old csv files (flag_base / ovr_uni512 / batch2_multi_w4096.csv etc.) still
    # exist on disk; keep old scenario entries only if their csv survives.
    for s in v1_kept:
        csv_ok = all((ROOT / x["csv"]).exists() for x in s["series"]) \
            and all((ROOT / x["csv"]).stat().st_size > 0 for x in s["series"])
        if csv_ok:
            scenarios.append(s)

    MANIFEST.write_text(json.dumps({"scenarios": scenarios}, indent=2))
    n_fwd = sum(
        (s["flags"] or {}).get("use_symmetric_averaging", True) and len(s["series"]) * 2
        or len(s["series"])
        for s in scenarios
    )
    print(f"wrote {MANIFEST.name}: {len(scenarios)} scenarios, "
          f"{sum(len(s['series']) for s in scenarios)} series "
          f"(~{n_fwd} forwards with sym averaging)")
    # summary by axis
    from collections import Counter
    fam = Counter()
    for s in scenarios:
        n = s["name"]
        if n.startswith("fg_"):
            fam["factor-grid"] += 1
        elif n.startswith("fx_"):
            fam["flag-x-dataset/combo"] += 1
        elif "_nan_" in n:
            fam["nan"] += 1
        elif "_batch" in n or "_mix" in n or n.startswith("mixed_batch"):
            fam["batch"] += 1
        elif "_h" in n and len(n.rsplit("_h", 1)[1].split("_")[0]) <= 4:
            fam["horizon"] += 1
        elif "_ctx" in n or "_c" in n.rsplit("_", 1)[-1]:
            fam["context"] += 1
        else:
            fam["variate/other"] += 1
    for k, v in fam.most_common():
        print(f"  {k:<16} {v}")


def _arg(key: str, default: int) -> str:
    a = sys.argv
    return a[a.index(key) + 1] if key in a else str(default)


if __name__ == "__main__":
    main()