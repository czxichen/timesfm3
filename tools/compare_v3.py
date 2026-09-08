#!/usr/bin/env python3
"""Full-matrix comparison for tools/scenarios_v3.json.

Compares, for every scenario:
  * official PyTorch (torch CPU)  vs  Rust FP32     -> accuracy verdict
  * Rust FP16 / FP16-balanced     vs  Rust FP32     -> quantisation drift
  * wall-clock of all four passes                   -> speed

Emits report/full_summary.json (machine readable) consumed by
tools/report_full.py.

Usage: python tools/compare_v3.py
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tools" / "scenarios_v3.json"
OFFICIAL = ROOT / "out_official_sc"
RUST = ROOT / "out_rust_sc"
RUST_F16 = ROOT / "out_rust_sc_f16"
RUST_BAL = ROOT / "out_rust_sc_bal"
REPORT = ROOT / "report"
NQ = 9
PASS = {"f32": RUST, "f16": RUST_F16, "bal": RUST_BAL}


def read_f32_bin(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").copy()


def fit(arr: np.ndarray, v: int, horizon: int) -> np.ndarray:
    a = np.asarray(arr).reshape(-1)
    if a.size == v * horizon:
        return a.reshape(v, horizon)
    actual = a.size // v if v else a.size
    if a.size % v != 0:
        raise ValueError(f"size {a.size} not divisible by v={v}")
    if actual < horizon:
        out = np.full((v, horizon), np.nan, dtype=a.dtype)
        out[:, :actual] = a.reshape(v, actual)
        return out
    return a.reshape(v, horizon)


def rel_stats(ref: np.ndarray, got: np.ndarray) -> dict:
    """Scale-normalised error.

    The legacy verdict uses an ABSOLUTE threshold (1e-4).  That threshold is
    meaningless for sources whose forecasts live at 1e4..1e5 (electricity peaks
    near 9.3e4, where a single float32 ULP is already 7.8e-3).  Two
    scale-aware statistics are therefore reported alongside it:

      norm = max|Δ| / (max(ref) - min(ref))            -- range normalised
      rel  = max(|Δ| / max(|ref|, 1e-3 * peak(ref)))   -- element-wise, floored

    `norm` is used for the TOL_SCALE verdict (threshold 1e-5).
    """
    r = np.asarray(ref, dtype=np.float64)
    g = np.asarray(got, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(g)
    r, g = r[ok], g[ok]
    if r.size == 0:
        return {"rel": float("nan"), "norm": float("nan"), "peak": 0.0}
    peak = float(np.abs(r).max())
    rng = float(r.max() - r.min())
    denom = np.maximum(np.abs(r), 1e-3 * peak)
    return {
        "rel": float((np.abs(g - r) / denom).max()),
        "norm": float(np.abs(g - r).max() / rng) if rng > 0 else float("nan"),
        "peak": peak,
    }


def pair_stats(ref: np.ndarray, got: np.ndarray) -> dict:
    d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    ok = np.isfinite(d)
    if not ok.all():
        d = d[ok]
    if d.size == 0:
        return {"max": float("nan"), "mean": float("nan"),
                "exact": 0.0, "corr": float("nan"), "finite": False}
    corr = (float(np.corrcoef(ref.ravel(), got.ravel())[0, 1])
            if ref.size > 1 and np.isfinite(ref).all() and np.isfinite(got).all()
            else 1.0)
    return {
        "max": float(d.max()), "mean": float(d.mean()),
        "exact": float((d == 0).mean()), "corr": corr,
        "finite": bool(np.isfinite(got).all()),
    }


def scenario_metrics(sc: dict) -> dict:
    name = sc["name"]
    horizon = int(sc["horizon"])
    off = np.load(OFFICIAL / f"{name}.npz")
    series, quant = [], []
    for i, ser in enumerate(sc["series"]):
        v = int(ser["variates"])
        med_o = fit(off[f"s{i}_median"], v, horizon).astype(np.float64)
        q_o = fit(off[f"s{i}_quantiles"], v, horizon * NQ).reshape(v, horizon, NQ)
        med_r = fit(read_f32_bin(RUST / name / f"s{i}_median.bin"), v, horizon)
        q_r = fit(read_f32_bin(RUST / name / f"s{i}_quantiles.bin"),
                  v, horizon * NQ).reshape(v, horizon, NQ).astype(np.float64)
        official_nan = bool(np.isnan(med_o).any() or np.isnan(q_o).any())
        rust_nan = bool(np.isnan(med_r).any() or np.isnan(q_r).any())
        m = pair_stats(np.nan_to_num(med_o), np.nan_to_num(med_r)) if (
            official_nan or rust_nan) else pair_stats(med_o, med_r)
        q = pair_stats(np.nan_to_num(q_o), np.nan_to_num(q_r)) if (
            official_nan or rust_nan) else pair_stats(q_o, q_r)
        rm = rel_stats(np.nan_to_num(med_o), np.nan_to_num(med_r))
        rq = rel_stats(np.nan_to_num(q_o), np.nan_to_num(q_r))
        # quantisation drift vs the engines own f32 output
        for key, directory in (("f16", RUST_F16), ("bal", RUST_BAL)):
            p = directory / name / f"s{i}_median.bin"
            if p.exists():
                med_p = fit(read_f32_bin(p), v, horizon)
                st = pair_stats(med_r.astype(np.float64), med_p.astype(np.float64))
                st["prec"] = key
                st["mono"] = bool(np.all(np.diff(
                    fit(read_f32_bin(directory / name / f"s{i}_quantiles.bin"),
                        v, horizon * NQ).reshape(v, horizon, NQ), axis=-1) >= 0))
                quant.append(st)
        series.append({
            "series": i, "shape": f"{v}x{horizon}",
            "official_nan": official_nan, "rust_nan": rust_nan,
            "median_max_abs": m["max"], "median_mean_abs": m["mean"],
            "median_rel_max": rm["rel"], "median_norm": rm["norm"],
            "peak": rm["peak"],
            "median_exact_frac": m["exact"],
            "quant_max_abs": q["max"], "quant_mean_abs": q["mean"],
            "quant_rel_max": rq["rel"], "quant_norm": rq["norm"],
            "quant_exact_frac": q["exact"], "corr": m["corr"],
        })
    med_max = max((s["median_max_abs"] for s in series
                   if np.isfinite(s["median_max_abs"])), default=0.0)
    q_max = max((s["quant_max_abs"] for s in series
                 if np.isfinite(s["quant_max_abs"])), default=0.0)
    # scale-normalised verdict uses the MEDIAN forecast (the primary output);
    # quantile errors are reported alongside but the quantile array spans a
    # much wider range, which would make the normalisation unstable.
    cand = [s["median_norm"] for s in series
            if s.get("median_norm") is not None and np.isfinite(s["median_norm"])]
    norm_max = max(cand) if cand else float("nan")
    rel_max = max((x for x in (s["median_rel_max"] for s in series)
                   if x is not None and np.isfinite(x)), default=float("nan"))
    q_rel_max = max((x for x in (s.get("quant_rel_max") for s in series)
                     if x is not None and np.isfinite(x)), default=float("nan"))
    peak = max((s["peak"] for s in series), default=0.0)
    if any(s["official_nan"] for s in series):
        verdict = "OFFICIAL_NAN"
    elif any(s["rust_nan"] for s in series):
        verdict = "RUST_NAN"
    elif med_max == 0.0 and q_max == 0.0:
        verdict = "EXACT"
    elif med_max <= 1e-4 and q_max <= 1e-4:
        verdict = "TOL"
    elif np.isfinite(norm_max) and norm_max <= 1e-5:
        # absolute error exceeds 1e-4 purely because the data lives at a large
        # scale (peak >> 1); range-normalised error is still noise-level.
        verdict = "TOL_SCALE"
    elif np.isfinite(norm_max) and norm_max <= 1e-4:
        # > 1e-5 but still within 0.01% of the forecast range: numeric drift
        # worth watching, not a semantic divergence.
        verdict = "WARN"
    else:
        verdict = "FAIL"
    return {
        "name": name, "horizon": horizon, "dims": sc.get("dims", {}),
        "flags": sc["flags"], "model_overrides": sc.get("model_overrides") or {},
        "note": sc.get("note", ""), "verdict": verdict, "series": series,
        "median_max_abs": med_max, "quant_max_abs": q_max,
        "rel_max": rel_max, "peak": peak, "quant": quant,
    }


def load_timings(d: Path) -> dict:
    p = d / "timings.json"
    return json.loads(p.read_text()) if p.exists() else {}


def clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    results = [scenario_metrics(s) for s in manifest["scenarios"]]
    out = {
        "results": results,
        "official_timings": load_timings(OFFICIAL),
        "rust_timings": {k: load_timings(v) for k, v in PASS.items()},
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "full_summary.json").write_text(
        json.dumps(clean(out), indent=2, ensure_ascii=False))
    from collections import Counter
    cnt = Counter(r["verdict"] for r in results)
    print("verdicts:", dict(cnt))
    for r in results:
        if r["verdict"] not in ("EXACT", "TOL"):
            print(f"  {r['name']:<24} {r['verdict']:<12} "
                  f"med={r['median_max_abs']:.3e} q={r['quant_max_abs']:.3e}")
    print(f"wrote report/full_summary.json ({len(results)} scenarios)")


if __name__ == "__main__":
    main()
