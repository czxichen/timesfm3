#!/usr/bin/env python3
"""Comprehensive precision evaluation: official PyTorch vs Rust engine at
f32 / f16 weight precision.

Loads:
  out_official/forecast.npz          -> official torch baseline
  out_rust/{median,quantiles}.bin    -> engine f32
  out_rust_f16/...                   -> engine f16

Reports per precision: max|d|, mean|d|, RMSE, corr, cosine vs official,
plus cross-precision (f16 vs f32) and monotonicity.
"""
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def read_f32_bin(p: Path) -> np.ndarray:
    return np.frombuffer(p.read_bytes(), dtype="<f4").copy()


def load_rust(prec: str):
    d = ROOT / f"out_rust{'' if prec == 'f32' else '_' + prec}"
    try:
        meta = json.loads((d / "meta.json").read_text())
        H = int(meta["horizon"])
        V = int(meta["variates"])
        med = read_f32_bin(d / "median.bin").reshape(V, H)
        q = read_f32_bin(d / "quantiles.bin").reshape(V, H, int(meta["quantiles"]))
        return med, q
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"cannot load {prec} outputs from {d}: {exc}") from exc


def stats(a, b, name):
    try:
        d = b - a
        corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        cos = float(
            b.ravel() @ a.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        )
        return {
            "name": name,
            "max_abs": float(np.abs(d).max()),
            "mean_abs": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d**2).mean())),
            "corr": float(corr),
            "cos": float(cos),
        }
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"stats fail for {name}: {exc}") from exc


def fmt(r):
    return (f"{r['name']:<14} max={r['max_abs']:.6g} mean={r['mean_abs']:.6g} "
            f"rmse={r['rmse']:.6g} corr={r['corr']:.7f} cos={r['cos']:.7f}")


def main():
    off = np.load(ROOT / "out_official" / "forecast.npz")
    med_o, q_o = off["median"], off["quantiles"]
    V, H = med_o.shape

    data = {}
    for prec in ("f32", "f16"):
        data[prec] = load_rust(prec)

    print(f"official median {med_o.shape}, quantiles {q_o.shape}\n")

    print("=== median (point) forecast: engine vs OFFICIAL ===")
    for prec in ("f32", "f16"):
        print("  " + fmt(stats(med_o, data[prec][0], f"med {prec} vs official")))

    print("\n=== full quantile grid: engine vs OFFICIAL ===")
    for prec in ("f32", "f16"):
        print("  " + fmt(stats(q_o, data[prec][1], f"q   {prec} vs official")))

    print("\n=== cross-precision (quantization-only loss) ===")
    for prec in ("f16",):
        print("  " + fmt(stats(data['f32'][0], data[prec][0], f"med {prec} vs f32")))
        print("  " + fmt(stats(data['f32'][1], data[prec][1], f"q   {prec} vs f32")))

    print("\n=== per-variate median max|d| vs official ===")
    print(f"{'v':>2} {'f32':>10} {'f16':>10}")
    for v in range(V):
        row = []
        for prec in ("f32", "f16"):
            d = data[prec][0][v] - med_o[v]
            row.append(f"{np.abs(d).max():>10.4f}")
        print(f"{v:>2} " + " ".join(row))

    print("\n=== monotonicity (q10<=...<=q90) ===")
    print(f"  official sorted: {np.all(np.diff(q_o, axis=-1) >= 0)}")
    for prec in ("f32", "f16"):
        print(f"  {prec} sorted: {np.all(np.diff(data[prec][1], axis=-1) >= 0)}")

    # Relative error on quantiles (scale-referenced to official magnitude)
    print("\n=== quantile relative max|d|/max|official| ===")
    scale = np.abs(q_o).max()
    for prec in ("f32", "f16"):
        d = np.abs(data[prec][1] - q_o).max()
        print(f"  {prec}: {d:.6g} / {scale:.4g} = {d/scale:.3%}")


if __name__ == "__main__":
    main()
