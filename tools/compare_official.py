#!/usr/bin/env python3
"""Multi-dimensional comparison: official torch vs Rust engine outputs.

Loads out_official/forecast.npz (torch) and out_rust/{median,quantiles}.bin
(Rust) and reports, per variate and across the batch:
  - median point forecast: max abs / mean abs / RMSE, by horizon
  - full quantile grid: max abs diff per (variate, horizon)
  - correlation / cosine similarity per variate
  - whether difference is a constant offset per variate (scale/translate check)

Usage: python3 tools/compare_official.py
"""

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def read_f32_bin(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    return np.frombuffer(raw, dtype="<f4").copy()


def main() -> None:
    try:
        off = np.load(ROOT / "out_official" / "forecast.npz")
        med_o = off["median"]           # (7, H)
        q_o = off["quantiles"]          # (7, H, 9)
        H = int(off["horizon"]) if "horizon" in off else med_o.shape[1]
        V = med_o.shape[0]
        meta = json.loads((ROOT / "out_rust" / "meta.json").read_text())
        med_r = read_f32_bin(ROOT / "out_rust" / "median.bin").reshape(V, H)
        q_r = read_f32_bin(ROOT / "out_rust" / "quantiles.bin").reshape(
            V, H, int(meta["quantiles"])
        )
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"cannot load comparison inputs: {exc}") from exc

    print(f"shapes: official median {med_o.shape} quantiles {q_o.shape}")
    print(f"        rust     median {med_r.shape} quantiles {q_r.shape}\n")

    # ---- 1. median point forecast per variate ----
    print("=== median forecast: per-variate error ===")
    print(f"{'v':>2} {'max_abs':>10} {'mean_abs':>10} {'rmse':>10} {'corr':>8} {'cos':>8}")
    for v in range(V):
        d = med_r[v] - med_o[v]
        corr = np.corrcoef(med_o[v], med_r[v])[0, 1]
        try:
            cos = float(
                med_r[v] @ med_o[v]
                / (np.linalg.norm(med_r[v]) * np.linalg.norm(med_o[v]) + 1e-12)
            )
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"cosine fail v{v}: {exc}") from exc
        print(f"{v:>2} {np.abs(d).max():>10.4f} {np.abs(d).mean():>10.4f} "
              f"{np.sqrt((d**2).mean()):>10.4f} {corr:>8.4f} {cos:>8.4f}")
    total = med_r - med_o
    print(f"ALL: max {np.abs(total).max():.4f} mean {np.abs(total).mean():.4f} "
          f"rmse {np.sqrt((total**2).mean()):.4f}")

    # ---- 2. per-horizon aggregation ----
    print("\n=== median error by horizon (over all variates) ===")
    print(f"{'h':>3} {'max_abs':>10} {'mean_abs':>10}")
    for h in range(H):
        d = med_r[:, h] - med_o[:, h]
        print(f"{h:>3} {np.abs(d).max():>10.4f} {np.abs(d).mean():>10.4f}")

    # ---- 3. full quantile grid ----
    print("\n=== quantile grid: max abs diff per (variate, horizon) ===")
    diff = np.abs(q_r - q_o)  # (V, H, 9)
    print(f"global max {diff.max():.4f} mean {diff.mean():.4f}")
    print("row = variate, show max over quantiles per horizon[0..H-1:3]:")
    head = "   " + "".join(f"{h:>8}" for h in range(0, H, 3))
    print(head)
    for v in range(V):
        row = "".join(f"{diff[v, h].max():>8.3f}" for h in range(0, H, 3))
        print(f"v{v:>1}  {row}")

    # ---- 4. constant-offset check (scale/translate failure hypothesis) ----
    print("\n=== per-variate delta stats (first 4 horizons) ===")
    for v in range(V):
        d = med_r[v, :4] - med_o[v, :4]
        print(f"v{v}: delta={np.round(d, 4)}  mean_delta={d.mean():.4f}")

    # ---- 5. symmetric-averaging sanity: q10 < q50 < q90 both sides ----
    print("\n=== monotonicity check (q10<=...<=q90) ===")
    mo = np.all(np.diff(q_o, axis=-1) >= 0)
    mr = np.all(np.diff(q_r, axis=-1) >= 0)
    print(f"official sorted: {mo}, rust sorted: {mr}")


if __name__ == "__main__":
    main()