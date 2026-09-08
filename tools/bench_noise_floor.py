#!/usr/bin/env python3
"""Numerical noise floor of the OFFICIAL implementation.

The official torch model is not bit-reproducible across thread counts (BLAS
reduction order changes).  Measuring "official@T threads vs Rust" for several T
tells us how much of the residual Rust-vs-official difference is simply the
official implementation disagreeing with itself.

usage: python tools/bench_noise_floor.py [scenario ...]

Prints one JSON line per (scenario, thread count).
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "repo"
sys.path.insert(0, str(REPO / "src"))

from _torch_compat import _install_rmsnorm  # noqa: E402

_install_rmsnorm()
from timesfm3 import ModelConfig  # noqa: E402
from timesfm3.timesfm3_forecaster import TimesFM3Forecaster  # noqa: E402

SCEN = ROOT / "tools" / "scenarios_v3.json"
RUST = ROOT / "out_rust_sc"
THREADS = [1, 4, 8, 16]
DEFAULT_SCENARIOS = ["etth1_h96", "etth2_h96", "exchange_h96", "electricity_h96"]


def load_series(rel: str) -> np.ndarray:
    rows = [[float(t) for t in line.split(",")]
            for line in (ROOT / rel).read_text().strip().splitlines() if line.strip()]
    return np.array(rows, dtype=np.float32)


def read_bin(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").copy()


def main() -> None:
    names = sys.argv[1:] or DEFAULT_SCENARIOS
    manifest = json.loads(SCEN.read_text())["scenarios"]
    wanted = [s for s in manifest if s["name"] in names]
    if not wanted:
        raise SystemExit(f"no matching scenario in {SCEN}")

    fc = TimesFM3Forecaster(config=ModelConfig(
        checkpoint_path=str(ROOT / "ckpt"), per_core_batch_size=32,
        use_stitching=True, use_linear_detrending=True,
        use_iterative_cpm_revin=True, use_frozen_running_stats=False,
        device="cpu"))

    out = []
    for sc in wanted:
        ctx = [load_series(x["csv"]) for x in sc["series"]]
        h = int(sc["horizon"])
        base = None
        for t in THREADS:
            torch.set_num_threads(t)
            r = list(fc.predict_batch(
                contexts=ctx, horizon=h, return_quantiles=True,
                use_symmetric_averaging=True, make_positive=False,
                sort_quantiles=True, use_znorm=False, padding_mode="none"))[0]
            med = np.asarray(r.forecast, dtype=np.float32).astype(np.float64)
            if t == 8:
                base = med
            rb = read_bin(RUST / sc["name"] / "s0_median.bin").astype(np.float64)
            rb = rb.reshape(med.shape)
            out.append({
                "scenario": sc["name"], "threads": t,
                "vs_rust_max": float(np.abs(med - rb).max()),
                "vs_rust_mean": float(np.abs(med - rb).mean()),
                "vs_official8_max": (None if base is None
                                     else float(np.abs(med - base).max())),
            })
            print(json.dumps(out[-1]), flush=True)
    torch.set_num_threads(8)
    print(json.dumps({"summary": {
        "rust_vs_official8_max": max(
            o["vs_rust_max"] for o in out if o["threads"] == 8),
        "official_self_noise_max": max(
            (o["vs_official8_max"] or 0) for o in out),
    }}, ensure_ascii=False))


if __name__ == "__main__":
    main()
