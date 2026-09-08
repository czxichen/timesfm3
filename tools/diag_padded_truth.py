#!/usr/bin/env python3
"""Validate the Rust engine's LEFT-PADDED batch path against per-series
official ground truth.

The official torch repo emits all-NaN forecasts for left-padded series in a
mixed-context-length batch (see tools/diag_official_batch.py), so the padded
rows cannot be compared directly. Instead: run the official forecaster on
EACH series alone (no padding) and diff against the Rust engine's padded
batch outputs (out_rust_sc/batch8_mixed_uni/s{i}_*.bin).

Usage: /tmp/venv312/bin/python tools/diag_padded_truth.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "repo" / "src"))

from _torch_compat import _install_rmsnorm  # pyright: ignore[reportMissingImports]

_install_rmsnorm()

from timesfm3 import ModelConfig  # pyright: ignore[reportMissingImports]
from timesfm3.timesfm3_forecaster import (  # pyright: ignore[reportMissingImports]
    TimesFM3Forecaster,
)

MANIFEST = ROOT / "tools" / "scenarios.json"
RUST = ROOT / "out_rust_sc"


def main() -> None:
    try:
        manifest = json.loads(MANIFEST.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {MANIFEST}: {exc}") from exc
    sc = next((s for s in manifest["scenarios"] if s["name"] == "batch8_mixed_uni"), None)
    if sc is None:
        raise SystemExit("scenario batch8_mixed_uni not found in manifest")
    fc = TimesFM3Forecaster(
        config=ModelConfig(
            checkpoint_path=str(ROOT / "ckpt"),
            per_core_batch_size=32,
            use_stitching=True,
            use_linear_detrending=True,
            use_iterative_cpm_revin=True,
            use_frozen_running_stats=False,
            device="cpu",
        )
    )
    rows = []
    for i, ser in enumerate(sc["series"]):
        path = ROOT / ser["csv"]
        try:
            ctx = np.array(
                [[float(t) for t in line.split(",")] for line in
                 path.read_text().strip().splitlines()],
                dtype=np.float32,
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot read {path}: {exc}") from exc
        out = fc.predict(
            context=ctx,
            horizon=96,
            return_quantiles=True,
            use_symmetric_averaging=True,
            make_positive=False,
            sort_quantiles=True,
            use_znorm=False,
            padding_mode="none",
        )
        med_o = np.asarray(out.forecast, dtype=np.float64).reshape(1, 96)
        q_o = np.asarray(out.quantiles, dtype=np.float64).reshape(1, 96, 9)
        med_r = np.fromfile(
            RUST / "batch8_mixed_uni" / f"s{i}_median.bin", dtype="<f4"
        ).reshape(1, 96).astype(np.float64)
        q_r = np.fromfile(
            RUST / "batch8_mixed_uni" / f"s{i}_quantiles.bin", dtype="<f4"
        ).reshape(1, 96, 9).astype(np.float64)
        dm = np.abs(med_r - med_o).max()
        dq = np.abs(q_r - q_o).max()
        c_len = ser["context_len"]
        try:
            rows.append({"series": i, "context_len": c_len,
                         "median_max_abs": float(dm), "quant_max_abs": float(dq)})
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"s{i}: cannot compute metrics: {exc}") from exc
        print(f"s{i} ctx={c_len:<5} official-alone vs rust-padded: "
              f"median max={dm:.3e}  quantiles max={dq:.3e}", flush=True)
        try:
            (ROOT / "report" / "padded_truth.json").write_text(
                json.dumps(rows, indent=2)
            )
        except OSError as exc:
            raise SystemExit(f"cannot write padded_truth.json: {exc}") from exc
        print("wrote report/padded_truth.json")


if __name__ == "__main__":
    main()
