#!/usr/bin/env python3
"""Official PyTorch reference run for TimesFM 3.0 comparison.

Loads the SAME checkpoint (ckpt/) with the official repo implementation
(repo/src/timesfm3), runs the SAME data (data/etth1_ctx.csv, the last 1024
points of ETTh1, 7 variates), and writes:

  out_official/{forecast.npz, summary.json}

forecast.npz contains flat arrays shaped like the Rust engine's outputs so the
comparison script can diff them element-wise:
  - median   : (v, H)
  - quantiles: (v, H, 9)  [sorted]
  - quantiles_raw: (v, H, 9) [before sorting, if applicable]

Only numpy/torch/safetensors are needed; pandas is avoided by parsing CSV
manually to keep the dependency surface small.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch  # pyright: ignore[reportMissingImports]  (venv-only: /tmp/venv312)



REPO = Path(__file__).resolve().parent.parent / "repo"
sys.path.insert(0, str(REPO / "src"))

from _torch_compat import _install_rmsnorm  # pyright: ignore[reportMissingImports]

_install_rmsnorm()

from timesfm3 import (  # pyright: ignore[reportMissingImports]  # venv-only
    TimesFM3Evaluator,
    ModelConfig,
)

CKPT = Path(__file__).resolve().parent.parent / "ckpt"
DATA = Path(__file__).resolve().parent.parent / "data" / "etth1_ctx.csv"
OUT = Path(__file__).resolve().parent.parent / "out_official"


def main() -> None:
    torch.manual_seed(0)
    np.set_printoptions(precision=6, suppress=True)

    # --- config matching the Rust CLI invocation ---
    config = ModelConfig(
        checkpoint_path=str(CKPT),
        per_core_batch_size=32,
        use_stitching=True,
        use_linear_detrending=True,
        use_iterative_cpm_revin=True,
        use_frozen_running_stats=False,
        device="cpu",
    )
    forecaster = TimesFM3Evaluator(config)

    # --- data: last 1024 points, 7 variates ---
    # DATA is the SAME transposed input the Rust CLI uses: each line is one
    # variate, comma-separated floats (7 lines x 1024 points).
    try:
        lines = DATA.read_text().strip().splitlines()
    except OSError as exc:  # pragma: no cover
        raise SystemExit(f"cannot read data file: {exc}") from exc
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            rows.append([float(t) for t in line.strip().split(",")])
        except ValueError as exc:  # pragma: no cover
            raise SystemExit(f"bad numeric row: {exc}") from exc
    arr = np.array(rows, dtype=np.float32)  # (v, N)
    nvariates, npoints = arr.shape
    context = arr
    print(f"built context: {context.shape} ({nvariates} variates x {npoints} points)")
    print("v0 tail:", context[0, -3:].round(3))

    horizon = 24
    use_sym = True
    # NOTE: keep symmetric averaging ON to match the Rust export exactly; the
    # official result is the parity target.
    outputs = list(
        forecaster.predict_batch(
            contexts=[context],
            horizon=horizon,
            return_quantiles=True,
            use_symmetric_averaging=use_sym,
            sort_quantiles=True,
        )
    )
    out = outputs[0]
    median = out.forecast                # (7, H)
    quantiles = out.quantiles            # (7, H, 9)
    print(f"median {median.shape}, quantiles {quantiles.shape}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "forecast.npz",
        median=median.astype(np.float32),
        quantiles=quantiles.astype(np.float32),
        horizon=np.array(horizon, dtype=np.float32),
        nvariates=np.array(median.shape[0], dtype=np.float32),
        context_len=np.array(npoints, dtype=np.float32),
        symmetric_averaging=np.array(1 if use_sym else 0, dtype=np.float32),
    )
    try:
        n_context = int(context.shape[1])
        n_variates = int(median.shape[0])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"cannot parse output shapes: {exc}") from exc
    summary = {
        "checkpoint": "timesfm-3.0-pytorch (google)",
        "context_len": n_context,
        "horizon": horizon,
        "variates": n_variates,
        "symmetric_averaging": use_sym,
        "device": str(forecaster.device),
        "model": str(forecaster.model),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote out_official/{forecast.npz, summary.json}")


if __name__ == "__main__":
    main()