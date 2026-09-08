#!/usr/bin/env python3
"""Controlled experiment: does the OFFICIAL forecaster emit NaN for
left-padded series inside a mixed-context-length batch?"""

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

CKPT = ROOT / "ckpt"
DATA = ROOT / "data" / "etth1.csv"


def series(n: int) -> np.ndarray:
    """(1, n) univariate OT, last n points."""
    try:
        lines = DATA.read_text().splitlines()[1:]
        col = [float(line.split(",")[6]) for line in lines[-n:]]
    except (OSError, ValueError, IndexError) as exc:
        raise SystemExit(f"cannot build series({n}) from {DATA}: {exc}") from exc
    return np.array([col], dtype=np.float32)


def run(label: str, ctxs, sym: bool, bs: int = 32) -> None:
    fc = TimesFM3Forecaster(
        config=ModelConfig(
            checkpoint_path=str(CKPT),
            per_core_batch_size=bs,
            use_stitching=True,
            use_linear_detrending=True,
            use_iterative_cpm_revin=True,
            use_frozen_running_stats=False,
            device="cpu",
        )
    )
    outs = list(
        fc.predict_batch(
            contexts=list(ctxs),
            horizon=96,
            return_quantiles=True,
            use_symmetric_averaging=sym,
            make_positive=False,
            sort_quantiles=True,
            use_znorm=False,
            padding_mode="none",
        )
    )
    status = []
    for i, o in enumerate(outs):
        m = np.asarray(o.forecast)
        status.append(f"s{i}: nan={np.isnan(m).sum()}/{m.size}")
    print(f"{label:<44} sym={sym} bs={bs} -> {'  '.join(status)}", flush=True)


def main() -> None:
    s1024 = series(1024)
    s2048 = series(2048)
    s3072 = series(3072)
    run("A [1024]", [s1024], sym=False)
    run("B [2048]", [s2048], sym=False)
    run("C [1024, 2048]", [s1024, s2048], sym=False)
    run("D [2048, 1024]", [s2048, s1024], sym=False)
    run("E [1024, 3072]", [s1024, s3072], sym=False)
    run("F [1024, 2048] sym", [s1024, s2048], sym=True)
    run("G [1024, 2048] bs=1", [s1024, s2048], sym=False, bs=1)
    run("H [1024, 1024]", [s1024, s1024], sym=False)


if __name__ == "__main__":
    main()
