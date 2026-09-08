#!/usr/bin/env python3
"""Official-PyTorch side of the speed benchmark.

usage: bench_official.py --variates V --context C --horizon H --repeats N
                         [--threads T] [--batch B] [--warmup W]

Prints one JSON line:
  {"shape":"7x1024","horizon":96,"threads":8,"load_ms":..,"median_ms":..,
   "min_ms":..,"iters_ms":[..],"peak_rss_mb":..,"rss_after_load_mb":..}
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "repo"
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from _torch_compat import _install_rmsnorm  # noqa: E402

_install_rmsnorm()
from timesfm3 import ModelConfig  # noqa: E402
from timesfm3.timesfm3_forecaster import TimesFM3Forecaster  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover - optional
    psutil = None  # type: ignore


def make_input(v: int, c: int) -> np.ndarray:
    """Same deterministic synthetic input as the Rust bench_speed binary."""
    t = np.arange(c, dtype=np.float32)
    out = np.empty((v, c), dtype=np.float32)
    for vi in range(v):
        out[vi] = (3.0 * np.sin(t * 0.05 + vi)
                   + 0.8 * np.sin(t * 0.43 + 1.7)
                   + 0.01 * t
                   + 0.2 * np.cos(t * 0.017 + vi * 0.3))
    return out


class RssSampler(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak = 0
        self._stop = threading.Event()

    def run(self) -> None:
        if psutil is None:
            return
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, proc.memory_info().rss)
            except Exception:
                break
            time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variates", type=int, default=7)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args()

    if a.threads > 0:
        torch.set_num_threads(a.threads)

    sampler = RssSampler()
    sampler.start()

    torch.manual_seed(0)
    t0 = time.perf_counter()
    config = ModelConfig(
        checkpoint_path=str(ROOT / "ckpt"),
        per_core_batch_size=32,
        use_stitching=True,
        use_linear_detrending=True,
        use_iterative_cpm_revin=True,
        use_frozen_running_stats=False,
        device="cpu",
    )
    fc = TimesFM3Forecaster(config=config)
    load_ms = (time.perf_counter() - t0) * 1e3
    rss_after_load = (psutil.Process(os.getpid()).memory_info().rss / 2**20
                      if psutil else -1.0)

    ctx = [make_input(a.variates, a.context) for _ in range(a.batch)]
    # NOTE: predict_batch is a GENERATOR - it must be consumed (list(...)) or
    # nothing is computed at all and the timing loop measures ~0 ms.
    for _ in range(a.warmup):
        list(fc.predict_batch(contexts=ctx, horizon=a.horizon, return_quantiles=True,
                              use_symmetric_averaging=True, make_positive=False,
                              sort_quantiles=True, use_znorm=False,
                              padding_mode="none"))
    iters = []
    for _ in range(a.repeats):
        t = time.perf_counter()
        list(fc.predict_batch(contexts=ctx, horizon=a.horizon, return_quantiles=True,
                              use_symmetric_averaging=True, make_positive=False,
                              sort_quantiles=True, use_znorm=False,
                              padding_mode="none"))
        iters.append((time.perf_counter() - t) * 1e3)
    sampler.stop()
    sampler.join(timeout=1)
    s = sorted(iters)
    print(json.dumps({
        "shape": f"{a.variates * a.batch}x{a.context}",
        "batch": a.batch,
        "horizon": a.horizon,
        "threads": torch.get_num_threads(),
        "load_ms": round(load_ms, 2),
        "median_ms": round(s[len(s) // 2], 2) if s else None,
        "min_ms": round(s[0], 2) if s else None,
        "iters_ms": [round(x, 1) for x in iters],
        "peak_rss_mb": round(sampler.peak / 2**20, 1) if psutil else None,
        "rss_after_load_mb": round(rss_after_load, 1),
    }))


if __name__ == "__main__":
    main()
