#!/usr/bin/env python3
"""Run ALL scenarios from tools/scenarios.json with the official PyTorch model.

Loads the checkpoint ONCE, then for each scenario:
  - applies model_overrides (use_stitching / use_linear_detrending /
    use_iterative_cpm_revin / use_frozen_running_stats) to the torch model,
    restoring the checkpoint values afterwards
  - runs one predict_batch call with the scenario's flags
  - writes out_official_sc/<name>.npz with arrays
      s{i}_median (v, H), s{i}_quantiles (v, H, 9)
    plus out_official_sc/timings.json

Usage: /tmp/venv312/bin/python tools/run_official_scenarios.py [scenario_name ...]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch  # pyright: ignore[reportMissingImports]  (venv-only: /tmp/venv312)

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "repo"
sys.path.insert(0, str(REPO / "src"))

from _torch_compat import _install_rmsnorm  # pyright: ignore[reportMissingImports]

_install_rmsnorm()

from timesfm3 import ModelConfig  # pyright: ignore[reportMissingImports]
from timesfm3.timesfm3_forecaster import (  # pyright: ignore[reportMissingImports]
    TimesFM3Forecaster,
)

CKPT = ROOT / "ckpt"
MANIFEST = ROOT / "tools" / "scenarios.json"
OUT = ROOT / "out_official_sc"
MODEL_KEYS = (
    "use_stitching",
    "use_linear_detrending",
    "use_iterative_cpm_revin",
    "use_frozen_running_stats",
)


def load_series(rel_csv: str) -> np.ndarray:
    """(v, C) float32 from a scenario csv (one line per variate)."""
    path = ROOT / rel_csv
    try:
        lines = path.read_text().strip().splitlines()
        rows = [
            [float(t) for t in line.split(",")] for line in lines if line.strip()
        ]
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    arr = np.array(rows, dtype=np.float32)
    if arr.ndim != 2:
        raise SystemExit(f"bad scenario csv {path}")
    return arr


def main() -> None:
    only = set(sys.argv[1:])
    manifest_path = MANIFEST
    if "--manifest" in sys.argv:
        i = sys.argv.index("--manifest")
        manifest_path = ROOT / sys.argv[i + 1]
        only = set(sys.argv[1:i]) | set(sys.argv[i + 2:])
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {manifest_path}: {exc}") from exc
    scenarios = [s for s in manifest["scenarios"] if not only or s["name"] in only]
    if not scenarios:
        raise SystemExit("no matching scenarios")

    torch.manual_seed(0)
    np.set_printoptions(precision=6, suppress=True)
    config = ModelConfig(
        checkpoint_path=str(CKPT),
        per_core_batch_size=32,
        use_stitching=True,
        use_linear_detrending=True,
        use_iterative_cpm_revin=True,
        use_frozen_running_stats=False,
        device="cpu",
    )
    fc = TimesFM3Forecaster(config=config)
    originals = {k: getattr(fc.model, k) for k in MODEL_KEYS}

    OUT.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    for sc in scenarios:
        name = sc["name"]
        try:
            horizon = int(sc["horizon"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"scenario {name}: bad horizon: {exc}") from exc
        flags = sc.get("flags") or {}
        overrides = sc.get("model_overrides") or {}
        try:
            for k, v in overrides.items():
                setattr(fc.model, k, bool(v))
            t0 = time.perf_counter()
            contexts = [load_series(x["csv"]) for x in sc["series"]]
            outs = list(
                fc.predict_batch(
                    contexts=contexts,
                    horizon=horizon,
                    return_quantiles=True,
                    use_symmetric_averaging=bool(flags["use_symmetric_averaging"]),
                    make_positive=bool(flags["make_positive"]),
                    sort_quantiles=bool(flags["sort_quantiles"]),
                    use_znorm=bool(flags["use_znorm"]),
                    padding_mode="none",
                )
            )
            elapsed = time.perf_counter() - t0
        finally:
            for k, v in originals.items():
                setattr(fc.model, k, v)

        arrays: dict[str, np.ndarray] = {}
        for i, out in enumerate(outs):
            med = np.asarray(out.forecast, dtype=np.float32)
            q = np.asarray(out.quantiles, dtype=np.float32)
            arrays[f"s{i}_median"] = med
            arrays[f"s{i}_quantiles"] = q
        # getattr keeps static analyzers from mis-binding **arrays to the
        # positional allow_pickle stub in numpy's savez signature.
        savez = getattr(np, "savez")
        savez(OUT / f"{name}.npz", **arrays)
        timings[name] = round(elapsed, 2)
        shapes = [f"{x['variates']}x{x['context_len']}" for x in sc["series"]]
        print(f"[official] {name:<20} h={horizon:<4} {shapes} {elapsed:7.1f}s",
              flush=True)
        (OUT / "timings.json").write_text(json.dumps(timings, indent=2))

    print(f"\nwrote {len(scenarios)} scenario npz files -> {OUT}")


if __name__ == "__main__":
    main()
