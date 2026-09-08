#!/usr/bin/env python3
"""Generate golden fixtures for the Rust engine's Phase 1 tests.

numpy-only (no torch needed) — a deterministic, independent oracle for the
residual block math. Writes per fixture:

  golden/<name>/weights.safetensors   weights in official tensor naming
  golden/<name>/input.bin             f32 LE input rows
  golden/<name>/expected.bin          f32 LE reference forward output

Re-run with:  python3 tools/gen_golden.py
"""

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "golden"

PREFIX = "pre_transformer_resblock"


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Writes tensors (torch layout, `(out, in)` weights) as a safetensors file."""
    header: dict = {"__metadata__": {"format": "timesfm3-rust-golden"}}
    offset = 0
    chunks: list[bytes] = []
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        chunks.append(raw)
    head = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(chunks))


def linear(name: str, out: int, inp: int, rng: np.random.Generator):
    t = {f"{name}.weight": rng.uniform(-0.08, 0.08, size=(out, inp))}
    t[f"{name}.bias"] = rng.uniform(-0.08, 0.08, size=(out,))
    return t


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def gen(
    name: str,
    inp_dim: int,
    hidden: int,
    out_dim: int,
    use_bias: bool,
    m: int,
    seed: int,
) -> None:
    d = GOLDEN / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    tensors: dict[str, np.ndarray] = {}
    tensors.update(linear(f"{PREFIX}.hidden_layer", hidden, inp_dim, rng))
    tensors.update(linear(f"{PREFIX}.output_layer", out_dim, hidden, rng))
    tensors.update(linear(f"{PREFIX}.residual_layer", out_dim, inp_dim, rng))
    if not use_bias:
        for k in list(tensors):
            if k.endswith(".bias"):
                del tensors[k]
    write_safetensors(d / "weights.safetensors", tensors)

    x = rng.normal(size=(m, inp_dim)).astype("<f4")

    def W(n: str) -> np.ndarray:
        return tensors[f"{n}.weight"]

    def b(n: str) -> np.ndarray:
        return tensors.get(f"{n}.bias")

    def fwd(xx: np.ndarray) -> np.ndarray:
        bh = b(f"{PREFIX}.hidden_layer")
        bo = b(f"{PREFIX}.output_layer")
        br = b(f"{PREFIX}.residual_layer")
        h = relu(xx @ W(f"{PREFIX}.hidden_layer").T + (bh if bh is not None else 0.0))
        y = h @ W(f"{PREFIX}.output_layer").T + (bo if bo is not None else 0.0)
        y = y + xx @ W(f"{PREFIX}.residual_layer").T + (br if br is not None else 0.0)
        return y

    y = fwd(x)
    (d / "input.bin").write_bytes(np.ascontiguousarray(x, "<f4").tobytes())
    (d / "expected.bin").write_bytes(np.ascontiguousarray(y, "<f4").tobytes())
    print(f"{name}: {x.shape} -> {y.shape} (bias={use_bias})")


if __name__ == "__main__":
    gen("residual_block_full", inp_dim=192, hidden=1280, out_dim=1280, use_bias=False, m=4, seed=0)
    gen("residual_block_bias", inp_dim=9, hidden=24, out_dim=16, use_bias=True, m=3, seed=1)