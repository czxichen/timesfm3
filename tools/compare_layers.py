#!/usr/bin/env python3
"""Layer-by-layer parity: official torch vs Rust engine intermediates.

Official: out_official/layers.npz (shapes (b,v,n,·)).
Rust:     out_rust_layers/*.bin    (shapes (bv*n, ·) flat; b and v merged).

Compares flattened row-major buffers, which are identical layouts for
(b,v,n,d) and (bv*n,d) when b*v rows are merged in the same order.

Reports max-abs diff per layer to find the first divergence point.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def read_f32_bin(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").copy()


def maxdiff(name: str, a: np.ndarray, b: np.ndarray) -> None:
    n = min(a.size, b.size)
    d = np.abs(a.reshape(-1)[:n] - b.reshape(-1)[:n])
    if d.size == 0:
        print(f"{name}: (no overlap)")
        return
    try:
        i = int(np.argmax(d))
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"{name} argmax: {exc}") from exc
    print(
        f"{name:>28} len {a.size:>8} vs {b.size:>8}  "
        f"maxdiff {d.max():.5e} at {i} (a={a.reshape(-1)[i]:.5f} b={b.reshape(-1)[i]:.5f})"
    )


def main() -> None:
    try:
        off = np.load(ROOT / "out_official" / "layers.npz")
        files = {
            p.stem: p for p in (ROOT / "out_rust_layers").glob("*.bin")
        }
    except OSError as exc:
        raise SystemExit(f"cannot load: {exc}") from exc

    rust_guard = {k: read_f32_bin(p) for k, p in files.items()}

    o_rb_out = off["__call__:transformer_input"]  # (b,v,n,d) == resblock out
    o_trans = off["__call__:transformer_output"]
    o_logits = off["logits"]  # (b,v,n,o,q)
    o_mu = off["pre_mu"]
    o_sigma = off["pre_sigma"]
    o_n = off["pre_n"]

    # Official uses symmetric averaging → 2 mirror batches; Rust export used
    # the single first batch (b=1). Align by taking official batch 0.
    print("official batch0 == batch1 (mirror)? mu:", np.allclose(o_mu[0], o_mu[1]))
    print("checking against official batch 0\n")

    maxdiff("resblock_out", o_rb_out[0], rust_guard["resblock_out"])
    for li in range(20):
        key = f"layer_{li}"
        if key not in rust_guard:
            break
        # official has no per-layer dump; we compare transformer_out only.
        _ = li
    maxdiff("transformer_out", o_trans[0], rust_guard["transformer_out"])
    maxdiff("logits_raw", o_logits[0], rust_guard["logits_raw"])
    maxdiff("revin_mean", o_mu[0], rust_guard["revin_mean"])
    maxdiff("revin_std", o_sigma[0], rust_guard["revin_std"])
    maxdiff("revin_n", o_n[0], rust_guard["revin_n"])

    # Also compare pre_resblock_input if present (shape (b,v,n,192))
    if "pre_resblock_input" in off:
        o_in = off["pre_resblock_input"]
        print("\nofficial pre_resblock_input shape:", o_in.shape)


if __name__ == "__main__":
    main()