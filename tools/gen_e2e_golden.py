#!/usr/bin/env python3
"""End-to-end golden generator for the Rust engine.

Builds a tiny TimesFM3-equivalent model with numpy-only math (translated from
the official `repo/src/timesfm3/model.py`), runs `decode()` on a fixed input,
and writes golden/e2e_micro/{config.json, model.safetensors, input.npz, expected.npz}.

The Rust tests load the same weights + input and must reproduce `expected`
(reverse-RevINed decode logits, shape (b, v_total, horizon, q)).

This is an independent oracle: every op is reimplemented from the official
math, NOT from the Rust code, so agreement catches indexing/ordering bugs.
"""

import json
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "golden" / "e2e_micro"

# ---- tiny model config (mirrors official model_test.py) ----
P = 4           # input_patch_len
O = 8           # output_patch_len (rolls = O//P = 2)
Q = 3           # num quantiles
D = 8           # model dims
H = 2           # heads  (head_dim = 4)
LAYERS = 2
HORIZON = 10    # forecast length (< padded horizon, exercises truncation)
CONTEXT = 13    # non-multiple of P → exercises left padding

rng = np.random.default_rng(7)


def weights(shape: tuple[int, ...], scale: float = 0.6) -> np.ndarray:
    return rng.uniform(-scale, scale, size=shape).astype(np.float32)


# ---------------------------------------------------------------------------
# layer math (independent numpy port of official model.py)
# ---------------------------------------------------------------------------

def rms_norm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: (..., d) → normalized; w: (d,)."""
    xf = x.astype(np.float64)
    rms = np.sqrt(np.mean(xf**2, axis=-1, keepdims=True) + 1e-7)
    return (xf / rms * w.astype(np.float64)).astype(np.float32)


def rope(x: np.ndarray, pos: np.ndarray, dims: int) -> np.ndarray:
    """x: (batch, n, h, hd); pos: (n,) int (broadcast over heads)."""
    half = dims // 2
    frac = 2.0 * np.arange(half, dtype=np.float64) / dims
    ts = 10000.0**frac  # (half,)
    angle = np.array(pos, dtype=np.float64)[:, None] / ts[None, :]  # (n, half)
    sin = np.sin(angle)[:, None, :]  # (n, 1, half)
    cos = np.cos(angle)[:, None, :]
    first, second = x[..., :half], x[..., half:]
    out = np.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=-1
    )
    return out.astype(np.float32)


def softplus(x: np.ndarray) -> np.ndarray:
    return (np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)).astype(np.float64)


def per_dim_scale(x: np.ndarray, param: np.ndarray) -> np.ndarray:
    """x: (batch, n, h, hd); param: (hd,)."""
    c = 1.442695041 / np.sqrt(x.shape[-1])
    s = (c * softplus(param.astype(np.float64))).astype(np.float32)
    return x * s


def softmax_rows(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def mha(x: np.ndarray, n: int, batch: int, W: dict[str, np.ndarray], rope_flag: bool, causal: bool, kv_mask: np.ndarray | None = None) -> np.ndarray:
    """x: (batch*n, d) → (batch*n, d). kv_mask: (batch*n,) bool, True = fully
    masked (excluded from attending), matching official make_attn_mask."""
    d = x.shape[-1]
    hd = d // H
    q = x @ W["wq"]
    k = x @ W["wk"]
    v = x @ W["wv"]
    q = q.reshape(batch, n, H, hd)
    k = k.reshape(batch, n, H, hd)
    v = v.reshape(batch, n, H, hd)
    if rope_flag:
        pos = np.arange(n, dtype=np.int32)
        q = rope(q, pos, hd)
        k = rope(k, pos, hd)
    q = np.stack([rms_norm(cast(np.ndarray, q[b]), W["qln"]) for b in range(batch)])
    k = np.stack([rms_norm(cast(np.ndarray, k[b]), W["kln"]) for b in range(batch)])
    q = np.stack([per_dim_scale(q[b], W["pds"]) for b in range(batch)])

    sq = np.sqrt(hd)
    outs = []
    for b in range(batch):
        attn = []
        for h in range(H):
            qh = q[b][:, h, :] * sq  # (n, hd)
            kh = k[b][:, h, :]
            logits = qh @ kh.T  # (n, n)
            if causal:
                logits = np.where(np.triu(np.ones((n, n), dtype=bool), 1), -1e9, logits)
            if kv_mask is not None:
                # exclude fully-masked kv positions (col index = global token)
                kv = kv_mask[b * n : (b + 1) * n][None, :]  # (1, n)
                logits = np.where(kv, -1e9, logits)
            p_ = softmax_rows(logits)
            attn.append(p_ @ v[b][:, h, :])
        outs.append(np.stack(attn, axis=1))  # (n, h, hd)
    out = np.stack(outs, axis=0).reshape(batch * n, d)
    return out @ W["wo"]


def mixing_layer(x: np.ndarray, b: int, v: int, n: int, W: dict[str, Any], eff_mask: np.ndarray | None = None) -> np.ndarray:
    """x: (b*v*n, d) flat (== (b,v,n,d) row-major). eff_mask: (b, v, n) bool of
    fully-masked patches (cumprod-ified) — seq attention excludes masked kv,
    var attention excludes masked variates."""
    d = x.shape[-1]
    bv = b * v
    seq_in = x.reshape(bv, n, d)

    seq_kv = eff_mask.reshape(bv, n).astype(bool) if eff_mask is not None else None
    # --- sequence attention (batch=bv, seq=n, causal) ---
    z = rms_norm(seq_in, cast(np.ndarray, W["pre_seq"]))
    z = mha(z.reshape(bv * n, d), n, bv, cast(dict[str, np.ndarray], W["seq"]), True, True, None if seq_kv is None else seq_kv.reshape(-1))
    z = rms_norm(z, cast(np.ndarray, W["post_seq"])) + seq_in

    # --- variate attention (batch=b*n, seq=v, non-causal) ---
    var_in = z.reshape(b, v, n, d).transpose(0, 2, 1, 3).reshape(b * n, v, d)
    var_kv = (
        eff_mask.transpose(0, 2, 1).reshape(b * n, v).astype(bool).reshape(-1)
        if eff_mask is not None else None
    )
    z2 = rms_norm(var_in, cast(np.ndarray, W["pre_var"]))
    z2 = mha(z2.reshape(b * n * v, d), v, b * n, cast(dict[str, np.ndarray], W["var"]), False, False, var_kv)
    z2 = z2.reshape(b, n, v, d).transpose(0, 2, 1, 3).reshape(bv, n, d)
    z2 = rms_norm(z2, cast(np.ndarray, W["post_var"])) + z

    # --- ffn ---
    ff = rms_norm(z2, cast(np.ndarray, W["pre_ff"]))
    ff = np.maximum(ff @ cast(np.ndarray, W["ff0"]), 0.0)
    ff = ff @ cast(np.ndarray, W["ff1"])
    out = rms_norm(ff, cast(np.ndarray, W["post_ff"])) + z2
    return out.reshape(bv * n, d)


def update_running_stats(
    n: np.ndarray, mu: np.ndarray, sigma: np.ndarray, x: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x/mask: (bv, p). n/mu/sigma: (bv,)."""
    legit = ~mask
    inc_n = legit.sum(-1).astype(np.float64)
    xm = np.where(legit, x, 0.0).astype(np.float64)
    inc_mu = np.where(inc_n == 0, 0.0, xm.sum(-1) / np.maximum(inc_n, 1))
    diff = np.where(legit, (x.astype(np.float64) - inc_mu[..., None]) ** 2, 0.0)
    inc_var = np.where(inc_n == 0, 0.0, diff.sum(-1) / np.maximum(inc_n, 1))
    inc_sigma = np.sqrt(inc_var)
    new_n = n.astype(np.float64) + inc_n
    new_mu = np.where(new_n == 0, 0.0, (n * mu + inc_mu * inc_n) / np.maximum(new_n, 1e-12))
    new_sigma = np.sqrt(
        np.where(
            new_n == 0,
            0.0,
            (
                n * sigma * sigma
                + inc_n * inc_sigma * inc_sigma
                + n * (mu - new_mu) ** 2
                + inc_n * (inc_mu - new_mu) ** 2
            )
            / np.maximum(new_n, 1e-12),
        )
    )
    return new_n.astype(np.float32), new_mu.astype(np.float32), new_sigma.astype(np.float32)


def get_running_stats(values: np.ndarray, masks: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """values/masks: (bv, n, p) → (running_mu, running_sigma) each (bv, n)."""
    bv, nn, p = values.shape
    assert (nn, p) == (n, values.shape[2])
    n_v, mu, sigma = np.zeros(bv), np.zeros(bv), np.zeros(bv)
    all_mu, all_sigma = [], []
    for i in range(n):
        n_v, mu, sigma = update_running_stats(
            n_v, mu, sigma, values[:, i, :], masks[:, i, :]
        )
        all_mu.append(mu.copy())
        all_sigma.append(sigma.copy())
    return np.stack(all_mu, axis=1), np.stack(all_sigma, axis=1)


# ---------------------------------------------------------------------------
# model assembly
# ---------------------------------------------------------------------------

def build_layer_weights(layer: int) -> dict[str, Any]:
    W: dict[str, Any] = {}
    for mha_name in ("seq", "var"):
        W[mha_name] = {
            "wq": weights((D, D)),
            "wk": weights((D, D)),
            "wv": weights((D, D)),
            "wo": weights((D, D)),
            "qln": np.ones(D // H),
            "kln": np.ones(D // H),
            "pds": np.zeros(D // H),
        }
    for name in ("pre_seq", "post_seq", "pre_var", "post_var", "pre_ff", "post_ff"):
        W[name] = weights((D,))
    W["ff0"] = weights((D, D))
    W["ff1"] = weights((D, D))
    return W


RESBLOCK_IN = 2 * (P + O)  # values|fcov|mask|fcov_mask
RESBLOCK = {
    "w1": weights((RESBLOCK_IN, D)),
    "w2": weights((D, D)),
    "wr": weights((RESBLOCK_IN, D)),
}
HEAD = weights((D, O * Q))
LAYERS_W = [build_layer_weights(layer) for layer in range(LAYERS)]


def preprocess_and_forward(
    values3: np.ndarray,
    masks3: np.ndarray,
    b: int,
    v: int,
    n: int,
    patch_is_target: np.ndarray | None = None,
    patch_cpm_mask: np.ndarray | None = None,
) -> tuple[
    np.ndarray,  # logits
    np.ndarray,  # running_mu
    np.ndarray,  # running_sigma
    np.ndarray,  # eff_mask
    np.ndarray,  # rb_out
    np.ndarray,  # transformer output
]:
    """values3/masks3: (bv, n, p). Runs official _preprocess + forward.
    patch_cpm_mask: (b, n) bool — CPM positions; masks target variates there
    (official decode passes the horizon CPM mask)."""
    bv = b * v
    rolls = O // P
    running_mu, running_sigma = get_running_stats(values3, masks3, n)

    # CPM mask: additionally mask target variates at CPM positions (official).
    if patch_cpm_mask is not None:
        assert patch_is_target is not None
        cpm_bvnp = patch_cpm_mask[None, :, None]  # (1, b, n) → (b, n) handled below
        cpm3 = patch_cpm_mask.reshape(b, 1, n)
        target3 = patch_is_target.reshape(b, v, n)
        extra = cpm3 & target3  # (b, v, n)
        masks3 = masks3 | extra[..., None].repeat(P, axis=-1)

    # RevIN normalize
    norms = (values3 - running_mu[:, :, None]) / np.where(
        running_sigma[:, :, None] < 1e-6, 1.0, running_sigma[:, :, None]
    )
    norms = np.where(masks3, 0.0, norms)
    # future covariates via roll (shift left by one patch, rolls times)
    fcov_parts = []
    src = norms
    for _ in range(rolls):
        src = np.concatenate([src[:, 1:, :], src[:, :1, :]], axis=1)
        fcov_parts.append(src)
    fcov = np.concatenate(fcov_parts, axis=-1)  # (bv, n, rolls*p)
    # decode path: horizon patches are masked & target → fcov fully masked
    fcov_mask = np.ones_like(fcov, dtype=bool)
    fcov_vals = np.where(fcov_mask, 0.0, fcov)
    rb_in = np.concatenate(
        [
            norms,
            fcov_vals,
            masks3.astype(np.float32),
            fcov_mask.astype(np.float32),
        ],
        axis=-1,
    )
    h = np.maximum(rb_in @ RESBLOCK["w1"], 0.0)
    rb_out = h @ RESBLOCK["w2"] + rb_in @ RESBLOCK["wr"]

    # fully-masked patch: every value AND fcov point masked → eff = cumprod
    full_mask = masks3.all(axis=-1) & fcov_mask.all(axis=-1)  # (bv, n)
    eff = np.cumprod(full_mask.astype(int), axis=-1).astype(bool)  # along patch axis
    eff3 = eff.reshape(b, v, n)

    x = rb_out.reshape(bv, n, D)
    for layer in range(LAYERS):
        x = mixing_layer(x, b, v, n, LAYERS_W[layer], eff3)
    logits = x @ HEAD
    return logits, running_mu, running_sigma, eff3, rb_out, x


def stitch(preds: np.ndarray, patch_len: int) -> np.ndarray:
    """preds: (b, v, n, extract_len, q) → (b, v, n*patch_len + overlap, q)."""
    b_, v_, n_, total, q_ = preds.shape
    overlap = total - patch_len
    if n_ == 1:
        return preds[:, :, 0, :, :]
    w = np.linspace(1.0, 0.0, overlap)[None, None, None, :, None]
    first = preds[:, :, 0, :patch_len, :]
    prev = preds[:, :, :-1, :, :]
    nxt = preds[:, :, 1:, :, :]
    so = w * prev[:, :, :, patch_len:, :] + (1 - w) * nxt[:, :, :, :overlap, :]
    mid = nxt[:, :, :, overlap:patch_len, :]
    chunks = np.concatenate([so, mid], axis=3).reshape(b_, v_, (n_ - 1) * patch_len, q_)
    tail = preds[:, :, -1, patch_len:, :]
    return np.concatenate([first, chunks, tail], axis=2)


def decode(target: np.ndarray) -> np.ndarray:
    """target: (b=1, v_total=1, C). Returns (b, v_total, horizon, q)."""
    b, v = 1, 1
    context = target.shape[-1]
    ctx_padding = (P - (context % P)) % P
    ctx_padded = context + ctx_padding

    tgt_padded = np.pad(target, ((0, 0), (0, 0), (ctx_padding, 0)))
    mask = np.zeros((b, ctx_padded), dtype=bool)
    mask[:, :ctx_padding] = True

    extract_len = min(2 * P, O)
    overlap = extract_len - P
    num_forecast_patches = max((HORIZON - overlap + P - 1) // P, 1)
    num_horizon_patches = num_forecast_patches + (O // P) - 1
    padded_horizon = num_horizon_patches * P
    num_context_patches = ctx_padded // P
    num_total_patches = num_context_patches + num_horizon_patches
    total_len = ctx_padded + padded_horizon

    all_vals = np.zeros((b, v, total_len), dtype=np.float32)
    all_vals[:, :, :ctx_padded] = tgt_padded
    all_masks = np.concatenate(
        [
            mask[:, None, :].repeat(v, axis=1),
            np.ones((b, v, padded_horizon), dtype=bool),
        ],
        axis=-1,
    )

    values_bvnp = all_vals.reshape(b, v, num_total_patches, P)
    masks_bvnp = all_masks.reshape(b, v, num_total_patches, P)
    # patch_is_target: for univariate decode everything is target
    patch_is_target = np.ones((b, v, num_total_patches), dtype=bool)

    horizon_cpm_mask_ = np.zeros((b, num_total_patches), dtype=bool)
    horizon_cpm_mask_[:, num_context_patches:] = True
    INTERMEDIATES["horizon_cpm"] = horizon_cpm_mask_.reshape(-1).astype("<f4")

    logits, running_mu, running_sigma, eff3, rb_out, trans_out = preprocess_and_forward(
        values_bvnp.reshape(b * v, num_total_patches, P),
        masks_bvnp.reshape(b * v, num_total_patches, P),
        b,
        v,
        num_total_patches,
        patch_is_target=patch_is_target,
        patch_cpm_mask=horizon_cpm_mask_,
    )
    INTERMEDIATES["logits_raw"] = logits.reshape(-1).astype("<f4")
    INTERMEDIATES["running_mu"] = running_mu.reshape(-1).astype("<f4")
    INTERMEDIATES["running_sigma"] = running_sigma.reshape(-1).astype("<f4")
    INTERMEDIATES["eff_mask"] = eff3.reshape(-1).astype("<f4")
    INTERMEDIATES["rb_out"] = rb_out.reshape(-1).astype("<f4")
    INTERMEDIATES["trans_out"] = trans_out.reshape(-1).astype("<f4")

    denorm = (
        logits.reshape(b, v, num_total_patches, O, Q)
        * running_sigma.reshape(b, v, num_total_patches, 1, 1)
        + running_mu.reshape(b, v, num_total_patches, 1, 1)
    )
    preds = denorm[
        :,
        :,
        num_context_patches - 1 : num_context_patches - 1 + num_forecast_patches,
        :extract_len,
        :,
    ]
    INTERMEDIATES["values_bvnp"] = values_bvnp.reshape(-1).astype("<f4")
    INTERMEDIATES["masks_bvnp"] = masks_bvnp.reshape(-1).astype("<f4")
    INTERMEDIATES["patch_is_target"] = patch_is_target.reshape(-1).astype("<f4")
    INTERMEDIATES["denorm"] = denorm.reshape(-1).astype("<f4")
    stitched = stitch(preds, P)
    INTERMEDIATES["stitched"] = stitched.reshape(-1).astype("<f4")
    INTERMEDIATES["num_context_patches"] = np.array(num_context_patches, dtype="<f4")
    INTERMEDIATES["num_forecast_patches"] = np.array(num_forecast_patches, dtype="<f4")
    return stitched[:, :, :HORIZON, :]


# ---------------------------------------------------------------------------
# data + checkpoint writer
# ---------------------------------------------------------------------------

INTERMEDIATES: dict[str, np.ndarray] = {}
TARGET = rng.normal(size=(1, 1, CONTEXT)).astype(np.float32)
EXPECTED = decode(TARGET)

NORM_KEYS = {
    "pre_seq_attn_ln": "pre_seq",
    "post_seq_attn_ln": "post_seq",
    "pre_var_attn_ln": "pre_var",
    "post_var_attn_ln": "post_var",
    "pre_ff_ln": "pre_ff",
    "post_ff_ln": "post_ff",
}


def write_safetensors(path: Path) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "timesfm3-e2e-golden"}}
    named: dict[str, np.ndarray] = {}
    named["pre_transformer_resblock.hidden_layer.weight"] = RESBLOCK["w1"].T
    named["pre_transformer_resblock.output_layer.weight"] = RESBLOCK["w2"].T
    named["pre_transformer_resblock.residual_layer.weight"] = RESBLOCK["wr"].T
    named["output_head.weight"] = HEAD.T
    named["output_head.bias"] = np.zeros(HEAD.shape[1])
    for layer in range(LAYERS):
        W = LAYERS_W[layer]
        for mha_name in ("seq", "var"):
            wq = cast(dict[str, np.ndarray], W[mha_name])
            base = f"transformer_stack.layers.{layer}.{mha_name}_attn"
            named[f"{base}.query_proj.weight"] = wq["wq"].T
            named[f"{base}.key_proj.weight"] = wq["wk"].T
            named[f"{base}.value_proj.weight"] = wq["wv"].T
            named[f"{base}.out_proj.weight"] = wq["wo"].T
            named[f"{base}.query_ln.weight"] = wq["qln"]
            named[f"{base}.key_ln.weight"] = wq["kln"]
            named[f"{base}.per_dim_scale"] = wq["pds"]
        for torch_name, wkey in NORM_KEYS.items():
            named[f"transformer_stack.layers.{layer}.{torch_name}.weight"] = cast(
                np.ndarray, W[wkey]
            )
        named[f"transformer_stack.layers.{layer}.ff0.weight"] = cast(np.ndarray, W["ff0"]).T
        named[f"transformer_stack.layers.{layer}.ff1.weight"] = cast(np.ndarray, W["ff1"]).T

    offset = 0
    chunks = []
    for name, arr in named.items():
        arr = np.ascontiguousarray(arr, dtype="<f4")
        raw = arr.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        chunks.append(raw)
    head = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(chunks))


def main() -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    write_safetensors(GOLDEN / "model.safetensors")
    cfg = {
        "input_patch_len": P,
        "output_patch_len": O,
        "quantiles": [0.1, 0.5, 0.9],
        "residual_block_config": {
            "hidden_dims": D,
            "output_dims": D,
            "use_bias": False,
            "activation": "relu",
            "prenorm": "none",
        },
        "transformer_config": {
            "num_layers": LAYERS,
            "transformer": {
                "model_dims": D,
                "hidden_dims": D,
                "num_heads": H,
                "attention_norm": "rms",
                "feedforward_norm": "rms",
                "qk_norm": "rms",
                "use_rope_seq": True,
                "use_rope_var": False,
                "use_bias": False,
                "ff_activation": "relu",
                "deterministic": True,
                "use_sdpa": True,
            },
        },
        "use_stitching": True,
        "use_linear_detrending": False,  # oracle omits detrending (simpler)
        "use_iterative_cpm_revin": False,
        "use_frozen_running_stats": False,
        "use_variate_attention": True,
        "value_clip": 1e20,
        "input_transform": "identity",
    }
    (GOLDEN / "config.json").write_text(json.dumps(cfg))
    np.savez(
        GOLDEN / "input.npz",
        target=TARGET,
        horizon=np.array(HORIZON, dtype=np.float32),
        context_len=np.array(CONTEXT, dtype=np.float32),
    )
    np.savez(GOLDEN / "expected.npz", logits=EXPECTED.astype("<f4"))
    np.savez(
        GOLDEN / "intermediates.npz",
        logits_raw=INTERMEDIATES["logits_raw"],
        running_mu=INTERMEDIATES["running_mu"],
        running_sigma=INTERMEDIATES["running_sigma"],
        eff_mask=INTERMEDIATES["eff_mask"],
        rb_out=INTERMEDIATES["rb_out"],
        trans_out=INTERMEDIATES["trans_out"],
        denorm=INTERMEDIATES["denorm"],
        values_bvnp=INTERMEDIATES["values_bvnp"],
        masks_bvnp=INTERMEDIATES["masks_bvnp"],
        patch_is_target=INTERMEDIATES["patch_is_target"],
        horizon_cpm=INTERMEDIATES["horizon_cpm"],
        stitched=INTERMEDIATES["stitched"],
        num_context_patches=INTERMEDIATES["num_context_patches"],
        num_forecast_patches=INTERMEDIATES["num_forecast_patches"],
    )
    print("wrote", GOLDEN)
    print("expected shape:", EXPECTED.shape)
    assert np.isfinite(EXPECTED).all(), "expected output must be finite"
    np.set_printoptions(precision=5, suppress=True)
    print(EXPECTED[0, 0])


if __name__ == "__main__":
    main()