#!/usr/bin/env python3
"""Speed / cold-start / thread-scaling / memory benchmark.

Runs BOTH engines on the same deterministic inputs and writes
report/bench.json:

  * cold_start : full process wall-clock for "start -> load -> 1 forecast"
  * steady     : median / min of N repeated predict_batch calls (model loaded once)
  * threads    : thread-scaling curve on one representative shape
  * memory     : peak RSS of each engine
  * artifacts  : on-disk footprint of each deployment

Usage: python tools/bench_v3.py [--quick]
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
EXE = ROOT / "target" / "release" / "bench_speed.exe"
REPORT = ROOT / "report"
OUT = REPORT / "bench.json"

REPEATS = 5
SHAPES = [
    ("S1 单变量小样本", 1, 512, 96, 1),
    ("S2 基线 7x1024", 7, 1024, 96, 1),
    ("S3 长视野 h720", 7, 1024, 720, 1),
    ("S4 宽表 24x1024", 24, 1024, 96, 1),
    ("S5 长上下文 3072", 7, 3072, 96, 1),
    ("S6 batch4 7x1024", 7, 1024, 96, 4),
]
PRECS = [("f32", "ckpt"), ("f16", "ckpt_f16"), ("balanced", "ckpt_f16_balanced")]
THREADS = [1, 2, 4, 8, 16]


def run_capture(cmd, env=None, timeout=3600):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True,
                       env=env or os.environ.copy(), timeout=timeout)
    return p, (time.perf_counter() - t0) * 1e3


def peak_rss(cmd, env=None, timeout=3600):
    """Run cmd while polling RSS; return (stdout, wall_ms, peak_rss_mb)."""
    import psutil
    env = env or os.environ.copy()
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    ps = psutil.Process(proc.pid)
    peak = 0
    while proc.poll() is None:
        try:
            for ch in ps.children(recursive=True):
                try:
                    peak = max(peak, ch.memory_info().rss)
                except Exception:
                    pass
            peak = max(peak, ps.memory_info().rss)
        except Exception:
            break
        time.sleep(0.05)
    out, _err = proc.communicate(timeout=timeout)
    wall = (time.perf_counter() - t0) * 1e3
    return out, wall, peak / 2**20


def last_json(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"no JSON line in output:\n{text[-2000:]}")


def bench_rust(ckpt: str, prec: str, v: int, c: int, h: int, batch: int,
               repeats: int, threads: int | None = None) -> dict:
    env = os.environ.copy()
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
        env["TIMESFM_NUM_THREADS"] = str(threads)
    cmd = [str(EXE), ckpt, str(v), str(c), str(h), str(repeats), str(batch)]
    if prec != "f32":
        cmd += ["--precision", prec]
    p, wall = run_capture(cmd, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"bench_speed failed: {p.stderr[-2000:]}")
    d = last_json(p.stdout)
    d["wall_ms"] = round(wall, 1)
    d["engine"] = f"rust-{prec}"
    d["precision"] = prec  # balanced == f16 config, tag by checkpoint preset
    return d


def bench_official(v: int, c: int, h: int, batch: int, repeats: int,
                   threads: int | None = None, warmup: int = 1) -> dict:
    cmd = [PY, str(ROOT / "tools" / "bench_official.py"),
           "--variates", str(v), "--context", str(c), "--horizon", str(h),
           "--repeats", str(repeats), "--warmup", str(warmup),
           "--batch", str(batch)]
    if threads:
        cmd += ["--threads", str(threads)]
    p, wall = run_capture(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"bench_official failed: {p.stderr[-2000:]}")
    d = last_json(p.stdout)
    d["wall_ms"] = round(wall, 1)
    d["engine"] = "official-torch"
    return d


def cold_start():
    out = {}
    cmd_r = [str(EXE), "ckpt", "7", "1024", "96", "1"]
    for i in range(3):
        _, wall = run_capture(cmd_r)
        out.setdefault("rust_f32", []).append(round(wall))
    for i in range(3):
        _, wall = run_capture([PY, str(ROOT / "tools" / "bench_official.py"),
                               "--variates", "7", "--context", "1024",
                               "--horizon", "96", "--repeats", "0", "--warmup", "0"])
        out.setdefault("official_torch", []).append(round(wall))
    return out


def memory():
    import psutil
    res = {}
    out_r, wall_r, rss_r = peak_rss([str(EXE), "ckpt", "7", "1024", "96", "3"])
    res["rust_f32"] = {"peak_rss_mb": round(rss_r, 1), "wall_ms": round(wall_r)}
    out_o, wall_o, rss_o = peak_rss(
        [PY, str(ROOT / "tools" / "bench_official.py"), "--variates", "7",
         "--context", "1024", "--horizon", "96", "--repeats", "2", "--warmup", "0"])
    d = last_json(out_o)
    res["official_torch"] = {
        "peak_rss_mb": round(max(rss_o, d.get("peak_rss_mb") or 0), 1),
        "rss_after_load_mb": d.get("rss_after_load_mb"),
        "wall_ms": round(wall_o),
    }
    return res


def artifacts():
    def size(p: Path) -> float:
        if p.is_file():
            return p.stat().st_size / 2**20
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20

    torch_dir = Path(sys.prefix) / "Lib" / "site-packages" / "torch"
    exe = ROOT / "target" / "release" / "forecast.exe"
    return {
        "rust_binary_mb": round(size(exe), 2) if exe.exists() else None,
        "rust_bench_binary_mb": round(size(EXE), 2) if EXE.exists() else None,
        "torch_package_mb": round(size(torch_dir), 1) if torch_dir.exists() else None,
        "python_env_mb": round(size(Path(sys.prefix)), 1),
        "ckpt_f32_mb": round(size(ROOT / "ckpt"), 1),
        "ckpt_f16_mb": round(size(ROOT / "ckpt_f16"), 1),
        "ckpt_balanced_mb": round(size(ROOT / "ckpt_f16_balanced"), 1),
    }


def main() -> None:
    quick = "--quick" in sys.argv
    reps = 2 if quick else REPEATS
    shapes = SHAPES[:2] if quick else SHAPES
    data: dict = {"repeats": reps, "shapes": [], "cold_start": {},
                  "threads": [], "memory": {}, "artifacts": {}}
    done_shapes = set()
    if "--resume" in sys.argv and OUT.exists():
        try:
            data = json.loads(OUT.read_text(encoding="utf-8"))
            done_shapes = {r["label"] for r in data.get("shapes", [])}
            print(f"[resume] skipping {len(done_shapes)} completed shapes")
        except Exception as e:  # pragma: no cover
            print(f"[resume] could not read existing file ({e}), starting fresh")

    for label, v, c, h, b in shapes:
        if label in done_shapes:
            continue
        row = {"label": label, "shape": f"{v * b}x{c}", "horizon": h, "batch": b,
               "runs": []}
        print(f"[steady] {label} {v}x{c} h={h} b={b}")
        row["runs"].append(bench_official(v, c, h, b, reps))
        for prec, ckpt in PRECS:
            row["runs"].append(bench_rust(ckpt, prec, v, c, h, b, reps))
        data["shapes"].append(row)
        OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    if not data.get("cold_start"):
        print("[cold start]")
        data["cold_start"] = cold_start()
        OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    if not data.get("memory"):
        print("[memory]")
        data["memory"] = memory()
        OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    if not quick:
        print("[thread scaling]")
        for t in THREADS:
            row = {"threads": t, "shape": "7x1024", "horizon": 96, "runs": []}
            row["runs"].append(bench_official(7, 1024, 96, 1, 3, threads=t, warmup=1))
            for prec, ckpt in PRECS:
                row["runs"].append(bench_rust(ckpt, prec, 7, 1024, 96, 1, 3, threads=t))
            data["threads"].append(row)
            print(f"  threads={t}: " + " ".join(
                f"{r['engine']}={r['median_ms']:.0f}ms" for r in row["runs"]))
            OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    print("[artifacts]")
    data["artifacts"] = artifacts()
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
