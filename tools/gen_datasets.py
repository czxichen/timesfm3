#!/usr/bin/env python3
"""Build extra REAL data sources for the parity/speed benchmark.

Downloaded from the public `laiguokun/multivariate-time-series-data` mirror
(Lai et al. 2018, same origin as the LSTNet / TSLib benchmark suite) and
converted to the project's CSV layout:

    <header line>, then one row per time step:  <index>,<v0>,<v1>,...

i.e. exactly the shape `tools/gen_scenarios.py::_load_csv` expects
(header skipped, first column dropped).

Wide datasets (electricity 321v, solar 137v, traffic 862v) are subsampled
evenly along the variate axis so that the scenario matrix stays affordable
while keeping real-world scale/seasonality characteristics.

Usage: python tools/gen_datasets.py [--src-dir DIR]
"""

import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# name -> (gz file stem, n_variates to keep, short note)
SPEC = {
    "exchange": ("exchange_rate", 8, "daily FX rates, 8 countries, 7588 steps"),
    "electricity": ("electricity", 24, "hourly household power load (321v -> 24v), 26304 steps"),
    "solar": ("solar_AL", 24, "10-min solar power (137v -> 24v), 52560 steps"),
    "traffic": ("traffic", 24, "hourly road occupancy (862v -> 24v), 17544 steps"),
    "electricity64": ("electricity", 64, "wide real table: 64 variates, 26304 steps"),
}


def _pick(n_total: int, k: int) -> list[int]:
    """Evenly spaced variate indices (deterministic, spread across the table)."""
    if k >= n_total:
        return list(range(n_total))
    return [i * n_total // k for i in range(k)]


def convert(src_dir: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for name, (stem, keep, note) in SPEC.items():
        gz = src_dir / f"{stem}.txt.gz"
        if not gz.exists():
            print(f"  skip {name}: {gz} missing (run the download step first)")
            continue
        with gzip.open(gz, "rt") as fh:
            rows = [line.strip() for line in fh if line.strip()]
        n_total = len(rows[0].split(","))
        idx = _pick(n_total, keep)
        header = "date," + ",".join(f"v{i}" for i in range(keep))
        out = [header]
        for t, line in enumerate(rows):
            vals = line.split(",")
            out.append(str(t) + "," + ",".join(f"{float(vals[i]):.6f}" for i in idx))
        path = DATA / f"{name}.csv"
        path.write_text("\n".join(out) + "\n")
        meta[name] = {
            "csv": str(path.relative_to(ROOT)),
            "variates": keep,
            "steps": len(rows),
            "note": note,
            "source_variates": n_total,
        }
        print(f"  {name:<14} {keep:>3}v x {len(rows):>6}  -> {path.name}  ({note})")
    return meta


def main() -> None:
    src = Path(sys.argv[sys.argv.index("--src-dir") + 1]) if "--src-dir" in sys.argv \
        else Path("/tmp/tsdl")
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"converting raw downloads from {src} -> {DATA}")
    meta = convert(src)
    (ROOT / "tools" / "datasets_extra.json").write_text(
        __import__("json").dumps(meta, indent=2, ensure_ascii=False)
    )
    print(f"wrote tools/datasets_extra.json ({len(meta)} extra sources)")
    if not meta:
        raise SystemExit("no dataset converted; see tools/gen_datasets.py docstring")


if __name__ == "__main__":
    main()
