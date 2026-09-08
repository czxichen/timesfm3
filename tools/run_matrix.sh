#!/usr/bin/env bash
# Full 4-pass matrix: official torch + Rust f32/f16/balanced over all 211 scenarios.
set -u
cd /d/tmp_r/timesfm3
PY="/c/Users/djl9460/.workbuddy/binaries/python/envs/default/Scripts/python"
EXE=./target/release/export_scenarios
M=tools/scenarios_v3.json

echo "=== [1/4] official torch ==="
"$PY" tools/run_official_scenarios.py --manifest "$M"

echo "=== [2/4] rust f32 ==="
"$EXE" ckpt "$M" out_rust_sc

echo "=== [3/4] rust f16 ==="
"$EXE" ckpt_f16 "$M" out_rust_sc_f16

echo "=== [4/4] rust f16-balanced ==="
"$EXE" ckpt_f16_balanced "$M" out_rust_sc_bal

echo "=== compare ==="
"$PY" tools/compare_v3.py
echo "ALL DONE"
