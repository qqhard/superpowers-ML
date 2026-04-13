#!/usr/bin/env bash
# Dry run: just verify the protocol/experiences files parse and the scripts execute.
set -euo pipefail

cd "$(dirname "$0")/base-project"
export OUT_DIR="$(mktemp -d)"
export CKPT="$OUT_DIR/ckpt.json"

# Exercise train + eval as Supervisor would
STEPS=20 python3 train.py
python3 evaluate.py > "$OUT_DIR/eval_out.txt"

cat "$OUT_DIR/eval_out.txt"

# Bump STEPS to demonstrate the "Researcher modifies script" behavior
STEPS=200 python3 train.py
python3 evaluate.py > "$OUT_DIR/eval_out_2.txt"

cat "$OUT_DIR/eval_out_2.txt"
