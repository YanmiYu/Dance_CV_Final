#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BENCHMARK="${BENCHMARK:-inputs/benchmark.mp4}"
LEARNER="${LEARNER:-inputs/user.mp4}"
OUT="${OUT:-results/final_demo}"
DEVICE="${DEVICE:-cpu}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

PYTHONPATH=. "$PYTHON_BIN" run.py \
  --benchmark "$BENCHMARK" \
  --learner "$LEARNER" \
  --out "$OUT" \
  --config configs/integrate/pipeline.yaml \
  --device "$DEVICE" \
  --require-lstm
