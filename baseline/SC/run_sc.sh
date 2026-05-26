#!/bin/bash
set -e

if [ $# -lt 2 ]; then
  echo "Usage: bash run_sc.sh <start_index> <end_index>"
  echo "Example: bash run_sc.sh 0 50"
  exit 1
fi

START_INDEX=$1
END_INDEX=$2

echo "======================================================="
echo "Start SC Sampling"
echo "Range: $START_INDEX to $END_INDEX"
echo "======================================================="

# deps for math-equivalence
pip install -U sympy antlr4-python3-runtime

# ===== Parameters (keep consistent with your baseline) =====
BATCH_SIZE=1
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
N_COMPLETIONS=512
MAX_TOKENS=38912
TP_SIZE=8

# ===== Paths =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${MODEL_NAME:-/path/to/model}"
QUESTION_FILE="${QUESTION_FILE:-/path/to/data/AIME_24.jsonl}"
OUT_DIR="${OUT_DIR:-/path/to/output}"

mkdir -p "$OUT_DIR"

python "${SCRIPT_DIR}/SC.py" \
  --model "$MODEL_NAME" \
  --input "$QUESTION_FILE" \
  --output "$OUT_DIR" \
  --n-completions "$N_COMPLETIONS" \
  --batch-size "$BATCH_SIZE" \
  --tensor-parallel-size "$TP_SIZE" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-tokens "$MAX_TOKENS" \
  --start-idx "$START_INDEX" \
  --end-idx "$END_INDEX"

echo "======================================================="
echo "Done."
echo "Outputs: $OUT_DIR/{idx}.json"
echo "======================================================="
