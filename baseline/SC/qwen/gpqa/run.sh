#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash baseline/SC/qwen/gpqa/run.sh <start_index> <end_index>"
  echo "Example: bash baseline/SC/qwen/gpqa/run.sh 0 50"
  exit 1
fi

START_INDEX="$1"
END_INDEX="$2"
if ! [[ "$START_INDEX" =~ ^[0-9]+$ && "$END_INDEX" =~ ^[0-9]+$ ]] || [ "$START_INDEX" -ge "$END_INDEX" ]; then
  echo "Error: start_index and end_index must be non-negative integers with start_index < end_index." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLER="${SCRIPT_DIR}/sample.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Override these placeholders with environment variables for your machine.
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"
DATA_DIR="${DATA_DIR:-/path/to/your/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/your/output/sc/qwen/gpqa}"
INPUT_EXT="${INPUT_EXT:-.jsonl}"

# Sampling parameters.
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TP_SIZE="${TP_SIZE:-8}"

DATASET="gpqa"
N_COMPLETIONS=64
QUESTION_FILE="${DATA_DIR}/${DATASET}${INPUT_EXT}"
OUT_DIR="${OUTPUT_ROOT}/${DATASET}/N${N_COMPLETIONS}"

echo "======================================================="
echo "SC sampling: Qwen / GPQA"
echo "Range: ${START_INDEX} to ${END_INDEX}"
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_ROOT}"
echo "======================================================="

mkdir -p "$OUT_DIR"
echo "Dataset=${DATASET} N=${N_COMPLETIONS}"
"$PYTHON_BIN" "$SAMPLER" \
  --model "$MODEL_PATH" \
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

echo "Qwen GPQA sampling completed: ${OUT_DIR}"
