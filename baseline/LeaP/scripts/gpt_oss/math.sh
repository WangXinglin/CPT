#!/bin/bash
set -euo pipefail

INSTALL_DEPS="${INSTALL_DEPS:-0}"
if [ "$INSTALL_DEPS" = "1" ]; then
    pip install Levenshtein jsonargparse
fi

if [ $# -gt 7 ]; then
    echo "Usage: bash scripts/gpt_oss/math.sh [max_turns] [max_tokens] [peer_top_k] [router] [batch_size] [start_idx] [end_idx]" >&2
    exit 1
fi

# Usage:
#   bash scripts/gpt_oss/math.sh [max_turns] [max_tokens] [peer_top_k] [router] [batch_size] [start_idx] [end_idx]
MAX_TURNS="${MAX_TURNS:-${1:-19}}"
MAX_TOKENS="${MAX_TOKENS:-${2:-2048}}"
PEER_TOP_K="${PEER_TOP_K:-${3:-4}}"
ROUTER="${ROUTER:-${4:-dispersed}}"
BS="${BS:-${5:-1}}"
START_IDX="${START_IDX:-${6:-0}}"
END_IDX="${END_IDX:-${7:-}}"

TASKS="HMMT_25"
N=64

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEAP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LEAP_SCRIPT="${LEAP_SCRIPT:-$LEAP_ROOT/inference/gpt_oss/math/runner.py}"
MODEL_PATH="${MODEL_PATH:-/path/to/your/gpt-oss-model}"
DATA_DIR="${DATA_DIR:-/path/to/your/data}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/path/to/your/output/leap/gpt_oss/math}"

RANGE_ARGS=(--start-idx "${START_IDX}")
if [[ -n "${END_IDX}" ]]; then
    RANGE_ARGS+=(--end-idx "${END_IDX}")
fi

echo "======================================================="
echo "LeaP gpt-oss-20b run"
echo "Task: ${TASKS}"
echo "N: ${N}"
echo "Range: ${START_IDX}${END_IDX:+ to ${END_IDX}}"
echo "Output base: ${OUTPUT_BASE_DIR}"
echo "======================================================="

SAVE_DIR="${OUTPUT_BASE_DIR}/leap_${TASKS}_gpt20b_N${N}"
mkdir -p "${SAVE_DIR}"
python "${LEAP_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --tasks "${TASKS}" \
    --max_turns "${MAX_TURNS}" \
    --peer_top_k "${PEER_TOP_K}" \
    --router "${ROUTER}" \
    --temperature 1.0 \
    --top_p 1.0 \
    --min_p 0.05 \
    --max_tokens "${MAX_TOKENS}" \
    --summarize_max_tokens 256 \
    --n "${N}" \
    --num_gpus 8 \
    --gpu_memory_utilization 0.95 \
    --tensor_parallel_size 8 \
    --batch_size "${BS}" \
    --resume true \
    "${RANGE_ARGS[@]}"

echo "Done: ${SAVE_DIR}"
