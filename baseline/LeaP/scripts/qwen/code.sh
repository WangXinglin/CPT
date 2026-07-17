#!/bin/bash
set -euo pipefail

INSTALL_DEPS="${INSTALL_DEPS:-0}"
if [ "${INSTALL_DEPS}" = "1" ]; then
    pip install Levenshtein jsonargparse
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEAP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LEAP_SCRIPT="${LEAP_SCRIPT:-$LEAP_ROOT/inference/qwen/code/runner.py}"

if [ $# -gt 2 ]; then
    echo "Usage: bash scripts/qwen/code.sh [START_IDX] [END_IDX]"
    echo "Example: bash scripts/qwen/code.sh 0 30"
    exit 1
fi

START_IDX=${1:-0}
END_IDX=${2:-}

MAX_TURNS="${MAX_TURNS:-16}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
N=64

PEER_TOP_K="${PEER_TOP_K:-4}"
ROUTER="${ROUTER:-dispersed}"
MODEL_PATH="${MODEL_PATH:-/path/to/your/qwen-model}"
INPUT_FILE="${INPUT_FILE:-/path/to/your/data/livecodebench_v6.jsonl}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/path/to/your/output/leap/qwen/code}"
BS="${BS:-1}"

RANGE_ARGS=(--start-idx "${START_IDX}")
if [[ -n "${END_IDX}" ]]; then
    RANGE_ARGS+=(--end-idx "${END_IDX}")
fi

OUTPUT_DIR="${OUTPUT_BASE_DIR}/leap_qwen4b_code_N${N}"
echo "Running LeaP 4B Code: N=${N}, output=${OUTPUT_DIR}"
python "${LEAP_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --input "${INPUT_FILE}" \
    --output "${OUTPUT_DIR}" \
    --max_turns "${MAX_TURNS}" \
    --peer_top_k "${PEER_TOP_K}" \
    --router "${ROUTER}" \
    --temperature 0.6 \
    --top_p 0.95 \
    --min_p 0.05 \
    --max_tokens "${MAX_TOKENS}" \
    --summarize_max_tokens 256 \
    --n "${N}" \
    --num_gpus 8 \
    --gpu_memory_utilization 0.95 \
    --tensor_parallel_size 8 \
    --batch_size "${BS}" \
    "${RANGE_ARGS[@]}"
