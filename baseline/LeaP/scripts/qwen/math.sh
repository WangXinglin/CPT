#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEAP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ $# -gt 2 ]; then
    echo "Usage: bash scripts/qwen/math.sh [START_IDX] [END_IDX]"
    exit 1
fi

START_IDX="${1:-0}"
END_IDX="${2:-}"
MAX_TURNS="${MAX_TURNS:-19}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
PEER_TOP_K="${PEER_TOP_K:-4}"
ROUTER="${ROUTER:-dispersed}"
BS="${BS:-1}"
TASKS="HMMT_25"
N=64

MODEL_PATH="${MODEL_PATH:-/path/to/your/qwen-model}"
DATA_DIR="${DATA_DIR:-/path/to/your/data}"
SAVE_DIR="${SAVE_DIR:-/path/to/your/output/leap/qwen/math/${TASKS}_N${N}}"

RANGE_ARGS=(--start-idx "${START_IDX}")
if [[ -n "${END_IDX}" ]]; then
    RANGE_ARGS+=(--end-idx "${END_IDX}")
fi

python "${LEAP_ROOT}/inference/qwen/math/runner.py" \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --tasks "${TASKS}" \
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
    --resume true \
    "${RANGE_ARGS[@]}"
