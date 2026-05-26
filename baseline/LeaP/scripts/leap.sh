#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEAP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MAX_TURNS=${1:-19}
MAX_TOKENS=${2:-2048}
N=${3:-64}
PEER_TOP_K=${4:-4}
ROUTER=${5:-"dispersed"}
TASKS=${6:-"aime"}
BS=${7:-1}
START_IDX=${8:-0}
END_IDX=${9:-}
RUN=${10:-1}

MODEL_PATH="${MODEL_PATH:-/path/to/model}"
DATA_DIR="${DATA_DIR:-/path/to/leap/data}"
SAVE_DIR="${SAVE_DIR:-/path/to/output/leap/${TASKS}_N${N}/run${RUN}}"

RANGE_ARGS=(--start-idx "${START_IDX}")
if [[ -n "${END_IDX}" ]]; then
    RANGE_ARGS+=(--end-idx "${END_IDX}")
fi


python "${LEAP_DIR}/leap.py" \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --tasks ${TASKS} \
    --max_turns ${MAX_TURNS} \
    --peer_top_k ${PEER_TOP_K} \
    --router ${ROUTER} \
    --temperature 0.6 \
    --top_p 0.95 \
    --min_p 0.05 \
    --max_tokens ${MAX_TOKENS} \
    --summarize_max_tokens 256 \
    --n ${N} \
    --num_gpus 8 \
    --gpu_memory_utilization 0.95 \
    --tensor_parallel_size 8 \
    --batch_size ${BS} \
    --resume true \
    "${RANGE_ARGS[@]}"
