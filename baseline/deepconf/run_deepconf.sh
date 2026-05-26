#!/bin/bash
# ==============================================================================
# DeepConf baseline inference script
#
# Usage:
#   bash run_deepconf.sh <start_index> <end_index>
#
# The defaults are placeholders and should be replaced with local model, data,
# and output paths before running experiments.
# ==============================================================================

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Error: missing arguments"
    echo "Usage: bash run_deepconf.sh <start_index> <end_index>"
    echo "Example: bash run_deepconf.sh 0 5"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_IDX="$1"
END_IDX="$2"

# ==================== Config (overridable by env vars) ====================

MODEL="${MODEL:-/path/to/model}"
INPUT="${INPUT:-/path/to/data/HMMT_24.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output}"
N_COMPLETIONS="${N_COMPLETIONS:-64}"
TP="${TP:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-38912}"
BATCH_SIZE="${BATCH_SIZE:-5}"
VOTING_BUDGETS="${VOTING_BUDGETS:-64}"

GPU_MEM="${GPU_MEM:-0.85}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
WINDOW_SIZE="${WINDOW_SIZE:-2048}"
LOGPROBS_TOPK="${LOGPROBS_TOPK:-20}"
DEEPCONF_MODE="${DEEPCONF_MODE:-online}"
WARMUP_TRACES="${WARMUP_TRACES:-16}"
CONFIDENCE_PERCENTILE="${CONFIDENCE_PERCENTILE:-90}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# Set INSTALL_DEEPCONF=1 to install the local package before running.
INSTALL_DEEPCONF="${INSTALL_DEEPCONF:-0}"

# ==============================================================================

echo "============================================================"
echo "  DeepConf Inference"
echo "============================================================"
echo "  Model:            $MODEL"
echo "  Input:            $INPUT"
echo "  Output:           $OUTPUT_DIR"
echo "  Range:            [$START_IDX, $END_IDX)"
echo "  TP:               $TP"
echo "  N_COMPLETIONS:    $N_COMPLETIONS"
echo "  Mode:             $DEEPCONF_MODE"
echo "  Temperature:      $TEMPERATURE"
echo "  Max tokens:       $MAX_TOKENS"
echo "  Batch size:       $BATCH_SIZE"
echo "  Window size:      $WINDOW_SIZE"
echo "  Voting budgets:   $VOTING_BUDGETS"
echo "============================================================"
echo ""

if [ "$INSTALL_DEEPCONF" = "1" ]; then
    echo "===== Install/Update local deepconf package ====="
    pip install -e "$SCRIPT_DIR"
    echo ""
fi

echo "===== DeepConf Inference ($DEEPCONF_MODE mode) ====="
echo ""

INFER_CMD=(
    python "$SCRIPT_DIR/df_sample.py"
    --model "$MODEL"
    --input "$INPUT"
    --output "$OUTPUT_DIR"
    --n-completions "$N_COMPLETIONS"
    --batch-size "$BATCH_SIZE"
    --tensor-parallel-size "$TP"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --max-tokens "$MAX_TOKENS"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM"
    --window-size "$WINDOW_SIZE"
    --logprobs-topk "$LOGPROBS_TOPK"
    --deepconf-mode "$DEEPCONF_MODE"
    --start-idx "$START_IDX"
    --end-idx "$END_IDX"
)

if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    INFER_CMD+=(--enable-prefix-caching)
else
    INFER_CMD+=(--no-prefix-caching)
fi

if [ "$DEEPCONF_MODE" = "online" ]; then
    INFER_CMD+=(
        --warmup-traces "$WARMUP_TRACES"
        --confidence-percentile "$CONFIDENCE_PERCENTILE"
    )
fi

if [ -n "$SYSTEM_PROMPT" ]; then
    INFER_CMD+=(--system-prompt "$SYSTEM_PROMPT")
fi

echo "Command:"
printf ' %q' "${INFER_CMD[@]}"
echo ""
echo ""

"${INFER_CMD[@]}"

echo ""
echo "===== Inference Completed ====="
echo "Results saved to: $OUTPUT_DIR"
