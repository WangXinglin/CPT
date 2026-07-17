#!/bin/bash
# ==============================================================================
# DeepConf single-run inference script
#
# Usage:
#   bash scripts/run_single.sh <start_index> <end_index>
#
# The defaults are placeholders and should be replaced with local model, data,
# and output paths before running experiments.
# ==============================================================================

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Error: missing arguments"
    echo "Usage: bash scripts/run_single.sh <start_index> <end_index>"
    echo "Example: bash scripts/run_single.sh 0 5"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPCONF_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

START_IDX="$1"
END_IDX="$2"

# ==================== Config (overridable by env vars) ====================

MODEL="${MODEL:-/path/to/your/model}"
INPUT="${INPUT:-/path/to/your/data/HMMT_25.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/your/output/deepconf/HMMT_25_N64}"
INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-$DEEPCONF_ROOT/inference/qwen/math.py}"
N_COMPLETIONS=64
TP="${TP:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-38912}"
BATCH_SIZE="${BATCH_SIZE:-5}"

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

# Set INSTALL_REQUIREMENTS=1 to install runtime dependencies before running.
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$DEEPCONF_ROOT/requirements-qwen.txt}"

# ==============================================================================

echo "============================================================"
echo "  DeepConf Inference"
echo "============================================================"
echo "  Model:            $MODEL"
echo "  Runner:           $INFERENCE_SCRIPT"
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
echo "============================================================"
echo ""

if [ ! -f "$INFERENCE_SCRIPT" ]; then
    echo "Error: inference script not found: $INFERENCE_SCRIPT"
    exit 1
fi

if [ "$INSTALL_REQUIREMENTS" = "1" ]; then
    echo "===== Install DeepConf runtime requirements ====="
    pip install -r "$REQUIREMENTS_FILE"
    echo ""
fi

echo "===== DeepConf Inference ($DEEPCONF_MODE mode) ====="
echo ""

INFER_CMD=(
    python "$INFERENCE_SCRIPT"
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
