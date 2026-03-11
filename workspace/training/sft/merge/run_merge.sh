#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cd workspace/training/sft/merge/
#   bash run_merge.sh sft
#   bash run_merge.sh grpo
#   bash run_merge.sh sft --save-full-precision
#
# 说明:
# - 本脚本统一管理两套合并参数（SFT / GRPO）。
# - 默认输出 bf16；附加 --save-full-precision 会输出 fp32。
# - 合并完成后请自行执行 vllm serve。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/merge_lora.py"

MODE="${1:-}"
EXTRA_ARG="${2:-}"

usage() {
  echo "Usage: bash run_merge.sh <sft|grpo> [--save-full-precision]"
}

if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

SAVE_FULL_PRECISION_FLAG=""
if [[ -n "$EXTRA_ARG" ]]; then
  if [[ "$EXTRA_ARG" == "--save-full-precision" ]]; then
    SAVE_FULL_PRECISION_FLAG="--save-full-precision"
  else
    echo "ERROR: unsupported arg: $EXTRA_ARG"
    usage
    exit 1
  fi
fi

BASE_MODEL_PATH=""
LORA_PATH=""
OUTPUT_PATH=""

case "$MODE" in
  sft)
    BASE_MODEL_PATH="/scratch/youliang/qwen2.5-vl-7b"
    LORA_PATH="/scratch/youliang/models/decider_lora_2"
    OUTPUT_PATH="/scratch/youliang/models/decider_lora_3_merged"
    ;;
  grpo)
    BASE_MODEL_PATH="/scratch/youliang/models/decider_lora_2_merged"
    LORA_PATH="/scratch/youliang/models/decider_grpo_1_500steps"
    OUTPUT_PATH="/scratch/youliang/models/decider_grpo_1_500steps_merged"
    ;;
  *)
    echo "ERROR: invalid mode '$MODE'"
    usage
    exit 1
    ;;
esac

echo "[merge mode] $MODE"
echo "[base model] $BASE_MODEL_PATH"
echo "[lora path ] $LORA_PATH"
echo "[output    ] $OUTPUT_PATH"
echo

python "$PYTHON_SCRIPT" \
  --base-model-path "$BASE_MODEL_PATH" \
  --lora-path "$LORA_PATH" \
  --output-path "$OUTPUT_PATH" \
  ${SAVE_FULL_PRECISION_FLAG}
