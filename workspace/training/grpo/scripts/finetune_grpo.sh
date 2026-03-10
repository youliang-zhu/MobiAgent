#!/bin/bash
# MobiMind Decider GRPO training entrypoint
# Prereq: Grounder service is running on GPU1 @ localhost:8001

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS
export PYTHONWARNINGS="ignore::UserWarning"

# Grounder config
export GROUNDER_URL="${GROUNDER_URL:-http://localhost:8001/v1/chat/completions}"
export GROUNDER_TIMEOUT="${GROUNDER_TIMEOUT:-30}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Paths
BASE_MODEL="${BASE_MODEL:-/scratch/youliang/models/decider_lora_2_merged}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/workspace/data/training_data/grpo_data/mobimind_decider_grpo_train.json}"
EVAL_PATH="${EVAL_PATH:-$REPO_ROOT/workspace/data/training_data/grpo_data/mobimind_decider_grpo_val.json}"
RUN_NAME="${RUN_NAME:-decider_grpo_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/workspace/training/grpo/output}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
TB_DIR="${TB_DIR:-$OUTPUT_DIR/tensorboard}"
MAX_STEPS="${MAX_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-50}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-$NUM_GENERATIONS}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-0}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$NUM_GENERATIONS}"

if (( GENERATION_BATCH_SIZE % NUM_GENERATIONS != 0 )); then
  echo "Error: GENERATION_BATCH_SIZE ($GENERATION_BATCH_SIZE) must be divisible by NUM_GENERATIONS ($NUM_GENERATIONS)."
  exit 1
fi

if (( PER_DEVICE_EVAL_BATCH_SIZE % NUM_GENERATIONS != 0 )); then
  echo "Error: PER_DEVICE_EVAL_BATCH_SIZE ($PER_DEVICE_EVAL_BATCH_SIZE) must be divisible by NUM_GENERATIONS ($NUM_GENERATIONS)."
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$TB_DIR"
TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
echo "Training log: $TRAIN_LOG"
echo "Base model : $BASE_MODEL"
echo "Train data : $DATA_PATH"
echo "Eval data  : $EVAL_PATH"
echo "Grounder   : $GROUNDER_URL"
echo "Run name   : $RUN_NAME"
echo "Output dir : $OUTPUT_DIR"
echo "TB dir     : $TB_DIR"
echo "Max steps  : $MAX_STEPS"
echo "Save steps : $SAVE_STEPS"
echo "Eval strat : $EVAL_STRATEGY"
echo "Eval steps : $EVAL_STEPS"
echo "Train bs   : $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Eval bs    : $PER_DEVICE_EVAL_BATCH_SIZE"
echo "Rollouts   : $NUM_GENERATIONS"
echo "Grad acc   : $GRADIENT_ACCUMULATION_STEPS"
echo "Temp/top_p : $TEMPERATURE / $TOP_P"
echo "Top_k      : $TOP_K"
echo "Gen batch  : $GENERATION_BATCH_SIZE"

# Move to workspace/training/grpo
cd "$GRPO_DIR"
export PYTHONPATH="src:../sft/src:${PYTHONPATH:-}"

conda run --no-capture-output -n grpo python -m src.train.train_grpo \
    --model_id "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --eval_path "$EVAL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --freeze_vision_tower True \
    --freeze_merger False \
    --freeze_llm True \
    --lora_enable True \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['visual']" \
    --image_min_pixels 200704 \
    --image_max_pixels 501760 \
    --grounder_url "$GROUNDER_URL" \
    --grounder_timeout "$GROUNDER_TIMEOUT" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --num_generations "$NUM_GENERATIONS" \
    --generation_batch_size "$GENERATION_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --max_completion_length 512 \
    --max_prompt_length 2048 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --beta 0.01 \
    --bf16 True \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --eval_strategy "$EVAL_STRATEGY" \
    --eval_steps "$EVAL_STEPS" \
    --save_strategy steps \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 5 \
    --max_steps "$MAX_STEPS" \
    --report_to tensorboard \
    --logging_dir "$TB_DIR" \
    --remove_unused_columns False \
    2>&1 | tee "$TRAIN_LOG"

echo "Training complete. Output: $OUTPUT_DIR"
