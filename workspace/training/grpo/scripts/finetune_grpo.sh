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

# Paths
BASE_MODEL="${BASE_MODEL:-/scratch/youliang/models/decider_lora_2_merged}"
DATA_PATH="${DATA_PATH:-workspace/data/training_data/grpo_data/mobimind_decider_grpo_train.json}"
EVAL_PATH="${EVAL_PATH:-workspace/data/training_data/grpo_data/mobimind_decider_grpo_val.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/youliang/models/decider_grpo_1}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/runs}"

mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
echo "Training log: $TRAIN_LOG"
echo "Base model : $BASE_MODEL"
echo "Train data : $DATA_PATH"
echo "Eval data  : $EVAL_PATH"
echo "Grounder   : $GROUNDER_URL"

# Move to workspace/training/grpo
cd "$(dirname "$0")/.."
export PYTHONPATH="src:../sft/src:$PYTHONPATH"

conda run -n grpo python -m src.train.train_grpo \
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
    --per_device_train_batch_size 1 \
    --num_generations 4 \
    --gradient_accumulation_steps 4 \
    --max_new_tokens 512 \
    --max_prompt_length 2048 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --beta 0.01 \
    --bf16 True \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 5 \
    --max_steps 500 \
    --report_to tensorboard \
    --logging_dir "$LOG_DIR" \
    --remove_unused_columns False \
    2>&1 | tee "$TRAIN_LOG"

echo "Training complete. Output: $OUTPUT_DIR"

