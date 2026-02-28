#!/bin/bash
# GRPO Training Script for MobiMind Decider
# Usage: bash scripts/finetune_grpo.sh

set -e

# ===== 配置区域 (请根据实际情况修改) =====

# GPU 配置
export CUDA_VISIBLE_DEVICES=0  # GPU0 用于训练，GPU1 部署 Grounder

# 模型路径
BASE_MODEL="/scratch/youliang/models/decider_lora_2_merged"  # Merged SFT 模型路径
OUTPUT_DIR="/scratch/youliang/models/decider_grpo_1"

# Grounder 服务配置
export GROUNDER_URL="http://localhost:8001/v1/chat/completions"

# 数据路径
DATA_PATH="/home/agent/mobiAgent/MobiAgent/workspace/data/training_data/grpo_data/mobimind_decider_grpo_train.json"
EVAL_PATH="/home/agent/mobiAgent/MobiAgent/workspace/data/training_data/grpo_data/mobimind_decider_grpo_val.json"

# 日志文件路径
LOG_FILE="/scratch/youliang/models/decider_grpo_1/training.log"
echo "Training log will be saved to: $LOG_FILE"

# 设置 Python logging 级别为 INFO（显示调试信息）
export PYTHONUNBUFFERED=1  # 禁用 Python 输出缓冲，实时显示日志

# ===== 训练参数 =====

cd "$(dirname "$0")/.."

python -m src.train.train_grpo \
    --model_id $BASE_MODEL \
    --data_path $DATA_PATH \
    --eval_path $EVAL_PATH \
    --output_dir $OUTPUT_DIR \
    --run_name "decider_grpo" \
    \
    --lora_enable True \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_bias "none" \
    \
    --freeze_vision_tower True \
    --freeze_merger False \
    --freeze_llm True \
    \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --num_generations 4 \
    --gradient_accumulation_steps 4 \
    --max_steps 200 \
    \
    --max_completion_length 512 \
    --max_prompt_length 4096 \
    \
    --learning_rate 5e-6 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    \
    --beta 0.01 \
    --scale_rewards "group" \
    \
    --bf16 True \
    --gradient_checkpointing True \
    \
    --logging_steps 10 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    \
    --report_to "tensorboard" \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    \
    --temperature 0.9 \
    --top_p 1.0 \
    --use_liger_loss True \
    --log_level "info" \
    2>&1 | tee "$LOG_FILE"

echo "GRPO training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "Training log saved to: $LOG_FILE"
