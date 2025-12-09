#!/bin/bash

# ============================================
# MobiMind-Decider LoRA Fine-tuning Script
# Model: Qwen2.5-VL-7B-Instruct
# Hardware: 2 x A100 40GB
# ============================================

MODEL_PATH="/scratch/youliang/qwen2.5-vl-7b"

# Disable NCCL debug logs (remove noisy communication traces)
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS

# Suppress repetitive warnings
export PYTHONWARNINGS="ignore::UserWarning"

# Increase NCCL timeout (default 600s -> 7200s)
export NCCL_TIMEOUT=7200

# Change to post_training directory
cd "$(dirname "$0")/.."
export PYTHONPATH=src:$PYTHONPATH

# ============================================
# Batch Configuration
# ============================================
GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=4
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "Training Configuration:"
echo "  Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "  Batch Per Device: $BATCH_PER_DEVICE"
echo "  Num Devices: $NUM_DEVICES"
echo "  Gradient Accumulation Steps: $GRAD_ACCUM_STEPS"

# ============================================
# Data Paths (MODIFY THESE FOR YOUR SETUP)
# ============================================
DATA_PATH="/home/agent/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/sft_data/mobimind_decider_train.json"
EVAL_PATH="/home/agent/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/sft_data/mobimind_decider_val.json"
OUTPUT_DIR="output/mobimind_decider_lora_sft"

# ============================================
# Training
# ============================================
deepspeed src/train/train_sft.py \
    --use_liger_kernel True \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_target_modules "['k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_PATH \
    --data_path $DATA_PATH \
    --eval_path $EVAL_PATH \
    --image_folder "" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --merger_lr 5e-5 \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --per_device_eval_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --eval_strategy "steps" \
    --eval_steps 36 \
    --save_strategy "steps" \
    --save_steps 144 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --dataloader_num_workers 4 \
    --dataloader_drop_last True