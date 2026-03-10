#!/usr/bin/env bash
# Deploy Grounder service for GRPO training (foreground).
# Usage:
#   bash workspace/training/grpo/scripts/deploy_grounder.sh

set -euo pipefail

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-MobiMind}"

MODEL_PATH="${MODEL_PATH:-/scratch/youliang/models/grounder}"
PORT="${PORT:-8001}"
GPU_ID="${GPU_ID:-1}"

exec bash -lc "source \"${CONDA_SH}\" && conda activate \"${ENV_NAME}\" && CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve \"${MODEL_PATH}\" --port ${PORT} --dtype float16 --max-model-len 32768 --gpu-memory-utilization 0.63 --enforce-eager"

