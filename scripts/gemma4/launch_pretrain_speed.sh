#!/bin/bash
# Mode B (Speed) pretrain launch — 4-way DDP on GPUs 4-7 (4× 40GB)
# Config: config/pretrain_taco_speed.yaml
#
# Usage (from repo root):
#   ./scripts/gemma4/launch_pretrain_speed.sh [extra args passed to finetune_gemma4.py]

set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

# Different master port to avoid collision with quality launch
.venv-gemma4/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    --master_port=29501 \
    VLA-Adapter/vla-scripts/finetune_gemma4.py \
    --config_path config/pretrain_taco_speed.yaml \
    "$@"
