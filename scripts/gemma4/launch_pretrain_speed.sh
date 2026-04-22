#!/bin/bash
# Mode B (Speed) pretrain launch — 2-way DDP on GPUs 6,7 (A100 40GB ×2)
# Config: config/pretrain_taco_speed.yaml
#
# GPU allocation (2026-04-22): 0,1 は Mode A (heavy, 80GB)、6,7 は Mode B (40GB)。
#
# Usage (from repo root):
#   ./scripts/gemma4/launch_pretrain_speed.sh [extra args passed to finetune_gemma4.py]

set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=6,7
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

# Different master port to avoid collision with quality launch
.venv-gemma4/bin/torchrun \
    --standalone \
    --nproc_per_node=2 \
    --master_port=29501 \
    VLA-Adapter/vla-scripts/finetune_gemma4.py \
    --config_path config/pretrain_taco_speed.yaml \
    "$@"
