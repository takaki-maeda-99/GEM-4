#!/bin/bash
# Mode A (Quality) pretrain launch — 4-way DDP on GPUs 0-3 (2× 80GB + 2× 40GB)
# Config: config/pretrain_taco_quality.yaml
#
# Usage (from repo root):
#   ./scripts/gemma4/launch_pretrain_quality.sh [extra args passed to finetune_gemma4.py]

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

.venv-gemma4/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    VLA-Adapter/vla-scripts/finetune_gemma4.py \
    --config_path config/pretrain_taco_quality.yaml \
    "$@"
