#!/bin/bash
# Mode A (Quality) pretrain launch — 2-way DDP on GPUs 0,1 (A100 80GB ×2)
# Config: config/pretrain_taco_quality.yaml
#
# GPU allocation (2026-04-22): 0,1 は Mode A (heavy, 80GB)、6,7 は Mode B。
# GPU 2,3 は別用途で使用中、4,5 は予備。
#
# Usage (from repo root):
#   ./scripts/gemma4/launch_pretrain_quality.sh [extra args passed to finetune_gemma4.py]

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4
# 2026-04-23 troubleshooting #012: Mode A (LoRA+GC) で step time が長時間 run で
# 突然 degrade する問題 (2.2→5.7s at step ~2000) の fix。default allocator の
# internal fragmentation が累積するのを expandable_segments で回避。
# Reserved memory も 37.6→30.7GB (-18%) に縮小する副効果あり。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv-gemma4/bin/torchrun \
    --standalone \
    --nproc_per_node=2 \
    VLA-Adapter/vla-scripts/finetune_gemma4.py \
    --config_path config/pretrain_taco_quality.yaml \
    "$@"
