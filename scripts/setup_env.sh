#!/bin/bash
# Setup uv environment for VLA-Gemma4
# Usage: bash scripts/setup_env.sh

set -e

echo "=== Setting up VLA-Gemma4 environment ==="

# 1. Sync base dependencies
echo "[1/3] Syncing base dependencies..."
uv sync

# 2. Install PyTorch-compatible torchvision
echo "[2/3] Installing torchvision (cu128)..."
uv pip install "torchvision==0.22.1+cu128" --index-strategy unsafe-best-match

# 3. Install lerobot (--no-deps due to transformers version conflict)
echo "[3/3] Installing lerobot and runtime deps..."
uv pip install lerobot --no-deps
uv pip install datasets huggingface_hub jsonlines draccus --index-strategy unsafe-best-match

echo ""
echo "=== Setup complete ==="
echo "Verify: uv run python3 -c \"import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')\""
echo "Quick test: uv run python3 scripts/quick_test.py"
