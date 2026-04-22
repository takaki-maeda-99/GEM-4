"""Verify that MLPResNet.forward correctly concats h_w, h_sp and trims.

Output must remain (B, NUM_ACTIONS_CHUNK, output_dim) regardless of h_w/h_sp.
Run:
    CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_action_head_concat.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
sys.path.insert(0, str(VLA_ROOT))

import torch
from prismatic.models.action_heads import MLPResNet
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2
    hidden_dim = 1536
    input_dim = hidden_dim * ACTION_DIM  # follows existing L1RegressionActionHead pattern
    output_dim = ACTION_DIM

    model = MLPResNet(
        num_blocks=24,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_pro_version=True,
    ).to(device, dtype=torch.bfloat16)

    # x: (B, NUM_ACTIONS_CHUNK, input_dim)
    x = torch.zeros(B, NUM_ACTIONS_CHUNK, input_dim, device=device, dtype=torch.bfloat16)
    # h_a: (B, 25, 64, hidden_dim)  -- Bridge adapter hidden
    h_a = torch.randn(B, 25, 64, hidden_dim, device=device, dtype=torch.bfloat16)
    # h_t: (B, 25, 280, hidden_dim) -- Bridge task hidden (assume soft_tokens=280)
    h_t = torch.randn(B, 25, 280, hidden_dim, device=device, dtype=torch.bfloat16)
    # p: (B, 1, hidden_dim) -- proprio
    p = torch.randn(B, 1, hidden_dim, device=device, dtype=torch.bfloat16)
    # h_w: (B, 49, hidden_dim) -- wrist
    h_w = torch.randn(B, 49, hidden_dim, device=device, dtype=torch.bfloat16)
    # h_sp: (B, 32, hidden_dim) -- soft prompt
    h_sp = torch.randn(B, 32, hidden_dim, device=device, dtype=torch.bfloat16)

    # Forward: expect output (B, NUM_ACTIONS_CHUNK, output_dim=7)
    out = model(x, h_a=h_a, h_t=h_t, p=p, h_w=h_w, h_sp=h_sp)
    assert out.shape == (B, NUM_ACTIONS_CHUNK, output_dim), \
        f"expected (B, {NUM_ACTIONS_CHUNK}, {output_dim}), got {tuple(out.shape)}"

    # Forward without h_w, h_sp should still work (graceful default)
    out2 = model(x, h_a=h_a, h_t=h_t, p=p)
    assert out2.shape == (B, NUM_ACTIONS_CHUNK, output_dim), \
        f"graceful default broken: {tuple(out2.shape)}"

    print("OK: MLPResNet concat-and-trim preserves output shape")


if __name__ == "__main__":
    main()
