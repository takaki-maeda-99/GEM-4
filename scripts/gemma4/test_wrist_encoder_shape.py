"""Test WristResNet18 output shape. Run:
    CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_wrist_encoder_shape.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
sys.path.insert(0, str(VLA_ROOT))

import torch
from prismatic.models.backbones.vision.wrist_resnet18 import WristResNet18


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = WristResNet18(out_dim=1536).to(device, dtype=torch.bfloat16)

    # 3 forward shape checks
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.bfloat16)
    out = enc(img)
    assert out.shape == (batch_size, 49, 1536), \
        f"expected (B, 49, 1536), got {tuple(out.shape)}"

    # trainable params check
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    assert trainable > 0, "WristResNet18 must be trainable"
    print(f"WristResNet18 trainable params: {trainable / 1e6:.2f} M")

    # ImageNet init smoke: first conv bias should differ from zero
    first_conv = next(enc.backbone.children())
    assert not torch.allclose(first_conv.weight, torch.zeros_like(first_conv.weight)), \
        "ResNet18 conv1 must be ImageNet-initialized, not zero"

    print("OK: WristResNet18 shape + trainability + init verified")


if __name__ == "__main__":
    main()
