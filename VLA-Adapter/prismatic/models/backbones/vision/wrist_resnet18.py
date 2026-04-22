"""WristResNet18: 7x7 feature map extractor + linear projection for VLA wrist camera.

Architecture:
    wrist_img (B, 3, 224, 224)
        -> ResNet18 (ImageNet init, up to layer4)  -> (B, 512, 7, 7)
        -> rearrange                               -> (B, 49, 512)
        -> Linear(512, llm_dim)                    -> (B, 49, llm_dim)

Used in VLAAdapterGemma4 to inject wrist visual signal directly into the action head,
bypassing the frozen LLM (which is pretrained on natural images and cannot adapt to wrist POV).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from einops import rearrange


class WristResNet18(nn.Module):
    """ResNet18 feature extractor for the wrist camera, producing 49 tokens * llm_dim."""

    def __init__(self, out_dim: int = 1536):
        super().__init__()

        # ResNet18 with ImageNet weights; drop avgpool + fc (we want the 7x7 feature map)
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        # backbone output: (B, 512, 7, 7) for 224x224 input
        self.proj = nn.Linear(512, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224), float tensor. dtype inherits from module.
        Returns:
            (B, 49, out_dim)
        """
        feat = self.backbone(x)                       # (B, 512, 7, 7)
        tokens = rearrange(feat, "b c h w -> b (h w) c")  # (B, 49, 512)
        return self.proj(tokens)                      # (B, 49, out_dim)
