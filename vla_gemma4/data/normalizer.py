import json
from pathlib import Path

import torch
from torch import Tensor


class Normalizer:
    """Action normalizer using mean/std statistics."""

    def __init__(self, mean: Tensor, std: Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, data: Tensor) -> "Normalizer":
        """Compute normalization statistics from data.

        Args:
            data: [N, action_dim] tensor of actions.
        """
        mean = data.mean(dim=0)
        std = data.std(dim=0).clamp(min=1e-6)
        return cls(mean, std)

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize actions. Works with [B, action_dim] or [B, T, action_dim]."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: Tensor) -> Tensor:
        """Denormalize actions back to original scale."""
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def save(self, path: str) -> None:
        data = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        data = json.loads(Path(path).read_text())
        return cls(
            mean=torch.tensor(data["mean"]),
            std=torch.tensor(data["std"]),
        )
