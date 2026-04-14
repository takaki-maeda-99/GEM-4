import pytest
import torch
from vla_gemma4.data.collate import vla_collate_fn


class TestCollate:
    def test_collates_batch(self):
        samples = [
            {
                "images": [torch.randn(3, 224, 224), torch.randn(3, 224, 224)],
                "instruction": "pick up block",
                "proprio": torch.randn(7),
                "actions": torch.randn(1, 7),
            },
            {
                "images": [torch.randn(3, 224, 224), torch.randn(3, 224, 224)],
                "instruction": "move cup",
                "proprio": torch.randn(7),
                "actions": torch.randn(1, 7),
            },
        ]
        batch = vla_collate_fn(samples, num_cameras=2)

        # images: list of [B, C, H, W] per camera
        assert len(batch["images"]) == 2
        assert batch["images"][0].shape == (2, 3, 224, 224)

        # instruction: list of str
        assert batch["instruction"] == ["pick up block", "move cup"]

        # proprio: [B, 7]
        assert batch["proprio"].shape == (2, 7)

        # actions: [B, T, 7]
        assert batch["actions"].shape == (2, 1, 7)
