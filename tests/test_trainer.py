import pytest
import torch
from unittest.mock import MagicMock, patch
from torch import nn

from vla_gemma4.training.trainer import VLATrainer


@pytest.fixture
def simple_policy():
    """A real nn.Module (not a mock) so Accelerator can wrap it."""

    class DummyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def compute_loss(self, batch):
            device = self.linear.weight.device
            x = self.linear(torch.randn(2, 10, device=device))
            loss = x.sum()
            return {"loss": loss, "mse_loss": loss * 0.8, "bce_loss": loss * 0.2}

    return DummyPolicy()


@pytest.fixture
def trainer_config():
    return {
        "training": {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "num_epochs": 2,
            "warmup_steps": 2,
            "max_grad_norm": 1.0,
            "mixed_precision": "no",
            "save_every_n_steps": 100,
            "eval_every_n_steps": 50,
            "batch_size": 4,
            "strategy": "full",
        },
    }


class TestVLATrainer:
    def test_init(self, simple_policy, trainer_config):
        trainer = VLATrainer(simple_policy, trainer_config)
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_train_step(self, simple_policy, trainer_config):
        trainer = VLATrainer(simple_policy, trainer_config)
        batch = {
            "images": [torch.randn(2, 3, 224, 224)],
            "instruction": ["pick up"],
            "proprio": torch.randn(2, 7),
            "actions": torch.randn(2, 1, 7),
        }
        loss_dict = trainer.train_step(batch)
        assert "loss" in loss_dict
        assert trainer.global_step == 1

    def test_train_loop_runs(self, simple_policy, trainer_config):
        trainer = VLATrainer(simple_policy, trainer_config)
        batch = {
            "images": [torch.randn(2, 3, 224, 224)],
            "instruction": ["pick up"],
            "proprio": torch.randn(2, 7),
            "actions": torch.randn(2, 1, 7),
        }
        mock_dataloader = [batch, batch]
        trainer.train_epoch(mock_dataloader)
        assert trainer.global_step == 2
