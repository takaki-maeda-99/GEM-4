"""Integration test: full pipeline with mocked Gemma 4 backbone."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from conftest import make_mock_gemma, make_mock_processor
from vla_gemma4.model.vla_policy import VLAPolicy
from vla_gemma4.training.trainer import VLATrainer


@patch("vla_gemma4.model.vla_policy.AutoModelForImageTextToText")
@patch("vla_gemma4.model.vla_policy.AutoProcessor")
def test_full_training_pipeline(mock_proc_cls, mock_model_cls, base_config):
    """End-to-end: build policy, run training steps, verify loss is finite."""
    mock_model_cls.from_pretrained.return_value = make_mock_gemma()
    mock_proc_cls.from_pretrained.return_value = make_mock_processor(batch_size=4)

    base_config["training"]["mixed_precision"] = "no"
    policy = VLAPolicy(base_config)

    trainer = VLATrainer(policy, base_config)

    batch = {
        "images": [torch.randn(4, 3, 224, 224), torch.randn(4, 3, 224, 224)],
        "instruction": ["pick up block"] * 4,
        "proprio": torch.randn(4, 7),
        "actions": torch.randn(4, 1, 7),
    }

    losses = []
    for _ in range(3):
        step_metrics = trainer.train_step(batch)
        losses.append(step_metrics["loss"])

    # All losses should be finite
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


@patch("vla_gemma4.model.vla_policy.AutoModelForImageTextToText")
@patch("vla_gemma4.model.vla_policy.AutoProcessor")
def test_predict_pipeline(mock_proc_cls, mock_model_cls, base_config):
    """End-to-end: build policy, run inference, verify output shape."""
    mock_model_cls.from_pretrained.return_value = make_mock_gemma()
    mock_proc_cls.from_pretrained.return_value = make_mock_processor()

    policy = VLAPolicy(base_config)

    batch = {
        "images": [torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224)],
        "instruction": ["pick up block", "move cup"],
        "proprio": torch.randn(2, 7),
    }

    pred = policy.predict(batch)
    assert pred.shape == (2, 1, 7)
    # Output should be finite
    assert torch.isfinite(pred).all()
