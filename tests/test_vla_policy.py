import pytest
import torch
from unittest.mock import MagicMock, patch
from torch import nn

from vla_gemma4.model.vla_policy import VLAPolicy
from vla_gemma4.model.action_heads.mlp_head import MLPHead
from conftest import make_mock_gemma, make_mock_processor


class TestVLAPolicy:
    @patch("vla_gemma4.model.vla_policy.AutoModelForImageTextToText")
    @patch("vla_gemma4.model.vla_policy.AutoProcessor")
    def test_init(self, mock_proc_cls, mock_model_cls, base_config):
        mock_model_cls.from_pretrained.return_value = make_mock_gemma()
        mock_proc_cls.from_pretrained.return_value = make_mock_processor()

        policy = VLAPolicy(base_config)
        assert policy.action_head is not None
        assert policy.proprio_encoder is not None

    @patch("vla_gemma4.model.vla_policy.AutoModelForImageTextToText")
    @patch("vla_gemma4.model.vla_policy.AutoProcessor")
    def test_encode_output_shape(self, mock_proc_cls, mock_model_cls, base_config):
        mock_model_cls.from_pretrained.return_value = make_mock_gemma()
        mock_proc_cls.from_pretrained.return_value = make_mock_processor()

        policy = VLAPolicy(base_config)

        batch = {
            "images": [torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224)],
            "instruction": ["pick up block", "move cup"],
            "proprio": torch.randn(2, 7),
            "actions": torch.randn(2, 1, 7),
        }
        features = policy.encode(batch)
        num_act = base_config["num_action_tokens"]
        assert features.shape == (2, num_act, 1536)

    @patch("vla_gemma4.model.vla_policy.AutoModelForImageTextToText")
    @patch("vla_gemma4.model.vla_policy.AutoProcessor")
    def test_compute_loss(self, mock_proc_cls, mock_model_cls, base_config):
        mock_model_cls.from_pretrained.return_value = make_mock_gemma()
        mock_proc_cls.from_pretrained.return_value = make_mock_processor()

        policy = VLAPolicy(base_config)

        batch = {
            "images": [torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224)],
            "instruction": ["pick up block", "move cup"],
            "proprio": torch.randn(2, 7),
            "actions": torch.randn(2, 1, 7),
        }
        loss_dict = policy.compute_loss(batch)
        assert "loss" in loss_dict
        assert loss_dict["loss"].requires_grad

    @patch("vla_gemma4.model.vla_policy.AutoModelForImageTextToText")
    @patch("vla_gemma4.model.vla_policy.AutoProcessor")
    def test_predict(self, mock_proc_cls, mock_model_cls, base_config):
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

    @patch("vla_gemma4.model.vla_policy.AutoModelForImageTextToText")
    @patch("vla_gemma4.model.vla_policy.AutoProcessor")
    def test_action_head_swappable(self, mock_proc_cls, mock_model_cls, base_config):
        mock_model_cls.from_pretrained.return_value = make_mock_gemma()
        mock_proc_cls.from_pretrained.return_value = make_mock_processor()

        policy = VLAPolicy(base_config)
        original_head = policy.action_head

        new_head = MLPHead(input_dim=1536, action_dim=7, chunk_size=4, hidden_dims=[256])
        policy.action_head = new_head

        assert policy.action_head is not original_head
        assert policy.action_head.chunk_size == 4
