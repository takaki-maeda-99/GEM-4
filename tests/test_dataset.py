import pytest
import torch
from unittest.mock import MagicMock, patch
from vla_gemma4.data.dataset import VLADataset


@pytest.fixture
def mock_lerobot_sample():
    """A sample dict mimicking LeRobot dataset output."""
    return {
        "observation.images.top": torch.randn(3, 224, 224),
        "observation.images.wrist": torch.randn(3, 224, 224),
        "observation.state": torch.randn(7),
        "action": torch.randn(7),
        "language_instruction": "pick up the red block",
    }


@pytest.fixture
def dataset_config():
    return {
        "cameras": ["observation.images.top", "observation.images.wrist"],
        "proprio_key": "observation.state",
        "action_key": "action",
        "language_instruction_key": "language_instruction",
        "default_instruction": "manipulation task",
        "chunk_size": 1,
    }


class TestVLADataset:
    @patch("vla_gemma4.data.dataset.LeRobotDataset")
    def test_getitem_returns_expected_keys(
        self, mock_lr_cls, mock_lerobot_sample, dataset_config
    ):
        mock_lr = MagicMock()
        mock_lr.__len__ = MagicMock(return_value=100)
        mock_lr.__getitem__ = MagicMock(return_value=mock_lerobot_sample)
        mock_lr_cls.return_value = mock_lr

        dataset = VLADataset("lerobot/bridge_v2", **dataset_config)
        sample = dataset[0]

        assert "images" in sample
        assert "instruction" in sample
        assert "proprio" in sample
        assert "actions" in sample

    @patch("vla_gemma4.data.dataset.LeRobotDataset")
    def test_images_list_matches_cameras(
        self, mock_lr_cls, mock_lerobot_sample, dataset_config
    ):
        mock_lr = MagicMock()
        mock_lr.__len__ = MagicMock(return_value=100)
        mock_lr.__getitem__ = MagicMock(return_value=mock_lerobot_sample)
        mock_lr_cls.return_value = mock_lr

        dataset = VLADataset("lerobot/bridge_v2", **dataset_config)
        sample = dataset[0]

        assert len(sample["images"]) == 2
        assert sample["images"][0].shape == (3, 224, 224)

    @patch("vla_gemma4.data.dataset.LeRobotDataset")
    def test_action_shape_single_step(
        self, mock_lr_cls, mock_lerobot_sample, dataset_config
    ):
        mock_lr = MagicMock()
        mock_lr.__len__ = MagicMock(return_value=100)
        mock_lr.__getitem__ = MagicMock(return_value=mock_lerobot_sample)
        mock_lr_cls.return_value = mock_lr

        dataset = VLADataset("lerobot/bridge_v2", **dataset_config)
        sample = dataset[0]

        assert sample["actions"].shape == (1, 7)

    @patch("vla_gemma4.data.dataset.LeRobotDataset")
    def test_missing_instruction_uses_default(
        self, mock_lr_cls, dataset_config
    ):
        sample_no_lang = {
            "observation.images.top": torch.randn(3, 224, 224),
            "observation.images.wrist": torch.randn(3, 224, 224),
            "observation.state": torch.randn(7),
            "action": torch.randn(7),
        }
        mock_lr = MagicMock()
        mock_lr.__len__ = MagicMock(return_value=100)
        mock_lr.__getitem__ = MagicMock(return_value=sample_no_lang)
        mock_lr_cls.return_value = mock_lr

        dataset = VLADataset("lerobot/bridge_v2", **dataset_config)
        sample = dataset[0]

        assert sample["instruction"] == "manipulation task"

    @patch("vla_gemma4.data.dataset.LeRobotDataset")
    def test_chunked_action(self, mock_lr_cls, dataset_config):
        dataset_config["chunk_size"] = 4
        chunked_sample = {
            "observation.images.top": torch.randn(3, 224, 224),
            "observation.images.wrist": torch.randn(3, 224, 224),
            "observation.state": torch.randn(7),
            "action": torch.randn(4, 7),
            "language_instruction": "pick up the red block",
        }
        mock_lr = MagicMock()
        mock_lr.__len__ = MagicMock(return_value=100)
        mock_lr.__getitem__ = MagicMock(return_value=chunked_sample)
        mock_lr_cls.return_value = mock_lr

        dataset = VLADataset("lerobot/bridge_v2", **dataset_config)
        sample = dataset[0]
        assert sample["actions"].shape == (4, 7)
