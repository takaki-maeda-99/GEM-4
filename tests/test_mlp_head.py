import pytest
import torch
from vla_gemma4.model.action_heads.base import ActionHead
from vla_gemma4.model.action_heads.mlp_head import MLPHead


@pytest.fixture
def mlp_head():
    """Default: all MSE (no binary gripper)."""
    return MLPHead(
        input_dim=1536,
        action_dim=7,
        chunk_size=1,
        hidden_dims=[512, 256],
    )


@pytest.fixture
def mlp_head_binary_gripper():
    """With binary gripper (last dim uses BCE)."""
    return MLPHead(
        input_dim=1536,
        action_dim=7,
        chunk_size=1,
        hidden_dims=[512, 256],
        gripper_as_binary=True,
    )


@pytest.fixture
def chunked_mlp_head():
    return MLPHead(
        input_dim=1536,
        action_dim=7,
        chunk_size=4,
        hidden_dims=[512, 256],
    )


class TestMLPHead:
    def test_is_action_head(self, mlp_head):
        assert isinstance(mlp_head, ActionHead)

    def test_predict_single_step(self, mlp_head):
        features = torch.randn(2, 1, 1536)
        pred = mlp_head.predict(features)
        assert pred.shape == (2, 1, 7)

    def test_predict_chunked(self, chunked_mlp_head):
        features = torch.randn(2, 1, 1536)
        pred = chunked_mlp_head.predict(features)
        assert pred.shape == (2, 4, 7)

    def test_compute_loss_mse_only(self, mlp_head):
        """Default mode: all MSE, no BCE."""
        features = torch.randn(2, 1, 1536)
        actions = torch.randn(2, 1, 7)
        loss_dict = mlp_head.compute_loss(features, actions)
        assert "loss" in loss_dict
        assert "mse_loss" in loss_dict
        assert "bce_loss" not in loss_dict
        assert loss_dict["loss"].requires_grad
        assert loss_dict["loss"].item() >= 0  # MSE is always non-negative

    def test_compute_loss_binary_gripper(self, mlp_head_binary_gripper):
        """Binary gripper mode: MSE + BCE."""
        features = torch.randn(2, 1, 1536)
        actions = torch.randn(2, 1, 7).abs().clamp(0, 1)  # BCE needs [0,1] targets
        loss_dict = mlp_head_binary_gripper.compute_loss(features, actions)
        assert "loss" in loss_dict
        assert "mse_loss" in loss_dict
        assert "bce_loss" in loss_dict
        assert loss_dict["loss"].requires_grad

    def test_compute_loss_chunked(self, chunked_mlp_head):
        features = torch.randn(2, 1, 1536)
        actions = torch.randn(2, 4, 7)
        loss_dict = chunked_mlp_head.compute_loss(features, actions)
        assert "loss" in loss_dict
        assert loss_dict["loss"].item() >= 0

    def test_gripper_sigmoid_bounded(self, mlp_head_binary_gripper):
        """Binary gripper: last dim should be in [0, 1] after sigmoid."""
        features = torch.randn(2, 1, 1536)
        pred = mlp_head_binary_gripper.predict(features)
        gripper = pred[:, :, 6]
        assert (gripper >= 0).all() and (gripper <= 1).all()

    def test_parameters_registered(self, mlp_head):
        params = list(mlp_head.parameters())
        assert len(params) > 0
