import pytest
import torch
from vla_gemma4.model.action_heads.base import ActionHead
from vla_gemma4.model.action_heads.mlp_head import MLPHead


@pytest.fixture
def mlp_head():
    return MLPHead(
        input_dim=1536,
        action_dim=7,
        chunk_size=1,
        hidden_dims=[512, 256],
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

    def test_compute_loss_returns_required_keys(self, mlp_head):
        features = torch.randn(2, 1, 1536)
        actions = torch.randn(2, 1, 7)
        loss_dict = mlp_head.compute_loss(features, actions)
        assert "loss" in loss_dict
        assert "mse_loss" in loss_dict
        assert "bce_loss" in loss_dict
        assert loss_dict["loss"].requires_grad

    def test_compute_loss_chunked(self, chunked_mlp_head):
        features = torch.randn(2, 1, 1536)
        actions = torch.randn(2, 4, 7)
        loss_dict = chunked_mlp_head.compute_loss(features, actions)
        assert "loss" in loss_dict

    def test_gripper_sigmoid_bounded(self, mlp_head):
        """Gripper output (dim 6) should be in [0, 1] after sigmoid."""
        features = torch.randn(2, 1, 1536)
        pred = mlp_head.predict(features)
        gripper = pred[:, :, 6]
        assert (gripper >= 0).all() and (gripper <= 1).all()

    def test_parameters_registered(self, mlp_head):
        params = list(mlp_head.parameters())
        assert len(params) > 0
