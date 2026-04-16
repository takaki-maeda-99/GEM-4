import pytest
import torch
from vla_gemma4.model.action_heads.base import ActionHead
from vla_gemma4.model.action_heads.act_head import ACTHead


@pytest.fixture
def act_head():
    return ACTHead(
        input_dim=1536,
        action_dim=7,
        chunk_size=20,
        d_model=256,
        nhead=4,
        num_layers=2,
    )


class TestACTHead:
    def test_is_action_head(self, act_head):
        assert isinstance(act_head, ActionHead)

    def test_compute_loss_output(self, act_head):
        features = torch.randn(2, 1, 1536)
        actions = torch.randn(2, 20, 7)
        loss_dict = act_head.compute_loss(features, actions)
        assert "loss" in loss_dict
        assert "l1_loss" in loss_dict
        assert loss_dict["loss"].requires_grad
        assert loss_dict["loss"].item() >= 0

    def test_predict_shape_with_ensemble(self, act_head):
        features = torch.randn(1, 1, 1536)
        act_head.reset_ensemble()
        pred = act_head.predict(features)
        assert pred.shape == (1, 1, 7)

    def test_predict_multiple_steps(self, act_head):
        act_head.reset_ensemble()
        for _ in range(5):
            pred = act_head.predict(torch.randn(1, 1, 1536))
            assert pred.shape == (1, 1, 7)

    def test_reset_ensemble(self, act_head):
        act_head.reset_ensemble()
        act_head.predict(torch.randn(1, 1, 1536))
        act_head.reset_ensemble()
        assert len(act_head._ensemble_buffer) == 0

    def test_parameters_registered(self, act_head):
        params = list(act_head.parameters())
        assert len(params) > 0

    def test_chunk_size_1(self):
        head = ACTHead(input_dim=1536, action_dim=7, chunk_size=1)
        features = torch.randn(2, 1, 1536)
        actions = torch.randn(2, 1, 7)
        loss_dict = head.compute_loss(features, actions)
        assert loss_dict["loss"].item() >= 0
