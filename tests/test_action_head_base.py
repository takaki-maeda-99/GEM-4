import pytest
import torch
from torch import nn
from vla_gemma4.model.action_heads.base import ActionHead


def test_action_head_is_abstract():
    """ActionHead cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ActionHead()


def test_action_head_inherits_nn_module():
    """ActionHead must be an nn.Module for parameter management."""
    assert issubclass(ActionHead, nn.Module)


def test_action_head_subclass_must_implement_methods():
    """Subclass that doesn't implement abstract methods raises TypeError."""

    class IncompleteHead(ActionHead):
        pass

    with pytest.raises(TypeError):
        IncompleteHead()


def test_action_head_subclass_works():
    """Subclass that implements all methods can be instantiated."""

    class DummyHead(ActionHead):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 7)

        def compute_loss(self, features, actions, **kwargs):
            pred = self.linear(features.squeeze(1))
            loss = torch.nn.functional.mse_loss(pred, actions.squeeze(1))
            return {"loss": loss}

        def predict(self, features, **kwargs):
            return self.linear(features.squeeze(1)).unsqueeze(1)

    head = DummyHead()
    assert isinstance(head, nn.Module)
    assert isinstance(head, ActionHead)

    features = torch.randn(2, 1, 10)
    actions = torch.randn(2, 1, 7)
    loss_dict = head.compute_loss(features, actions)
    assert "loss" in loss_dict
    assert loss_dict["loss"].requires_grad

    pred = head.predict(features)
    assert pred.shape == (2, 1, 7)
