import pytest
import torch
from vla_gemma4.model.proprio_encoder import ProprioEncoder


@pytest.fixture
def encoder():
    return ProprioEncoder(proprio_dim=7, hidden_dim=1536)


class TestProprioEncoder:
    def test_output_shape(self, encoder):
        proprio = torch.randn(4, 7)
        out = encoder(proprio)
        assert out.shape == (4, 1, 1536)

    def test_output_differentiable(self, encoder):
        proprio = torch.randn(4, 7, requires_grad=True)
        out = encoder(proprio)
        out.sum().backward()
        assert proprio.grad is not None

    def test_parameters_registered(self, encoder):
        params = list(encoder.parameters())
        assert len(params) > 0
