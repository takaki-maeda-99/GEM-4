import json
import pytest
import torch
from vla_gemma4.data.normalizer import Normalizer


class TestNormalizer:
    def test_fit_computes_stats(self):
        data = torch.randn(100, 7)
        normalizer = Normalizer.fit(data)
        assert normalizer.mean.shape == (7,)
        assert normalizer.std.shape == (7,)

    def test_normalize_zero_mean_unit_std(self):
        data = torch.randn(1000, 7) * 3 + 5
        normalizer = Normalizer.fit(data)
        normalized = normalizer.normalize(data)
        assert normalized.mean(dim=0).abs().max() < 0.1
        assert (normalized.std(dim=0) - 1.0).abs().max() < 0.1

    def test_denormalize_recovers_original(self):
        data = torch.randn(50, 7) * 2 + 1
        normalizer = Normalizer.fit(data)
        normalized = normalizer.normalize(data)
        recovered = normalizer.denormalize(normalized)
        assert torch.allclose(data, recovered, atol=1e-5)

    def test_save_and_load(self, tmp_path):
        data = torch.randn(100, 7)
        normalizer = Normalizer.fit(data)
        path = tmp_path / "stats.json"
        normalizer.save(str(path))

        loaded = Normalizer.load(str(path))
        assert torch.allclose(normalizer.mean, loaded.mean)
        assert torch.allclose(normalizer.std, loaded.std)

    def test_std_clamp_prevents_division_by_zero(self):
        data = torch.zeros(100, 7)
        data[:, 0] = 5.0
        normalizer = Normalizer.fit(data)
        normalized = normalizer.normalize(data)
        assert not torch.isnan(normalized).any()

    def test_normalize_chunked_actions(self):
        """Normalizer works with [B, T, action_dim] shaped actions."""
        data = torch.randn(100, 7)
        normalizer = Normalizer.fit(data)
        chunked = torch.randn(8, 4, 7)
        normalized = normalizer.normalize(chunked)
        assert normalized.shape == (8, 4, 7)
        recovered = normalizer.denormalize(normalized)
        assert torch.allclose(chunked, recovered, atol=1e-5)
