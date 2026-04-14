import pytest
import torch
from unittest.mock import MagicMock
from torch import nn


def make_mock_gemma(hidden_dim=1536):
    """Create a mock Gemma 4 model with the expected interface.

    Shared across test_vla_policy.py and test_integration.py.
    """
    mock_model = MagicMock()

    # model.model.embed_tokens: maps token IDs to embeddings
    embed_tokens = nn.Embedding(100, hidden_dim)
    mock_model.model.embed_tokens = embed_tokens

    # model.config
    config = MagicMock()
    config.hidden_size = hidden_dim
    mock_model.config = config

    # model.dtype
    mock_model.dtype = torch.float32

    # Vision tower: returns object with last_hidden_state
    def mock_vision_tower(pixel_values=None, **kwargs):
        B = pixel_values.shape[0]
        output = MagicMock()
        output.last_hidden_state = torch.randn(B, 64, 768)
        return output

    mock_model.model.vision_tower.side_effect = mock_vision_tower

    # embed_vision: projects vision features to LLM space
    def mock_embed_vision(inputs_embeds=None, **kwargs):
        B, S, _ = inputs_embeds.shape
        return torch.randn(B, S, hidden_dim)

    mock_model.model.embed_vision.side_effect = mock_embed_vision

    # model() forward returns an object with last_hidden_state
    def mock_forward(**kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        B, S, D = inputs_embeds.shape
        output = MagicMock()
        output.last_hidden_state = torch.randn(B, S, D)
        return output

    mock_model.model.side_effect = mock_forward

    return mock_model


@pytest.fixture
def base_config():
    """Minimal config dict for testing without loading YAML."""
    return {
        "model_name": "google/gemma-4-E2B-it",
        "num_action_tokens": 1,
        "visual_token_budget": 280,
        "action_dim": 7,
        "chunk_size": 1,
        "cameras": ["observation.images.top", "observation.images.wrist"],
        "proprio_dim": 7,
        "proprio_key": "observation.state",
        "action_head": {
            "type": "mlp",
            "hidden_dims": [512, 256],
        },
        "training": {
            "strategy": "full",
            "lora": {"r": 16, "alpha": 32, "target_modules": ["q_proj", "v_proj"]},
            "batch_size": 8,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "num_epochs": 100,
            "warmup_steps": 1000,
            "max_grad_norm": 1.0,
            "mixed_precision": "bf16",
            "save_every_n_steps": 5000,
            "eval_every_n_steps": 1000,
        },
        "data": {
            "dataset_name": "lerobot/bridge_v2",
            "language_instruction_key": "language_instruction",
            "default_instruction": "manipulation task",
            "normalizer_path": None,
        },
        "inference": {
            "quantization": None,
        },
    }


@pytest.fixture
def hidden_dim():
    """Gemma 4 E2B hidden dimension."""
    return 1536


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def device():
    return torch.device("cpu")
