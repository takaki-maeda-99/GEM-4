import pytest
import torch
from unittest.mock import MagicMock
from torch import nn


def make_mock_gemma(hidden_dim=1536):
    """Create a mock Gemma 4 model with the expected interface.

    Mocks the PLE-aware encode() flow:
    - model.model.language_model.embed_tokens (real nn.Embedding)
    - model.model.language_model.hidden_size_per_layer_input = False (skip PLE in tests)
    - model.model.language_model.forward(inputs_embeds=..., per_layer_inputs=..., attention_mask=...)
    - model.model.get_placeholder_mask(input_ids, mm_token_type_ids) → (mask, None, None)
    - model.model.get_image_features(pixel_values, image_position_ids) → BaseModelOutputWithPast
    """
    mock_model = MagicMock()

    # model.config — mock without text_config so VLAPolicy falls back to config.hidden_size
    config = MagicMock()
    config.hidden_size = hidden_dim
    # Make text_config.hidden_size not be an int so the isinstance check fails
    config.text_config.hidden_size = MagicMock()  # Not an int → falls back
    mock_model.config = config

    # model.dtype
    mock_model.dtype = torch.float32

    # --- Language model sub-module ---
    embed_tokens = nn.Embedding(100, hidden_dim)
    mock_model.model.language_model.embed_tokens = embed_tokens

    # Disable PLE in tests (simplifies mocking significantly)
    mock_model.model.language_model.hidden_size_per_layer_input = False

    # language_model.forward: takes inputs_embeds, returns object with last_hidden_state
    def mock_lang_forward(inputs_embeds=None, per_layer_inputs=None, attention_mask=None, **kwargs):
        B, S, D = inputs_embeds.shape
        output = MagicMock()
        output.last_hidden_state = torch.randn(B, S, D)
        return output

    mock_model.model.language_model.side_effect = mock_lang_forward
    mock_model.model.language_model.parameters = lambda: iter([embed_tokens.weight])

    # --- get_placeholder_mask: returns all-False mask (no image tokens in mock) ---
    def mock_get_placeholder_mask(input_ids, mm_token_type_ids=None):
        mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        return mask, None, None

    mock_model.model.get_placeholder_mask.side_effect = mock_get_placeholder_mask

    # --- get_image_features: not called when mask is all-False ---
    # But define it just in case
    def mock_get_image_features(pixel_values, image_position_ids=None, **kwargs):
        output = MagicMock()
        output.last_hidden_state = torch.randn(1, 64, 768)
        return output

    mock_model.model.get_image_features.side_effect = mock_get_image_features

    return mock_model


def make_mock_processor(batch_size=2, seq_len=10):
    """Create a mock processor that returns proper tensors."""
    mock_processor = MagicMock()

    def mock_call(text=None, images=None, return_tensors=None, padding=None, **kwargs):
        B = len(text) if isinstance(text, list) else 1
        result = MagicMock()
        result.__getitem__ = lambda self, key: getattr(self, key)
        result.get = lambda key, default=None: getattr(result, key, default)
        result.input_ids = torch.randint(0, 100, (B, seq_len))
        result.attention_mask = torch.ones(B, seq_len, dtype=torch.long)
        result.mm_token_type_ids = torch.zeros(B, seq_len, dtype=torch.long)
        result.pixel_values = None  # No actual images in mock
        result.image_position_ids = None
        return result

    mock_processor.side_effect = mock_call
    mock_processor.tokenizer.pad_token_id = 0

    return mock_processor


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
