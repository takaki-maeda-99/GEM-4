# VLA-Gemma4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a VLA model using Gemma 4 E2B as backbone, with swappable action heads, for robot manipulation from multi-camera images + language instructions + proprioception.

**Architecture:** Token-integration approach — all inputs (multi-camera images, language, proprioception) are tokenized and fed through the Gemma 4 LLM backbone. A learnable `[ACT]` token's hidden state is extracted and passed to a swappable action head (initially MLP) that predicts EEF delta poses.

**Tech Stack:** PyTorch, HuggingFace Transformers (>=5.5.0), HuggingFace PEFT, HuggingFace Accelerate, LeRobot, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-04-14-vla-gemma4-design.md`

---

## File Structure

```
vla_gemma4/
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── proprio_encoder.py     # ProprioEncoder: 7d proprio → LLM embedding dim
│   ├── vla_policy.py          # VLAPolicy: orchestrates backbone + head
│   └── action_heads/
│       ├── __init__.py
│       ├── base.py             # ActionHead ABC (nn.Module + ABC)
│       └── mlp_head.py         # MLPHead: deterministic regression
├── data/
│   ├── __init__.py
│   ├── normalizer.py           # Normalizer: action mean/std normalization
│   ├── collate.py              # Custom collate_fn for DataLoader
│   ├── transforms.py           # Image data augmentation for training
│   └── dataset.py              # VLADataset: LeRobot wrapper
├── training/
│   ├── __init__.py
│   └── trainer.py              # VLATrainer: training loop
├── scripts/
│   ├── train.py                # CLI entrypoint for training
│   └── eval.py                 # CLI entrypoint for evaluation
└── configs/
    └── base.yaml               # Default configuration

tests/
├── __init__.py
├── test_proprio_encoder.py
├── test_action_head_base.py
├── test_mlp_head.py
├── test_normalizer.py
├── test_dataset.py
├── test_vla_policy.py
├── test_trainer.py
└── conftest.py                 # Shared fixtures (dummy config, tensors)
```

---

## Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `vla_gemma4/__init__.py`
- Create: `vla_gemma4/model/__init__.py`
- Create: `vla_gemma4/model/action_heads/__init__.py`
- Create: `vla_gemma4/data/__init__.py`
- Create: `vla_gemma4/training/__init__.py`
- Create: `vla_gemma4/scripts/` (directory only)
- Create: `vla_gemma4/configs/base.yaml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "vla-gemma4"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.50.0",
    "peft>=0.7.0",
    "accelerate>=0.25.0",
    "lerobot>=0.1.0",
    "pyyaml>=6.0",
    "bitsandbytes>=0.41.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[tool.setuptools.packages.find]
include = ["vla_gemma4*"]
```

- [ ] **Step 2: Create `vla_gemma4/configs/base.yaml`**

```yaml
# Model
model_name: "google/gemma-4-E2B-it"
num_action_tokens: 1
visual_token_budget: 280

# Action space
action_dim: 7
chunk_size: 1

# Cameras
cameras:
  - "observation.images.top"
  - "observation.images.wrist"

# Proprioception
proprio_dim: 7
proprio_key: "observation.state"

# Action head
action_head:
  type: "mlp"
  hidden_dims: [512, 256]

# Training
training:
  strategy: "full"  # "full" or "lora"
  lora:
    r: 16
    alpha: 32
    target_modules: ["q_proj", "v_proj"]
  batch_size: 8
  lr: 1.0e-4
  weight_decay: 0.01
  num_epochs: 100
  warmup_steps: 1000
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  save_every_n_steps: 5000
  eval_every_n_steps: 1000

# Data
data:
  dataset_name: "lerobot/bridge_v2"
  language_instruction_key: "language_instruction"
  default_instruction: "manipulation task"
  normalizer_path: null  # Path to pre-computed normalizer stats JSON

# Inference
inference:
  quantization: null  # null, "8bit", or "4bit"
```

- [ ] **Step 3: Create package `__init__.py` files**

`vla_gemma4/__init__.py`:
```python
```

`vla_gemma4/model/__init__.py`:
```python
```

`vla_gemma4/model/action_heads/__init__.py`:
```python
```

`vla_gemma4/data/__init__.py`:
```python
```

`vla_gemma4/training/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 4: Create `tests/conftest.py` with shared fixtures**

```python
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
        output.last_hidden_state = torch.randn(B, 64, 768)  # SiglipVisionModel output
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
```

- [ ] **Step 5: Create scripts directory placeholder**

```bash
mkdir -p vla_gemma4/scripts
touch vla_gemma4/scripts/__init__.py
```

- [ ] **Step 6: Install package in dev mode and verify**

Run: `pip install -e ".[dev]"`
Expected: Successful installation

- [ ] **Step 7: Run pytest to verify empty test suite works**

Run: `pytest tests/ -v`
Expected: "no tests ran" or "0 items collected", exit code 5 (no tests)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml vla_gemma4/ tests/ 
git commit -m "feat: project scaffolding with config and test fixtures"
```

---

## Task 2: ActionHead ABC

**Files:**
- Create: `vla_gemma4/model/action_heads/base.py`
- Create: `tests/test_action_head_base.py`

- [ ] **Step 1: Write the failing test**

`tests/test_action_head_base.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_action_head_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vla_gemma4.model.action_heads.base'`

- [ ] **Step 3: Write implementation**

`vla_gemma4/model/action_heads/base.py`:
```python
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class ActionHead(nn.Module, ABC):
    """Base class for all action heads.

    All action heads receive features from the LLM backbone and
    produce action predictions. They encapsulate their own loss
    computation so head-specific logic (KL, VQ, denoising) stays
    internal.
    """

    @abstractmethod
    def compute_loss(self, features: Tensor, actions: Tensor, **kwargs) -> dict:
        """Compute training loss.

        Args:
            features: [B, N_act, D] features from backbone [ACT] tokens.
            actions: [B, T, action_dim] ground-truth actions.

        Returns:
            Dict with at least "loss" key (scalar Tensor), plus any metrics.
        """

    @abstractmethod
    def predict(self, features: Tensor, **kwargs) -> Tensor:
        """Predict actions at inference time.

        Args:
            features: [B, N_act, D] features from backbone [ACT] tokens.

        Returns:
            [B, T, action_dim] predicted actions.
        """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_action_head_base.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vla_gemma4/model/action_heads/base.py tests/test_action_head_base.py
git commit -m "feat: add ActionHead abstract base class"
```

---

## Task 3: MLPHead

**Files:**
- Create: `vla_gemma4/model/action_heads/mlp_head.py`
- Create: `tests/test_mlp_head.py`

- [ ] **Step 1: Write the failing test**

`tests/test_mlp_head.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mlp_head.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`vla_gemma4/model/action_heads/mlp_head.py`:
```python
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import ActionHead


class MLPHead(ActionHead):
    """MLP action head with separate MSE (position/rotation) and BCE (gripper) losses."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 7,
        chunk_size: int = 1,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        # dims 0-5: position/rotation (continuous), dim 6: gripper (sigmoid)
        self.pose_dim = action_dim - 1
        self.gripper_dim = 1

        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim

        # Output: chunk_size * (pose_dim + gripper_dim)
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(self, features: Tensor, actions: Tensor, **kwargs) -> dict:
        pred = self._forward(features)  # [B, T, action_dim]
        # Split pose and gripper
        pred_pose = pred[:, :, :self.pose_dim]
        pred_gripper = pred[:, :, self.pose_dim:]
        target_pose = actions[:, :, :self.pose_dim]
        target_gripper = actions[:, :, self.pose_dim:]

        mse_loss = F.mse_loss(pred_pose, target_pose)
        bce_loss = F.binary_cross_entropy_with_logits(pred_gripper, target_gripper)
        loss = mse_loss + bce_loss

        return {"loss": loss, "mse_loss": mse_loss, "bce_loss": bce_loss}

    def predict(self, features: Tensor, **kwargs) -> Tensor:
        pred = self._forward(features)  # [B, T, action_dim]
        # Apply sigmoid to gripper dimension only
        pred_pose = pred[:, :, :self.pose_dim]
        pred_gripper = torch.sigmoid(pred[:, :, self.pose_dim:])
        return torch.cat([pred_pose, pred_gripper], dim=-1)

    def _forward(self, features: Tensor) -> Tensor:
        # features: [B, N_act, D] -> squeeze to [B, D] for single token
        x = features.squeeze(1)  # [B, D]
        x = self.mlp(x)  # [B, chunk_size * action_dim]
        return x.reshape(x.shape[0], self.chunk_size, self.action_dim)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mlp_head.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vla_gemma4/model/action_heads/mlp_head.py tests/test_mlp_head.py
git commit -m "feat: add MLPHead with MSE + BCE loss"
```

---

## Task 4: ProprioEncoder

**Files:**
- Create: `vla_gemma4/model/proprio_encoder.py`
- Create: `tests/test_proprio_encoder.py`

- [ ] **Step 1: Write the failing test**

`tests/test_proprio_encoder.py`:
```python
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
        # Output is a single token embedding: [B, 1, hidden_dim]
        assert out.shape == (4, 1, 1536)

    def test_output_differentiable(self, encoder):
        proprio = torch.randn(4, 7, requires_grad=True)
        out = encoder(proprio)
        out.sum().backward()
        assert proprio.grad is not None

    def test_parameters_registered(self, encoder):
        params = list(encoder.parameters())
        assert len(params) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_proprio_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`vla_gemma4/model/proprio_encoder.py`:
```python
from torch import Tensor, nn


class ProprioEncoder(nn.Module):
    """Encodes proprioception (EEF pose + gripper) into a single token embedding."""

    def __init__(self, proprio_dim: int = 7, hidden_dim: int = 1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, proprio: Tensor) -> Tensor:
        """
        Args:
            proprio: [B, proprio_dim]
        Returns:
            [B, 1, hidden_dim] — single token embedding
        """
        return self.mlp(proprio).unsqueeze(1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_proprio_encoder.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vla_gemma4/model/proprio_encoder.py tests/test_proprio_encoder.py
git commit -m "feat: add ProprioEncoder"
```

---

## Task 5: Normalizer

**Files:**
- Create: `vla_gemma4/data/normalizer.py`
- Create: `tests/test_normalizer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_normalizer.py`:
```python
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
        # Constant column should not cause NaN
        data = torch.zeros(100, 7)
        data[:, 0] = 5.0  # constant
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalizer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`vla_gemma4/data/normalizer.py`:
```python
import json
from pathlib import Path

import torch
from torch import Tensor


class Normalizer:
    """Action normalizer using mean/std statistics."""

    def __init__(self, mean: Tensor, std: Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, data: Tensor) -> "Normalizer":
        """Compute normalization statistics from data.

        Args:
            data: [N, action_dim] tensor of actions.
        """
        mean = data.mean(dim=0)
        std = data.std(dim=0).clamp(min=1e-6)
        return cls(mean, std)

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize actions. Works with [B, action_dim] or [B, T, action_dim]."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: Tensor) -> Tensor:
        """Denormalize actions back to original scale."""
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def save(self, path: str) -> None:
        data = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        data = json.loads(Path(path).read_text())
        return cls(
            mean=torch.tensor(data["mean"]),
            std=torch.tensor(data["std"]),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_normalizer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vla_gemma4/data/normalizer.py tests/test_normalizer.py
git commit -m "feat: add Normalizer for action normalization"
```

---

## Task 6: VLADataset (LeRobot Wrapper)

**Files:**
- Create: `vla_gemma4/data/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

`tests/test_dataset.py`:
```python
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
        """When chunk_size > 1, actions from delta_timestamps are stacked."""
        dataset_config["chunk_size"] = 4
        chunked_sample = {
            "observation.images.top": torch.randn(3, 224, 224),
            "observation.images.wrist": torch.randn(3, 224, 224),
            "observation.state": torch.randn(7),
            "action": torch.randn(4, 7),  # LeRobot returns [T, action_dim] with delta_timestamps
            "language_instruction": "pick up the red block",
        }
        mock_lr = MagicMock()
        mock_lr.__len__ = MagicMock(return_value=100)
        mock_lr.__getitem__ = MagicMock(return_value=chunked_sample)
        mock_lr_cls.return_value = mock_lr

        dataset = VLADataset("lerobot/bridge_v2", **dataset_config)
        sample = dataset[0]
        assert sample["actions"].shape == (4, 7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`vla_gemma4/data/dataset.py`:
```python
import torch
from torch import Tensor
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .normalizer import Normalizer


class VLADataset(Dataset):
    """Wraps a LeRobot dataset into a unified format for VLA training."""

    def __init__(
        self,
        dataset_name: str,
        cameras: list[str],
        proprio_key: str = "observation.state",
        action_key: str = "action",
        language_instruction_key: str = "language_instruction",
        default_instruction: str = "manipulation task",
        chunk_size: int = 1,
        normalizer: Normalizer | None = None,
        **kwargs,
    ):
        self.cameras = cameras
        self.proprio_key = proprio_key
        self.action_key = action_key
        self.language_instruction_key = language_instruction_key
        self.default_instruction = default_instruction
        self.chunk_size = chunk_size
        self.normalizer = normalizer

        # Build delta_timestamps for action chunking
        delta_timestamps = None
        if chunk_size > 1:
            # Request future action steps from LeRobot
            fps = kwargs.pop("fps", 10)
            dt = 1.0 / fps
            delta_timestamps = {
                action_key: [i * dt for i in range(chunk_size)],
            }

        self.lerobot_dataset = LeRobotDataset(
            dataset_name,
            delta_timestamps=delta_timestamps,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.lerobot_dataset[idx]

        images = [sample[cam] for cam in self.cameras]
        proprio = sample[self.proprio_key]
        actions = sample[self.action_key]

        # Ensure actions shape is [T, action_dim]
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        if self.normalizer is not None:
            actions = self.normalizer.normalize(actions)

        instruction = sample.get(
            self.language_instruction_key, self.default_instruction
        )

        return {
            "images": images,
            "instruction": instruction,
            "proprio": proprio,
            "actions": actions,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vla_gemma4/data/dataset.py tests/test_dataset.py
git commit -m "feat: add VLADataset LeRobot wrapper"
```

---

## Task 6b: Collate Function & Transforms

**Files:**
- Create: `vla_gemma4/data/collate.py`
- Create: `vla_gemma4/data/transforms.py`
- Create: `tests/test_collate.py`

The default PyTorch DataLoader collation cannot handle the VLADataset output format (list of tensors for images, strings for instructions). We need a custom collate function.

- [ ] **Step 1: Write the failing test**

`tests/test_collate.py`:
```python
import pytest
import torch
from vla_gemma4.data.collate import vla_collate_fn


class TestCollate:
    def test_collates_batch(self):
        samples = [
            {
                "images": [torch.randn(3, 224, 224), torch.randn(3, 224, 224)],
                "instruction": "pick up block",
                "proprio": torch.randn(7),
                "actions": torch.randn(1, 7),
            },
            {
                "images": [torch.randn(3, 224, 224), torch.randn(3, 224, 224)],
                "instruction": "move cup",
                "proprio": torch.randn(7),
                "actions": torch.randn(1, 7),
            },
        ]
        batch = vla_collate_fn(samples, num_cameras=2)

        # images: list of [B, C, H, W] per camera
        assert len(batch["images"]) == 2
        assert batch["images"][0].shape == (2, 3, 224, 224)

        # instruction: list of str
        assert batch["instruction"] == ["pick up block", "move cup"]

        # proprio: [B, 7]
        assert batch["proprio"].shape == (2, 7)

        # actions: [B, T, 7]
        assert batch["actions"].shape == (2, 1, 7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write collate implementation**

`vla_gemma4/data/collate.py`:
```python
import torch


def vla_collate_fn(samples: list[dict], num_cameras: int) -> dict:
    """Custom collate for VLADataset output.

    Handles: list-of-tensors (images), strings (instruction), tensors (proprio, actions).
    """
    # Stack images per camera: list of [B, C, H, W]
    images = [
        torch.stack([s["images"][cam_idx] for s in samples])
        for cam_idx in range(num_cameras)
    ]

    return {
        "images": images,
        "instruction": [s["instruction"] for s in samples],
        "proprio": torch.stack([s["proprio"] for s in samples]),
        "actions": torch.stack([s["actions"] for s in samples]),
    }
```

- [ ] **Step 4: Write transforms module**

Note: Image preprocessing (resize, normalize) is handled by Gemma 4's `AutoProcessor`. These transforms provide *additional* training-time augmentation. Passed to `VLADataset` via config when needed.

`vla_gemma4/data/transforms.py`:
```python
from torchvision import transforms as T


def get_train_transforms(image_size: tuple[int, int] = (224, 224)):
    """Training-time image augmentation."""
    return T.Compose([
        T.Resize(image_size),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_eval_transforms(image_size: tuple[int, int] = (224, 224)):
    """Evaluation-time image preprocessing (no augmentation)."""
    return T.Compose([
        T.Resize(image_size),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_collate.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add vla_gemma4/data/collate.py vla_gemma4/data/transforms.py tests/test_collate.py
git commit -m "feat: add custom collate function and image transforms"
```

---

## Task 7: VLAPolicy

**Files:**
- Create: `vla_gemma4/model/vla_policy.py`
- Create: `tests/test_vla_policy.py`

This is the core module. It integrates Gemma 4, ProprioEncoder, and ActionHead. Tests use mocks for the heavy Gemma 4 model to keep tests fast and runnable without GPU.

- [ ] **Step 1: Write the failing test**

`tests/test_vla_policy.py`:
```python
import pytest
import torch
from unittest.mock import MagicMock, patch
from torch import nn

from vla_gemma4.model.vla_policy import VLAPolicy
from vla_gemma4.model.action_heads.mlp_head import MLPHead
from conftest import make_mock_gemma


class TestVLAPolicy:
    @patch("vla_gemma4.model.vla_policy.AutoModelForImageTextToText")
    @patch("vla_gemma4.model.vla_policy.AutoProcessor")
    def test_init(self, mock_proc_cls, mock_model_cls, base_config):
        mock_model_cls.from_pretrained.return_value = make_mock_gemma()
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        policy = VLAPolicy(base_config)
        assert policy.action_head is not None
        assert policy.proprio_encoder is not None

    @patch("vla_gemma4.model.vla_policy.AutoModelForImageTextToText")
    @patch("vla_gemma4.model.vla_policy.AutoProcessor")
    def test_encode_output_shape(self, mock_proc_cls, mock_model_cls, base_config):
        mock_model_cls.from_pretrained.return_value = make_mock_gemma()
        mock_proc_cls.from_pretrained.return_value = MagicMock()

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
        mock_proc_cls.from_pretrained.return_value = MagicMock()

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
        mock_proc_cls.from_pretrained.return_value = MagicMock()

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
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        policy = VLAPolicy(base_config)
        original_head = policy.action_head

        new_head = MLPHead(input_dim=1536, action_dim=7, chunk_size=4, hidden_dims=[256])
        policy.action_head = new_head

        assert policy.action_head is not original_head
        assert policy.action_head.chunk_size == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vla_policy.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`vla_gemma4/model/vla_policy.py`:
```python
import torch
from torch import Tensor, nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from .action_heads.base import ActionHead
from .action_heads.mlp_head import MLPHead
from .proprio_encoder import ProprioEncoder


def _build_action_head(config: dict, input_dim: int) -> ActionHead:
    """Factory to build action head from config."""
    head_config = config["action_head"]
    head_type = head_config["type"]

    if head_type == "mlp":
        return MLPHead(
            input_dim=input_dim,
            action_dim=config["action_dim"],
            chunk_size=config["chunk_size"],
            hidden_dims=head_config.get("hidden_dims"),
        )
    else:
        raise ValueError(f"Unknown action head type: {head_type}")


class VLAPolicy(nn.Module):
    """VLA policy that integrates Gemma 4 backbone with swappable action heads."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Load Gemma 4 model and processor
        self.gemma = AutoModelForImageTextToText.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(config["model_name"])

        hidden_dim = self.gemma.config.hidden_size

        # Proprio encoder
        self.proprio_encoder = ProprioEncoder(
            proprio_dim=config["proprio_dim"],
            hidden_dim=hidden_dim,
        )

        # Learnable [ACT] tokens
        num_act = config["num_action_tokens"]
        self.act_tokens = nn.Parameter(torch.randn(1, num_act, hidden_dim) * 0.02)

        # Action head
        self.action_head = _build_action_head(config, input_dim=hidden_dim)

    def encode(self, batch: dict) -> Tensor:
        """Encode observations into features for the action head.

        Returns:
            [B, N_act, D] features from [ACT] token positions.
        """
        images = batch["images"]  # list of [B, C, H, W]
        instructions = batch["instruction"]  # list of str
        proprio = batch["proprio"]  # [B, proprio_dim]

        B = proprio.shape[0]
        device = proprio.device

        # 1. Process images through Gemma 4 vision encoder
        #    Each camera image is processed as a separate image input
        image_embeds_list = []
        for cam_images in images:
            # Get image features via the vision tower (SiglipVisionModel)
            vision_outputs = self.gemma.model.vision_tower(
                pixel_values=cam_images.to(self.gemma.dtype),
            )
            img_features = vision_outputs.last_hidden_state
            # Project to LLM space via embed_vision
            img_embeds = self.gemma.model.embed_vision(inputs_embeds=img_features)
            image_embeds_list.append(img_embeds)

        # 2. Encode language instruction
        text_inputs = self.processor.tokenizer(
            instructions, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        text_embeds = self.gemma.model.embed_tokens(text_inputs.input_ids)
        # Track text attention mask for padding
        text_attention_mask = text_inputs.attention_mask  # [B, text_len]

        # 3. Encode proprioception
        proprio_embeds = self.proprio_encoder(proprio)  # [B, 1, D]

        # 4. Expand [ACT] tokens for batch
        act_embeds = self.act_tokens.expand(B, -1, -1)  # [B, N_act, D]

        # 5. Concatenate all embeddings: [images...] [text] [proprio] [ACT]
        all_embeds = torch.cat(
            image_embeds_list + [text_embeds, proprio_embeds, act_embeds],
            dim=1,
        )

        # 6. Build attention mask (1 for real tokens, 0 for text padding)
        num_image_tokens = sum(e.shape[1] for e in image_embeds_list)
        num_proprio_tokens = 1
        num_act_tokens = self.config["num_action_tokens"]
        # Image, proprio, and ACT tokens are always attended to (ones)
        non_text_mask = torch.ones(
            B, num_image_tokens + num_proprio_tokens + num_act_tokens,
            device=device, dtype=text_attention_mask.dtype,
        )
        # Insert text mask between image tokens and proprio/ACT tokens
        attention_mask = torch.cat(
            [
                non_text_mask[:, :num_image_tokens],
                text_attention_mask,
                non_text_mask[:, num_image_tokens:],
            ],
            dim=1,
        )

        # 7. Forward through LLM backbone
        outputs = self.gemma.model(
            inputs_embeds=all_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        # 8. Extract [ACT] token features from the end
        num_act = self.config["num_action_tokens"]
        features = hidden_states[:, -num_act:, :]  # [B, N_act, D]

        return features

    def compute_loss(self, batch: dict) -> dict:
        features = self.encode(batch)
        return self.action_head.compute_loss(features, batch["actions"])

    def predict(self, batch: dict) -> Tensor:
        with torch.no_grad():
            features = self.encode(batch)
            return self.action_head.predict(features)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vla_policy.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vla_gemma4/model/vla_policy.py tests/test_vla_policy.py
git commit -m "feat: add VLAPolicy integrating Gemma 4 backbone"
```

---

## Task 8: VLATrainer

**Files:**
- Create: `vla_gemma4/training/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Write the failing test**

`tests/test_trainer.py`:
```python
import pytest
import torch
from unittest.mock import MagicMock, patch
from torch import nn

from vla_gemma4.training.trainer import VLATrainer


@pytest.fixture
def simple_policy():
    """A real nn.Module (not a mock) so Accelerator can wrap it."""

    class DummyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def compute_loss(self, batch):
            x = self.linear(torch.randn(2, 10))
            loss = x.sum()
            return {"loss": loss, "mse_loss": loss * 0.8, "bce_loss": loss * 0.2}

    return DummyPolicy()


@pytest.fixture
def trainer_config():
    return {
        "training": {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "num_epochs": 2,
            "warmup_steps": 2,
            "max_grad_norm": 1.0,
            "mixed_precision": "no",
            "save_every_n_steps": 100,
            "eval_every_n_steps": 50,
            "batch_size": 4,
            "strategy": "full",
        },
    }


class TestVLATrainer:
    def test_init(self, simple_policy, trainer_config):
        trainer = VLATrainer(simple_policy, trainer_config)
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_train_step(self, simple_policy, trainer_config):
        trainer = VLATrainer(simple_policy, trainer_config)
        batch = {
            "images": [torch.randn(2, 3, 224, 224)],
            "instruction": ["pick up"],
            "proprio": torch.randn(2, 7),
            "actions": torch.randn(2, 1, 7),
        }
        loss_dict = trainer.train_step(batch)
        assert "loss" in loss_dict
        assert trainer.global_step == 1

    def test_train_loop_runs(self, simple_policy, trainer_config):
        trainer = VLATrainer(simple_policy, trainer_config)
        batch = {
            "images": [torch.randn(2, 3, 224, 224)],
            "instruction": ["pick up"],
            "proprio": torch.randn(2, 7),
            "actions": torch.randn(2, 1, 7),
        }
        mock_dataloader = [batch, batch]
        trainer.train_epoch(mock_dataloader)
        assert trainer.global_step == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`vla_gemma4/training/trainer.py`:
```python
import logging

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate import Accelerator

logger = logging.getLogger(__name__)


class VLATrainer:
    """Training loop for VLAPolicy with Accelerate for multi-GPU and mixed precision."""

    def __init__(self, policy: nn.Module, config: dict, num_training_steps: int = 10000):
        self.config = config
        train_cfg = config["training"]

        # Accelerator handles device placement, mixed precision, and multi-GPU
        self.accelerator = Accelerator(
            mixed_precision=train_cfg.get("mixed_precision", "no"),
        )

        self.optimizer = AdamW(
            policy.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )

        # LR scheduler: linear warmup then cosine decay
        warmup_steps = train_cfg["warmup_steps"]
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(num_training_steps - warmup_steps, 1)
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # Wrap with Accelerator for multi-GPU + AMP
        self.policy, self.optimizer, self.scheduler = self.accelerator.prepare(
            policy, self.optimizer, self.scheduler
        )

        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.num_epochs = train_cfg["num_epochs"]
        self.save_every_n_steps = train_cfg["save_every_n_steps"]
        self.eval_every_n_steps = train_cfg["eval_every_n_steps"]
        self.global_step = 0

    def train_step(self, batch: dict) -> dict:
        self.policy.train()
        self.optimizer.zero_grad()

        with self.accelerator.autocast():
            loss_dict = self.policy.compute_loss(batch)

        self.accelerator.backward(loss_dict["loss"])

        self.accelerator.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        return {k: v.item() if hasattr(v, "item") else v for k, v in loss_dict.items()}

    def train_epoch(self, dataloader) -> list[dict]:
        metrics = []
        for batch in dataloader:
            step_metrics = self.train_step(batch)
            metrics.append(step_metrics)

            if self.global_step % self.save_every_n_steps == 0:
                self._save_checkpoint()

        return metrics

    def train(self, train_dataloader, eval_dataloader=None) -> None:
        # Prepare dataloader with Accelerator
        train_dataloader = self.accelerator.prepare(train_dataloader)

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            metrics = self.train_epoch(train_dataloader)

            avg_loss = sum(m["loss"] for m in metrics) / len(metrics)
            logger.info(f"  avg_loss: {avg_loss:.4f}")

        self._save_checkpoint()

    def _save_checkpoint(self, path: str | None = None) -> None:
        if path is None:
            path = f"checkpoint_step_{self.global_step}.pt"
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrapped = self.accelerator.unwrap_model(self.policy)
            torch.save(
                {
                    "step": self.global_step,
                    "model_state_dict": unwrapped.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                path,
            )
            # If using LoRA, also save adapter in PEFT format for easy loading
            if hasattr(unwrapped, "gemma") and hasattr(unwrapped.gemma, "save_pretrained"):
                adapter_path = path.replace(".pt", "_adapter")
                unwrapped.gemma.save_pretrained(adapter_path)
                logger.info(f"LoRA adapter saved to {adapter_path}")
            logger.info(f"Checkpoint saved to {path}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trainer.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vla_gemma4/training/trainer.py tests/test_trainer.py
git commit -m "feat: add VLATrainer with training loop"
```

---

## Task 9: Training Script

**Files:**
- Create: `vla_gemma4/scripts/train.py`

- [ ] **Step 1: Write the training script**

`vla_gemma4/scripts/train.py`:
```python
"""Training entrypoint for VLA-Gemma4."""

import argparse
import logging
import yaml

import torch
from torch.utils.data import DataLoader

from vla_gemma4.data.collate import vla_collate_fn
from vla_gemma4.data.dataset import VLADataset
from vla_gemma4.data.normalizer import Normalizer
from vla_gemma4.model.vla_policy import VLAPolicy
from vla_gemma4.training.trainer import VLATrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_lora(policy: VLAPolicy, config: dict) -> VLAPolicy:
    """Apply LoRA to the LLM backbone and freeze ViT."""
    from peft import LoraConfig, get_peft_model

    lora_cfg = config["training"]["lora"]

    # Freeze vision tower
    for param in policy.gemma.model.vision_tower.parameters():
        param.requires_grad = False

    # Apply LoRA to LLM
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    policy.gemma = get_peft_model(policy.gemma, lora_config)
    return policy


def main():
    parser = argparse.ArgumentParser(description="Train VLA-Gemma4")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config loaded: {args.config}")

    # Build dataset
    normalizer = None
    if config["data"].get("normalizer_path"):
        normalizer = Normalizer.load(config["data"]["normalizer_path"])

    dataset = VLADataset(
        dataset_name=config["data"]["dataset_name"],
        cameras=config["cameras"],
        proprio_key=config.get("proprio_key", "observation.state"),
        language_instruction_key=config["data"]["language_instruction_key"],
        default_instruction=config["data"]["default_instruction"],
        chunk_size=config["chunk_size"],
        normalizer=normalizer,
    )

    num_cameras = len(config["cameras"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: vla_collate_fn(batch, num_cameras=num_cameras),
    )

    # Build model
    policy = VLAPolicy(config)

    # Apply LoRA if configured
    if config["training"]["strategy"] == "lora":
        policy = setup_lora(policy, config)
        logger.info("LoRA applied to LLM backbone, ViT frozen")

    # Train (Accelerator in VLATrainer handles device placement and multi-GPU)
    num_training_steps = len(dataloader) * config["training"]["num_epochs"]
    trainer = VLATrainer(policy, config, num_training_steps=num_training_steps)
    trainer.train(dataloader)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses without errors**

Run: `python -c "import vla_gemma4.scripts.train"`
Expected: No import errors (actual training requires GPU + model download)

- [ ] **Step 3: Commit**

```bash
git add vla_gemma4/scripts/train.py
git commit -m "feat: add training script with LoRA support"
```

---

## Task 10: Evaluation Script

**Files:**
- Create: `vla_gemma4/scripts/eval.py`

- [ ] **Step 1: Write the evaluation script**

`vla_gemma4/scripts/eval.py`:
```python
"""Evaluation entrypoint for VLA-Gemma4."""

import argparse
import json
import logging

import torch
from torch.utils.data import DataLoader

import yaml

from vla_gemma4.data.collate import vla_collate_fn
from vla_gemma4.data.dataset import VLADataset
from vla_gemma4.data.normalizer import Normalizer
from vla_gemma4.model.vla_policy import VLAPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(policy, dataloader, normalizer=None) -> dict:
    """Run offline evaluation on a dataset.

    Returns:
        Dict with evaluation metrics.
    """
    policy.eval()

    all_mse = []
    all_l1 = []
    gripper_correct = 0
    gripper_total = 0

    with torch.no_grad():
        for batch in dataloader:
            pred = policy.predict(batch)  # [B, T, 7]
            actions = batch["actions"]

            # Denormalize if needed
            if normalizer is not None:
                pred = normalizer.denormalize(pred)
                actions = normalizer.denormalize(actions)

            # Position/rotation metrics (dims 0-5)
            pose_pred = pred[:, :, :6]
            pose_target = actions[:, :, :6]
            all_mse.append(((pose_pred - pose_target) ** 2).mean().item())
            all_l1.append((pose_pred - pose_target).abs().mean().item())

            # Gripper metrics (dim 6)
            gripper_pred = (pred[:, :, 6] > 0.5).float()
            gripper_target = (actions[:, :, 6] > 0.5).float()
            gripper_correct += (gripper_pred == gripper_target).sum().item()
            gripper_total += gripper_target.numel()

    metrics = {
        "mse": sum(all_mse) / len(all_mse),
        "l1": sum(all_l1) / len(all_l1),
        "gripper_accuracy": gripper_correct / gripper_total if gripper_total > 0 else 0.0,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA-Gemma4")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    config = load_config(args.config)

    normalizer = None
    if config["data"].get("normalizer_path"):
        normalizer = Normalizer.load(config["data"]["normalizer_path"])

    dataset = VLADataset(
        dataset_name=config["data"]["dataset_name"],
        cameras=config["cameras"],
        proprio_key=config.get("proprio_key", "observation.state"),
        language_instruction_key=config["data"]["language_instruction_key"],
        default_instruction=config["data"]["default_instruction"],
        chunk_size=config["chunk_size"],
        normalizer=normalizer,
    )

    num_cameras = len(config["cameras"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: vla_collate_fn(batch, num_cameras=num_cameras),
    )

    # Load model (handles both full and LoRA checkpoints)
    policy = VLAPolicy(config)
    if config["training"]["strategy"] == "lora":
        from peft import PeftModel
        policy.gemma = PeftModel.from_pretrained(
            policy.gemma, args.checkpoint
        )
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        policy.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)

    metrics = evaluate(policy, dataloader, normalizer)

    logger.info(f"Evaluation results: {metrics}")
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses without errors**

Run: `python -c "import vla_gemma4.scripts.eval"`
Expected: No import errors

- [ ] **Step 3: Commit**

```bash
git add vla_gemma4/scripts/eval.py
git commit -m "feat: add evaluation script with offline metrics"
```

---

## Task 11: Integration Test

**Files:**
- Create: `tests/test_integration.py`

A lightweight end-to-end test that verifies the full pipeline (dataset → policy → trainer) works together using mocks for the heavy model.

- [ ] **Step 1: Write integration test**

`tests/test_integration.py`:
```python
"""Integration test: full pipeline with mocked Gemma 4 backbone."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from conftest import make_mock_gemma
from vla_gemma4.model.vla_policy import VLAPolicy
from vla_gemma4.training.trainer import VLATrainer


@patch("vla_gemma4.model.vla_policy.AutoModelForImageTextToText")
@patch("vla_gemma4.model.vla_policy.AutoProcessor")
def test_full_training_pipeline(mock_proc_cls, mock_model_cls, base_config):
    """End-to-end: build policy, run training steps, verify loss decreases or is finite."""
    mock_model_cls.from_pretrained.return_value = make_mock_gemma()
    mock_proc_cls.from_pretrained.return_value = MagicMock()

    base_config["training"]["mixed_precision"] = "no"
    policy = VLAPolicy(base_config)

    trainer_config = base_config.copy()
    trainer = VLATrainer(policy, trainer_config)

    batch = {
        "images": [torch.randn(4, 3, 224, 224), torch.randn(4, 3, 224, 224)],
        "instruction": ["pick up block"] * 4,
        "proprio": torch.randn(4, 7),
        "actions": torch.randn(4, 1, 7),
    }

    losses = []
    for _ in range(3):
        step_metrics = trainer.train_step(batch)
        losses.append(step_metrics["loss"])

    # All losses should be finite
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


@patch("vla_gemma4.model.vla_policy.AutoModelForImageTextToText")
@patch("vla_gemma4.model.vla_policy.AutoProcessor")
def test_predict_pipeline(mock_proc_cls, mock_model_cls, base_config):
    """End-to-end: build policy, run inference, verify output shape."""
    mock_model_cls.from_pretrained.return_value = make_mock_gemma()
    mock_proc_cls.from_pretrained.return_value = MagicMock()

    policy = VLAPolicy(base_config)

    batch = {
        "images": [torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224)],
        "instruction": ["pick up block", "move cup"],
        "proprio": torch.randn(2, 7),
    }

    pred = policy.predict(batch)
    assert pred.shape == (2, 1, 7)
    # Gripper should be in [0, 1]
    assert (pred[:, :, 6] >= 0).all()
    assert (pred[:, :, 6] <= 1).all()
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for full pipeline"
```

---

## Task 12: Final Verification & Cleanup

- [ ] **Step 1: Run full test suite with coverage**

Run: `pytest tests/ -v --cov=vla_gemma4 --cov-report=term-missing`
Expected: All tests PASS, reasonable coverage

- [ ] **Step 2: Verify package structure**

Run: `find vla_gemma4 -name "*.py" | sort`
Expected:
```
vla_gemma4/__init__.py
vla_gemma4/data/__init__.py
vla_gemma4/data/dataset.py
vla_gemma4/data/normalizer.py
vla_gemma4/model/__init__.py
vla_gemma4/model/action_heads/__init__.py
vla_gemma4/model/action_heads/base.py
vla_gemma4/model/action_heads/mlp_head.py
vla_gemma4/model/proprio_encoder.py
vla_gemma4/model/vla_policy.py
vla_gemma4/scripts/__init__.py
vla_gemma4/scripts/eval.py
vla_gemma4/scripts/train.py
vla_gemma4/training/__init__.py
vla_gemma4/training/trainer.py
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
```
