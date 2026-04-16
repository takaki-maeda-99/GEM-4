# ACTHead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ACT (Action Chunking with Transformers) action head that predicts chunk_size future actions via a Transformer decoder with temporal ensemble at inference.

**Architecture:** ACTHead inherits ActionHead ABC, uses `nn.TransformerDecoder` with learnable chunk queries that cross-attend to backbone features. Training uses MSE on action chunks. Inference uses temporal ensemble (exponential decay weighted average of overlapping predictions) to output 1 action per step.

**Tech Stack:** PyTorch (nn.TransformerDecoder), existing ActionHead ABC

**Spec:** `docs/superpowers/specs/2026-04-16-act-head-design.md`

---

## File Structure

```
vla_gemma4/
├── model/
│   ├── action_heads/
│   │   └── act_head.py          # ACTHead implementation (NEW)
│   └── vla_policy.py            # Add "act" to _build_action_head (MODIFY)
├── configs/
│   └── libero_spatial_act.yaml  # ACT config for LIBERO (NEW)

scripts/
└── eval_libero.py               # Add reset_ensemble() call (MODIFY)

tests/
└── test_act_head.py             # ACTHead tests (NEW)
```

---

## Task 1: ACTHead Implementation

**Files:**
- Create: `vla_gemma4/model/action_heads/act_head.py`
- Create: `tests/test_act_head.py`

- [ ] **Step 1: Write the failing test**

`tests/test_act_head.py`:
```python
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
        assert "mse_loss" in loss_dict
        assert loss_dict["loss"].requires_grad
        assert loss_dict["loss"].item() >= 0

    def test_predict_shape_with_ensemble(self, act_head):
        """predict() returns [B, 1, action_dim] after temporal ensemble."""
        features = torch.randn(1, 1, 1536)
        act_head.reset_ensemble()
        pred = act_head.predict(features)
        assert pred.shape == (1, 1, 7)

    def test_predict_multiple_steps(self, act_head):
        """Multiple predict() calls should accumulate ensemble buffer."""
        act_head.reset_ensemble()
        for _ in range(5):
            pred = act_head.predict(torch.randn(1, 1, 1536))
            assert pred.shape == (1, 1, 7)

    def test_reset_ensemble(self, act_head):
        """reset_ensemble() should clear the buffer."""
        act_head.reset_ensemble()
        act_head.predict(torch.randn(1, 1, 1536))
        act_head.reset_ensemble()
        # After reset, internal buffer should be empty
        assert len(act_head._ensemble_buffer) == 0

    def test_parameters_registered(self, act_head):
        params = list(act_head.parameters())
        assert len(params) > 0

    def test_chunk_size_1(self):
        """ACTHead should work with chunk_size=1 (degenerate case)."""
        head = ACTHead(input_dim=1536, action_dim=7, chunk_size=1)
        features = torch.randn(2, 1, 1536)
        actions = torch.randn(2, 1, 7)
        loss_dict = head.compute_loss(features, actions)
        assert loss_dict["loss"].item() >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_act_head.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

`vla_gemma4/model/action_heads/act_head.py`:
```python
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import ActionHead


class ACTHead(ActionHead):
    """ACT (Action Chunking with Transformers) action head.

    Uses a Transformer decoder with learnable chunk queries to predict
    chunk_size future actions. At inference, applies temporal ensemble
    (exponential decay weighted average) over overlapping predictions.

    CVAE is not included in this implementation (future extension).
    Always uses MSE loss on all action dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 7,
        chunk_size: int = 20,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        temporal_ensemble_m: float = 0.01,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.temporal_ensemble_m = temporal_ensemble_m

        # Project backbone features to decoder dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable chunk queries — one per predicted timestep
        self.chunk_queries = nn.Parameter(
            torch.randn(chunk_size, d_model) * 0.02
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, action_dim)

        # Temporal ensemble state (inference only)
        self._ensemble_buffer: list[Tensor] = []

    def _decode(self, features: Tensor) -> Tensor:
        """Run Transformer decoder.

        Args:
            features: [B, N_act, D] from backbone.
        Returns:
            [B, chunk_size, action_dim] predicted action chunk.
        """
        B = features.shape[0]

        # Project features to decoder dim
        memory = self.input_proj(features)  # [B, N_act, d_model]

        # Expand chunk queries for batch
        queries = self.chunk_queries.unsqueeze(0).expand(B, -1, -1)  # [B, chunk_size, d_model]

        # Decode: queries cross-attend to backbone features
        decoded = self.decoder(tgt=queries, memory=memory)  # [B, chunk_size, d_model]

        # Project to action space
        return self.output_proj(decoded)  # [B, chunk_size, action_dim]

    def compute_loss(self, features: Tensor, actions: Tensor, **kwargs) -> dict:
        pred = self._decode(features)  # [B, chunk_size, action_dim]
        mse_loss = F.mse_loss(pred, actions)
        return {"loss": mse_loss, "mse_loss": mse_loss}

    def predict(self, features: Tensor, **kwargs) -> Tensor:
        pred_chunk = self._decode(features)  # [B, chunk_size, action_dim]

        # Add to ensemble buffer
        self._ensemble_buffer.append(pred_chunk.detach())

        # Keep only last chunk_size predictions
        if len(self._ensemble_buffer) > self.chunk_size:
            self._ensemble_buffer = self._ensemble_buffer[-self.chunk_size:]

        # Temporal ensemble: weighted average of overlapping predictions
        m = self.temporal_ensemble_m
        weighted_sum = torch.zeros_like(pred_chunk[:, 0:1, :])  # [B, 1, action_dim]
        weight_sum = 0.0

        for k, past_pred in enumerate(reversed(self._ensemble_buffer)):
            # past_pred[:, k, :] is the prediction for "current timestep"
            # made k steps ago
            if k < past_pred.shape[1]:
                w = math.exp(-m * k)
                weighted_sum += w * past_pred[:, k:k+1, :]
                weight_sum += w

        action = weighted_sum / weight_sum  # [B, 1, action_dim]
        return action

    def reset_ensemble(self):
        """Clear temporal ensemble buffer. Call at episode start."""
        self._ensemble_buffer = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_act_head.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add vla_gemma4/model/action_heads/act_head.py tests/test_act_head.py
git commit -m "feat: add ACTHead with Transformer decoder and temporal ensemble"
```

---

## Task 2: Integration (factory + config + eval)

**Files:**
- Modify: `vla_gemma4/model/vla_policy.py` — add `"act"` to factory
- Create: `vla_gemma4/configs/libero_spatial_act.yaml`
- Modify: `scripts/eval_libero.py` — add `reset_ensemble()` call

- [ ] **Step 1: Add `"act"` to `_build_action_head` factory**

In `vla_gemma4/model/vla_policy.py`, add after the `"mlp"` case in `_build_action_head`:

```python
    elif head_type == "act":
        from .action_heads.act_head import ACTHead
        return ACTHead(
            input_dim=input_dim,
            action_dim=config["action_dim"],
            chunk_size=config["chunk_size"],
            d_model=head_config.get("d_model", 256),
            nhead=head_config.get("nhead", 4),
            num_layers=head_config.get("num_layers", 2),
            dim_feedforward=head_config.get("dim_feedforward", 1024),
            temporal_ensemble_m=head_config.get("temporal_ensemble_m", 0.01),
        )
```

- [ ] **Step 2: Create ACT config**

`vla_gemma4/configs/libero_spatial_act.yaml`: Copy `libero_spatial.yaml` and change:
```yaml
chunk_size: 20  # was 1

action_head:
  type: "act"
  d_model: 256
  nhead: 4
  num_layers: 2
  dim_feedforward: 1024
  temporal_ensemble_m: 0.01
```

- [ ] **Step 3: Add reset_ensemble to eval_libero.py**

In `scripts/eval_libero.py`, after `env.reset()` inside the episode loop, add:

```python
            # Reset temporal ensemble buffer for ACTHead
            if hasattr(policy.action_head, "reset_ensemble"):
                policy.action_head.reset_ensemble()
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS (existing 37 + new 7 = 44)

- [ ] **Step 5: Commit**

```bash
git add vla_gemma4/model/vla_policy.py vla_gemma4/configs/libero_spatial_act.yaml scripts/eval_libero.py
git commit -m "feat: integrate ACTHead into factory, config, and eval"
```

---

## Task 3: End-to-End Smoke Test

Verify the full pipeline works: train a few steps with ACTHead, then run eval_libero.

- [ ] **Step 1: Quick training smoke test**

Run:
```bash
uv run python3 -c "
import torch, yaml
from vla_gemma4.model.action_heads.act_head import ACTHead

head = ACTHead(input_dim=1536, action_dim=7, chunk_size=20)
features = torch.randn(4, 1, 1536)
actions = torch.randn(4, 20, 7)

# Training
loss = head.compute_loss(features, actions)
print(f'Loss: {loss[\"loss\"].item():.4f}')
loss['loss'].backward()
print('Backward OK')

# Inference with ensemble
head.reset_ensemble()
for step in range(5):
    pred = head.predict(torch.randn(1, 1, 1536))
    print(f'Step {step}: pred shape={pred.shape}, values={pred[0,0,:3].tolist()}')
print('Smoke test PASSED')
"
```
Expected: Loss printed, backward OK, 5 predict steps with shape [1, 1, 7].

- [ ] **Step 2: LIBERO eval smoke test (1 task, 1 episode, 10 steps)**

Run:
```bash
MUJOCO_GL=egl uv run python3 scripts/eval_libero.py \
  --config vla_gemma4/configs/libero_spatial_act.yaml \
  --n_episodes 1 \
  --max_steps 10 \
  --output outputs/act_smoke_test.json
```
Expected: Script runs to completion, outputs JSON.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: ACTHead smoke test fixes"
```
