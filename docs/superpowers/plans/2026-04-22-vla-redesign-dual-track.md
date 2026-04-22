# VLA Redesign Dual-Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement VLA architecture redesign with dual-track (Quality vs Speed) configuration so two pretrain variants can run in parallel on 8× A100 GPUs and be evaluated against each other on LIBERO.

**Architecture:** Scene image goes through Gemma 4 native vision encoder (frozen) + Gemma 4 LM (frozen params) to produce multi-layer Bridge hidden states. Wrist image goes through a trainable ResNet18 directly to the action head. Soft prompts (dataset conditioning) are concatenated into the action latent sequence instead of being prepended to the LLM input. A `training_mode` YAML flag selects **Mode A** (`action_queries` trainable + LLM backward + GC + LoRA r=16) vs **Mode B** (`action_queries` frozen + LLM forward under `torch.no_grad()`).

**Tech Stack:** PyTorch 2.11, Transformers 5.5.4, Flash Attention 2, torchvision ResNet18, peft (LoRA, Mode A only), RLDS / tfds (Taco Play), DDP via `torchrun`, `.venv-gemma4/bin/python`.

**Spec:** [docs/superpowers/specs/2026-04-22-vla-redesign-scene-wrist-split-design.md](../specs/2026-04-22-vla-redesign-scene-wrist-split-design.md)

---

## Context for the Engineer

### Repository layout (where things live)

- `VLA-Adapter/` — forked upstream VLA-Adapter with our Gemma 4 additions under `prismatic/extern/hf/modeling_prismatic_gemma4.py`, `prismatic/models/action_heads.py`, `vla-scripts/finetune_gemma4.py`
- `scripts/gemma4/` — smoke test scripts (numbered `test_NN_*.py`). These are **not pytest**, they are standalone Python scripts run via `.venv-gemma4/bin/python scripts/gemma4/test_NN_*.py`
- `scripts/stage3/` — OXE pretrain data loaders (e.g. `multi_dataset_loader.py`)
- `config/` — **does not exist yet**, create in Task 13
- `runs/gemma4/` — output directory for training runs
- `docs/` — design docs, plans, specs

### Python env

**All commands must use `.venv-gemma4/bin/python`**. Never use system `python` or plain `python` (may default to wrong env).

### Git commit conventions

Follow existing pattern from `git log --oneline`:
```
feat(stage3-3c-0): <description>
fix(stage3-3b-3): <description>
docs(redesign): <description>
```

Every commit must end with:
```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

### Uncommitted state to handle first

`VLA-Adapter/vla-scripts/finetune_gemma4.py` has uncommitted modifications from a prior session. **Before starting, stash or review/commit them**:

```bash
git status
git diff VLA-Adapter/vla-scripts/finetune_gemma4.py
```

Decision point (ask user or examine):
- If the changes are part of this redesign → proceed on top of them
- If they are unrelated → `git stash` them first

### Key facts from the spec

- `action_queries` is a `nn.Embedding(64, 1536)` inside `VLAAdapterGemma4` (not inside `gemma`) — default `requires_grad=True`
- All `gemma.parameters()` are set to `requires_grad=False` (frozen params) but forward/backward still executes normally
- Bridge Attention = extracting `out.hidden_states[0:25]` at scene_vision positions (→ h_t) and action_placeholder positions (→ h_a), shape (B, 25, num_tokens, 1536)
- `llm_dim = 1536` for Gemma 4 E2B (verified via `AutoConfig.from_pretrained('google/gemma-4-E2B').text_config.hidden_size`)
- Gemma 4 E2B **has native vision** (`architectures: ['Gemma4ForConditionalGeneration']`, `vision_config.hidden_size=768, num_hidden_layers=16, patch_size=16`)
- Soft tokens for Gemma 4 vision processor: supported set is `(70, 140, 280, 560, 1120)`, default 280

### 60k checkpoint disposition

The 60k Stage 3c-0 checkpoint is **discarded**. Fresh pretrain from step 0 is planned. Do not attempt to load or migrate old checkpoints.

---

## File Structure

### Files to create

| Path | Responsibility |
|---|---|
| `VLA-Adapter/prismatic/models/backbones/vision/wrist_resnet18.py` | ResNet18 wrist encoder + linear projector (trainable) |
| `scripts/stage3/taco_solo_loader.py` | Taco Play single-dataset loader (2-cam: rgb_static + rgb_gripper, proprio enabled) |
| `config/pretrain_taco_quality.yaml` | Mode A config (action_queries trainable, LoRA, GC) |
| `config/pretrain_taco_speed.yaml` | Mode B config (action_queries frozen, no_grad, no LoRA) |
| `scripts/gemma4/test_13_dual_track_smoke.py` | End-to-end smoke test: both modes, 100 step |
| `scripts/gemma4/launch_pretrain_quality.sh` | DDP 4-way launch for Mode A on GPU 0-3 |
| `scripts/gemma4/launch_pretrain_speed.sh` | DDP 4-way launch for Mode B on GPU 4-7 |

### Files to modify

| Path | Change summary |
|---|---|
| `VLA-Adapter/prismatic/models/action_heads.py` | Remove `film_gen` / `apply_film` dead code; add `h_w`, `h_sp` args to `MLPResNet.forward` with concat-and-trim; update `L1RegressionActionHead.predict_action` signature |
| `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` | Switch from `AutoModelForCausalLM` to `Gemma4ForConditionalGeneration`; remove `DinoSigLIPViTBackbone` + `VisionProjector`; add `WristEncoder`; relocate `SoftPromptLibrary` from LLM input to action head; add `training_mode` flag that wraps LLM forward in `torch.no_grad()` for Mode B; add `action_queries.requires_grad` control |
| `VLA-Adapter/vla-scripts/finetune_gemma4.py` | Read `training_mode` from config; set `action_queries.requires_grad` accordingly; enable GC (Mode A); apply LoRA via `peft` (Mode A); branch optimizer groups; update trainable param assertion |

### Files NOT to modify

- `VLA-Adapter/prismatic/models/backbones/vision/dinosiglip_vit.py` — legacy, leave for other code paths
- `VLA-Adapter/prismatic/extern/hf/modeling_prismatic.py` — upstream base class, not our fork
- `VLA-Adapter/vla-scripts/finetune.py` — upstream legacy, not Gemma 4

---

## Phase 1: Core Architecture (Tasks 1–8)

### Task 1: Create WristResNet18 encoder module

**Files:**
- Create: `VLA-Adapter/prismatic/models/backbones/vision/wrist_resnet18.py`
- Test: `scripts/gemma4/test_wrist_encoder_shape.py`

**What and why:** The new architecture routes the wrist camera through a small trainable CNN (ResNet18) that outputs a 7×7×512 feature map, reshaped to 49 tokens × 512 dim, then projected to `llm_dim=1536`. This is what we concat into the action head. ResNet18 starts from ImageNet weights.

- [ ] **Step 1: Write the failing test (shape verification)**

Create `scripts/gemma4/test_wrist_encoder_shape.py`:

```python
"""Test WristResNet18 output shape. Run:
    CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_wrist_encoder_shape.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
sys.path.insert(0, str(VLA_ROOT))

import torch
from prismatic.models.backbones.vision.wrist_resnet18 import WristResNet18


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = WristResNet18(out_dim=1536).to(device, dtype=torch.bfloat16)

    # 3 forward shape checks
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.bfloat16)
    out = enc(img)
    assert out.shape == (batch_size, 49, 1536), \
        f"expected (B, 49, 1536), got {tuple(out.shape)}"

    # trainable params check
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    assert trainable > 0, "WristResNet18 must be trainable"
    print(f"WristResNet18 trainable params: {trainable / 1e6:.2f} M")

    # ImageNet init smoke: first conv bias should differ from zero
    first_conv = next(enc.backbone.children())
    assert not torch.allclose(first_conv.weight, torch.zeros_like(first_conv.weight)), \
        "ResNet18 conv1 must be ImageNet-initialized, not zero"

    print("OK: WristResNet18 shape + trainability + init verified")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /misc/dl00/takaki/vla-gemma-4
CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_wrist_encoder_shape.py
```

Expected: `ModuleNotFoundError: No module named 'prismatic.models.backbones.vision.wrist_resnet18'`

- [ ] **Step 3: Implement WristResNet18**

Create `VLA-Adapter/prismatic/models/backbones/vision/wrist_resnet18.py`:

```python
"""WristResNet18: 7x7 feature map extractor + linear projection for VLA wrist camera.

Architecture:
    wrist_img (B, 3, 224, 224)
        -> ResNet18 (ImageNet init, up to layer4)  -> (B, 512, 7, 7)
        -> rearrange                               -> (B, 49, 512)
        -> Linear(512, llm_dim)                    -> (B, 49, llm_dim)

Used in VLAAdapterGemma4 to inject wrist visual signal directly into the action head,
bypassing the frozen LLM (which is pretrained on natural images and cannot adapt to wrist POV).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from einops import rearrange


class WristResNet18(nn.Module):
    """ResNet18 feature extractor for the wrist camera, producing 49 tokens * llm_dim."""

    def __init__(self, out_dim: int = 1536):
        super().__init__()

        # ResNet18 with ImageNet weights; drop avgpool + fc (we want the 7x7 feature map)
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        # backbone output: (B, 512, 7, 7) for 224x224 input
        self.proj = nn.Linear(512, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224), float tensor. dtype inherits from module.
        Returns:
            (B, 49, out_dim)
        """
        feat = self.backbone(x)                       # (B, 512, 7, 7)
        tokens = rearrange(feat, "b c h w -> b (h w) c")  # (B, 49, 512)
        return self.proj(tokens)                      # (B, 49, out_dim)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_wrist_encoder_shape.py
```

Expected: `OK: WristResNet18 shape + trainability + init verified`

- [ ] **Step 5: Commit**

```bash
git add VLA-Adapter/prismatic/models/backbones/vision/wrist_resnet18.py \
        scripts/gemma4/test_wrist_encoder_shape.py
git commit -m "$(cat <<'EOF'
feat(redesign): WristResNet18 encoder + shape smoke test

ResNet18 (ImageNet init) -> (B, 512, 7, 7) -> rearrange (49 tokens) -> Linear
(512 -> llm_dim=1536) -> (B, 49, 1536). 空間情報保持のため GAP なし。action head
の concat-to-x に流す前提。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Remove `film_gen` dead code from MLPResNetBlock_Pro

**Files:**
- Modify: `VLA-Adapter/prismatic/models/action_heads.py:287-410`

**What and why:** Spec §5.4. The `film_gen` `nn.Linear(1536, 3072)` and `apply_film` method are defined but fully commented out in the forward pass. They contribute 113M dead params × 24 blocks. Since we are discarding the 60k checkpoint, we can remove them safely (no state_dict compat concern).

- [ ] **Step 1: Verify no external references to `film_gen` / `apply_film`**

```bash
cd /misc/dl00/takaki/vla-gemma-4
grep -rn "film_gen\|apply_film" VLA-Adapter/prismatic/ scripts/ 2>/dev/null
```

Expected: only lines inside `action_heads.py` itself. The `FiLMedPrismaticVisionBackbone` and related files in `film_vit_wrapper.py` are a separate system (vision-side FiLM), not related. Do not touch those.

- [ ] **Step 2: Delete the `film_gen` attribute from `__init__`**

In `VLA-Adapter/prismatic/models/action_heads.py`, locate the `MLPResNetBlock_Pro.__init__` method (around line 290–330). Delete the block starting with the comment `# ---- FiLM ----`:

Before (lines ~324–329):
```python
        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output γ and β
            )
```

After:
```python
        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)
```

- [ ] **Step 3: Delete the `apply_film` method**

In the same file, delete the whole `apply_film` method (lines ~332–334):

```python
    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)
```

- [ ] **Step 4: Delete the commented-out FiLM call in `forward`**

In `MLPResNetBlock_Pro.forward`, around lines ~403–406:

```python
        # # ---- FiLM ----
        # gamma_beta = self.film_gen(p)  # [B, 2C]
        # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        # output = self.apply_film(output, gamma, beta)
```

Delete these 4 commented lines entirely.

- [ ] **Step 5: Update the class docstring to reflect removal**

At the top of `MLPResNetBlock_Pro` (line ~288):

Before:
```python
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""
```

After:
```python
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE."""
```

- [ ] **Step 6: Run a syntax + import smoke check**

```bash
.venv-gemma4/bin/python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'VLA-Adapter')
from prismatic.models.action_heads import MLPResNetBlock_Pro
import torch
b = MLPResNetBlock_Pro(dim=1536)
assert not hasattr(b, 'film_gen'), 'film_gen should be removed'
assert not hasattr(b, 'apply_film'), 'apply_film should be removed'
# param count: block should no longer have film_gen's ~4.72M params
p = sum(x.numel() for x in b.parameters())
print(f'MLPResNetBlock_Pro params: {p/1e6:.3f} M (expected ~23.7M, was ~28.4M with film_gen)')
assert p < 25e6, f'still too many params: {p/1e6:.3f} M; film_gen may not be fully removed'
print('OK: film_gen removed, param count reduced')
"
```

Expected: `OK: film_gen removed, param count reduced`

- [ ] **Step 7: Commit**

```bash
git add VLA-Adapter/prismatic/models/action_heads.py
git commit -m "$(cat <<'EOF'
refactor(redesign): remove dead FiLM code from MLPResNetBlock_Pro

film_gen / apply_film は forward で全コメントアウト済の dead code
(Linear(1536, 3072) × 24 blocks = 113M dead weight)。60k ckpt 破棄
決定で state_dict 互換の縛りがなくなったため safe に削除。
param count smoke で block あたり ~28M → ~24M 削減確認。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add `h_w`, `h_sp` concat to `MLPResNet.forward`

**Files:**
- Modify: `VLA-Adapter/prismatic/models/action_heads.py:84-121`
- Test: `scripts/gemma4/test_action_head_concat.py`

**What and why:** Spec §5.3. The action head should accept wrist features and soft prompt as additional tokens that participate in the 24-block self-attention pool. We concat them into `x` after `fc1` and trim them off before `fc2`.

- [ ] **Step 1: Write the failing test (shape preservation under concat)**

Create `scripts/gemma4/test_action_head_concat.py`:

```python
"""Verify that MLPResNet.forward correctly concats h_w, h_sp and trims.

Output must remain (B, NUM_ACTIONS_CHUNK, output_dim) regardless of h_w/h_sp.
Run:
    CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_action_head_concat.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
sys.path.insert(0, str(VLA_ROOT))

import torch
from prismatic.models.action_heads import MLPResNet
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2
    hidden_dim = 1536
    input_dim = hidden_dim * ACTION_DIM  # follows existing L1RegressionActionHead pattern
    output_dim = ACTION_DIM

    model = MLPResNet(
        num_blocks=24,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_pro_version=True,
    ).to(device, dtype=torch.bfloat16)

    # x: (B, NUM_ACTIONS_CHUNK, input_dim)
    x = torch.zeros(B, NUM_ACTIONS_CHUNK, input_dim, device=device, dtype=torch.bfloat16)
    # h_a: (B, 25, 64, hidden_dim)  -- Bridge adapter hidden
    h_a = torch.randn(B, 25, 64, hidden_dim, device=device, dtype=torch.bfloat16)
    # h_t: (B, 25, 280, hidden_dim) -- Bridge task hidden (assume soft_tokens=280)
    h_t = torch.randn(B, 25, 280, hidden_dim, device=device, dtype=torch.bfloat16)
    # p: (B, 1, hidden_dim) -- proprio
    p = torch.randn(B, 1, hidden_dim, device=device, dtype=torch.bfloat16)
    # h_w: (B, 49, hidden_dim) -- wrist
    h_w = torch.randn(B, 49, hidden_dim, device=device, dtype=torch.bfloat16)
    # h_sp: (B, 32, hidden_dim) -- soft prompt
    h_sp = torch.randn(B, 32, hidden_dim, device=device, dtype=torch.bfloat16)

    # Forward: expect output (B, NUM_ACTIONS_CHUNK, output_dim=7)
    out = model(x, h_a=h_a, h_t=h_t, p=p, h_w=h_w, h_sp=h_sp)
    assert out.shape == (B, NUM_ACTIONS_CHUNK, output_dim), \
        f"expected (B, {NUM_ACTIONS_CHUNK}, {output_dim}), got {tuple(out.shape)}"

    # Forward without h_w, h_sp should still work (graceful default)
    out2 = model(x, h_a=h_a, h_t=h_t, p=p)
    assert out2.shape == (B, NUM_ACTIONS_CHUNK, output_dim), \
        f"graceful default broken: {tuple(out2.shape)}"

    print("OK: MLPResNet concat-and-trim preserves output shape")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_action_head_concat.py
```

Expected: `TypeError: forward() got an unexpected keyword argument 'h_w'`

- [ ] **Step 3: Modify `MLPResNet.forward` to accept `h_w`, `h_sp` and concat/trim**

In `VLA-Adapter/prismatic/models/action_heads.py`, replace the `MLPResNet.forward` method (around lines 111–121):

Before:
```python
    def forward(self, x, h_a=None, h_t=None, p= None):

        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x
```

After:
```python
    def forward(self, x, h_a=None, h_t=None, p=None, h_w=None, h_sp=None):
        """X-VLA-style concat-to-x: wrist and soft_prompt join the self-attention pool.

        Args:
            x:   (B, NUM_ACTIONS_CHUNK, input_dim) action latent
            h_a: (B, num_layers, K_a, hidden_dim) Bridge adapter hidden
            h_t: (B, num_layers, K_t, hidden_dim) Bridge task hidden
            p:   (B, 1, hidden_dim) proprio
            h_w: (B, 49, hidden_dim) wrist tokens (optional)
            h_sp:(B, 32, hidden_dim) soft prompt tokens (optional)
        Returns:
            (B, NUM_ACTIONS_CHUNK, output_dim)
        """
        action_len = x.shape[1]  # remember action token count for trim

        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)

        # Concat aux streams (X-VLA self-attention pool style)
        if h_w is not None:
            x = torch.cat([x, h_w], dim=1)
        if h_sp is not None:
            x = torch.cat([x, h_sp], dim=1)

        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t=h_t[:, i + 1, :], h_a=h_a[:, i + 1, :], p=p)

        # Trim: keep only action positions
        x = x[:, :action_len, :]

        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x
```

- [ ] **Step 4: Run test to verify it passes**

```bash
CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_action_head_concat.py
```

Expected: `OK: MLPResNet concat-and-trim preserves output shape`

- [ ] **Step 5: Commit**

```bash
git add VLA-Adapter/prismatic/models/action_heads.py \
        scripts/gemma4/test_action_head_concat.py
git commit -m "$(cat <<'EOF'
feat(redesign): MLPResNet.forward に h_w / h_sp concat-to-x サポート追加

X-VLA 原実装の self-attn pool 設計に合わせて、wrist (49 tokens) と
soft_prompt (32 tokens) を fc1 直後で x に concat、24 block self-attn
+ Bridge cross-attn を通過後、action 位置だけ trim して fc2 へ。

新 K/V projection 追加なし (MLPResNetBlock_Pro 構造不変)。
shape smoke test で動作確認。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Update `L1RegressionActionHead.predict_action` signature

**Files:**
- Modify: `VLA-Adapter/prismatic/models/action_heads.py:43-80`

**What and why:** The outer `predict_action` method needs to thread `h_w` and `h_sp` through to `self.model` (MLPResNet).

- [ ] **Step 1: Modify the method signature and pass args through**

In `VLA-Adapter/prismatic/models/action_heads.py`, locate `L1RegressionActionHead.predict_action` (around lines 43–80) and update:

Before (signature + final call):
```python
    def predict_action(
            self,
            actions_hidden_states,
            proprio=None,
            proprio_projector=None,
            phase="Inference"
            ):
        ...
        action = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states
            )
```

After (add h_w, h_sp args; pass them to self.model):
```python
    def predict_action(
            self,
            actions_hidden_states,
            proprio=None,
            proprio_projector=None,
            phase="Inference",
            h_w=None,
            h_sp=None,
            ):
        ...
        action = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states,
            h_w=h_w,
            h_sp=h_sp,
            )
```

(The `...` preserves the intermediate logic; only signature and the final `self.model(...)` call change.)

- [ ] **Step 2: Verify import syntax**

```bash
.venv-gemma4/bin/python -c "
import sys; sys.path.insert(0, 'VLA-Adapter')
from prismatic.models.action_heads import L1RegressionActionHead
import inspect
sig = inspect.signature(L1RegressionActionHead.predict_action)
assert 'h_w' in sig.parameters
assert 'h_sp' in sig.parameters
print('OK:', list(sig.parameters.keys()))
"
```

Expected: `OK: ['self', 'actions_hidden_states', 'proprio', 'proprio_projector', 'phase', 'h_w', 'h_sp']`

- [ ] **Step 3: Commit**

```bash
git add VLA-Adapter/prismatic/models/action_heads.py
git commit -m "$(cat <<'EOF'
feat(redesign): L1RegressionActionHead.predict_action に h_w/h_sp 引数追加

MLPResNet.forward に pass through するためのシグネチャ更新。default None で
既存呼び出しとの backward compat 維持。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Switch model load to `Gemma4ForConditionalGeneration`

**Files:**
- Modify: `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py`
- Modify: `VLA-Adapter/vla-scripts/finetune_gemma4.py`

**What and why:** Spec §5.1. The current code loads `AutoModelForCausalLM` which gives the text-only Gemma 4 LM. We need `Gemma4ForConditionalGeneration` to get both `vision_tower` (native vision, frozen) and `multi_modal_projector` (native projector) alongside the language model.

- [ ] **Step 1: Inspect the loaded `Gemma4ForConditionalGeneration` structure**

```bash
.venv-gemma4/bin/python -c "
from transformers import Gemma4ForConditionalGeneration
import torch
m = Gemma4ForConditionalGeneration.from_pretrained(
    'google/gemma-4-E2B', dtype=torch.bfloat16, attn_implementation='sdpa'
)
print('Top-level children:')
for name, _ in m.named_children():
    print(' ', name)
print()
print('model.* children:')
for name, _ in m.model.named_children():
    print(' ', name)
print()
print('type(m.model.vision_tower):', type(m.model.vision_tower).__name__)
print('type(m.model.multi_modal_projector):', type(m.model.multi_modal_projector).__name__)
print('type(m.model.language_model):', type(m.model.language_model).__name__)
" 2>&1 | tail -30
```

Record the output. Expected: `vision_tower`, `multi_modal_projector`, `language_model` all present on `m.model`. This exact attribute layout is needed in Task 6.

- [ ] **Step 2: Update the import at top of `modeling_prismatic_gemma4.py`**

In `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py`, find the imports section (around lines 1–40):

Before (partial):
```python
# (import of AutoModelForCausalLM in finetune_gemma4.py or similar)
```

Add at the top of the file:
```python
from transformers import Gemma4ForConditionalGeneration  # native multimodal
```

- [ ] **Step 3: Update `finetune_gemma4.py:241-243` model load**

In `VLA-Adapter/vla-scripts/finetune_gemma4.py`, find the model load (around lines 241–243):

Before:
```python
    gemma = AutoModelForCausalLM.from_pretrained(
        cfg.gemma_model_id, dtype=torch.bfloat16, attn_implementation=cfg.attn_implementation,
    ).to(device).eval()
```

After:
```python
    from transformers import Gemma4ForConditionalGeneration
    gemma = Gemma4ForConditionalGeneration.from_pretrained(
        cfg.gemma_model_id, dtype=torch.bfloat16, attn_implementation=cfg.attn_implementation,
    ).to(device).eval()
```

Also change the default `attn_implementation` on `FinetuneConfig` (around line 157) from `"sdpa"` to `"flash_attention_2"` (spec §7.1). First verify `flash-attn` is installed:

```bash
.venv-gemma4/bin/pip show flash-attn 2>&1 | head -2
```

If not installed:
```bash
.venv-gemma4/bin/pip install flash-attn --no-build-isolation
```

Then change the attr:

Before (line ~157):
```python
    attn_implementation: str = "sdpa"
```

After:
```python
    attn_implementation: str = "flash_attention_2"
```

- [ ] **Step 4: Run a 1-forward smoke**

This will still fail for architecture reasons (the wrapper still expects DinoSigLIP), but it should at least load the model. Just run the import:

```bash
.venv-gemma4/bin/python -c "
import sys; sys.path.insert(0, 'VLA-Adapter')
import torch
from transformers import Gemma4ForConditionalGeneration
m = Gemma4ForConditionalGeneration.from_pretrained(
    'google/gemma-4-E2B', dtype=torch.bfloat16, attn_implementation='flash_attention_2'
).to('cuda').eval()
assert hasattr(m.model, 'vision_tower')
assert hasattr(m.model, 'multi_modal_projector')
assert hasattr(m.model, 'language_model')
print('OK: Gemma4ForConditionalGeneration loaded with vision + projector + LM')
"
```

Expected: `OK: Gemma4ForConditionalGeneration loaded with vision + projector + LM`

- [ ] **Step 5: Commit**

```bash
git add VLA-Adapter/vla-scripts/finetune_gemma4.py
git commit -m "$(cat <<'EOF'
feat(redesign): switch model load to Gemma4ForConditionalGeneration + FA-2

AutoModelForCausalLM (text-only) → Gemma4ForConditionalGeneration (multimodal)
に切替。vision_tower + multi_modal_projector + language_model が一括ロード
される。併せて attn_implementation default を sdpa → flash_attention_2 に変更
(Gemma 4 HF 側は _supports_flash_attn=True 宣言済)。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Replace DinoSigLIP path with Gemma 4 native vision in `VLAAdapterGemma4`

**Files:**
- Modify: `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` (major)

**What and why:** Spec §5.1. The outer `VLAAdapterGemma4` currently takes a separate `DinoSigLIPViTBackbone` and projects its output to LLM dim via `VisionProjector`. We replace both with Gemma 4's native `vision_tower` + `multi_modal_projector`. This removes ~400M DinoSigLIP + VisionProjector params.

- [ ] **Step 1: Note the current `VLAAdapterGemma4.__init__` signature and fields to remove**

Open `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` and locate `VLAAdapterGemma4.__init__` (around lines 93–158). Fields to remove:
- `vision_backbone` parameter
- `self.vision_backbone = ...`
- `self.vision_projector = VisionProjector(...)`
- The `VisionProjector` class itself (around lines 66–77)

Fields to keep:
- `self.llm` (now a `Gemma4ForConditionalGeneration` instance)
- `self.proprio_projector`
- `self.action_queries`
- `self.feature_norm`
- `self.action_head`
- `self.soft_prompt_library`

- [ ] **Step 2: Refactor `__init__` to remove DinoSigLIP dependencies**

In `VLAAdapterGemma4.__init__`, delete the `VisionProjector` class definition and the `vision_backbone` / `vision_projector` references. Example of new `__init__` signature:

Before:
```python
    def __init__(
        self,
        gemma_model: nn.Module,
        vision_backbone: DinoSigLIPViTBackbone,
        feature_norm: Optional[nn.Module] = None,
        proprio_dim: int = 8,
        action_dim: int = 7,
        num_action_chunks: int = 8,
        initial_projection_dim: int = 8192,
        num_pretrain_datasets: int = 0,
        num_soft_prompt_tokens: int = 32,
    ):
```

After:
```python
    def __init__(
        self,
        gemma_model: nn.Module,   # Gemma4ForConditionalGeneration instance
        feature_norm: Optional[nn.Module] = None,
        proprio_dim: int = 8,
        action_dim: int = 7,
        num_action_chunks: int = 8,
        num_pretrain_datasets: int = 0,
        num_soft_prompt_tokens: int = 32,
    ):
```

Delete the `VisionProjector` class (lines ~66–77) entirely.

Delete these lines from `__init__`:
```python
        self.vision_backbone = vision_backbone
        for p in self.vision_backbone.parameters():
            p.requires_grad = False
        self.vision_backbone.eval()

        ...

        self.vision_projector = VisionProjector(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            initial_projection_dim=initial_projection_dim,
        )
```

Also delete or update `vision_dim = self.vision_backbone.embed_dim` line.

Freeze Gemma 4 vision_tower + multi_modal_projector explicitly:

```python
        # Freeze all of Gemma 4 (vision_tower, multi_modal_projector, language_model)
        for p in self.llm.parameters():
            p.requires_grad = False
```

(If this was already present, leave it — Gemma 4 ForConditionalGeneration includes all three as submodules, so one loop freezes them all.)

- [ ] **Step 3: Update the `forward` to use native vision**

In `VLAAdapterGemma4.forward` (around lines 178–268), the section that processes vision needs to change. The current code does:

```python
        # ---- Vision features ----
        vision_features = self.vision_backbone(pixel_values)           # (B, 512, vision_dim)
        vision_projected = self.vision_projector(vision_features)      # (B, 512, llm_dim)
```

Replace with Gemma 4 native vision. Note: Gemma 4's `vision_tower` expects image pixel values in its own processor format. For now, accept pixel_values in the shape it expects:

```python
        # ---- Vision features (Gemma 4 native) ----
        # pixel_values: (B, 3, H, W) or processed via Gemma 4 image processor (depends on caller)
        # self.llm.model.vision_tower returns (B, num_patches, 768) before pooling
        # multi_modal_projector projects 768 -> 1536
        # For VLA we only use the scene view (slot 0); wrist is handled separately (Task 8)
        scene_pixel_values = pixel_values["scene"]   # (B, 3, H, W)
        vision_out = self.llm.model.vision_tower(scene_pixel_values)
        vision_features = vision_out.last_hidden_state   # (B, num_tokens, 768)
        vision_projected = self.llm.model.multi_modal_projector(vision_features)  # (B, num_tokens, 1536)
```

Note: the exact API for `vision_tower` invocation (whether it returns `BaseModelOutputWithPooling` or similar) must match Task 5 Step 1 inspection output. Adjust based on actual structure — if `vision_tower` takes additional args like `pixel_position_ids`, include them per HF docs. **Inspect the Gemma4VisionModel signature before implementing:**

```bash
.venv-gemma4/bin/python -c "
import inspect
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel
print(inspect.signature(Gemma4VisionModel.forward))
"
```

Record the exact signature and adapt the call accordingly.

- [ ] **Step 4: Update `NUM_VISION_TOKENS` constant if needed**

In `VLA-Adapter/prismatic/vla/constants_gemma4.py`, check the value of `NUM_VISION_TOKENS`. It is currently set for DinoSigLIP (512 tokens for 2 views). For Gemma 4 native vision with soft_tokens=280 (default) and scene-only (1 view), this should become **280**.

```bash
grep -n "NUM_VISION_TOKENS" VLA-Adapter/prismatic/vla/constants_gemma4.py
```

Update:

Before:
```python
NUM_VISION_TOKENS = 512   # DinoSigLIP 2 views
```

After:
```python
NUM_VISION_TOKENS = 280   # Gemma 4 native vision, default max_soft_tokens, scene view only
```

(If the constant file uses a different value, update to 280 and document in comment.)

- [ ] **Step 5: Commit**

```bash
git add VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py \
        VLA-Adapter/prismatic/vla/constants_gemma4.py
git commit -m "$(cat <<'EOF'
feat(redesign): DinoSigLIP 廃止、Gemma 4 native vision に切替

VisionProjector クラス削除、vision_backbone 引数削除。
scene 画像は self.llm.model.vision_tower + multi_modal_projector で処理
(Gemma 4 同時 pretrain で feature alignment 獲得済)。
NUM_VISION_TOKENS: 512 (DinoSigLIP 2view) → 280 (Gemma4 scene only、
max_soft_tokens default)。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Relocate `SoftPromptLibrary` from LLM input to action head

**Files:**
- Modify: `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py:218-244` (forward)

**What and why:** Spec §5.5. Currently `soft_prompts` are concatenated to the LLM `inputs_embeds`. X-VLA original places them at the action head input instead. We remove the LLM-side concat and pass `h_sp` to the action head.

- [ ] **Step 1: Remove soft_prompt concat from LLM input embeddings**

In `VLAAdapterGemma4.forward`, find the section (around lines 218–229):

Before:
```python
        # ---- Stage 3 (optional): Soft Prompt を inputs_embeds 前段に concat (案 B) ----
        if use_soft_prompt:
            soft_prompts = self.soft_prompt_library(dataset_id)
            embeddings = torch.cat([soft_prompts, embeddings], dim=1)
            zero_ple = torch.zeros(
                B, num_sp, per_layer_inputs.size(2), per_layer_inputs.size(3),
                device=per_layer_inputs.device, dtype=per_layer_inputs.dtype,
            )
            per_layer_inputs = torch.cat([zero_ple, per_layer_inputs], dim=1)

        L_total = num_sp + L
        ...
```

After (remove the soft_prompt concat, compute h_sp separately for action head):
```python
        # Stage 3 Soft Prompt is now fed directly to the action head (X-VLA original design),
        # NOT prepended to the LLM input. We compute h_sp here for later use in predict_action.
        if use_soft_prompt:
            h_sp = self.soft_prompt_library(dataset_id)   # (B, 32, llm_dim)
        else:
            h_sp = None

        L_total = L   # no longer extended by num_sp
        ...
```

Also remove `num_sp = self.num_soft_prompt_tokens if use_soft_prompt else 0` if it becomes unused.

- [ ] **Step 2: Remove soft_prompt offset from position masks**

In the same forward, around lines 251–252:

Before:
```python
        apos0 = amask[0].nonzero(as_tuple=True)[0] + num_sp
        vpos0 = vmask[0].nonzero(as_tuple=True)[0] + num_sp
```

After:
```python
        apos0 = amask[0].nonzero(as_tuple=True)[0]
        vpos0 = vmask[0].nonzero(as_tuple=True)[0]
```

- [ ] **Step 3: Pass `h_sp` to action head in the forward**

At the end of `VLAAdapterGemma4.forward`, around lines 258–263:

Before:
```python
        predicted = self.action_head.predict_action(
            actions_hidden_states=combined,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            phase="Training" if self.training else "Inference",
        )
```

After:
```python
        predicted = self.action_head.predict_action(
            actions_hidden_states=combined,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            phase="Training" if self.training else "Inference",
            h_w=h_w,      # set by Task 8
            h_sp=h_sp,
        )
```

Note: `h_w` will be defined by Task 8's forward update. For now this line references it; Task 8 will add the wrist feature computation before this line.

- [ ] **Step 4: Commit**

```bash
git add VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py
git commit -m "$(cat <<'EOF'
feat(redesign): SoftPromptLibrary を LLM 入力 → action head 入力に relocate

X-VLA 原実装と一致させる: soft_prompt を LLM inputs_embeds に concat する案 B
deviation を廃止、代わりに action head の predict_action に h_sp 引数で渡す。
LLM 側の attention_mask / position_ids / Bridge offset が num_sp 不要に。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Add wrist path to `VLAAdapterGemma4`

**Files:**
- Modify: `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` (__init__ + forward)

**What and why:** Spec §5.2. The wrist camera now goes through `WristResNet18` (Task 1) and its output is threaded into the action head as `h_w`.

- [ ] **Step 1: Add `WristResNet18` attribute in `__init__`**

In `VLAAdapterGemma4.__init__`, after the existing trainable modules, add:

```python
        from prismatic.models.backbones.vision.wrist_resnet18 import WristResNet18
        self.wrist_encoder = WristResNet18(out_dim=llm_dim)
```

(Keep the import local to avoid circular import risk, or put it at the top of the file. Either works.)

- [ ] **Step 2: Update `forward` signature to accept wrist image**

In `VLAAdapterGemma4.forward`, update the `pixel_values` type annotation and expected shape:

Before:
```python
        pixel_values: Dict[str, torch.Tensor],  # {"dino": (B, T, 3, H, W), "siglip": (B, T, 3, H, W)}
```

After:
```python
        pixel_values: Dict[str, torch.Tensor],  # {"scene": (B, 3, 224, 224), "wrist": (B, 3, 224, 224)}
```

- [ ] **Step 3: Compute wrist features in forward**

In `VLAAdapterGemma4.forward`, after the scene vision processing (from Task 6) and before the LLM call, add:

```python
        # ---- Wrist features (trainable ResNet18) ----
        wrist_pixel_values = pixel_values["wrist"]       # (B, 3, 224, 224)
        h_w = self.wrist_encoder(wrist_pixel_values)     # (B, 49, llm_dim)
```

`h_w` is already referenced in Task 7's `predict_action` call.

- [ ] **Step 4: Update trainable param assertion**

In `VLAAdapterGemma4.__init__`, there was an assertion on trainable param count. Update the expected total:

```python
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        # New baseline:
        #   action_head (~562M, film_gen 削除後)
        #   + action_queries (0.1M)
        #   + soft_prompt_library (0.05M per num_datasets)
        #   + wrist_encoder (ResNet18 11.7M + projector 1.2M = ~12.9M)
        #   + proprio_projector (~2.4M)
        #   ≈ 577M base, + LoRA (4M) if Mode A
        expected_base = 577.0 + cfg_num_pretrain_datasets * 0.05   # rough
        # Use a wider tolerance since params fluctuate slightly:
        # assert abs(total_trainable - expected_base) < 5.0, ...
        # Or just log and skip hard assertion for now:
        print(f"[VLAAdapterGemma4] trainable params: {total_trainable:.3f} M")
```

Since exact param counts are sensitive to implementation (e.g. whether proprio_projector has biases, whether ResNet18 has bn affine, etc.), replace the tight assertion with a log + loose bound. Example:

```python
        assert 550.0 < total_trainable < 600.0, \
            f"trainable params out of expected range: got {total_trainable:.3f} M"
```

- [ ] **Step 5: Smoke import + instantiation**

```bash
.venv-gemma4/bin/python -c "
import sys; sys.path.insert(0, 'VLA-Adapter')
import torch
from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4

device = 'cuda'
gemma = Gemma4ForConditionalGeneration.from_pretrained(
    'google/gemma-4-E2B', dtype=torch.bfloat16, attn_implementation='flash_attention_2'
).to(device).eval()
model = VLAAdapterGemma4(
    gemma_model=gemma,
    proprio_dim=8,
    action_dim=7,
    num_action_chunks=8,
    num_pretrain_datasets=1,
    num_soft_prompt_tokens=32,
).to(device, dtype=torch.bfloat16)

total = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f'trainable: {total:.2f} M')
assert hasattr(model, 'wrist_encoder')
assert hasattr(model, 'soft_prompt_library')
assert hasattr(model, 'action_queries')
print('OK: model instantiated with new structure')
"
```

Expected: `trainable: 577 M (approx)` and `OK: model instantiated with new structure`.

- [ ] **Step 6: Commit**

```bash
git add VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py
git commit -m "$(cat <<'EOF'
feat(redesign): VLAAdapterGemma4 に wrist 経路を追加

WristResNet18 を self.wrist_encoder として統合、forward で
pixel_values["wrist"] から h_w (B, 49, 1536) を出して
action head の predict_action に pass。
pixel_values の key schema を {"dino", "siglip"} から {"scene", "wrist"} に変更。
trainable params 目標 ~577M (Mode A の LoRA 追加前) の範囲で assertion。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Dual-Track Configuration (Tasks 9–12)

### Task 9: Add `training_mode` config flag to `FinetuneConfig`

**Files:**
- Modify: `VLA-Adapter/vla-scripts/finetune_gemma4.py` (FinetuneConfig dataclass + CLI)

**What and why:** Spec §4.3. A single flag `training_mode ∈ {"quality", "speed"}` selects Mode A vs Mode B. All other mode-specific settings (LoRA, GC, action_queries freeze, no_grad wrap) derive from this flag.

- [ ] **Step 1: Add `training_mode` field to `FinetuneConfig`**

In `VLA-Adapter/vla-scripts/finetune_gemma4.py`, find the `FinetuneConfig` dataclass (around line 100+):

Add these fields (after existing fields, e.g. after `attn_implementation`):

```python
    # --- Dual-Track ---
    training_mode: str = "quality"       # "quality" (Mode A) | "speed" (Mode B)
    lora_r: int = 16                     # Mode A: LoRA rank
    lora_alpha: int = 32                 # Mode A: LoRA alpha
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
```

Also add a validator (simple assert) in `__post_init__` or at top of `main`:

```python
    assert cfg.training_mode in ("quality", "speed"), \
        f"training_mode must be 'quality' or 'speed', got {cfg.training_mode!r}"
```

- [ ] **Step 2: Echo the mode at startup for visibility**

Near the top of `main` (after loading config), add:

```python
    rprint(f"[dual-track] training_mode = {cfg.training_mode}")
```

- [ ] **Step 3: Commit**

```bash
git add VLA-Adapter/vla-scripts/finetune_gemma4.py
git commit -m "$(cat <<'EOF'
feat(dual-track): add training_mode flag ("quality" | "speed") to FinetuneConfig

Dual-track 実装の root flag。後続 task で action_queries / GC / LoRA /
torch.no_grad wrap がこの flag から分岐する。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Implement Mode A (action_queries trainable + GC)

**Files:**
- Modify: `VLA-Adapter/vla-scripts/finetune_gemma4.py` (model setup block)

**What and why:** Spec §GA1-GA2. In Quality mode, `action_queries` stays trainable (default), and we enable gradient checkpointing on the LLM text model.

- [ ] **Step 1: Branch model setup on `training_mode`**

In `VLA-Adapter/vla-scripts/finetune_gemma4.py`, find the block where the `VLAAdapterGemma4` is built (around `build_model` function). After model creation, add:

```python
    if cfg.training_mode == "quality":
        # Mode A: action_queries trainable (default, already set by nn.Embedding init),
        # LLM gradient checkpointing enabled.
        assert model_vla.action_queries.weight.requires_grad is True
        model_vla.llm.model.language_model.gradient_checkpointing_enable()
        rprint("[mode-A] action_queries trainable; LLM GC enabled")
    elif cfg.training_mode == "speed":
        # Mode B: see Task 11
        pass   # placeholder, filled by Task 11
```

- [ ] **Step 2: Verify GC enables without error**

Run a quick import-level smoke (no full forward needed yet):

```bash
.venv-gemma4/bin/python -c "
import sys; sys.path.insert(0, 'VLA-Adapter')
import torch
from transformers import Gemma4ForConditionalGeneration
m = Gemma4ForConditionalGeneration.from_pretrained(
    'google/gemma-4-E2B', dtype=torch.bfloat16, attn_implementation='flash_attention_2'
).to('cuda').eval()
m.model.language_model.gradient_checkpointing_enable()
print('OK: GC enabled on Gemma 4 text model')
print('is_gradient_checkpointing:', m.model.language_model.is_gradient_checkpointing)
"
```

Expected: `OK: GC enabled ... is_gradient_checkpointing: True`. If it errors with something like "gradient_checkpointing has known issues on this version", investigate spec §11 risk row "GC が transformers 5.5.4 で動かない" and work around.

- [ ] **Step 3: Commit**

```bash
git add VLA-Adapter/vla-scripts/finetune_gemma4.py
git commit -m "$(cat <<'EOF'
feat(dual-track): Mode A (Quality) — action_queries trainable + LLM GC 有効化

training_mode == "quality" で LLM の gradient_checkpointing を enable。
action_queries は既に default trainable (nn.Embedding 標準挙動)。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Implement Mode B (action_queries frozen + `torch.no_grad()` LLM forward)

**Files:**
- Modify: `VLA-Adapter/vla-scripts/finetune_gemma4.py` (setup branch)
- Modify: `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` (forward branch)

**What and why:** Spec §GB1-GB2 and §5.10. In Speed mode, `action_queries` is frozen (so LLM input has no trainable source), and we wrap the LLM forward in `torch.no_grad()` to skip activation storage and backward entirely.

- [ ] **Step 1: Add `training_mode` attribute to `VLAAdapterGemma4`**

In `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py`, add a `training_mode` attribute in `__init__`:

```python
    def __init__(
        self,
        gemma_model: nn.Module,
        ...
        num_pretrain_datasets: int = 0,
        num_soft_prompt_tokens: int = 32,
        training_mode: str = "quality",    # "quality" | "speed"
    ):
        super().__init__()
        self.training_mode = training_mode
        assert training_mode in ("quality", "speed")
        ...
```

Propagate this from `finetune_gemma4.py:build_model` to the constructor:

```python
    model_vla = VLAAdapterGemma4(
        gemma_model=gemma,
        ...
        num_pretrain_datasets=cfg.num_pretrain_datasets,
        num_soft_prompt_tokens=cfg.num_soft_prompt_tokens,
        training_mode=cfg.training_mode,   # NEW
    ).to(device, dtype=torch.bfloat16)
```

- [ ] **Step 2: Wrap the LLM forward in `torch.no_grad()` when mode is speed**

In `VLAAdapterGemma4.forward`, find the LLM call (around line 236–244):

Before:
```python
        out = llm(
            inputs_embeds=embeddings,
            per_layer_inputs=per_layer_inputs,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
```

After:
```python
        if self.training_mode == "speed":
            with torch.no_grad():
                out = llm(
                    inputs_embeds=embeddings,
                    per_layer_inputs=per_layer_inputs,
                    use_cache=True,
                    output_hidden_states=True,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
        else:  # quality
            out = llm(
                inputs_embeds=embeddings,
                per_layer_inputs=per_layer_inputs,
                use_cache=True,
                output_hidden_states=True,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
```

- [ ] **Step 3: Implement the Mode B branch in `finetune_gemma4.py`**

Fill in the speed-mode branch from Task 10 Step 1:

Replace:
```python
    elif cfg.training_mode == "speed":
        pass   # placeholder
```

With:
```python
    elif cfg.training_mode == "speed":
        # Mode B: action_queries frozen (zero init + requires_grad=False),
        # LLM forward wrapped in torch.no_grad() (done inside VLAAdapterGemma4.forward).
        # NO GC, NO LoRA.
        model_vla.action_queries.weight.data.zero_()
        model_vla.action_queries.weight.requires_grad = False
        rprint("[mode-B] action_queries frozen (zero init); LLM no_grad wrap active; no GC; no LoRA")
```

- [ ] **Step 4: Verify trainable param assertion allows Mode B path**

The trainable param loose bound (Task 8 Step 4) is `550 < total < 600`. In Mode B, action_queries (0.1M) is removed from the trainable set, so total should drop to ~577M. This is still within the bound.

Double-check:

```bash
.venv-gemma4/bin/python -c "
import sys; sys.path.insert(0, 'VLA-Adapter')
import torch
from transformers import Gemma4ForConditionalGeneration
from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4
gemma = Gemma4ForConditionalGeneration.from_pretrained(
    'google/gemma-4-E2B', dtype=torch.bfloat16, attn_implementation='flash_attention_2'
).to('cuda').eval()
for mode in ('quality', 'speed'):
    m = VLAAdapterGemma4(
        gemma_model=gemma,
        num_pretrain_datasets=1,
        num_soft_prompt_tokens=32,
        training_mode=mode,
    ).to('cuda', dtype=torch.bfloat16)
    if mode == 'speed':
        m.action_queries.weight.requires_grad = False
    total = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    print(f'{mode}: trainable = {total:.2f} M')
    del m
"
```

Expected: `quality: ~577.xx M`, `speed: ~577.xx M` minus 0.1M for action_queries ≈ ~576.xx M.

- [ ] **Step 5: Commit**

```bash
git add VLA-Adapter/vla-scripts/finetune_gemma4.py \
        VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py
git commit -m "$(cat <<'EOF'
feat(dual-track): Mode B (Speed) — action_queries frozen + LLM no_grad wrap

training_mode == "speed" で:
- action_queries.weight.data.zero_() + requires_grad=False
- VLAAdapterGemma4.forward 内で LLM call を with torch.no_grad() で wrap
- GC / LoRA は不採用 (LLM backward が走らないため意味がない)

期待効果: LLM backward ~F 削減、activation memory ほぼゼロ、batch 拡大余地。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: LoRA integration for Mode A

**Files:**
- Modify: `VLA-Adapter/vla-scripts/finetune_gemma4.py`

**What and why:** Spec §5.8. In Mode A, apply LoRA to Gemma 4 LM's q/k/v/o_proj with r=16, alpha=32. We use the `peft` library.

- [ ] **Step 1: Verify `peft` is installed**

```bash
.venv-gemma4/bin/pip show peft 2>&1 | head -2
```

If not installed:
```bash
.venv-gemma4/bin/pip install peft
```

- [ ] **Step 2: Wrap LLM with LoRA in Mode A**

In `VLA-Adapter/vla-scripts/finetune_gemma4.py`, modify the Mode A branch (Task 10):

Before:
```python
    if cfg.training_mode == "quality":
        assert model_vla.action_queries.weight.requires_grad is True
        model_vla.llm.model.language_model.gradient_checkpointing_enable()
        rprint("[mode-A] action_queries trainable; LLM GC enabled")
```

After:
```python
    if cfg.training_mode == "quality":
        assert model_vla.action_queries.weight.requires_grad is True
        model_vla.llm.model.language_model.gradient_checkpointing_enable()

        # Apply LoRA to Gemma 4 LM's attention projections
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.lora_target_modules),
            lora_dropout=0.0,
            bias="none",
            task_type=None,    # not a standard PEFT task, raw Module wrap
        )
        # Wrap only the text model (language_model), not vision_tower / multi_modal_projector
        model_vla.llm.model.language_model = get_peft_model(
            model_vla.llm.model.language_model, lora_cfg
        )
        # LoRA params require grad; base model params stay frozen.
        lora_params = sum(
            p.numel() for p in model_vla.llm.model.language_model.parameters() if p.requires_grad
        ) / 1e6
        rprint(f"[mode-A] LoRA wrapped: r={cfg.lora_r}, alpha={cfg.lora_alpha}, "
               f"targets={cfg.lora_target_modules}, LoRA trainable = {lora_params:.3f} M")
        rprint("[mode-A] action_queries trainable; LLM GC enabled")
```

- [ ] **Step 3: Update optimizer group collection to include LoRA params**

In `finetune_gemma4.py`, the optimizer setup already collects `model_vla.parameters()` filtered by `requires_grad=True`. Since LoRA params have `requires_grad=True`, they will be picked up automatically. Verify by printing trainable param count after LoRA wrap:

```python
        total_trainable = sum(p.numel() for p in model_vla.parameters() if p.requires_grad) / 1e6
        rprint(f"[mode-A] total trainable (incl LoRA): {total_trainable:.3f} M")
```

- [ ] **Step 4: Smoke test — instantiate Mode A with LoRA**

```bash
.venv-gemma4/bin/python -c "
import sys; sys.path.insert(0, 'VLA-Adapter')
import torch
from transformers import Gemma4ForConditionalGeneration
from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4
from peft import LoraConfig, get_peft_model

gemma = Gemma4ForConditionalGeneration.from_pretrained(
    'google/gemma-4-E2B', dtype=torch.bfloat16, attn_implementation='flash_attention_2'
).to('cuda').eval()
m = VLAAdapterGemma4(
    gemma_model=gemma, num_pretrain_datasets=1, training_mode='quality',
).to('cuda', dtype=torch.bfloat16)
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','k_proj','v_proj','o_proj'],
                 lora_dropout=0.0, bias='none', task_type=None)
m.llm.model.language_model = get_peft_model(m.llm.model.language_model, cfg)
lora_p = sum(p.numel() for p in m.llm.model.language_model.parameters() if p.requires_grad) / 1e6
print(f'LoRA trainable: {lora_p:.3f} M')
assert 1.0 < lora_p < 10.0, 'LoRA param count out of range'
print('OK: LoRA wrap successful')
"
```

Expected: `LoRA trainable: ~3-5 M`, `OK: LoRA wrap successful`.

- [ ] **Step 5: Commit**

```bash
git add VLA-Adapter/vla-scripts/finetune_gemma4.py
git commit -m "$(cat <<'EOF'
feat(dual-track): Mode A に LoRA (r=16 on q/k/v/o_proj) 統合

peft.get_peft_model で Gemma 4 LM の language_model サブツリーのみ LoRA wrap。
vision_tower / multi_modal_projector は frozen のまま (scene feature は
pretrained alignment に任せる設計)。LoRA trainable ~4M 程度。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Configs and Data Loader (Tasks 13–14)

### Task 13: Create YAML configs for Mode A / Mode B

**Files:**
- Create: `config/pretrain_taco_quality.yaml`
- Create: `config/pretrain_taco_speed.yaml`

**What and why:** Spec §4.3. Each mode has its own config file for clarity and reproducibility.

- [ ] **Step 1: Create `config/pretrain_taco_quality.yaml`**

```yaml
# Mode A (Quality) — Taco Play pretrain
# action_queries trainable + LLM GC + LoRA r=16

training_mode: "quality"

# Model
gemma_model_id: "google/gemma-4-E2B"
attn_implementation: "flash_attention_2"

# LoRA
lora_r: 16
lora_alpha: 32
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# Pretrain
pretrain_mode: true
num_pretrain_datasets: 1
num_soft_prompt_tokens: 32
pretrain_freeze_steps: 1000
pretrain_warmup_steps: 2000

# Batch / LR
batch_size: 16                # per-GPU
grad_accumulation_steps: 1
learning_rate: 2.0e-4
coef: 0.25                    # soft_prompt LR multiplier

# Data
dataset_name: "taco_play"
data_root_dir: "data/stage3_openx"
pretrain_num_workers: 4
shuffle_buffer_size: 256000

# Run name
run_name: "gemma-4-e2b+pretrain-taco-solo+mode-quality"
wandb_project: "vla-gemma4"
wandb_entity: "takaki-maeda-1999-toyota-technological-institute"
```

- [ ] **Step 2: Create `config/pretrain_taco_speed.yaml`**

```yaml
# Mode B (Speed) — Taco Play pretrain
# action_queries frozen + LLM no_grad + no LoRA

training_mode: "speed"

# Model
gemma_model_id: "google/gemma-4-E2B"
attn_implementation: "flash_attention_2"

# LoRA (ignored when training_mode=="speed")
lora_r: 0
lora_alpha: 0
lora_target_modules: []

# Pretrain
pretrain_mode: true
num_pretrain_datasets: 1
num_soft_prompt_tokens: 32
pretrain_freeze_steps: 1000
pretrain_warmup_steps: 2000

# Batch / LR (larger batch enabled by memory savings from no_grad LLM)
batch_size: 24                # per-GPU
grad_accumulation_steps: 1
learning_rate: 2.0e-4
coef: 0.25

# Data
dataset_name: "taco_play"
data_root_dir: "data/stage3_openx"
pretrain_num_workers: 4
shuffle_buffer_size: 256000

# Run name
run_name: "gemma-4-e2b+pretrain-taco-solo+mode-speed"
wandb_project: "vla-gemma4"
wandb_entity: "takaki-maeda-1999-toyota-technological-institute"
```

- [ ] **Step 3: Verify the configs can be loaded by draccus / the config parser**

Look at the existing config loading pattern in `finetune_gemma4.py`. If it uses `draccus.parse`, the YAML must match the dataclass structure. Smoke test:

```bash
.venv-gemma4/bin/python -c "
import yaml
with open('config/pretrain_taco_quality.yaml') as f:
    cfg_a = yaml.safe_load(f)
with open('config/pretrain_taco_speed.yaml') as f:
    cfg_b = yaml.safe_load(f)
assert cfg_a['training_mode'] == 'quality'
assert cfg_b['training_mode'] == 'speed'
print('OK: both configs parse as YAML and have correct training_mode')
"
```

Expected: `OK: both configs parse as YAML and have correct training_mode`

- [ ] **Step 4: Commit**

```bash
git add config/pretrain_taco_quality.yaml config/pretrain_taco_speed.yaml
git commit -m "$(cat <<'EOF'
feat(dual-track): YAML configs for Mode A / Mode B pretrain

config/pretrain_taco_quality.yaml : Mode A (LoRA + GC, batch=16/GPU)
config/pretrain_taco_speed.yaml   : Mode B (no_grad + frozen queries, batch=24/GPU)
両 config は finetune_gemma4.py が training_mode で分岐して読む。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Create `taco_solo_loader.py`

**Files:**
- Create: `scripts/stage3/taco_solo_loader.py`

**What and why:** Spec §6.1. The existing `multi_dataset_loader.py` handles Taco + Fractal with a problematic Fractal-image-duplication hack. We derive a Taco-only version that properly uses Taco's 2-cam (rgb_static + rgb_gripper) and enables proprio.

- [ ] **Step 1: Inspect the current `multi_dataset_loader.py`**

```bash
head -150 scripts/stage3/multi_dataset_loader.py
```

Understand the structure: `Gemma4OXEDataset` class, batch transform, canonical action extractor, weight-based sampler. The new file will remove Fractal entirely, keep only Taco.

- [ ] **Step 2: Inspect Taco Play proprio format (one-shot investigation)**

```bash
.venv-gemma4/bin/python <<'PY'
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
ds = tfds.load("taco_play", data_dir="data/stage3_openx", split="train", shuffle_files=False)
for ep in ds.take(1):
    for step in ep["steps"].take(1):
        obs = step["observation"]
        print("observation keys:", list(obs.keys()))
        for k, v in obs.items():
            try:
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            except Exception:
                print(f"  {k}: (non-tensor)")
        break
    break
PY
```

Record the proprio field name and dimensions. Common Taco Play fields: `state_eef` (8-dim) or `robot_obs` (15-dim). Use this to choose the canonical proprio extractor.

- [ ] **Step 3: Create `scripts/stage3/taco_solo_loader.py`**

```python
"""Taco Play single-dataset pretrain loader.

Derived from scripts/stage3/multi_dataset_loader.py with:
  - Fractal20220817_data and its duplicate-image hack removed
  - Proprio enabled (extracted from Taco's observation.state_eef or robot_obs)
  - 2-cam fixed: rgb_static (scene) + rgb_gripper (wrist)
  - canonical 7-dim action = rel_actions_world (same as Taco entry in multi_dataset_loader)
  - dataset_id always = 0 (num_pretrain_datasets=1 in VLAAdapterGemma4)

VLAAdapterGemma4 forward 呼び出し互換 batch dict:
  {
    "pixel_values": {"scene": (B, 3, 224, 224), "wrist": (B, 3, 224, 224)},
    "input_ids":    (B, L) long,   # placeholder tokens for vision + action positions
    "proprio":      (B, 8) bf16,
    "actions":      (B, 8, 7) bf16,
    "dataset_id":   (B,) long = 0,
    "languages":    [str] * B,
  }

Run (standalone verify):
  CUDA_VISIBLE_DEVICES=6 .venv-gemma4/bin/python scripts/stage3/taco_solo_loader.py --verify
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
import torch
from PIL import Image
from torch.utils.data import IterableDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "stage3_openx"
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
SCRIPTS_GEMMA4 = REPO_ROOT / "scripts" / "gemma4"
sys.path.insert(0, str(VLA_ROOT))
sys.path.insert(0, str(SCRIPTS_GEMMA4))

from prismatic.vla.constants_gemma4 import (  # noqa: E402
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTION_TOKENS,
    NUM_VISION_TOKENS,
    PROPRIO_PLACEHOLDER_IDX,
    VISION_PLACEHOLDER_BEGIN_IDX,
)


# Constants
PROMPT_MAX_LEN = 20
PROPRIO_DIM = 8
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8
IMAGE_SIZE = 224

DATASET_NAME = "taco_play"
DATASET_ID = 0   # single-dataset pretrain


def extract_canonical_action(step: Dict[str, Any]) -> np.ndarray:
    """Taco Play: action.rel_actions_world (7 dim)."""
    return step["action"]["rel_actions_world"].numpy().astype(np.float32)


def extract_proprio(step: Dict[str, Any]) -> np.ndarray:
    """Taco Play: observation.state_eef (8 dim, xyz + euler + gripper, typical).

    If Step 2 inspection shows a different field name / shape, adapt here.
    Fallback: pad/truncate to PROPRIO_DIM=8.
    """
    obs = step["observation"]
    if "state_eef" in obs:
        p = obs["state_eef"].numpy().astype(np.float32)
    elif "robot_obs" in obs:
        # robot_obs is typically 15-dim; take first 8 as EEF-like
        p = obs["robot_obs"].numpy().astype(np.float32)[:8]
    else:
        # Fallback: zeros
        p = np.zeros(PROPRIO_DIM, dtype=np.float32)
    # Pad/truncate to PROPRIO_DIM
    if p.shape[0] < PROPRIO_DIM:
        p = np.pad(p, (0, PROPRIO_DIM - p.shape[0]))
    elif p.shape[0] > PROPRIO_DIM:
        p = p[:PROPRIO_DIM]
    return p


def resize_image(img_np: np.ndarray) -> np.ndarray:
    """Resize to (224, 224, 3) via PIL."""
    img = Image.fromarray(img_np).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(img)


class TacoSoloDataset(IterableDataset):
    """Iterable dataset yielding Taco Play samples in VLAAdapterGemma4-compatible format."""

    def __init__(self, data_dir: Path = DATA_ROOT, shuffle_buffer_size: int = 256_000):
        super().__init__()
        self.data_dir = data_dir
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self):
        ds = tfds.load(DATASET_NAME, data_dir=str(self.data_dir), split="train", shuffle_files=True)
        ds = ds.flat_map(lambda ep: ep["steps"])
        ds = ds.shuffle(self.shuffle_buffer_size)
        for step in ds:
            scene_img = resize_image(step["observation"]["rgb_static"].numpy())
            wrist_img = resize_image(step["observation"]["rgb_gripper"].numpy())
            action = extract_canonical_action(step)   # (7,)
            proprio = extract_proprio(step)           # (8,)
            language = step.get("language_instruction", b"").numpy().decode("utf-8", errors="replace") \
                if "language_instruction" in step else ""
            yield {
                "scene_img": scene_img,          # (224, 224, 3) uint8
                "wrist_img": wrist_img,          # (224, 224, 3) uint8
                "action": action,                # (7,)
                "proprio": proprio,              # (8,)
                "language": language,
            }


def taco_solo_collate(batch: list, tokenizer) -> Dict[str, torch.Tensor]:
    """Collate into VLAAdapterGemma4 forward signature.

    NOTE: action chunk building (chunk size = NUM_ACTIONS_CHUNK=8) is handled by
    the upstream data pipeline elsewhere (see finetune_gemma4.py build_dataloader).
    Here we just yield per-step samples; chunking is a wrapper layer.

    This collate is a scaffold; full chunking logic will be ported from
    multi_dataset_loader.py's existing chunking code at Task 14 Step X (next step).
    """
    raise NotImplementedError("Chunking layer not yet ported — see Step 4 below")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        ds = TacoSoloDataset()
        for i, sample in enumerate(ds):
            print(f"sample {i}:")
            print(f"  scene_img: {sample['scene_img'].shape}")
            print(f"  wrist_img: {sample['wrist_img'].shape}")
            print(f"  action:    {sample['action'].shape}")
            print(f"  proprio:   {sample['proprio'].shape}")
            print(f"  language:  {sample['language']!r}")
            if i >= 2:
                break
        print("OK: TacoSoloDataset yields samples with expected shapes")
```

- [ ] **Step 4: Port the chunking and batch_transform logic from `multi_dataset_loader.py`**

This is a **mechanical port**. Read `multi_dataset_loader.py` — specifically the `Gemma4OXEDataset`, `Gemma4BatchTransform`, and `collate_gemma4` (or equivalent) sections. In `taco_solo_loader.py`, port:

1. Action chunking (sliding window of NUM_ACTIONS_CHUNK=8 actions)
2. Tokenizer + placeholder construction
3. `pixel_values` dict building with keys `"scene"` and `"wrist"` (instead of dino/siglip)
4. BOUNDS_Q99 normalization for actions (only for dim 0-5, gripper dim 6 is binary pass-through)

Because this is mechanical and specific to the existing code, the implementer should copy the relevant methods from `multi_dataset_loader.py`, remove Fractal branches, rename `slot0/slot1` to `scene/wrist`, and replace `DATASET_ID_MAP` logic with a hardcoded `dataset_id=0`.

Add a smoke main that produces a valid batch:

```python
if __name__ == "__main__":
    parser.add_argument("--smoke-batch", action="store_true")
    args = parser.parse_args()

    if args.smoke_batch:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
        # Assuming Gemma4BatchTransform is ported as TacoSoloBatchTransform:
        # batch = next(iter(DataLoader(
        #     TacoSoloDataset(), batch_size=2, collate_fn=taco_solo_collate_with_tok(tok)
        # )))
        # print(batch['pixel_values']['scene'].shape, batch['actions'].shape)
        print("batch smoke requires the chunking/tokenization logic ported in Step 4 below")
```

- [ ] **Step 5: Run `--verify` to confirm the dataset iterator works**

```bash
CUDA_VISIBLE_DEVICES=6 .venv-gemma4/bin/python scripts/stage3/taco_solo_loader.py --verify
```

Expected output: 3 samples with shapes `(224, 224, 3)`, `(7,)`, `(8,)` and a language string.

- [ ] **Step 6: Commit**

```bash
git add scripts/stage3/taco_solo_loader.py
git commit -m "$(cat <<'EOF'
feat(redesign): Taco Play 単独 pretrain loader (scripts/stage3/taco_solo_loader.py)

multi_dataset_loader.py から Fractal 関連と image duplication hack を削除、
2-cam (rgb_static=scene, rgb_gripper=wrist) + proprio (state_eef 8dim) 有効化
の Taco Play 専用 pipeline。pixel_values dict は {"scene", "wrist"} スキーマ
(新 VLAAdapterGemma4.forward シグネチャ互換)。--verify で 3 サンプル動作確認。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: Smoke Tests and Launch Scripts (Tasks 15–17)

### Task 15: End-to-end dual-track smoke test

**Files:**
- Create: `scripts/gemma4/test_13_dual_track_smoke.py`

**What and why:** Spec §7.2. Verify both modes run forward+backward without NaN and produce valid action predictions. This is the gate before starting full pretrain runs.

- [ ] **Step 1: Create the smoke test script**

```python
"""Phase 0.5 dual-track smoke test.

For each mode in {quality, speed}:
  1. Build VLAAdapterGemma4 with the correct config
  2. Get one batch from TacoSoloDataset
  3. Run forward (and backward for trainable modules)
  4. Assert no NaN in loss
  5. Print samples/sec and peak memory
  6. For speed mode: assert action_queries.grad is None (frozen)
  7. For quality mode: assert action_queries.grad is not None (trainable)

Run:
  CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_13_dual_track_smoke.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
sys.path.insert(0, str(VLA_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stage3"))

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model

from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4


def build_model(mode: str, device="cuda"):
    gemma = Gemma4ForConditionalGeneration.from_pretrained(
        "google/gemma-4-E2B",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device).eval()

    model = VLAAdapterGemma4(
        gemma_model=gemma,
        num_pretrain_datasets=1,
        num_soft_prompt_tokens=32,
        training_mode=mode,
    ).to(device, dtype=torch.bfloat16)

    if mode == "quality":
        model.llm.model.language_model.gradient_checkpointing_enable()
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none", task_type=None,
        )
        model.llm.model.language_model = get_peft_model(
            model.llm.model.language_model, lora_cfg
        )
    elif mode == "speed":
        model.action_queries.weight.data.zero_()
        model.action_queries.weight.requires_grad = False

    model.train()
    return model


def build_dummy_batch(B=2, device="cuda"):
    """Construct a synthetic batch matching the new forward signature."""
    return {
        "pixel_values": {
            "scene": torch.randn(B, 3, 224, 224, device=device, dtype=torch.bfloat16),
            "wrist": torch.randn(B, 3, 224, 224, device=device, dtype=torch.bfloat16),
        },
        "input_ids": torch.zeros(B, 300, device=device, dtype=torch.long),   # placeholder; adapt per constants_gemma4
        "proprio": torch.randn(B, 8, device=device, dtype=torch.bfloat16),
        "actions": torch.randn(B, 8, 7, device=device, dtype=torch.bfloat16),
        "dataset_id": torch.zeros(B, device=device, dtype=torch.long),
    }


def main():
    for mode in ("quality", "speed"):
        print(f"\n=== mode = {mode} ===")
        model = build_model(mode)
        batch = build_dummy_batch(B=2)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        predicted, loss = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            proprio=batch["proprio"],
            actions=batch["actions"],
            dataset_id=batch["dataset_id"],
        )
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        print(f"loss = {loss.item():.4f}, forward+backward = {t1-t0:.3f} s, peak mem = {peak_gb:.2f} GB")

        # Mode-specific assertions
        if mode == "speed":
            assert model.action_queries.weight.grad is None, \
                "speed mode: action_queries.weight.grad should be None (frozen)"
            print("✓ action_queries grad is None (frozen)")
        elif mode == "quality":
            assert model.action_queries.weight.grad is not None, \
                "quality mode: action_queries.weight.grad should be populated"
            print("✓ action_queries grad is populated (trainable)")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Note on `input_ids` placeholder construction**

The script uses `torch.zeros(..., dtype=torch.long)` as input_ids, which will not pass the vision/action placeholder masks in `VLAAdapterGemma4.forward`. Before running, the engineer must construct a valid `input_ids` tensor that contains `VISION_PLACEHOLDER_BEGIN_IDX` for scene tokens and `ACTION_TOKEN_BEGIN_IDX` for action tokens. Copy this construction from an existing smoke test such as `scripts/gemma4/test_06_full_forward.py` or from `scripts/stage3/multi_dataset_loader.py`'s batch_transform.

Specifically, replace:
```python
"input_ids": torch.zeros(B, 300, device=device, dtype=torch.long),
```

with a function `build_input_ids(tok, B, device)` that constructs the exact placeholder sequence. Port from `test_06_full_forward.py` (around lines 47–80 has a `build_input_ids` helper).

- [ ] **Step 3: Run the smoke test**

```bash
CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_13_dual_track_smoke.py
```

Expected output:
```
=== mode = quality ===
loss = 1.xxxx, forward+backward = x.xxx s, peak mem = xx.xx GB
✓ action_queries grad is populated (trainable)

=== mode = speed ===
loss = 1.xxxx, forward+backward = x.xxx s, peak mem = xx.xx GB   (less than quality)
✓ action_queries grad is None (frozen)
```

If speed mode's peak memory is NOT significantly lower than quality mode's, the `torch.no_grad()` wrap may not be effective — debug `VLAAdapterGemma4.forward`.

- [ ] **Step 4: Commit**

```bash
git add scripts/gemma4/test_13_dual_track_smoke.py
git commit -m "$(cat <<'EOF'
feat(dual-track): end-to-end smoke test for Mode A / Mode B

single-GPU 1 batch forward+backward を両 mode で走らせ、
- loss finite (NaN なし)
- speed mode: action_queries.grad is None (frozen 確認)
- quality mode: action_queries.grad is populated (trainable 確認)
- 両 mode の peak memory / samples/sec を比較 print
を実施。Phase 2 並行 pretrain 開始前のゲート。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Multi-GPU DDP launch scripts

**Files:**
- Create: `scripts/gemma4/launch_pretrain_quality.sh`
- Create: `scripts/gemma4/launch_pretrain_speed.sh`

**What and why:** Spec §9.8. 8 GPUs available (2× A100 80GB + 6× A100 40GB). Mode A on GPUs 0-3 (prefer 80GB for memory-tight config), Mode B on GPUs 4-7.

- [ ] **Step 1: Inspect current DDP launch pattern**

```bash
ls scripts/gemma4/ | grep -i "launch\|ddp\|pretrain" 2>&1
grep -l "torchrun\|accelerate\|torch.distributed" scripts/gemma4/*.py 2>&1 | head
```

Confirm what launcher is used. Assume `torchrun` based on standard PyTorch patterns.

- [ ] **Step 2: Create `scripts/gemma4/launch_pretrain_quality.sh`**

```bash
#!/bin/bash
# Mode A (Quality) pretrain launch — 4-way DDP on GPUs 0-3 (2× 80GB + 2× 40GB)
# Config: config/pretrain_taco_quality.yaml

set -euo pipefail

cd "$(dirname "$0")/../.."   # cd to repo root

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

.venv-gemma4/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    VLA-Adapter/vla-scripts/finetune_gemma4.py \
    --config-file config/pretrain_taco_quality.yaml \
    "$@"
```

Make it executable:
```bash
chmod +x scripts/gemma4/launch_pretrain_quality.sh
```

- [ ] **Step 3: Create `scripts/gemma4/launch_pretrain_speed.sh`**

```bash
#!/bin/bash
# Mode B (Speed) pretrain launch — 4-way DDP on GPUs 4-7 (4× 40GB)
# Config: config/pretrain_taco_speed.yaml

set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

# Different master port to avoid collision with quality launch
.venv-gemma4/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    --master_port=29501 \
    VLA-Adapter/vla-scripts/finetune_gemma4.py \
    --config-file config/pretrain_taco_speed.yaml \
    "$@"
```

```bash
chmod +x scripts/gemma4/launch_pretrain_speed.sh
```

- [ ] **Step 4: Verify scripts are syntactically valid and launcher is present**

```bash
bash -n scripts/gemma4/launch_pretrain_quality.sh
bash -n scripts/gemma4/launch_pretrain_speed.sh
.venv-gemma4/bin/torchrun --help 2>&1 | head -5
```

Expected: no syntax errors, torchrun help prints.

- [ ] **Step 5: Verify `finetune_gemma4.py` supports `--config-file` flag**

If the current `finetune_gemma4.py` uses `draccus` or `simple_parsing` or a custom CLI, the `--config-file` flag may not be supported out-of-the-box. Inspect:

```bash
grep -n "draccus\|simple_parsing\|ArgumentParser\|config_file\|config-file" VLA-Adapter/vla-scripts/finetune_gemma4.py | head -20
```

If `--config-file` is not supported, add a small CLI wrapper at the top of `main()`:

```python
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None)
    args, _ = parser.parse_known_args()
    if args.config_file is not None:
        import yaml
        with open(args.config_file) as f:
            yaml_cfg = yaml.safe_load(f)
        # Override FinetuneConfig defaults with yaml values
        cfg = FinetuneConfig(**yaml_cfg)
    else:
        # fall back to existing CLI
        cfg = FinetuneConfig()   # or existing parser
```

Adapt this to match the existing config mechanism. If draccus is used, follow its `draccus.parse(FinetuneConfig, config_path=args.config_file)` idiom.

- [ ] **Step 6: Commit**

```bash
git add scripts/gemma4/launch_pretrain_quality.sh scripts/gemma4/launch_pretrain_speed.sh \
        VLA-Adapter/vla-scripts/finetune_gemma4.py
git commit -m "$(cat <<'EOF'
feat(dual-track): DDP launch scripts for Mode A / Mode B parallel pretrain

launch_pretrain_quality.sh : GPUs 0-3 (80GB×2 + 40GB×2) でのMode A 4-way DDP
launch_pretrain_speed.sh   : GPUs 4-7 (40GB×4) での Mode B 4-way DDP
  → 両 mode を同時並行 pretrain 可能
(master_port をずらして collision 回避。)
finetune_gemma4.py に --config-file flag 対応を追加。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 17: Trainable param groups + optimizer setup for dual-track

**Files:**
- Modify: `VLA-Adapter/vla-scripts/finetune_gemma4.py` (`create_param_groups` or equivalent)

**What and why:** Spec §6.1/6.2. The existing param group setup (for Stage 3 pretrain) expects specific named groups. With LoRA (Mode A) and frozen action_queries (Mode B), we need to ensure the optimizer picks up the right params with the right LRs.

- [ ] **Step 1: Inspect existing param group logic**

```bash
grep -n "param_groups\|learning_coef\|AdamW\|torch.optim" VLA-Adapter/vla-scripts/finetune_gemma4.py | head -30
```

Locate the function that builds optimizer param groups. Read it to understand current group names and LR ratios.

- [ ] **Step 2: Adapt param groups for both modes**

In the param group builder, add dual-track-aware logic:

```python
def build_param_groups(model, cfg):
    """Build optimizer param groups for Mode A or Mode B."""
    base_lr = cfg.learning_rate
    coef = cfg.coef

    # Common: action_head, proprio_projector, wrist_encoder get base LR
    # soft_prompt_library gets base * coef (X-VLA convention)
    groups = []

    # wrist_encoder (new)
    wrist_params = [p for p in model.wrist_encoder.parameters() if p.requires_grad]
    if wrist_params:
        groups.append({"name": "wrist_encoder", "params": wrist_params, "lr": base_lr, "weight_decay": cfg.weight_decay})

    # proprio_projector
    pp_params = [p for p in model.proprio_projector.parameters() if p.requires_grad]
    groups.append({"name": "proprio_projector", "params": pp_params, "lr": base_lr, "weight_decay": cfg.weight_decay})

    # action_head
    ah_params = [p for p in model.action_head.parameters() if p.requires_grad]
    groups.append({"name": "action_head", "params": ah_params, "lr": base_lr, "weight_decay": cfg.weight_decay})

    # action_queries (Mode A: trainable; Mode B: frozen so empty list)
    aq_params = [p for p in [model.action_queries.weight] if p.requires_grad]
    if aq_params:
        groups.append({"name": "action_queries", "params": aq_params, "lr": base_lr, "weight_decay": cfg.weight_decay})

    # soft_prompt_library (both modes)
    if model.soft_prompt_library is not None:
        sp_params = list(model.soft_prompt_library.parameters())
        groups.append({"name": "soft_prompt_library", "params": sp_params, "lr": base_lr * coef, "weight_decay": cfg.weight_decay})

    # LoRA (Mode A only)
    if cfg.training_mode == "quality":
        lora_params = [p for n, p in model.llm.model.language_model.named_parameters()
                       if p.requires_grad and ("lora_" in n or "LoRA" in n)]
        if lora_params:
            groups.append({"name": "lora", "params": lora_params, "lr": base_lr, "weight_decay": 0.0})

    total = sum(sum(p.numel() for p in g["params"]) for g in groups) / 1e6
    for g in groups:
        n = sum(p.numel() for p in g["params"]) / 1e6
        print(f"[param-groups] {g['name']}: {n:.3f} M, lr={g['lr']}")
    print(f"[param-groups] total trainable across groups: {total:.3f} M")
    return groups
```

- [ ] **Step 3: Wire into optimizer construction**

Replace any existing `AdamW(model.parameters(), ...)` call with:

```python
    param_groups = build_param_groups(model_vla, cfg)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
```

- [ ] **Step 4: Smoke verify**

Run the dual-track smoke (Task 15) again, but this time with the full param group setup. Loss should still be finite, and the log should print all expected groups.

- [ ] **Step 5: Commit**

```bash
git add VLA-Adapter/vla-scripts/finetune_gemma4.py
git commit -m "$(cat <<'EOF'
feat(dual-track): optimizer param groups を Mode A / Mode B 両対応化

group list を動的に構築:
- wrist_encoder, proprio_projector, action_head: 常時 (base LR)
- action_queries: Mode A のみ (Mode B は frozen で自動除外)
- soft_prompt_library: 常時 (base × coef)
- lora: Mode A のみ (language_model 配下の lora_* param を収集)

param count を起動時に log、visibility 向上。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (Run by the Implementing Engineer Before Executing)

After completing each task's tests, run this checklist before moving to the next task:

- [ ] Relevant test scripts pass (`scripts/gemma4/test_*.py`)
- [ ] `git status` shows a clean tree (all changes committed)
- [ ] New file paths match the plan (no typos)
- [ ] Commit message includes the `Co-Authored-By: Claude Opus 4.7 (1M context)` trailer
- [ ] No TODO/placeholder/TBD left in committed code

After completing **all** tasks, run:
- [ ] `scripts/gemma4/test_13_dual_track_smoke.py` passes for both modes
- [ ] `scripts/stage3/taco_solo_loader.py --verify` yields valid samples
- [ ] `bash -n scripts/gemma4/launch_pretrain_quality.sh && bash -n scripts/gemma4/launch_pretrain_speed.sh` (syntax OK)
- [ ] Update `docs/troubleshooting.md` with any issues encountered and their resolutions (per CLAUDE.md rules)

## Out of Scope for This Plan

The following are explicitly deferred (not plan tasks, but listed for clarity):

- **Phase 2 full pretrain execution** — running the two parallel pretrains on 8 GPUs is operational, not coding. Monitor via wandb per spec §6.3.
- **LIBERO fine-tune** — done in Week 3 after pretrain checkpoints exist, using existing `Gemma4RLDSDataset` (adapted for new forward signature — one mechanical port, ~2 hours work).
- **LIBERO eval + deliverable selection** — Week 4, per spec §6.5.
- **Troubleshooting** — any issues will be logged per CLAUDE.md rules to `docs/troubleshooting.md`.
- **README.md updates** — repo has no README.md at root; CLAUDE.md rule doesn't apply here. If one is created, update Project Structure / Results per CLAUDE.md rules.

---

**Plan word count:** ~8,000 words across 17 tasks. Expected total engineering effort: 2–3 days of focused coding, followed by 2–3 weeks of compute for pretrain + fine-tune + eval.
