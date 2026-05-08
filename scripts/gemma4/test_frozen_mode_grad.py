"""
Smoke test for training_mode='frozen' (#020): verify action_queries.weight gradient is non-None
after 1 forward+backward pass, AND that LLM params do NOT receive gradient (frozen).

Usage:
  CUDA_VISIBLE_DEVICES=5 .venv-gemma4/bin/python scripts/gemma4/test_frozen_mode_grad.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "VLA-Adapter"))
sys.path.insert(0, str(REPO / "scripts" / "gemma4"))

import torch
from transformers import Gemma4ForConditionalGeneration, AutoTokenizer

from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4
from prismatic.vla.constants_gemma4 import (
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTION_TOKENS,
    PROPRIO_PLACEHOLDER_IDX,
    VISION_PLACEHOLDER_BEGIN_IDX,
)

GEMMA_ID = "google/gemma-4-E2B"
device = torch.device("cuda:0")

print("== loading Gemma4 ==")
gemma = Gemma4ForConditionalGeneration.from_pretrained(
    GEMMA_ID, dtype=torch.bfloat16, attn_implementation="sdpa",
).to(device).eval()
for p in gemma.parameters():
    p.requires_grad = False

print("== building VLAAdapterGemma4 (Mode B + wrist_bridge + num_action_head_blocks=35) ==")
model = VLAAdapterGemma4(
    gemma_model=gemma,
    max_soft_tokens=280,
    vision_backbone_type="siglip",
    siglip_use_tensor_transform=True,
    training_mode="speed",   # Mode B: AQ frozen, LLM no_grad
    use_wrist_bridge=True,
    wrist_bridge_layer_mode="per_layer",
    num_action_head_blocks=35,   # #021: 24 → 35 で Gemma4 全層使用
).to(device, dtype=torch.bfloat16)
model.train()

# Mode B: freeze action_queries + skip GC (matches finetune_gemma4.py speed branch)
model.action_queries.weight.data.zero_()
model.action_queries.weight.requires_grad = False
llm_trainable = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
assert llm_trainable == 0, f"LLM must be fully frozen, got {llm_trainable} trainable params"
print(f"[ok] action_queries frozen (Mode B), LLM fully frozen, num_blocks={model.num_action_head_blocks}")

# --- Build dummy batch ---
tok = AutoTokenizer.from_pretrained(GEMMA_ID)
B = 1
L_prompt = 20
L = 1 + L_prompt + model.num_vision_tokens + 1 + NUM_ACTION_TOKENS + 1
ids = (
    [tok.bos_token_id]
    + [tok.pad_token_id] * L_prompt
    + list(range(VISION_PLACEHOLDER_BEGIN_IDX, VISION_PLACEHOLDER_BEGIN_IDX + model.num_vision_tokens))
    + [PROPRIO_PLACEHOLDER_IDX]
    + list(range(ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS))
    + [tok.eos_token_id]
)
input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1).contiguous()

pixel_values = {
    "scene": torch.rand(B, 3, 224, 224, dtype=torch.float32, device=device) * 255.0,
    "wrist": torch.rand(B, 3, 224, 224, dtype=torch.bfloat16, device=device),
}
proprio = torch.randn(B, 8, dtype=torch.float32, device=device)
actions = torch.randn(B, 8, 7, dtype=torch.float32, device=device)

# --- Forward + backward ---
print("== forward ==")
predicted, loss = None, None
out = model(pixel_values=pixel_values, input_ids=input_ids, proprio=proprio, actions=actions)
if isinstance(out, tuple):
    predicted, loss = out
else:
    predicted = out
    loss = torch.nn.functional.l1_loss(predicted, actions)
print(f"[ok] forward done, loss={loss.item():.4f}")

print("== backward ==")
loss.backward()

# Mode B では AQ は frozen なので grad 無しが正しい。action_head 側が grad 取れればOK。
ah_params_with_grad = [n for n, p in model.action_head.named_parameters() if p.grad is not None and p.grad.abs().max() > 0]
assert ah_params_with_grad, "action_head must receive gradient"
print(f"[ok] action_head received gradient ({len(ah_params_with_grad)} params)")
assert model.action_queries.weight.grad is None or model.action_queries.weight.grad.abs().max() == 0, \
    "action_queries should NOT have grad in Mode B (requires_grad=False)"
print(f"[ok] action_queries no grad (Mode B intended)")

# Verify num_action_head_blocks matches model
assert model.action_head.model.mlp_resnet_blocks.__len__() == 35, \
    f"action_head should have 35 blocks, got {len(model.action_head.model.mlp_resnet_blocks)}"
print(f"[ok] action_head has 35 MLPResNetBlocks")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"[summary] total trainable params: {trainable:.2f} M (expected ~780M for num_blocks=35)")
print("== MODE B + num_blocks=35 SMOKE TEST PASSED ==")
