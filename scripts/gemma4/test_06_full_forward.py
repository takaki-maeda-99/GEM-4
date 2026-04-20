"""
Phase 1b.6: 統合 VLAAdapterGemma4 + 機械的 frozen verify

目的:
  - VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py の VLAAdapterGemma4 を import
  - 実 DINO+SigLIP を weight download 込みでロード (Step 0)
  - LLM 完全凍結 + vision backbone 凍結の機械的 verify (Plan v5.2 の Check-in #5)
  - Total trainable が 600-750M (Plan v5.2 → v5.3 の修正値)
  - Full forward + backward + 全 trainable module grad > 0
  - `predicted.shape == (B, 8, 7)`
  - Prompt dependency test (1b.4 から維持されているか最終確認、User 提案)
  - Peak memory < 20 GB (User 予測目安、実測して 1c 着手判断に使用)

Run: CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_06_full_forward.py
"""
import json
import os
import sys
import time
from pathlib import Path

# TF の verbose log 抑制 (prismatic import で TF 初期化が走るため)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
sys.path.insert(0, str(VLA_ROOT))

from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4  # noqa: E402
from prismatic.vla.constants_gemma4 import (  # noqa: E402
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTION_TOKENS,
    NUM_VISION_TOKENS,
    PROPRIO_PLACEHOLDER_IDX,
    VISION_PLACEHOLDER_BEGIN_IDX,
)
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone  # noqa: E402


MODEL_ID = "google/gemma-4-E2B"
VISION_BACKBONE_ID = "dinosiglip-vit-so-224px"


def build_input_ids(tok, prompt, device):
    prompt_ids_full = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_ids = prompt_ids_full[:, :50]
    prompt_len = prompt_ids.shape[1]

    bos = torch.tensor([[tok.bos_token_id]], dtype=torch.long, device=device)
    eos = torch.tensor([[tok.eos_token_id]], dtype=torch.long, device=device)
    vision_ids = torch.arange(
        VISION_PLACEHOLDER_BEGIN_IDX,
        VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS,
        device=device,
    ).unsqueeze(0)
    proprio_id = torch.tensor([[PROPRIO_PLACEHOLDER_IDX]], dtype=torch.long, device=device)
    action_ids = torch.arange(
        ACTION_TOKEN_BEGIN_IDX,
        ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS,
        device=device,
    ).unsqueeze(0)

    input_ids = torch.cat([bos, prompt_ids, vision_ids, proprio_id, action_ids, eos], dim=1)
    return input_ids, prompt_len


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # =================================================================
    # Step 0: DINO+SigLIP weight download (最初に済ませる、User 提案 (a))
    # =================================================================
    print("\n=== Step 0: Downloading DINO+SigLIP weights ===")
    t0 = time.time()
    vision_backbone = DinoSigLIPViTBackbone(
        vision_backbone_id=VISION_BACKBONE_ID,
        image_resize_strategy="resize-naive",
        default_image_size=224,
        image_sequence_len=2,           # LIBERO: agentview + wrist
    )
    dl_sec = time.time() - t0
    vision_backbone = vision_backbone.to(device, dtype=torch.bfloat16).eval()
    print(f"Vision backbone loaded in {dl_sec:.1f}s")
    print(f"  DINO params:   {sum(p.numel() for p in vision_backbone.dino_featurizer.parameters()) / 1e6:.1f}M")
    print(f"  SigLIP params: {sum(p.numel() for p in vision_backbone.siglip_featurizer.parameters()) / 1e6:.1f}M")
    print(f"  embed_dim (concat):  {vision_backbone.embed_dim}  (DINO + SigLIP)")
    print(f"  num_patches (total): {vision_backbone.num_patches}  (2 cam × per-cam)")
    assert vision_backbone.num_patches == NUM_VISION_TOKENS, \
        f"num_patches {vision_backbone.num_patches} != NUM_VISION_TOKENS {NUM_VISION_TOKENS}"

    # =================================================================
    # Gemma 4 load (1b.1-1b.5 と同条件)
    # =================================================================
    print("\n=== Loading Gemma 4 (bf16, sdpa) ===")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    gemma = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    gemma.config.use_cache = True
    for p in gemma.parameters():
        p.requires_grad = False

    # =================================================================
    # 統合クラス 構築
    # =================================================================
    print("\n=== Building VLAAdapterGemma4 ===")
    model_vla = VLAAdapterGemma4(
        gemma_model=gemma,
        vision_backbone=vision_backbone,
        feature_norm=torch.nn.Identity(),   # 1b.1 判定
        proprio_dim=8,
        action_dim=7,
        num_action_chunks=8,
    ).to(device, dtype=torch.bfloat16)

    # =================================================================
    # 凍結 + trainable 機械 verify (Check-in Point #5 の核)
    # =================================================================
    print("\n=== LLM / Vision backbone 凍結 verify ===")
    llm_trainable = sum(p.numel() for p in model_vla.llm.parameters() if p.requires_grad)
    vb_trainable = sum(p.numel() for p in model_vla.vision_backbone.parameters() if p.requires_grad)
    print(f"LLM trainable:            {llm_trainable} (expected 0)")
    print(f"Vision backbone trainable: {vb_trainable} (expected 0)")
    assert llm_trainable == 0, f"LLM should be frozen, got {llm_trainable}"
    assert vb_trainable == 0, f"Vision backbone should be frozen, got {vb_trainable}"

    total_trainable = sum(p.numel() for p in model_vla.parameters() if p.requires_grad)
    print(f"Total trainable:          {total_trainable / 1e6:.2f} M")

    # v5.3 修正値 (1b.5 で Pro action head ~640M が判明したため、218M → 675M ベース)
    assert 600e6 < total_trainable < 750e6, \
        f"Unexpected trainable: {total_trainable/1e6:.1f}M (expected ~675M for Pro action head)"
    print("Trainable range check: PASS")

    # trainable 内訳
    breakdown = {
        "action_head": sum(p.numel() for p in model_vla.action_head.parameters() if p.requires_grad),
        "vision_projector": sum(p.numel() for p in model_vla.vision_projector.parameters() if p.requires_grad),
        "proprio_projector": sum(p.numel() for p in model_vla.proprio_projector.parameters() if p.requires_grad),
        "action_queries": sum(p.numel() for p in model_vla.action_queries.parameters() if p.requires_grad),
    }
    for k, v in breakdown.items():
        print(f"  {k:20s}: {v/1e6:>6.2f} M")

    # =================================================================
    # Dummy input 構築
    # =================================================================
    B = 1
    prompt = "pick up the red cube and place it in the blue tray"
    input_ids, prompt_len = build_input_ids(tok, prompt, device)
    L = input_ids.shape[1]
    print(f"\ninput_ids.shape = {tuple(input_ids.shape)}  (prompt_len={prompt_len})")

    # pixel_values: dict {"dino": (B, T=2, 3, 224, 224), "siglip": (B, T=2, 3, 224, 224)}
    pixel_values = {
        "dino": torch.randn(B, 2, 3, 224, 224, dtype=torch.bfloat16, device=device),
        "siglip": torch.randn(B, 2, 3, 224, 224, dtype=torch.bfloat16, device=device),
    }
    proprio = torch.randn(B, 8, dtype=torch.bfloat16, device=device)
    actions_target = torch.randn(B, 8, 7, dtype=torch.bfloat16, device=device)

    # =================================================================
    # Forward
    # =================================================================
    print("\n=== Full forward ===")
    model_vla.train()      # 学習 mode (action_head は Training phase で learnable perturbation を使う)
    torch.cuda.reset_peak_memory_stats(0)
    predicted, loss = model_vla(pixel_values, input_ids, proprio, actions_target)
    fwd_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"predicted.shape = {tuple(predicted.shape)}  (expected (B, 8, 7))")
    print(f"L1 loss value   = {loss.item():.4f}")
    print(f"Forward peak: {fwd_peak_gb:.2f} GB")
    assert predicted.shape == (B, 8, 7)
    assert torch.isfinite(loss).item()
    # User 予想 0.3-2.0 (1b.5 で 0.6445)
    assert 0.1 < loss.item() < 5.0, f"loss out of expected range: {loss.item()}"

    # =================================================================
    # Backward + 全 trainable module grad verify
    # =================================================================
    print("\n=== Backward + Gradient verification ===")
    torch.cuda.reset_peak_memory_stats(0)
    loss.backward()
    bwd_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3

    grad_checks = [
        ("action_queries.weight", model_vla.action_queries.weight.grad),
        ("vision_projector.fc1.weight", model_vla.vision_projector.fc1.weight.grad),
        ("vision_projector.fc3.weight", model_vla.vision_projector.fc3.weight.grad),
        ("proprio_projector.fc1.weight", model_vla.proprio_projector.fc1.weight.grad),
        ("action_head.model.fc1.weight", model_vla.action_head.model.fc1.weight.grad),
        ("action_head.model.fc2.weight", model_vla.action_head.model.fc2.weight.grad),
    ]
    grad_summary = {}
    for name, g in grad_checks:
        assert g is not None, f"{name}.grad is None"
        val = g.abs().sum().item()
        print(f"  {name:40s} grad.abs().sum() = {val:.4e}")
        assert val > 0, f"{name} grad is zero"
        grad_summary[name] = val

    # LLM / vision backbone leak check
    llm_leak = sum(
        1 for p in model_vla.llm.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    vb_leak = sum(
        1 for p in model_vla.vision_backbone.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    print(f"  LLM grad leak:            {llm_leak} (expected 0)")
    print(f"  Vision backbone leak:     {vb_leak} (expected 0)")
    assert llm_leak == 0
    assert vb_leak == 0

    print(f"\nBackward peak: {bwd_peak_gb:.2f} GB")
    print(f"Backward delta: {bwd_peak_gb - fwd_peak_gb:+.2f} GB")

    # =================================================================
    # Prompt dependency test (User 追加、1b.4 から維持されているか最終確認)
    # =================================================================
    print("\n=== Prompt dependency test (language → last action path) ===")
    model_vla.eval()
    pv_fixed = {
        "dino": torch.zeros(B, 2, 3, 224, 224, dtype=torch.bfloat16, device=device),
        "siglip": torch.zeros(B, 2, 3, 224, 224, dtype=torch.bfloat16, device=device),
    }
    prompt_a, prompt_b = "pick up the red cube", "pick up the blue cube"
    input_ids_a, _ = build_input_ids(tok, prompt_a, device)
    input_ids_b, _ = build_input_ids(tok, prompt_b, device)

    # forward で hidden を取り出せるようにするため、vla の forward で last_hidden_state を保存する簡易版:
    # 直接 llm を呼んで比較
    llm_inner = model_vla.text_model

    def get_last_action_hidden(input_ids_x, pv_x):
        vision_features = model_vla.vision_backbone(pv_x)
        vision_projected = model_vla.vision_projector(vision_features)
        with torch.no_grad():
            per_layer_inputs = llm_inner.get_per_layer_inputs(input_ids_x, None)
            raw_embeddings = llm_inner.embed_tokens(input_ids_x)
        embeddings = raw_embeddings.clone()
        amask = (input_ids_x >= ACTION_TOKEN_BEGIN_IDX) & (
            input_ids_x < ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS
        )
        vmask = (input_ids_x >= VISION_PLACEHOLDER_BEGIN_IDX) & (
            input_ids_x < VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS
        )
        for b in range(input_ids_x.shape[0]):
            apos = amask[b].nonzero(as_tuple=True)[0]
            vpos = vmask[b].nonzero(as_tuple=True)[0]
            embeddings[b, apos] = model_vla.action_queries.weight
            embeddings[b, vpos] = vision_projected[b]
        attn = torch.ones_like(input_ids_x, dtype=torch.long)
        pos = torch.arange(input_ids_x.shape[1], dtype=torch.long, device=device).unsqueeze(0).expand(
            input_ids_x.shape[0], -1)
        out_x = llm_inner(
            inputs_embeds=embeddings,
            per_layer_inputs=per_layer_inputs,
            use_cache=True,
            output_hidden_states=False,
            attention_mask=attn,
            position_ids=pos,
        )
        apos0 = amask[0].nonzero(as_tuple=True)[0]
        return out_x.last_hidden_state[0, apos0[-1]].float()

    with torch.no_grad():
        h_a = get_last_action_hidden(input_ids_a, pv_fixed)
        h_b = get_last_action_hidden(input_ids_b, pv_fixed)
    prompt_diff_max = (h_a - h_b).abs().max().item()
    prompt_diff_mean = (h_a - h_b).abs().mean().item()
    print(f"|h_a - h_b|_max  = {prompt_diff_max:.4e}")
    print(f"|h_a - h_b|_mean = {prompt_diff_mean:.4e}")
    assert not torch.allclose(h_a, h_b, atol=1e-3), \
        "last action hidden does not depend on prompt — 統合後に言語 grounding が壊れた"

    # =================================================================
    # Exit criteria (peak memory)
    # =================================================================
    print("\n=== Exit criteria ===")
    soft_limit = 20.0
    hard_limit = 25.0
    print(f"Forward peak:  {fwd_peak_gb:.2f} GB  (soft < {soft_limit}, hard < {hard_limit})")
    print(f"Backward peak: {bwd_peak_gb:.2f} GB  (soft < {soft_limit}, hard < {hard_limit})")
    if bwd_peak_gb > soft_limit:
        print(f"  WARN: backward peak exceeds soft limit {soft_limit} GB")
    assert bwd_peak_gb < hard_limit, f"Backward peak {bwd_peak_gb:.2f} exceeds hard limit {hard_limit}"

    # =================================================================
    # JSON dump
    # =================================================================
    summary = {
        "phase": "1b.6",
        "model_id": MODEL_ID,
        "vision_backbone_id": VISION_BACKBONE_ID,
        "vision_backbone_download_sec": round(dl_sec, 2),
        "B": B,
        "L": L,
        "prompt_len": prompt_len,
        "vision_embed_dim": vision_backbone.embed_dim,
        "num_vision_patches": vision_backbone.num_patches,
        "llm_trainable": llm_trainable,
        "vision_backbone_trainable": vb_trainable,
        "total_trainable_M": round(total_trainable / 1e6, 3),
        "breakdown_M": {k: round(v / 1e6, 3) for k, v in breakdown.items()},
        "forward_peak_gb": round(fwd_peak_gb, 3),
        "backward_peak_gb": round(bwd_peak_gb, 3),
        "backward_delta_gb": round(bwd_peak_gb - fwd_peak_gb, 3),
        "predicted_shape": list(predicted.shape),
        "loss": float(loss.item()),
        "grad_abs_sums": grad_summary,
        "llm_leak": llm_leak,
        "vb_leak": vb_leak,
        "prompt_diff_max": prompt_diff_max,
        "prompt_diff_mean": prompt_diff_mean,
    }
    json_path = Path(__file__).resolve().parent / "test_06_result.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {json_path}")
    print("\n=== Phase 1b.6 OK: all Check-in Point #5 assertions pass ===")
    return summary


if __name__ == "__main__":
    main()
