"""
Phase 1c: Backward batch scaling + Optimizer state 測定

目的:
  - VLAAdapterGemma4 を B=1,2,4,8 で forward/backward/optimizer step
  - 各 B の fwd / bwd / opt_first / opt_steady peak memory を測定
  - 1b.6 基準値 (B=1: fwd 14.97 / bwd 15.78 GB, trainable 675.14M) で回帰確認
  - 1d 採用 batch size を数値ベースで決定する根拠を得る

依存:
  - VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py::VLAAdapterGemma4 (Phase 1b.6)
  - test_06 の build_input_ids ロジックを流用 (LIBERO 固定 layout、B 方向は tile)

Run: CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_07_batch_scaling.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from torch.optim import AdamW
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

# 1b.6 baseline (docs/gemma4_migration_log.md Phase 1b.6 の数値)
BASELINE_1B6 = {"fwd_gb": 14.97, "bwd_gb": 15.78, "trainable_M": 675.14}


def build_input_ids(tok, prompt, device):
    """test_06 の build_input_ids と同一 (B=1 出力)."""
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


def build_batched_inputs(tok, prompt, B, device, seed):
    """LIBERO 固定 layout なので B 方向は identical tile. Random は pixel/proprio/actions のみ."""
    gen = torch.Generator(device=device).manual_seed(seed)
    ids_1, plen = build_input_ids(tok, prompt, device)
    input_ids = ids_1.expand(B, -1).contiguous()

    pixel_values = {
        "dino":   torch.randn(B, 2, 3, 224, 224, generator=gen, dtype=torch.bfloat16, device=device),
        "siglip": torch.randn(B, 2, 3, 224, 224, generator=gen, dtype=torch.bfloat16, device=device),
    }
    proprio = torch.randn(B, 8, generator=gen, dtype=torch.bfloat16, device=device)
    actions = torch.randn(B, 8, 7, generator=gen, dtype=torch.bfloat16, device=device)
    return pixel_values, input_ids, proprio, actions, plen


def measure_one_batch(model_vla, optimizer, tok, prompt, B, device):
    """指定 B で fwd/bwd/opt_first/opt_steady の peak を測定."""
    # ---- Step 1: first forward ----
    pv, ids, pro, act, _ = build_batched_inputs(tok, prompt, B, device, seed=100 + B)
    L = ids.shape[1]

    torch.cuda.reset_peak_memory_stats(device)
    predicted, loss = model_vla(pv, ids, pro, act)
    fwd_peak = torch.cuda.max_memory_allocated(device) / 1024**3

    assert predicted.shape == (B, 8, 7), f"B={B} bad predicted shape {predicted.shape}"
    assert loss.dim() == 0, f"B={B} loss is not scalar: shape={loss.shape}"
    assert torch.isfinite(loss).item(), f"B={B} loss not finite: {loss.item()}"
    loss1_val = float(loss.item())

    # ---- Step 2: first backward (peak is cumulative since reset above) ----
    loss.backward()
    bwd_peak = torch.cuda.max_memory_allocated(device) / 1024**3

    # ---- Step 3: first optimizer.step (AdamW state 初回 allocate は B=1 のみ) ----
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.step()
    optimizer.zero_grad()
    opt_first_peak = torch.cuda.max_memory_allocated(device) / 1024**3

    # free step-1 activation tensors
    del predicted, loss, pv, ids, pro, act

    # ---- Step 4: second forward+backward+step (steady state) ----
    pv2, ids2, pro2, act2, _ = build_batched_inputs(tok, prompt, B, device, seed=200 + B)
    predicted2, loss2 = model_vla(pv2, ids2, pro2, act2)
    assert loss2.dim() == 0 and torch.isfinite(loss2).item()
    loss2_val = float(loss2.item())
    loss2.backward()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.step()
    optimizer.zero_grad()
    opt_steady_peak = torch.cuda.max_memory_allocated(device) / 1024**3

    del predicted2, loss2, pv2, ids2, pro2, act2

    return {
        "B": B,
        "L": L,
        "fwd_gb": round(fwd_peak, 3),
        "bwd_gb": round(bwd_peak, 3),
        "opt_first_gb": round(opt_first_peak, 3),
        "opt_steady_gb": round(opt_steady_peak, 3),
        "loss1": round(loss1_val, 6),
        "loss2": round(loss2_val, 6),
    }


def main():
    assert torch.cuda.is_available(), "CUDA unavailable"
    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    torch.manual_seed(42)

    # =================================================================
    # Vision backbone
    # =================================================================
    print("\n=== Loading DINO+SigLIP ===")
    t0 = time.time()
    vision_backbone = DinoSigLIPViTBackbone(
        vision_backbone_id=VISION_BACKBONE_ID,
        image_resize_strategy="resize-naive",
        default_image_size=224,
        image_sequence_len=2,
    ).to(device, dtype=torch.bfloat16).eval()
    print(f"  loaded in {time.time()-t0:.1f}s, embed_dim={vision_backbone.embed_dim}, "
          f"num_patches={vision_backbone.num_patches}")
    assert vision_backbone.num_patches == NUM_VISION_TOKENS

    # =================================================================
    # Gemma 4 (bf16, sdpa, 凍結)
    # =================================================================
    print("\n=== Loading Gemma 4 ===")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    gemma = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    gemma.config.use_cache = True
    for p in gemma.parameters():
        p.requires_grad = False

    # =================================================================
    # VLAAdapterGemma4
    # =================================================================
    print("\n=== Building VLAAdapterGemma4 ===")
    model_vla = VLAAdapterGemma4(
        gemma_model=gemma,
        vision_backbone=vision_backbone,
        feature_norm=torch.nn.Identity(),
        proprio_dim=8,
        action_dim=7,
        num_action_chunks=8,
    ).to(device, dtype=torch.bfloat16)
    model_vla.train()  # 1b.6 と同じ (action_head の Training phase を使うため)

    total_trainable = sum(p.numel() for p in model_vla.parameters() if p.requires_grad)
    total_trainable_M = total_trainable / 1e6
    print(f"Total trainable: {total_trainable_M:.3f} M  (baseline 1b.6: {BASELINE_1B6['trainable_M']} M)")
    assert 670.0 < total_trainable_M < 680.0, \
        f"trainable params regressed (expected ~675.14M, got {total_trainable_M:.3f} M)"

    # LLM 凍結 regression verify
    llm_trainable = sum(p.numel() for p in model_vla.llm.parameters() if p.requires_grad)
    vb_trainable = sum(p.numel() for p in model_vla.vision_backbone.parameters() if p.requires_grad)
    assert llm_trainable == 0, f"LLM should be frozen, got {llm_trainable}"
    assert vb_trainable == 0, f"vision backbone should be frozen, got {vb_trainable}"

    # Optimizer (AdamW, 1d と同 hyperparams)
    trainable_params = [p for p in model_vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=2e-4, weight_decay=0.01)

    # =================================================================
    # Batch scaling sweep
    # =================================================================
    prompt = "pick up the red cube and place it in the blue tray"
    cases = [1, 2, 4, 8]
    results = []
    oom_at = None

    for B in cases:
        print(f"\n=== B={B} ===")
        try:
            r = measure_one_batch(model_vla, optimizer, tok, prompt, B, device)
            results.append(r)
            print(f"  fwd={r['fwd_gb']:.2f} bwd={r['bwd_gb']:.2f} "
                  f"opt_first={r['opt_first_gb']:.2f} opt_steady={r['opt_steady_gb']:.2f}  "
                  f"loss1={r['loss1']:.4f} loss2={r['loss2']:.4f}")
            # Safety: 次の B で 2x 超過予測、70GB 超えたら stop
            if r["bwd_gb"] > 70.0:
                print(f"  ABORT: bwd peak {r['bwd_gb']:.2f} GB > 70 GB, skipping larger B")
                break
        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower() or "CUDA out of memory" in msg:
                print(f"  OOM at B={B}")
                oom_at = B
                results.append({"B": B, "oom": True, "err": msg[:300]})
                # Cleanup for safe exit
                try:
                    optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
                torch.cuda.empty_cache()
                break
            raise
        torch.cuda.empty_cache()

    # =================================================================
    # Table print
    # =================================================================
    print("\n=== Batch scaling table ===")
    header = f"{'B':>3} | {'fwd(GB)':>8} | {'bwd(GB)':>8} | {'opt_1st(GB)':>11} | {'opt_steady(GB)':>14} | {'loss1':>8} | {'loss2':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r.get("oom"):
            print(f"{r['B']:>3} | ---OOM---")
        else:
            print(f"{r['B']:>3} | {r['fwd_gb']:>8.2f} | {r['bwd_gb']:>8.2f} | "
                  f"{r['opt_first_gb']:>11.2f} | {r['opt_steady_gb']:>14.2f} | "
                  f"{r['loss1']:>8.4f} | {r['loss2']:>8.4f}")

    # =================================================================
    # 1b.6 regression compare (B=1)
    # =================================================================
    regression = None
    if results and not results[0].get("oom") and results[0]["B"] == 1:
        b1 = results[0]
        regression = {
            "fwd_delta": round(b1["fwd_gb"] - BASELINE_1B6["fwd_gb"], 3),
            "bwd_delta": round(b1["bwd_gb"] - BASELINE_1B6["bwd_gb"], 3),
        }
        print("\n=== 1b.6 回帰確認 (B=1) ===")
        print(f"  fwd:  measured {b1['fwd_gb']:.2f} GB  baseline {BASELINE_1B6['fwd_gb']} GB  delta {regression['fwd_delta']:+.2f}")
        print(f"  bwd:  measured {b1['bwd_gb']:.2f} GB  baseline {BASELINE_1B6['bwd_gb']} GB  delta {regression['bwd_delta']:+.2f}")
        # Note: opt 測定 (AdamW state) が 1b.6 にはなかったので delta は比較しない

    # =================================================================
    # 1d recommended batch
    # =================================================================
    ok = [r for r in results if not r.get("oom") and r["opt_steady_gb"] < 75.0]
    recommended_B = max((r["B"] for r in ok), default=None)
    print("\n=== 1d 採用候補 ===")
    print(f"  opt_steady < 75 GB の最大 B = {recommended_B}")
    # grad accum 提案
    if recommended_B is not None and recommended_B < 8:
        ga = 8 // recommended_B
        print(f"  (effective batch=8 狙いなら grad_accum={ga})")

    # =================================================================
    # Exit criteria
    # =================================================================
    # Escalation 検知
    escalation = None
    if oom_at == 1:
        escalation = "ESCALATION: B=1 OOM (Phase 1c Escalation #1)"
    elif oom_at == 2:
        escalation = "ESCALATION: B=2 OOM (Phase 1c Escalation #2, 1d 実行不能)"
    if escalation:
        print(f"\n!!! {escalation} !!!")

    # =================================================================
    # JSON dump
    # =================================================================
    summary = {
        "phase": "1c",
        "model_id": MODEL_ID,
        "vision_backbone_id": VISION_BACKBONE_ID,
        "total_trainable_M": round(total_trainable_M, 3),
        "baseline_1b6": BASELINE_1B6,
        "regression_vs_1b6": regression,
        "results": results,
        "oom_at": oom_at,
        "recommended_B": recommended_B,
        "escalation": escalation,
    }
    json_path = Path(__file__).resolve().parent / "test_07_result.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {json_path}")


if __name__ == "__main__":
    main()
