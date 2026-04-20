"""
Phase 1d.b: 単 step (×2) 学習 — AdamW step + LLM leak check + opt_first/opt_steady 測定

目的:
  - 1d.a の data pipeline (Gemma4BatchTransform + Gemma4RLDSDataset) を維持
  - AdamW (lr=2e-4, weight_decay=0.01) で 2 step 連続実行
  - 各 step で: zero_grad(set_to_none=True) → forward → loss → backward → step()
  - fwd / bwd / opt_first / opt_steady の peak memory を記録
  - 1c B=8 の opt ≈16.4 GB との比較、乖離があれば切り分け

成功条件 (1d.b Exit Criteria):
  - 2 step 連続で loss finite
  - LLM / vision backbone grad leak = 0 (各 step で確認)
  - trainable params = 675.138M ±0.1
  - Loss が発散しない (step 2 が step 1 の 2 倍以上にならない)
  - opt_first / opt_steady が 1c B=8 の基準値 ±2 GB

Run: CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_09_single_step.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
SCRIPTS_GEMMA4 = Path(__file__).resolve().parent
sys.path.insert(0, str(VLA_ROOT))
sys.path.insert(0, str(SCRIPTS_GEMMA4))

# 1d.a の data pipeline 要素を再利用 (重複実装を避ける)
from test_08_data_pipeline import (  # noqa: E402
    Gemma4BatchTransform,
    Gemma4RLDSDataset,
    collate_gemma4,
    PROMPT_MAX_LEN,
)
from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4  # noqa: E402
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone  # noqa: E402


MODEL_ID = "google/gemma-4-E2B"
VISION_BACKBONE_ID = "dinosiglip-vit-so-224px"

# 1c B=8 baseline (docs/gemma4_migration_log.md Phase 1c)
BASELINE_1C_B8 = {
    "fwd_gb": 36.82, "bwd_gb": 36.82,
    "opt_first_gb": 16.46, "opt_steady_gb": 16.45,
}

# 1d.a config 継承
DATA_ROOT_DIR = "data/modified_libero_rlds"
DATASET_NAME = "libero_spatial_no_noops"
BATCH_SIZE = 8
NUM_STEPS = 2

# Optimizer (1d.c と同じ hyperparams、1c で AdamW 使用)
LR = 2e-4
WEIGHT_DECAY = 0.01

# Dims
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8


def move_batch_to_device(batch, device, bf16_dtype=torch.bfloat16):
    pv = {
        "dino":   batch["pixel_values"]["dino"].to(device, dtype=bf16_dtype),
        "siglip": batch["pixel_values"]["siglip"].to(device, dtype=bf16_dtype),
    }
    input_ids = batch["input_ids"].to(device)
    proprio = batch["proprio"].to(device, dtype=bf16_dtype)
    actions = batch["actions"].to(device, dtype=bf16_dtype)
    return pv, input_ids, proprio, actions, batch["languages"]


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    torch.manual_seed(42)

    # --- Vision backbone (1d.a と同じ) ---
    print("\n=== Loading DINO+SigLIP ===")
    t0 = time.time()
    vision_backbone = DinoSigLIPViTBackbone(
        vision_backbone_id=VISION_BACKBONE_ID,
        image_resize_strategy="resize-naive",
        default_image_size=224,
        image_sequence_len=2,
    ).to(device, dtype=torch.bfloat16).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # --- Gemma 4 ---
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

    # --- VLAAdapterGemma4 ---
    print("\n=== Building VLAAdapterGemma4 ===")
    model_vla = VLAAdapterGemma4(
        gemma_model=gemma,
        vision_backbone=vision_backbone,
        feature_norm=torch.nn.Identity(),
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        num_action_chunks=NUM_ACTIONS_CHUNK,
    ).to(device, dtype=torch.bfloat16)
    model_vla.train()

    total_trainable = sum(p.numel() for p in model_vla.parameters() if p.requires_grad)
    total_trainable_M = total_trainable / 1e6
    print(f"Total trainable: {total_trainable_M:.3f} M")
    # 既定値 675.138M (1c/1d.a 実測) ±0.1
    assert abs(total_trainable_M - 675.138) < 0.1, \
        f"trainable regressed: expected 675.138M ±0.1, got {total_trainable_M:.3f}M"

    # LLM / vision backbone 凍結再確認
    llm_trainable = sum(p.numel() for p in model_vla.llm.parameters() if p.requires_grad)
    vb_trainable = sum(p.numel() for p in model_vla.vision_backbone.parameters() if p.requires_grad)
    assert llm_trainable == 0, f"LLM not frozen: {llm_trainable}"
    assert vb_trainable == 0, f"Vision backbone not frozen: {vb_trainable}"

    # --- Data pipeline (1d.a と同じ) ---
    print(f"\n=== Building Gemma4RLDSDataset ({DATA_ROOT_DIR}) ===")
    t0 = time.time()
    batch_transform = Gemma4BatchTransform(
        tokenizer=tok,
        image_transform=vision_backbone.image_transform,
        prompt_max_len=PROMPT_MAX_LEN,
    )
    rlds_dataset = Gemma4RLDSDataset(
        data_root_dir=DATA_ROOT_DIR,
        dataset_name=DATASET_NAME,
        batch_transform=batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=1000,
        train=True,
    )
    loader = DataLoader(
        rlds_dataset,
        batch_size=BATCH_SIZE,
        sampler=None,
        collate_fn=collate_gemma4,
        num_workers=0,
    )
    print(f"  built in {time.time()-t0:.1f}s")

    # --- Optimizer ---
    trainable_params = [p for p in model_vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"\nOptimizer: AdamW(lr={LR}, weight_decay={WEIGHT_DECAY})")

    # =================================================================
    # 2 step 学習
    # =================================================================
    print(f"\n=== {NUM_STEPS}-step training (B={BATCH_SIZE}) ===")
    step_results = []
    data_iter = iter(loader)

    for step in range(NUM_STEPS):
        t_step = time.time()
        # zero_grad (set_to_none=True: grad tensor を完全解放)
        optimizer.zero_grad(set_to_none=True)

        # batch load
        batch = next(data_iter)
        pv, input_ids, proprio, actions, languages = move_batch_to_device(batch, device)
        B, L = input_ids.shape

        # --- Forward ---
        torch.cuda.reset_peak_memory_stats(device)
        predicted, loss = model_vla(pv, input_ids, proprio, actions)
        fwd_peak = torch.cuda.max_memory_allocated(device) / 1024**3

        # Asserts
        assert predicted.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM), \
            f"step {step}: bad predicted shape {predicted.shape}"
        assert loss.dim() == 0, f"step {step}: loss not scalar"
        assert torch.isfinite(loss).item(), f"step {step}: loss not finite: {loss.item()}"
        loss_val = float(loss.item())

        # --- Backward ---
        loss.backward()
        bwd_peak = torch.cuda.max_memory_allocated(device) / 1024**3

        # Grad leak check (R4)
        llm_leak = sum(
            1 for p in model_vla.llm.parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        )
        vb_leak = sum(
            1 for p in model_vla.vision_backbone.parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        )
        assert llm_leak == 0, f"step {step}: LLM grad leak: {llm_leak} params"
        assert vb_leak == 0, f"step {step}: vision backbone grad leak: {vb_leak} params"

        # Grad norm (観察用、clip はしない — 1d.c で導入)
        with torch.no_grad():
            grad_norms = [p.grad.norm().item() for p in trainable_params if p.grad is not None]
            total_grad_norm = sum(g * g for g in grad_norms) ** 0.5

        # --- Optimizer step (AdamW state は step 0 で初回 allocate) ---
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.step()
        opt_peak = torch.cuda.max_memory_allocated(device) / 1024**3
        # step 0 → opt_first, step 1 → opt_steady

        step_sec = time.time() - t_step
        step_results.append({
            "step": step,
            "loss": round(loss_val, 6),
            "fwd_gb": round(fwd_peak, 3),
            "bwd_gb": round(bwd_peak, 3),
            "opt_gb": round(opt_peak, 3),
            "total_grad_norm": round(total_grad_norm, 4),
            "llm_leak": llm_leak,
            "vb_leak": vb_leak,
            "L": L,
            "first_language": languages[0],
            "step_sec": round(step_sec, 3),
        })

        print(f"  [step {step}] loss={loss_val:.4f}  fwd={fwd_peak:.2f}  bwd={bwd_peak:.2f}  "
              f"opt={opt_peak:.2f}  grad_norm={total_grad_norm:.3f}  "
              f"llm_leak={llm_leak}  vb_leak={vb_leak}  ({step_sec:.1f}s)")
        print(f"               language: '{languages[0]}'")

    # =================================================================
    # Loss 発散チェック
    # =================================================================
    loss1 = step_results[0]["loss"]
    loss2 = step_results[1]["loss"]
    loss_ratio = loss2 / loss1 if loss1 > 0 else float("nan")
    print(f"\n=== Loss trajectory ===")
    print(f"  step 0: {loss1:.4f}")
    print(f"  step 1: {loss2:.4f}  (ratio {loss_ratio:.3f})")
    if loss_ratio > 2.0:
        print(f"  !!! WARN: loss doubled (ratio {loss_ratio:.2f}), 発散兆候 !!!")

    # =================================================================
    # 1c B=8 regression compare
    # =================================================================
    print(f"\n=== 1c B=8 回帰確認 ===")
    opt_first = step_results[0]["opt_gb"]
    opt_steady = step_results[1]["opt_gb"]
    fwd0 = step_results[0]["fwd_gb"]
    bwd0 = step_results[0]["bwd_gb"]
    regression = {
        "fwd_delta": round(fwd0 - BASELINE_1C_B8["fwd_gb"], 3),
        "bwd_delta": round(bwd0 - BASELINE_1C_B8["bwd_gb"], 3),
        "opt_first_delta": round(opt_first - BASELINE_1C_B8["opt_first_gb"], 3),
        "opt_steady_delta": round(opt_steady - BASELINE_1C_B8["opt_steady_gb"], 3),
    }
    print(f"  fwd:        measured {fwd0:.2f} GB   baseline {BASELINE_1C_B8['fwd_gb']} GB   delta {regression['fwd_delta']:+.2f}")
    print(f"  bwd:        measured {bwd0:.2f} GB   baseline {BASELINE_1C_B8['bwd_gb']} GB   delta {regression['bwd_delta']:+.2f}")
    print(f"  opt_first:  measured {opt_first:.2f} GB   baseline {BASELINE_1C_B8['opt_first_gb']} GB   delta {regression['opt_first_delta']:+.2f}")
    print(f"  opt_steady: measured {opt_steady:.2f} GB   baseline {BASELINE_1C_B8['opt_steady_gb']} GB   delta {regression['opt_steady_delta']:+.2f}")

    opt_within = abs(regression["opt_first_delta"]) < 2.0 and abs(regression["opt_steady_delta"]) < 2.0
    if opt_within:
        print("  → opt_first/opt_steady within ±2 GB: OK")
    else:
        print(f"  → WARN: opt memory 乖離 > 2 GB、要切り分け")

    # =================================================================
    # JSON dump
    # =================================================================
    summary = {
        "phase": "1d.b",
        "data_root_dir": DATA_ROOT_DIR,
        "dataset_name": DATASET_NAME,
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "optimizer": f"AdamW(lr={LR}, weight_decay={WEIGHT_DECAY})",
        "prompt_max_len": PROMPT_MAX_LEN,
        "total_trainable_M": round(total_trainable_M, 3),
        "baseline_1c_B8": BASELINE_1C_B8,
        "steps": step_results,
        "loss_trajectory": {"step0": loss1, "step1": loss2, "ratio": round(loss_ratio, 4)},
        "regression_vs_1c_B8": regression,
        "opt_within_tol": opt_within,
    }
    json_path = Path(__file__).resolve().parent / "test_09_result.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {json_path}")


if __name__ == "__main__":
    main()
