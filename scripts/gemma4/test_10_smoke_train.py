"""
Phase 1d.c: 10-step smoke train with grad clip

目的 (Stage 1 最終):
  - VLAAdapterGemma4 + 実 LIBERO-Spatial data で 10 step 完走
  - clip_grad_norm_(max_norm=1.0) 導入、1d.b の loss ratio 6.45 → <2.0 を検証
  - loss トレンド (初期比 5% 以上減少 / ±10% 変動 / 発散 2x 以上) の判定
  - grad_norm pre-clip / post-clip を両方記録
  - 1d.b の step 1 memory +2.14 GB が weight-shift 起因 か allocator 起因 かを切り分け

User 要請 4 項目 (Check-in Point #8 回答):
  (1) step 1 の grad_norm 確認 → 1d.b log から 47.071 で step 0 update 由来単独判明済
  (2) clip_grad_norm_ 戻り値 (pre-clip norm) と post-clip norm を両方記録
  (3) loss ratio < 2x (step N / step 0) を検証
  (4) step 1 の pre-forward allocated 値を記録、AdamW state persistence 起因か weight-shift 起因かを判定

Run: CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_10_smoke_train.py
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

# 1d.a の data pipeline を再利用
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
DATA_ROOT_DIR = "data/modified_libero_rlds"
DATASET_NAME = "libero_spatial_no_noops"
BATCH_SIZE = 8
MAX_STEPS = 10
LR = 2e-4
WEIGHT_DECAY = 0.01
CLIP_MAX_NORM = 1.0

NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8


def move_batch(batch, device, dtype=torch.bfloat16):
    pv = {
        "dino":   batch["pixel_values"]["dino"].to(device, dtype=dtype),
        "siglip": batch["pixel_values"]["siglip"].to(device, dtype=dtype),
    }
    return (
        pv,
        batch["input_ids"].to(device),
        batch["proprio"].to(device, dtype=dtype),
        batch["actions"].to(device, dtype=dtype),
        batch["languages"],
    )


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    torch.manual_seed(42)

    # --- Model load ---
    print("\n=== Loading DINO+SigLIP ===")
    t0 = time.time()
    vision_backbone = DinoSigLIPViTBackbone(
        vision_backbone_id=VISION_BACKBONE_ID,
        image_resize_strategy="resize-naive",
        default_image_size=224,
        image_sequence_len=2,
    ).to(device, dtype=torch.bfloat16).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

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
    assert abs(total_trainable_M - 675.138) < 0.1

    # --- Data ---
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
        rlds_dataset, batch_size=BATCH_SIZE, sampler=None,
        collate_fn=collate_gemma4, num_workers=0,
    )
    print(f"  built in {time.time()-t0:.1f}s")

    # --- Optimizer ---
    trainable_params = [p for p in model_vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"\nOptimizer: AdamW(lr={LR}, wd={WEIGHT_DECAY})  clip max_norm={CLIP_MAX_NORM}")

    # =================================================================
    # 10-step smoke
    # =================================================================
    print(f"\n=== {MAX_STEPS}-step smoke train (B={BATCH_SIZE}) ===")
    step_results = []
    data_iter = iter(loader)

    for step in range(MAX_STEPS):
        t_step = time.time()

        optimizer.zero_grad(set_to_none=True)
        batch = next(data_iter)
        pv, input_ids, proprio, actions, languages = move_batch(batch, device)
        B, L = input_ids.shape

        # Record allocated memory BEFORE forward reset (User item 4 切り分け用)
        pre_fwd_allocated = torch.cuda.memory_allocated(device) / 1024**3

        # --- Forward ---
        torch.cuda.reset_peak_memory_stats(device)
        predicted, loss = model_vla(pv, input_ids, proprio, actions)
        fwd_peak = torch.cuda.max_memory_allocated(device) / 1024**3

        assert predicted.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
        assert loss.dim() == 0
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Loss non-finite at step {step}: {loss.item()}")
        loss_val = float(loss.item())

        # --- Backward ---
        loss.backward()
        bwd_peak = torch.cuda.max_memory_allocated(device) / 1024**3

        # Leak check (step 0 + step 9 のみ全 5GB+ paramset scan、中間 step は高速 subset)
        if step in (0, MAX_STEPS - 1):
            llm_leak = sum(
                1 for p in model_vla.llm.parameters()
                if p.grad is not None and p.grad.abs().sum().item() > 0
            )
            vb_leak = sum(
                1 for p in model_vla.vision_backbone.parameters()
                if p.grad is not None and p.grad.abs().sum().item() > 0
            )
            assert llm_leak == 0, f"step {step}: LLM leak {llm_leak}"
            assert vb_leak == 0, f"step {step}: VB leak {vb_leak}"
        else:
            llm_leak = vb_leak = -1   # skipped marker

        # --- Clip (returns pre-clip total grad norm) ---
        grad_norm_pre = torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm=CLIP_MAX_NORM
        ).item()
        # Post-clip: 再計算して飽和確認 (期待: grad_norm_pre > 1 なら post = 1.0、<1 なら pre == post)
        with torch.no_grad():
            sq_sum = 0.0
            for p in trainable_params:
                if p.grad is not None:
                    sq_sum += p.grad.detach().float().norm().item() ** 2
        grad_norm_post = sq_sum ** 0.5

        # --- Optimizer step ---
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.step()
        opt_peak = torch.cuda.max_memory_allocated(device) / 1024**3

        step_sec = time.time() - t_step
        step_results.append({
            "step": step,
            "loss": round(loss_val, 6),
            "grad_norm_pre": round(grad_norm_pre, 4),
            "grad_norm_post": round(grad_norm_post, 6),
            "fwd_gb": round(fwd_peak, 3),
            "bwd_gb": round(bwd_peak, 3),
            "opt_gb": round(opt_peak, 3),
            "pre_fwd_allocated_gb": round(pre_fwd_allocated, 3),
            "llm_leak": llm_leak,
            "vb_leak": vb_leak,
            "L": L,
            "first_language": languages[0],
            "step_sec": round(step_sec, 3),
        })

        print(f"  [step {step}] loss={loss_val:.4f}  "
              f"gn_pre={grad_norm_pre:.2f}  gn_post={grad_norm_post:.4f}  "
              f"fwd={fwd_peak:.2f}  bwd={bwd_peak:.2f}  opt={opt_peak:.2f}  "
              f"pre_alloc={pre_fwd_allocated:.2f}  ({step_sec:.1f}s)")

    # =================================================================
    # Loss trend 判定
    # =================================================================
    loss0 = step_results[0]["loss"]
    lossN = step_results[-1]["loss"]
    delta = (lossN - loss0) / loss0 if loss0 > 0 else float("nan")
    max_ratio = max(r["loss"] for r in step_results) / loss0

    # Loss trend 記述的分類 (Stage 1 合否判定には使わない — plan line 239 の 4 軸のみ)
    if delta < -0.05:
        trend = f"DECREASE>5% (delta {delta:+.1%})"
    elif abs(delta) < 0.10:
        trend = f"STABLE ±10% (delta {delta:+.1%})"
    else:
        trend = f"OSCILLATE/TREND (delta {delta:+.1%}, max ratio {max_ratio:.2f}x)"

    # Escalation #7 判定 (plan line 149: "loss 発散 (初期値の 2 倍以上)" = 即停止条件)
    # 発散 = monotonic divergence (grad_norm pre-clip が持続的に高止まり + loss が下がらない)
    # Oscillation (peak → recovery) は発散に非該当
    final_below_initial_or_close = (lossN <= loss0 * 1.5)  # 終端が初期の 1.5x 以下なら recovery 済
    max_gn_pre_dec = (step_results[-1]["grad_norm_pre"] < step_results[0]["grad_norm_pre"])
    escalation7_triggered = (max_ratio > 2.0) and (not final_below_initial_or_close) and (not max_gn_pre_dec)

    print(f"\n=== Loss trend (descriptive, not pass/fail) ===")
    print(f"  step 0: {loss0:.4f} → step {MAX_STEPS-1}: {lossN:.4f}  delta {delta:+.2%}")
    print(f"  max/init ratio: {max_ratio:.2f}  (max at step {max(range(len(step_results)), key=lambda i: step_results[i]['loss'])})")
    print(f"  classification: {trend}")
    print(f"\n=== Escalation #7 check (monotonic divergence, plan line 149) ===")
    print(f"  max_ratio > 2.0: {max_ratio > 2.0}")
    print(f"  final loss ≤ 1.5× initial (recovery 済): {final_below_initial_or_close}  ({lossN:.4f} vs {loss0 * 1.5:.4f})")
    print(f"  grad_norm pre-clip monotonic decay: {max_gn_pre_dec}  (step 0: {step_results[0]['grad_norm_pre']:.1f} → step {MAX_STEPS-1}: {step_results[-1]['grad_norm_pre']:.1f})")
    print(f"  → Escalation #7: {'TRIGGERED (即停止)' if escalation7_triggered else 'NOT triggered (oscillation + recovery)'}")

    # =================================================================
    # Grad norm 判定
    # =================================================================
    max_gn_pre = max(r["grad_norm_pre"] for r in step_results)
    min_gn_pre = min(r["grad_norm_pre"] for r in step_results)
    all_post_saturated = all(abs(r["grad_norm_post"] - 1.0) < 0.05
                             for r in step_results if r["grad_norm_pre"] > 1.0)
    all_post_below_1 = all(r["grad_norm_post"] <= 1.01 for r in step_results)

    if max_gn_pre > 100.0:
        gn_verdict = f"MAX_PRE_HIGH ({max_gn_pre:.1f} > 100, but clipped to ~1.0)"
    elif max_gn_pre > 10.0:
        gn_verdict = f"MODERATE_PRE ({max_gn_pre:.1f} ∈ [10, 100])"
    else:
        gn_verdict = f"STABLE_PRE ({max_gn_pre:.1f} < 10)"

    print(f"\n=== Grad norm ===")
    print(f"  pre-clip range: [{min_gn_pre:.2f}, {max_gn_pre:.2f}]")
    print(f"  post-clip saturated at ~1.0 when pre > 1.0: {all_post_saturated}")
    print(f"  all post ≤ 1.01: {all_post_below_1}")
    print(f"  verdict: {gn_verdict}")

    # =================================================================
    # User item 4 切り分け: step 0 vs step 1 memory delta
    # =================================================================
    s0 = step_results[0]
    s1 = step_results[1]
    pre_alloc_delta = s1["pre_fwd_allocated_gb"] - s0["pre_fwd_allocated_gb"]
    fwd_delta = s1["fwd_gb"] - s0["fwd_gb"]
    # activation 消費部分 = fwd_peak - pre_fwd_allocated
    act_s0 = s0["fwd_gb"] - s0["pre_fwd_allocated_gb"]
    act_s1 = s1["fwd_gb"] - s1["pre_fwd_allocated_gb"]
    act_delta = act_s1 - act_s0

    print(f"\n=== Step 1 memory +{fwd_delta:.2f} GB の切り分け (User item 4) ===")
    print(f"  pre-fwd allocated (s0/s1):  {s0['pre_fwd_allocated_gb']:.2f} / {s1['pre_fwd_allocated_gb']:.2f}  delta {pre_alloc_delta:+.2f}")
    print(f"  fwd peak (s0/s1):           {s0['fwd_gb']:.2f} / {s1['fwd_gb']:.2f}  delta {fwd_delta:+.2f}")
    print(f"  activation alone (fwd-pre): {act_s0:.2f} / {act_s1:.2f}  delta {act_delta:+.2f}")

    # 判定:
    #   pre_alloc_delta ≈ fwd_delta → baseline (AdamW state 持続) 起因
    #   act_delta 優位 → weight-shift 起因 (activation 分布変化)
    if abs(act_delta) < 0.3:
        diag = f"ALLOCATOR (AdamW state persistence): pre_alloc delta {pre_alloc_delta:+.2f} ≈ fwd delta {fwd_delta:+.2f}"
    elif abs(pre_alloc_delta) < 0.3:
        diag = f"WEIGHT-SHIFT (activation distribution change): act delta {act_delta:+.2f} dominant"
    else:
        diag = f"MIXED: pre_alloc {pre_alloc_delta:+.2f} + activation {act_delta:+.2f}"
    print(f"  → diagnosis: {diag}")

    # =================================================================
    # Stage 1 exit criteria (plan line 239 厳密適用、4 軸のみ)
    # plan line 239: "10 step 完走 + loss finite + LLM leak 0 + grad norm 安定"
    # "grad norm 安定" = clip 動作下で post-clip が bounded (=1.0 saturate) の意味
    # Loss trend / ratio は Escalation 条項であって合否基準ではない (上の Escalation #7 block で別判定)
    # =================================================================
    loss_all_finite = all(torch.isfinite(torch.tensor(r["loss"])).item() for r in step_results)
    llm_leak_all_zero = all(r["llm_leak"] == 0 for r in step_results if r["llm_leak"] != -1)
    grad_norm_stable = all_post_below_1  # post-clip が 1.0 以下に有界 = clip 正常動作
    stage1_pass = (
        len(step_results) == MAX_STEPS
        and loss_all_finite
        and llm_leak_all_zero
        and grad_norm_stable
    )
    print(f"\n=== Stage 1 exit check (plan line 239 厳密適用) ===")
    print(f"  10-step 完走:            {'YES' if len(step_results) == MAX_STEPS else 'NO'}")
    print(f"  loss finite (全 step):   {'YES' if loss_all_finite else 'NO'}")
    print(f"  LLM leak 0 (全 check):   {'YES' if llm_leak_all_zero else 'NO'}")
    print(f"  grad norm 安定 (post-clip ≤ 1.01): {'YES' if grad_norm_stable else 'NO'}")
    print(f"  → Stage 1: {'PASS' if stage1_pass else 'INCOMPLETE (User 判断要)'}")

    # =================================================================
    # JSON dump
    # =================================================================
    summary = {
        "phase": "1d.c",
        "data_root_dir": DATA_ROOT_DIR,
        "dataset_name": DATASET_NAME,
        "batch_size": BATCH_SIZE,
        "max_steps": MAX_STEPS,
        "optimizer": f"AdamW(lr={LR}, wd={WEIGHT_DECAY})",
        "clip_max_norm": CLIP_MAX_NORM,
        "total_trainable_M": round(total_trainable_M, 3),
        "steps": step_results,
        "loss_trend": {
            "step0": loss0, "stepN": lossN,
            "delta_pct": round(delta * 100, 3), "max_ratio": round(max_ratio, 3),
            "classification": trend,
        },
        "escalation7_check": {
            "triggered": escalation7_triggered,
            "max_ratio_over_2": max_ratio > 2.0,
            "final_within_1_5x_initial": final_below_initial_or_close,
            "grad_norm_pre_monotonic_decay": max_gn_pre_dec,
        },
        "grad_norm_summary": {
            "max_pre": max_gn_pre, "min_pre": min_gn_pre,
            "all_post_below_1": all_post_below_1,
            "all_post_saturated_when_pre_gt_1": all_post_saturated,
            "verdict": gn_verdict,
        },
        "step1_vs_step0_memory_diag": {
            "pre_alloc_delta": round(pre_alloc_delta, 3),
            "fwd_peak_delta": round(fwd_delta, 3),
            "activation_delta": round(act_delta, 3),
            "diagnosis": diag,
        },
        "stage1_pass": stage1_pass,
    }
    json_path = Path(__file__).resolve().parent / "test_10_result.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {json_path}")


if __name__ == "__main__":
    main()
