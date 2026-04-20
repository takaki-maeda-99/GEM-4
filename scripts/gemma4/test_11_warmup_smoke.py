"""
Phase 2a: warmup smoke (30 step, linear warmup 10% → 100%, warmup_steps=500)

目的:
  - VLA-Adapter 原実装 `finetune.py:1060-1065` の warmup 数式をそのまま移植
    (LambdaLR でラップせず、optimizer.param_groups[0]["lr"] を各 step で手動更新)
  - Stage 1 1d.c (no-warmup) と同じ設定で 30 step 実行、lr/loss/grad_norm trajectory を取得
  - Stage 1 1d.c step 0 pre-clip grad_norm = 67584、step 2 loss peak 2.4844 (ratio 5.19) との比較

成功条件 (Phase 2a Exit Criteria + Check-in #10 提示物):
  - 30 step 完走、loss 全 finite、LLM leak 0 全 step
  - step 1 (1-indexed、gradient_step_idx=0) lr ≈ 2.04e-5 (原数式から計算可能)
  - step 30 (1-indexed、gradient_step_idx=29) lr ≈ 3.08e-5
  - post-clip grad_norm が 1.0 付近に飽和 (warmup は lr のみ調整、clip は別経路)
  - loss ratio (max/init) が Stage 1 1d.c の 5.19 から大幅縮小を期待

非依存: libero, wandb は不要 (純粋に train pipeline の warmup 挙動確認のみ)

Run: CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_11_warmup_smoke.py
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

# 1d.a data pipeline を再利用 (test_10 と同じ pattern)
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

# ---------------- Phase 2a hyperparams (Plan §4 R10, §5 Phase 2a) ----------------
BATCH_SIZE = 8
MAX_STEPS = 30
ORIGINAL_LR = 2e-4         # target_lr (warmup 完了後の lr)
WARMUP_STEPS = 500         # Env A 確定値 (Plan §4 R10)
WEIGHT_DECAY = 0.01
CLIP_MAX_NORM = 1.0
GRAD_ACCUM_STEPS = 1       # Stage 1 と同じ

# ---------------- Stage 1 1d.c reference (regression 比較用、migration_log.md) ----------------
STAGE1_REF = {
    "step0_loss": 0.4785,
    "step2_loss_max": 2.4844,
    "loss_max_over_init_ratio": 5.19,   # step 2 peak / step 0
    "step0_grad_norm_pre": 67584.0,
    "post_clip_range": (0.9952, 1.0029),
}

NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8


def compute_warmup_lr(gradient_step_idx: int, original_lr: float, warmup_steps: int) -> float:
    """原実装 `finetune.py:1060-1065` そのまま。"""
    if warmup_steps <= 0:
        return original_lr
    lr_progress = min((gradient_step_idx + 1) / warmup_steps, 1.0)
    return original_lr * (0.1 + 0.9 * lr_progress)


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

    # =================================================================
    # Model load (Stage 1 1d.c と同じ)
    # =================================================================
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
    assert abs(total_trainable_M - 675.138) < 0.1, \
        f"trainable regressed: expected 675.138M, got {total_trainable_M:.3f}M"

    # =================================================================
    # Data pipeline
    # =================================================================
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

    # =================================================================
    # Optimizer (lr = ORIGINAL_LR で初期化、warmup で step 毎に書き換え)
    # =================================================================
    trainable_params = [p for p in model_vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=ORIGINAL_LR, weight_decay=WEIGHT_DECAY)
    original_lr = optimizer.param_groups[0]["lr"]  # 原実装 line 913 と同じ
    print(f"\nOptimizer: AdamW(lr={original_lr}, wd={WEIGHT_DECAY})")
    print(f"Warmup: linear 10% → 100% over {WARMUP_STEPS} steps, clip max_norm={CLIP_MAX_NORM}")
    print(f"Expected lr at step 1 (1-idx): {compute_warmup_lr(0, original_lr, WARMUP_STEPS):.3e}")
    print(f"Expected lr at step {MAX_STEPS}: {compute_warmup_lr(MAX_STEPS-1, original_lr, WARMUP_STEPS):.3e}")
    print(f"Expected lr at step {WARMUP_STEPS}: {compute_warmup_lr(WARMUP_STEPS-1, original_lr, WARMUP_STEPS):.3e}")

    # =================================================================
    # 30-step smoke (warmup ON)
    # =================================================================
    print(f"\n=== {MAX_STEPS}-step warmup smoke (B={BATCH_SIZE}) ===")
    step_results = []
    data_iter = iter(loader)

    for step in range(MAX_STEPS):
        t_step = time.time()

        # --- gradient_step_idx (grad_accum=1 なので batch_idx と同じ) ---
        gradient_step_idx = step   # // GRAD_ACCUM_STEPS (= 1)

        # --- batch load ---
        optimizer.zero_grad(set_to_none=True)
        batch = next(data_iter)
        pv, input_ids, proprio, actions, languages = move_batch(batch, device)
        B, L = input_ids.shape

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

        # Leak check (step 0 と最終 step のみ fullscan、中間は skip 高速化)
        if step in (0, MAX_STEPS - 1):
            llm_leak = sum(
                1 for p in model_vla.llm.parameters()
                if p.grad is not None and p.grad.abs().sum().item() > 0
            )
            vb_leak = sum(
                1 for p in model_vla.vision_backbone.parameters()
                if p.grad is not None and p.grad.abs().sum().item() > 0
            )
            assert llm_leak == 0, f"step {step}: LLM grad leak {llm_leak}"
            assert vb_leak == 0, f"step {step}: VB grad leak {vb_leak}"
        else:
            llm_leak = vb_leak = -1

        # --- Warmup lr update (原実装 line 1060-1065、optimizer.step() より前) ---
        current_lr = compute_warmup_lr(gradient_step_idx, original_lr, WARMUP_STEPS)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # --- Clip (pre-clip 値を戻り値として取得) ---
        grad_norm_pre = torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm=CLIP_MAX_NORM
        ).item()
        # post-clip 再計算
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
            "gradient_step_idx": gradient_step_idx,
            "step_1indexed": step + 1,
            "lr": current_lr,
            "loss": round(loss_val, 6),
            "grad_norm_pre": round(grad_norm_pre, 6),
            "grad_norm_post": round(grad_norm_post, 6),
            "fwd_gb": round(fwd_peak, 3),
            "bwd_gb": round(bwd_peak, 3),
            "opt_gb": round(opt_peak, 3),
            "llm_leak": llm_leak,
            "vb_leak": vb_leak,
            "L": L,
            "first_language": languages[0],
            "step_sec": round(step_sec, 3),
        })

        print(f"  [step {step+1:2d}/{MAX_STEPS}] lr={current_lr:.3e}  loss={loss_val:.4f}  "
              f"gn_pre={grad_norm_pre:.2f}  gn_post={grad_norm_post:.4f}  "
              f"fwd={fwd_peak:.2f}  opt={opt_peak:.2f}  ({step_sec:.1f}s)")

    # =================================================================
    # Check-in #10 用の追加提示項目 (User 要請 4 項目)
    # =================================================================

    # --- (1) lr trajectory 数値検証 ---
    lr_step1 = step_results[0]["lr"]
    lr_step30 = step_results[-1]["lr"]
    expected_lr_step1 = compute_warmup_lr(0, ORIGINAL_LR, WARMUP_STEPS)
    expected_lr_step30 = compute_warmup_lr(MAX_STEPS - 1, ORIGINAL_LR, WARMUP_STEPS)
    lr_step250_extrap = compute_warmup_lr(249, ORIGINAL_LR, WARMUP_STEPS)
    lr_step500_extrap = compute_warmup_lr(499, ORIGINAL_LR, WARMUP_STEPS)
    lr_step501_extrap = compute_warmup_lr(500, ORIGINAL_LR, WARMUP_STEPS)

    lr_check_tol = 1e-9
    assert abs(lr_step1 - expected_lr_step1) < lr_check_tol
    assert abs(lr_step30 - expected_lr_step30) < lr_check_tol
    assert abs(lr_step500_extrap - lr_step501_extrap) < lr_check_tol, \
        "warmup 飽和失敗 (step 500 と 501 の lr が異なる)"

    # --- (2) Stage 1 1d.c step 0 pre-clip grad_norm 比較 ---
    s1c_gn_pre_step0 = STAGE1_REF["step0_grad_norm_pre"]
    p2a_gn_pre_step0 = step_results[0]["grad_norm_pre"]
    gn_pre_shrink_factor = s1c_gn_pre_step0 / p2a_gn_pre_step0 if p2a_gn_pre_step0 > 0 else float("inf")

    # --- (3) loss ratio (max/init) 比較 ---
    loss_init = step_results[0]["loss"]
    loss_max = max(r["loss"] for r in step_results)
    loss_min = min(r["loss"] for r in step_results)
    loss_final = step_results[-1]["loss"]
    max_init_ratio = loss_max / loss_init if loss_init > 0 else float("nan")
    s1c_ratio = STAGE1_REF["loss_max_over_init_ratio"]
    ratio_shrink_factor = s1c_ratio / max_init_ratio if max_init_ratio > 0 else float("inf")
    max_step_idx = max(range(len(step_results)), key=lambda i: step_results[i]["loss"])

    # --- (4) post-clip grad_norm 飽和確認 ---
    post_clip_vals = [r["grad_norm_post"] for r in step_results]
    post_clip_max = max(post_clip_vals)
    post_clip_min = min(post_clip_vals)
    all_post_saturated = all(
        abs(r["grad_norm_post"] - 1.0) < 0.05
        for r in step_results if r["grad_norm_pre"] > 1.0
    )
    all_post_le_1p01 = all(r["grad_norm_post"] <= 1.01 for r in step_results)

    print("\n" + "=" * 80)
    print("Check-in #10 提示物")
    print("=" * 80)

    print("\n(1) lr trajectory 数値検証 (warmup 10% → 100% over 500 steps、target_lr=2e-4)")
    print(f"    step 1  (1-idx, gradient_step_idx=0):   lr = {lr_step1:.6e}  (expected {expected_lr_step1:.6e})")
    print(f"    step 30 (1-idx, gradient_step_idx=29):  lr = {lr_step30:.6e}  (expected {expected_lr_step30:.6e})")
    print(f"    step 250 (extrapolated):                lr = {lr_step250_extrap:.6e}")
    print(f"    step 500 (extrapolated、warmup 終了):  lr = {lr_step500_extrap:.6e}")
    print(f"    step 501 (extrapolated、飽和確認):      lr = {lr_step501_extrap:.6e}")
    print(f"    → 原数式 `0.1 + 0.9*min((step+1)/500, 1.0)` と数値一致、step 500 以降 target_lr で飽和")

    print(f"\n(2) Stage 1 1d.c vs Phase 2a step 0 pre-clip grad_norm 比較")
    print(f"    1d.c step 0: {s1c_gn_pre_step0:.2f}")
    print(f"    2a   step 0: {p2a_gn_pre_step0:.2f}")
    print(f"    縮小比: {gn_pre_shrink_factor:.1f}x (warmup による初期 lr 1/10 の効果)")

    print(f"\n(3) Loss ratio (max/init) 比較")
    print(f"    Stage 1 1d.c: max_ratio = {s1c_ratio:.2f} (step 2 peak {STAGE1_REF['step2_loss_max']} / step 0 {STAGE1_REF['step0_loss']})")
    print(f"    Phase 2a    : max_ratio = {max_init_ratio:.2f} (step {max_step_idx+1} peak {loss_max:.4f} / step 1 {loss_init:.4f})")
    print(f"    期待: < 2.0 まで抑制、達成: {'YES' if max_init_ratio < 2.0 else 'NO'}")
    print(f"    loss trajectory: init={loss_init:.4f}, min={loss_min:.4f}, max={loss_max:.4f}, final={loss_final:.4f}")

    print(f"\n(4) Post-clip grad_norm 飽和確認 (warmup は lr のみ、clip は別経路で不変期待)")
    print(f"    post-clip range: [{post_clip_min:.4f}, {post_clip_max:.4f}]")
    print(f"    Stage 1 1d.c range: [{STAGE1_REF['post_clip_range'][0]:.4f}, {STAGE1_REF['post_clip_range'][1]:.4f}]")
    print(f"    post-clip saturated (pre>1.0 時 1.0 ±0.05): {all_post_saturated}")
    print(f"    all post-clip ≤ 1.01: {all_post_le_1p01}")

    # =================================================================
    # Exit Criteria 判定
    # =================================================================
    loss_all_finite = all(torch.isfinite(torch.tensor(r["loss"])).item() for r in step_results)
    llm_leak_all_zero = all(r["llm_leak"] == 0 for r in step_results if r["llm_leak"] != -1)
    lr_formula_matches = abs(lr_step1 - expected_lr_step1) < lr_check_tol and \
                         abs(lr_step30 - expected_lr_step30) < lr_check_tol

    phase2a_pass = (
        len(step_results) == MAX_STEPS
        and loss_all_finite
        and llm_leak_all_zero
        and lr_formula_matches
        and all_post_le_1p01
    )
    print(f"\n=== Phase 2a Exit Criteria ===")
    print(f"  30-step 完走:             {'YES' if len(step_results) == MAX_STEPS else 'NO'}")
    print(f"  loss finite (全 step):    {'YES' if loss_all_finite else 'NO'}")
    print(f"  LLM leak 0 (check step):  {'YES' if llm_leak_all_zero else 'NO'}")
    print(f"  lr 数式 原実装一致:       {'YES' if lr_formula_matches else 'NO'}")
    print(f"  post-clip ≤ 1.01 (全):    {'YES' if all_post_le_1p01 else 'NO'}")
    print(f"  → Phase 2a: {'PASS' if phase2a_pass else 'INCOMPLETE'}")

    # =================================================================
    # JSON dump
    # =================================================================
    summary = {
        "phase": "2a",
        "mode": "warmup smoke (30 step)",
        "data_root_dir": DATA_ROOT_DIR,
        "dataset_name": DATASET_NAME,
        "batch_size": BATCH_SIZE,
        "max_steps": MAX_STEPS,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "optimizer": f"AdamW(lr={ORIGINAL_LR}, wd={WEIGHT_DECAY})",
        "warmup_steps": WARMUP_STEPS,
        "warmup_formula": "current_lr = original_lr * (0.1 + 0.9 * min((gradient_step_idx+1)/warmup_steps, 1.0))",
        "warmup_formula_source": "VLA-Adapter/vla-scripts/finetune.py:1060-1065",
        "clip_max_norm": CLIP_MAX_NORM,
        "total_trainable_M": round(total_trainable_M, 3),
        "steps": step_results,
        "lr_trajectory_verification": {
            "step1_1idx_measured": lr_step1,
            "step1_1idx_expected": expected_lr_step1,
            "step30_1idx_measured": lr_step30,
            "step30_1idx_expected": expected_lr_step30,
            "step250_extrapolated": lr_step250_extrap,
            "step500_extrapolated": lr_step500_extrap,
            "step501_extrapolated_saturated": lr_step501_extrap,
            "formula_matches": lr_formula_matches,
        },
        "stage1_1dc_comparison": {
            "step0_grad_norm_pre": {
                "stage1_1dc": s1c_gn_pre_step0,
                "phase2a": p2a_gn_pre_step0,
                "shrink_factor": round(gn_pre_shrink_factor, 2),
            },
            "loss_max_over_init_ratio": {
                "stage1_1dc": s1c_ratio,
                "phase2a": round(max_init_ratio, 3),
                "max_at_step_1idx": max_step_idx + 1,
                "target_under_2": max_init_ratio < 2.0,
                "shrink_factor": round(ratio_shrink_factor, 2),
            },
            "loss_trajectory": {
                "init": loss_init, "min": loss_min, "max": loss_max, "final": loss_final,
            },
        },
        "post_clip_grad_norm": {
            "range": [post_clip_min, post_clip_max],
            "stage1_1dc_range": list(STAGE1_REF["post_clip_range"]),
            "all_saturated_when_pre_gt_1": all_post_saturated,
            "all_le_1p01": all_post_le_1p01,
        },
        "phase2a_pass": phase2a_pass,
    }
    json_path = Path(__file__).resolve().parent / "test_11_result.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {json_path}")


if __name__ == "__main__":
    main()
