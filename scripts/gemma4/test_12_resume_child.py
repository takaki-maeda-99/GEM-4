"""
test_12_resume_child.py

Phase 2b checkpoint resume 検証の **子 process** (subprocess 経由で finetune_gemma4.py から spawn).

役割:
  1. fresh な VLAAdapterGemma4 を build (model rebuild from pretrained)
  2. parent が保存した checkpoint (.pt) を load (trainable_state_dict + optimizer_state_dict + scheduler + gradient_step_idx + current_lr)
  3. parent が pickle した batch (.pt) を load
  4. 同じ batch で forward → loss_b を計算
  5. checkpoint に保存された state と **load 直後の in-memory state** の diff を計算
     - state_diff_max: trainable_state_dict 全 tensor の (saved - loaded).abs().max()
     - optim_diff_max: optimizer state の exp_avg (m) / exp_avg_sq (v) 全 param の diff max
  6. lr_after_resume, gradient_step_idx_after_resume を回収
  7. 結果を JSON で --output-path に書き出し

Acceptance criteria (parent 側で評価、User 仕様):
  (i)   |loss_a - loss_b| < 1e-3
  (ii)  state_diff_max < 1e-6
  (iii) optim_diff_max < 1e-6
  (iv)  lr_after_resume == current_lr_at_save (bit-exact)

Run (parent から spawn される、通常は手動実行しない):
  .venv-gemma4/bin/python scripts/gemma4/test_12_resume_child.py \\
      --checkpoint-path runs/gemma4/.../smoke_step50_checkpoint.pt \\
      --batch-path runs/gemma4/.../_resume_verify_batch.pt \\
      --output-path runs/gemma4/.../_resume_verify_child_out.json \\
      --gemma-model-id google/gemma-4-E2B \\
      --vision-backbone-id dinosiglip-vit-so-224px
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
sys.path.insert(0, str(VLA_ROOT))

from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4  # noqa: E402
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone  # noqa: E402


def build_model(gemma_model_id: str, vision_backbone_id: str, device: torch.device) -> VLAAdapterGemma4:
    vision_backbone = DinoSigLIPViTBackbone(
        vision_backbone_id=vision_backbone_id,
        image_resize_strategy="resize-naive",
        default_image_size=224,
        image_sequence_len=2,
    ).to(device, dtype=torch.bfloat16).eval()

    gemma = AutoModelForCausalLM.from_pretrained(
        gemma_model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    gemma.config.use_cache = True
    for p in gemma.parameters():
        p.requires_grad = False

    model_vla = VLAAdapterGemma4(
        gemma_model=gemma,
        vision_backbone=vision_backbone,
        feature_norm=torch.nn.Identity(),
        proprio_dim=8,
        action_dim=7,
        num_action_chunks=8,
    ).to(device, dtype=torch.bfloat16)
    model_vla.train()
    return model_vla


def compute_state_diff(saved_state: dict, loaded_state: dict) -> float:
    """trainable_state_dict の全 tensor について (saved - loaded).abs().max() の maximum."""
    max_diff = 0.0
    for k, saved_t in saved_state.items():
        assert k in loaded_state, f"loaded missing key: {k}"
        loaded_t = loaded_state[k]
        if saved_t.shape != loaded_t.shape:
            raise RuntimeError(f"shape mismatch for {k}: {saved_t.shape} vs {loaded_t.shape}")
        diff = (saved_t.detach().float().cpu() - loaded_t.detach().float().cpu()).abs().max().item()
        if diff > max_diff:
            max_diff = diff
    return max_diff


def compute_optim_diff(saved_optim_state: dict, loaded_optim_state: dict) -> float:
    """AdamW の state['state'][param_id]['exp_avg'] / ['exp_avg_sq'] 全てについて diff max."""
    saved_state = saved_optim_state["state"]
    loaded_state = loaded_optim_state["state"]
    max_diff = 0.0
    for pid, saved_entry in saved_state.items():
        assert pid in loaded_state, f"loaded optim missing param id {pid}"
        loaded_entry = loaded_state[pid]
        for tkey in ("exp_avg", "exp_avg_sq"):
            if tkey in saved_entry:
                assert tkey in loaded_entry, f"loaded optim missing {tkey} for pid {pid}"
                s = saved_entry[tkey].detach().float().cpu()
                l = loaded_entry[tkey].detach().float().cpu()
                diff = (s - l).abs().max().item()
                if diff > max_diff:
                    max_diff = diff
    return max_diff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--batch-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--gemma-model-id", type=str, required=True)
    p.add_argument("--vision-backbone-id", type=str, required=True)
    args = p.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    print(f"[resume-child] checkpoint:        {args.checkpoint_path}")
    print(f"[resume-child] batch:             {args.batch_path}")
    print(f"[resume-child] gemma_model_id:    {args.gemma_model_id}")

    # --- Load saved payload (parent が書いたもの) ---
    saved_payload = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    saved_trainable_state = saved_payload["trainable_state_dict"]
    saved_optim_state = saved_payload["optimizer_state_dict"]
    saved_scheduler_state = saved_payload["scheduler_state_dict"]
    gradient_step_idx_at_save = saved_payload["gradient_step_idx"]
    current_lr_at_save = saved_payload["current_lr"]

    # --- Build fresh model + optimizer + scheduler ---
    print("[resume-child] Building fresh model from pretrained...")
    model_vla = build_model(args.gemma_model_id, args.vision_backbone_id, device)
    trainable_params = [p for p in model_vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=2e-4, weight_decay=0.01)
    scheduler = MultiStepLR(optimizer, milestones=[100_000], gamma=0.1)

    # --- Load checkpoint state into fresh model ---
    print("[resume-child] Loading checkpoint state into fresh model...")
    missing, unexpected = model_vla.load_state_dict(saved_trainable_state, strict=False)
    relevant_missing = [k for k in missing if not k.startswith("llm.") and not k.startswith("vision_backbone.")]
    assert not relevant_missing, f"unexpected missing: {relevant_missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    optimizer.load_state_dict(saved_optim_state)
    scheduler.load_state_dict(saved_scheduler_state)

    # After load, param_groups[0]["lr"] は save 時 lr を反映 (optimizer state に入っている)
    lr_after_resume = optimizer.param_groups[0]["lr"]

    # --- State diff (saved on-disk state vs in-memory state after load) ---
    # load 直後の model.state_dict() から trainable のみ抽出 (llm., vision_backbone. を除く)
    loaded_full = model_vla.state_dict()
    loaded_trainable_state = {
        k: v for k, v in loaded_full.items()
        if not k.startswith("llm.") and not k.startswith("vision_backbone.")
    }
    state_diff_max = compute_state_diff(saved_trainable_state, loaded_trainable_state)
    optim_diff_max = compute_optim_diff(saved_optim_state, optimizer.state_dict())

    # --- Load batch + forward ---
    batch_payload = torch.load(args.batch_path, map_location="cpu", weights_only=False)
    pv = {
        "dino":   batch_payload["pixel_values_dino"].to(device, dtype=torch.bfloat16),
        "siglip": batch_payload["pixel_values_siglip"].to(device, dtype=torch.bfloat16),
    }
    input_ids = batch_payload["input_ids"].to(device)
    proprio = batch_payload["proprio"].to(device, dtype=torch.bfloat16)
    actions = batch_payload["actions"].to(device, dtype=torch.bfloat16)

    model_vla.eval()
    with torch.no_grad():
        _, loss_b = model_vla(pv, input_ids, proprio, actions)
    loss_b_val = float(loss_b.item())

    # --- Write result ---
    result = {
        "loss_b": loss_b_val,
        "state_diff_max": state_diff_max,
        "optim_diff_max": optim_diff_max,
        "lr_after_resume": lr_after_resume,
        "current_lr_at_save": current_lr_at_save,
        "gradient_step_idx_after_resume": gradient_step_idx_at_save,  # scheduler に埋まっている値を使う: saved payload で渡された値
        "gradient_step_idx_at_save": gradient_step_idx_at_save,
        "scheduler_last_epoch": scheduler.last_epoch,
    }
    args.output_path.write_text(json.dumps(result, indent=2))
    print(f"[resume-child] loss_b={loss_b_val:.6f}  state_diff_max={state_diff_max:.3e}  "
          f"optim_diff_max={optim_diff_max:.3e}  lr={lr_after_resume:.3e}")
    print(f"[resume-child] wrote {args.output_path}")


if __name__ == "__main__":
    main()
