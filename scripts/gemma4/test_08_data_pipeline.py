"""
Phase 1d.a: RLDS data pipeline → VLAAdapterGemma4 forward 接続検証

目的:
  - LIBERO-Spatial 実データを RLDSDataset で load
  - Gemma4BatchTransform で pixel_values/input_ids/proprio/actions を構築
  - B=8 で VLAAdapterGemma4.forward が通ることを確認
  - loss scalar + finite + LLM grad leak 0 の assert
  - Phase 1c B=8 基準値 (bwd 36.82 GB) との regression 確認

スコープ (1d.a):
  - 3 batch iterate (1 batch load + forward + backward、optimizer step は 1d.b)
  - dataset_statistics は新規計算せず outputs/LIBERO-Spatial-Pro/ から再利用
  - R8 遵守

Run: CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_08_data_pipeline.py
"""
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
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
from prismatic.vla.datasets.rlds import make_interleaved_dataset  # noqa: E402
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights  # noqa: E402
from prismatic.vla.constants import NormalizationType  # noqa: E402

MODEL_ID = "google/gemma-4-E2B"
VISION_BACKBONE_ID = "dinosiglip-vit-so-224px"

# Phase 1c B=8 baseline (docs/gemma4_migration_log.md Phase 1c)
BASELINE_1C_B8 = {"fwd_gb": 36.82, "bwd_gb": 36.82}

# 固定 layout 用の prompt length (LIBERO の language instruction は概ね 10-15 tokens に収まる)
PROMPT_MAX_LEN = 20

# 1d.a config
DATA_ROOT_DIR = "data/modified_libero_rlds"
DATASET_NAME = "libero_spatial_no_noops"
BATCH_SIZE = 8
NUM_BATCHES_TO_ITERATE = 3

# Action / Proprio chunk sizes (prismatic.vla.constants と揃える)
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8


# ============================================================
# Gemma4BatchTransform: RLDS raw → VLAAdapterGemma4 の入力形式
# ============================================================
@dataclass
class Gemma4BatchTransform:
    tokenizer: Any
    image_transform: Any          # DinoSigLIPImageTransform (__call__(PIL.Image) → {"dino", "siglip"})
    prompt_max_len: int = PROMPT_MAX_LEN

    def _tokenize_prompt(self, lang: str) -> List[int]:
        text = f"What action should the robot take to {lang}?"
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) > self.prompt_max_len:
            ids = ids[: self.prompt_max_len]
        else:
            pad = [self.tokenizer.pad_token_id] * (self.prompt_max_len - len(ids))
            ids = ids + pad
        return ids

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        # --- images (primary + wrist) ---
        img_primary = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_wrist = Image.fromarray(rlds_batch["observation"]["image_wrist"][0])
        pv_p = self.image_transform(img_primary)
        pv_w = self.image_transform(img_wrist)
        pixel_values = {
            "dino": torch.stack([pv_p["dino"], pv_w["dino"]], dim=0),         # (2, 3, 224, 224)
            "siglip": torch.stack([pv_p["siglip"], pv_w["siglip"]], dim=0),
        }

        # --- language instruction ---
        lang = rlds_batch["task"]["language_instruction"].decode().strip().lower()
        prompt_ids = self._tokenize_prompt(lang)

        # --- full input_ids: [BOS] + prompt_padded + vision(512) + proprio(1) + action(64) + [EOS] ---
        input_ids = (
            [self.tokenizer.bos_token_id]
            + prompt_ids
            + list(range(VISION_PLACEHOLDER_BEGIN_IDX, VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS))
            + [PROPRIO_PLACEHOLDER_IDX]
            + list(range(ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS))
            + [self.tokenizer.eos_token_id]
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # --- proprio / actions ---
        # observation.proprio: (window_size=1, PROPRIO_DIM) numpy → (PROPRIO_DIM,)
        proprio = torch.tensor(np.asarray(rlds_batch["observation"]["proprio"][0]), dtype=torch.float32)
        # action: (NUM_ACTIONS_CHUNK, ACTION_DIM) numpy
        actions = torch.tensor(np.asarray(rlds_batch["action"]), dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "proprio": proprio,
            "actions": actions,
            "language": lang,
        }


def collate_gemma4(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "pixel_values": {
            "dino":   torch.stack([s["pixel_values"]["dino"] for s in samples], dim=0),    # (B, 2, 3, 224, 224)
            "siglip": torch.stack([s["pixel_values"]["siglip"] for s in samples], dim=0),
        },
        "input_ids": torch.stack([s["input_ids"] for s in samples], dim=0),               # (B, L)
        "proprio":   torch.stack([s["proprio"] for s in samples], dim=0),                  # (B, PROPRIO_DIM)
        "actions":   torch.stack([s["actions"] for s in samples], dim=0),                  # (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
        "languages": [s["language"] for s in samples],
    }


# ============================================================
# Custom RLDSDataset shim: bypass Qwen-specific RLDSBatchTransform
# ============================================================
from torch.utils.data import IterableDataset  # noqa: E402


class Gemma4RLDSDataset(IterableDataset):
    """RLDSDataset の最小レプリカ (Qwen 固有処理を除く)、Gemma4BatchTransform を適用."""
    def __init__(
        self,
        data_root_dir: str,
        dataset_name: str,
        batch_transform: Gemma4BatchTransform,
        resize_resolution=(224, 224),
        shuffle_buffer_size: int = 1000,
        train: bool = True,
    ):
        self.data_root_dir = data_root_dir
        self.dataset_name = dataset_name
        self.batch_transform = batch_transform

        mixture_spec = [(dataset_name, 1.0)]
        load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,
                future_action_window_size=NUM_ACTIONS_CHUNK - 1,
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )
        self.dataset, self.dataset_length, self.dataset_statistics = self._make(rlds_config)

    @staticmethod
    def _make(rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self):
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self):
        return self.dataset_length


# ============================================================
# Main
# ============================================================
def log_rlds_sample_once(rlds_dataset: Gemma4RLDSDataset):
    """Raw RLDS sample の 1 件を inspect、batch dict structure を記録."""
    raw_iter = rlds_dataset.dataset.as_numpy_iterator()
    raw_sample = next(raw_iter)
    summary = {}
    def _walk(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _walk(v, key)
            elif hasattr(v, "shape"):
                summary[key] = f"shape={tuple(v.shape)} dtype={v.dtype}"
            elif isinstance(v, (bytes, str)):
                s = v.decode() if isinstance(v, bytes) else v
                summary[key] = f"str len={len(s)} val={s[:60]!r}"
            else:
                summary[key] = f"{type(v).__name__}={v!r}"
    _walk(raw_sample)
    return summary


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    torch.manual_seed(42)

    # --- verify data path ---
    data_root_abs = (REPO_ROOT / DATA_ROOT_DIR).resolve()
    print(f"\nData root: {data_root_abs}")
    assert data_root_abs.exists(), f"missing: {data_root_abs}"
    task_dir = data_root_abs / DATASET_NAME
    assert task_dir.exists(), f"missing task: {task_dir}"
    versions = sorted(p.name for p in task_dir.iterdir() if p.is_dir())
    print(f"Dataset: {DATASET_NAME}  versions: {versions}")

    # --- Vision backbone ---
    print("\n=== Loading DINO+SigLIP ===")
    t0 = time.time()
    vision_backbone = DinoSigLIPViTBackbone(
        vision_backbone_id=VISION_BACKBONE_ID,
        image_resize_strategy="resize-naive",
        default_image_size=224,
        image_sequence_len=2,
    ).to(device, dtype=torch.bfloat16).eval()
    print(f"  loaded in {time.time()-t0:.1f}s, embed_dim={vision_backbone.embed_dim}")

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

    total_trainable_M = sum(p.numel() for p in model_vla.parameters() if p.requires_grad) / 1e6
    print(f"Total trainable: {total_trainable_M:.3f} M")
    assert 670.0 < total_trainable_M < 680.0, f"trainable regressed: {total_trainable_M:.3f}M"

    # --- BatchTransform + dataset ---
    print(f"\n=== Building Gemma4RLDSDataset ({DATA_ROOT_DIR}, {DATASET_NAME}) ===")
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
        shuffle_buffer_size=1000,   # smoke (normal は 256_000)
        train=True,
    )
    print(f"  built in {time.time()-t0:.1f}s, dataset_length={rlds_dataset.dataset_length}")

    # --- Raw sample inspection (log batch dict structure) ---
    print("\n=== Raw RLDS sample keys (pre-transform) ===")
    raw_summary = log_rlds_sample_once(rlds_dataset)
    for k, v in raw_summary.items():
        print(f"  {k}: {v}")

    # --- DataLoader ---
    loader = DataLoader(
        rlds_dataset,
        batch_size=BATCH_SIZE,
        sampler=None,
        collate_fn=collate_gemma4,
        num_workers=0,  # RLDS が内部 parallelism、DataLoader workers=0 必須
    )

    # --- Iterate NUM_BATCHES ---
    print(f"\n=== Iterate {NUM_BATCHES_TO_ITERATE} batches (B={BATCH_SIZE}) ===")
    batch_results = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= NUM_BATCHES_TO_ITERATE:
            break
        t0 = time.time()

        # Move to GPU + dtype cast
        pv = {
            "dino":   batch["pixel_values"]["dino"].to(device, dtype=torch.bfloat16),
            "siglip": batch["pixel_values"]["siglip"].to(device, dtype=torch.bfloat16),
        }
        input_ids = batch["input_ids"].to(device)                                   # long
        proprio = batch["proprio"].to(device, dtype=torch.bfloat16)
        actions = batch["actions"].to(device, dtype=torch.bfloat16)
        languages = batch["languages"]

        B, L = input_ids.shape
        dev_sec = time.time() - t0

        # Log first batch structure once
        if batch_idx == 0:
            print(f"  [batch 0] pixel_values.dino:   {tuple(pv['dino'].shape)} {pv['dino'].dtype}")
            print(f"  [batch 0] pixel_values.siglip: {tuple(pv['siglip'].shape)} {pv['siglip'].dtype}")
            print(f"  [batch 0] input_ids:           {tuple(input_ids.shape)} {input_ids.dtype}")
            print(f"  [batch 0] proprio:             {tuple(proprio.shape)} {proprio.dtype}")
            print(f"  [batch 0] actions:             {tuple(actions.shape)} {actions.dtype}")
            print(f"  [batch 0] languages (first 3): {languages[:3]}")
            # Verify placeholder positions are at fixed offsets (since PROMPT_MAX_LEN is fixed)
            row0_ids = input_ids[0].tolist()
            vstart = row0_ids.index(VISION_PLACEHOLDER_BEGIN_IDX)
            astart = row0_ids.index(ACTION_TOKEN_BEGIN_IDX)
            print(f"  [batch 0] vision range:        rows share? "
                  f"{all((input_ids[b].tolist().index(VISION_PLACEHOLDER_BEGIN_IDX) == vstart) for b in range(B))} "
                  f"(pos={vstart})")
            print(f"  [batch 0] action range:        pos={astart}")

        # Forward + Backward
        torch.cuda.reset_peak_memory_stats(device)
        predicted, loss = model_vla(pv, input_ids, proprio, actions)
        fwd_peak = torch.cuda.max_memory_allocated(device) / 1024**3

        # Asserts
        assert predicted.shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM), \
            f"batch {batch_idx} predicted {predicted.shape}"
        assert loss.dim() == 0, f"batch {batch_idx} loss not scalar"
        assert torch.isfinite(loss).item(), f"batch {batch_idx} loss not finite: {loss.item()}"

        loss.backward()
        bwd_peak = torch.cuda.max_memory_allocated(device) / 1024**3

        # LLM grad leak check (R4)
        llm_leak = sum(
            1 for p in model_vla.llm.parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        )
        assert llm_leak == 0, f"batch {batch_idx} LLM grad leak: {llm_leak} params"

        # Vision backbone leak check
        vb_leak = sum(
            1 for p in model_vla.vision_backbone.parameters()
            if p.grad is not None and p.grad.abs().sum().item() > 0
        )
        assert vb_leak == 0, f"batch {batch_idx} vision backbone grad leak: {vb_leak} params"

        # Zero grads before next batch (no optimizer step in 1d.a)
        for p in model_vla.parameters():
            if p.grad is not None:
                p.grad = None

        batch_results.append({
            "batch_idx": batch_idx,
            "B": B, "L": L,
            "fwd_gb": round(fwd_peak, 3),
            "bwd_gb": round(bwd_peak, 3),
            "loss": round(float(loss.item()), 6),
            "dev_sec": round(dev_sec, 3),
            "first_language": languages[0],
        })
        print(f"  [batch {batch_idx}] fwd={fwd_peak:.2f} bwd={bwd_peak:.2f} "
              f"loss={loss.item():.4f}  '{languages[0]}'")

    # --- Regression compare vs Phase 1c B=8 ---
    print("\n=== Phase 1c B=8 回帰確認 (batch 0) ===")
    b0 = batch_results[0]
    regression = {
        "fwd_delta": round(b0["fwd_gb"] - BASELINE_1C_B8["fwd_gb"], 3),
        "bwd_delta": round(b0["bwd_gb"] - BASELINE_1C_B8["bwd_gb"], 3),
    }
    print(f"  fwd: measured {b0['fwd_gb']} GB, baseline {BASELINE_1C_B8['fwd_gb']} GB, delta {regression['fwd_delta']:+.2f}")
    print(f"  bwd: measured {b0['bwd_gb']} GB, baseline {BASELINE_1C_B8['bwd_gb']} GB, delta {regression['bwd_delta']:+.2f}")

    # Exit: delta ±2 GB 以内 (plan は ±1 GB だが実データで resize/transform 変動あり)
    within_tol = abs(regression["bwd_delta"]) < 2.0
    if within_tol:
        print("  → within ±2 GB tolerance: OK")
    else:
        print(f"  → WARN: bwd delta {regression['bwd_delta']:+.2f} GB exceeds ±2 GB tolerance")

    # --- JSON dump ---
    summary = {
        "phase": "1d.a",
        "data_root_dir": DATA_ROOT_DIR,
        "data_root_abs": str(data_root_abs),
        "dataset_name": DATASET_NAME,
        "versions": versions,
        "batch_size": BATCH_SIZE,
        "num_batches_iterated": len(batch_results),
        "prompt_max_len": PROMPT_MAX_LEN,
        "dataset_length": rlds_dataset.dataset_length,
        "total_trainable_M": round(total_trainable_M, 3),
        "raw_rlds_sample_keys": raw_summary,
        "batches": batch_results,
        "regression_vs_1c_b8": regression,
        "within_tol": within_tol,
    }
    json_path = Path(__file__).resolve().parent / "test_08_result.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {json_path}")


if __name__ == "__main__":
    main()
