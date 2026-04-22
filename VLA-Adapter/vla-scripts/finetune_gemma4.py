"""
finetune_gemma4.py

Phase 2b: VLA-Adapter + Gemma 4 E2B 本番学習スクリプト (single A100 80GB)

Stage 2 scope 変更 (2026-04-20, User 判断) 反映:
  - Env B (A100 40GB × 8) は dormant、single A100 80GB 構成
  - DDP / FSDP / R15 / R16 / 'module.' prefix handling 全削除
  - LoRA / PEFT / gradient_checkpointing は import のみ、使用は全コメントアウト (R6, R14)
  - Checkpoint save/load/resume は本 script 内で完結 (旧 Phase 2d 移管)

原 `finetune.py` からの派生関係:
  - CLI/dataclass pattern: `FinetuneConfig` (draccus + dataclass)  … from finetune.py:67-128
  - warmup (line 1060-1065) + MultiStepLR pattern (line 915-921)   … from finetune.py:910-1080
  - save_latest_checkpoint_only / save_freq config                … from finetune.py:99-100
  - run_id 命名規則                                                 … 簡略版 (Env B 関連 id 省略)

Stage 1 Phase 1d 資産の再利用 (import):
  - Gemma4BatchTransform / Gemma4RLDSDataset / collate_gemma4      … scripts/gemma4/test_08_data_pipeline.py
  - VLAAdapterGemma4                                                … VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py
  - Warmup lambda 実装                                              … 原 finetune.py line 1060-1065 準拠 (LambdaLR 不使用)

Phase 2b smoke mode 仕様:
  - max_steps=100, B=8, warmup_steps=500, lr=2e-4
  - step 50 で checkpoint save
  - step 50 save 直後に resume 検証 (subprocess spawn で別 Python process が state load → forward → loss 比較)
  - 4 acceptance criteria (Check-in #11 提示):
    (i)   |loss_a - loss_b| < 1e-3  (forward 結果一致)
    (ii)  model state_dict tensor diff max < 1e-6
    (iii) optimizer m/v diff max < 1e-6
    (iv)  lr continuity (save 時 lr と resume 後 param_group[0]['lr'] 一致)

Phase 2e 本番 mode (smoke_mode=False):
  - max_steps=200000, save_freq=10000, save_latest_checkpoint_only=True
  - WandB on (wandb_project != "" なら有効)
  - resume 検証は smoke でのみ実行、本番では skip

Run (smoke):
  CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python VLA-Adapter/vla-scripts/finetune_gemma4.py --smoke_mode True

Run (prod, Phase 2e):
  CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python VLA-Adapter/vla-scripts/finetune_gemma4.py --smoke_mode False
"""
# NOTE: `from __future__ import annotations` を使うと draccus の dataclass 検出が壊れる
#       (annotation が string 化され、dataclasses.fields が TypeError で fail)
# そのため本 file では future import を使わない。

import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# R16 (Stage 2 後半で active 化): TF が CUDA を grab する前に GPU を無効化。
# DDP subprocess では各 worker で本 import が走り、TF が全 GPU を grab するのを防ぐ。
# single-GPU 実行でも冪等、副作用なし。test_08 経由で TF がロードされる前に実行することが重要。
import tensorflow as _tf_for_disable  # noqa: E402
_tf_for_disable.config.set_visible_devices([], "GPU")
del _tf_for_disable

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- R14: LoRA/PEFT は import のみ維持、使用禁止 ----
# from peft import LoraConfig, get_peft_model   # (intentionally commented, R14 永久 OoS)

import wandb  # noqa: F401 (Phase 2e で有効化、smoke は no-op)

REPO_ROOT = Path(__file__).resolve().parents[2]
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
SCRIPTS_GEMMA4 = REPO_ROOT / "scripts" / "gemma4"
sys.path.insert(0, str(VLA_ROOT))
sys.path.insert(0, str(SCRIPTS_GEMMA4))

from test_08_data_pipeline import (  # noqa: E402
    Gemma4BatchTransform,
    Gemma4RLDSDataset,
    collate_gemma4,
    PROMPT_MAX_LEN,
)
from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4  # noqa: E402
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone  # noqa: E402


# =============================================================================
# Config (原 finetune.py FinetuneConfig から DDP/LoRA/FiLM/diffusion を除去)
# =============================================================================
@dataclass
class FinetuneConfig:
    # fmt: off

    # --- Model ---
    gemma_model_id: str = "google/gemma-4-E2B"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"

    # --- Dataset ---
    data_root_dir: Path = Path("data/modified_libero_rlds")
    dataset_name: str = "libero_spatial_no_noops"
    shuffle_buffer_size: int = 1000           # smoke default、prod は 256_000 相当

    # --- Algorithm ---
    num_action_chunks: int = 8
    action_dim: int = 7
    proprio_dim: int = 8

    # --- Training ---
    batch_size: int = 8
    learning_rate: float = 2e-4
    lr_warmup_steps: int = 500                # 原 finetune.py:92 基準
    num_steps_before_decay: int = 100_000     # MultiStepLR milestone (gamma=0.1)
    grad_accumulation_steps: int = 1
    max_steps: int = 200_000                  # 本番値。smoke_mode で 100 に override
    weight_decay: float = 0.01
    clip_max_norm: float = 1.0

    # --- Checkpointing ---
    save_freq: int = 10_000
    save_latest_checkpoint_only: bool = True
    run_root_dir: Path = Path("runs/gemma4")

    # --- Weight init from prior checkpoint (Stage 3 → Stage 2 transfer 比較用) ---
    # 非空なら model build 後に load_state_dict(strict=False) で trainable weights を上書き。
    # optimizer/scheduler/step は resume せず、新規 run として扱う。
    # Stage 3 ckpt (soft_prompt_library 含む) を Stage 2 で load する用途想定、
    # 不整合 key は strict=False で無視 (soft_prompt_library.* 等)。
    init_weights_from: str = ""

    # --- WandB (smoke は空 string で無効化) ---
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_log_freq: int = 10
    run_id_note: Optional[str] = None

    # --- DDP (R15 active、Stage 2 後半) ---
    # torchrun 起動時のみ True、single-GPU は False (default)
    # 各 rank は LOCAL_RANK env var から GPU 選択、init_process_group で NCCL 接続
    ddp_mode: bool = False
    ddp_backend: str = "nccl"
    ddp_bucket_cap_mb: int = 25                # Phase 3c-0 T3、PyTorch default 25 → 拡大で 5-10% 通信 overhead 減

    # --- Phase 3c-0 optimization knobs (low-risk algorithmic optimizations) ---
    optim_fused: bool = False                  # T1: AdamW(fused=True)、期待 5-15% forward+backward overhead 削減
    attn_implementation: str = "sdpa"          # T5: "sdpa" or "flash_attention_2"、FA-2 は Gemma 4 互換性要検証

    # --- Smoke mode (Phase 2b) ---
    smoke_mode: bool = False                  # True: max_steps=100, save@50, resume check on
    smoke_max_steps: int = 100
    smoke_save_step: int = 50
    smoke_run_resume_check: bool = True

    # --- Stage 3 Pretrain mode (X-VLA Soft Prompt、5/13 hard deadline) ---
    # pretrain_mode=True で multi-dataset pretrain、SoftPromptLibrary 有効化 + dataset_id 付与 + custom LR
    # false の場合 Stage 1-2 動作完全互換 (backward compat)
    pretrain_mode: bool = False
    pretrain_num_workers: int = 4             # Phase 3b-5 data pipeline 最適化、DDP throughput 改善
    num_pretrain_datasets: int = 0            # 0=disabled、2=Taco+Fractal、3=+自前等
    num_soft_prompt_tokens: int = 32          # X-VLA 原実装 len_soft_prompts=32 と一致
    # Custom LR: X-VLA の learning_coef (vision_projector + soft_prompt に backbone LR × coef)
    learning_coef: float = 0.25               # User plan §R21 "1/4 backbone" = 0.25
    # X-VLA 流 two-step adaptation (Plan §3b-3):
    #   step < freeze_steps: soft_prompt のみ train (他 LR=0、prompt warmup)
    #   freeze_steps ≤ step: joint、linear warmup warmup_steps、以降 base LR
    pretrain_freeze_steps: int = 1000
    pretrain_warmup_steps: int = 2000
    # X-VLA recipe default (Stage 2 default と差分、明示上書き):
    #   weight_decay 0.0 (Stage 2 は 0.01、pretrain は 多数 data で 正則化過多回避)
    #   betas (0.9, 0.95) (Stage 2 は PyTorch default (0.9, 0.999))
    pretrain_weight_decay: float = 0.0        # 上書き、pretrain 時のみ weight_decay の代わりに使用
    pretrain_betas_beta2: float = 0.95        # AdamW beta2 (beta1=0.9 固定、X-VLA 準拠)
    # 5/13 hard deadline (R19)、empty で disabled
    hard_stop_datetime: str = ""              # e.g. "2026-05-13 00:00:00"
    # fmt: on


def build_run_id(cfg: FinetuneConfig) -> str:
    gemma_short = cfg.gemma_model_id.split("/")[-1].lower().replace(".", "-")
    if cfg.pretrain_mode:
        # Stage 3 pretrain identifier (dataset_name の代わりに pretrain prefix)
        parts = [
            gemma_short,
            f"pretrain-nd{cfg.num_pretrain_datasets}-sp{cfg.num_soft_prompt_tokens}",
            f"b{cfg.batch_size * cfg.grad_accumulation_steps}",
            f"lr-{cfg.learning_rate}",
            f"coef-{cfg.learning_coef}",
            f"fr-{cfg.pretrain_freeze_steps}-wu-{cfg.pretrain_warmup_steps}",
        ]
    else:
        parts = [
            gemma_short,
            cfg.dataset_name,
            f"b{cfg.batch_size * cfg.grad_accumulation_steps}",
            f"lr-{cfg.learning_rate}",
            f"wu-{cfg.lr_warmup_steps}",
        ]
    if cfg.smoke_mode:
        parts.append("smoke")
    if cfg.run_id_note:
        parts.append(cfg.run_id_note)
    return "+".join(parts)


# =============================================================================
# Warmup (原 finetune.py:1060-1065 準拠、LambdaLR 不使用)
# =============================================================================
def compute_warmup_lr(gradient_step_idx: int, original_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return original_lr
    lr_progress = min((gradient_step_idx + 1) / warmup_steps, 1.0)
    return original_lr * (0.1 + 0.9 * lr_progress)


# =============================================================================
# Model / Data builders
# =============================================================================
def build_model(cfg: FinetuneConfig, device: torch.device) -> tuple[VLAAdapterGemma4, AutoTokenizer, DinoSigLIPViTBackbone]:
    vision_backbone = DinoSigLIPViTBackbone(
        vision_backbone_id=cfg.vision_backbone_id,
        image_resize_strategy="resize-naive",
        default_image_size=224,
        image_sequence_len=2,
    ).to(device, dtype=torch.bfloat16).eval()

    tok = AutoTokenizer.from_pretrained(cfg.gemma_model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    gemma = AutoModelForCausalLM.from_pretrained(
        cfg.gemma_model_id, dtype=torch.bfloat16, attn_implementation=cfg.attn_implementation,
    ).to(device).eval()
    gemma.config.use_cache = True                       # R6: use_cache=True 必須
    # gemma.gradient_checkpointing_enable()             # R6: GC 永久禁止 (HF #45242)
    for p in gemma.parameters():
        p.requires_grad = False

    model_vla = VLAAdapterGemma4(
        gemma_model=gemma,
        vision_backbone=vision_backbone,
        feature_norm=torch.nn.Identity(),
        proprio_dim=cfg.proprio_dim,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        num_pretrain_datasets=cfg.num_pretrain_datasets,   # Stage 3: 0 で disable、>=1 で SoftPromptLibrary 構築
        num_soft_prompt_tokens=cfg.num_soft_prompt_tokens,
    ).to(device, dtype=torch.bfloat16)
    model_vla.train()

    total_trainable = sum(p.numel() for p in model_vla.parameters() if p.requires_grad) / 1e6
    # Soft Prompt 有効化時は trainable 数 +num_datasets × 32 × 1536 × 4 byte / 1M param 上乗せ許容
    soft_prompt_expected_M = cfg.num_pretrain_datasets * cfg.num_soft_prompt_tokens * 1536 / 1e6
    expected = 675.138 + soft_prompt_expected_M
    assert abs(total_trainable - expected) < 0.5, \
        f"trainable regressed: expected {expected:.3f}M (675.138 + {soft_prompt_expected_M:.3f}M soft prompt), got {total_trainable:.3f}M"

    return model_vla, tok, vision_backbone


def build_dataloader(cfg: FinetuneConfig, tok, vision_backbone) -> DataLoader:
    batch_transform = Gemma4BatchTransform(
        tokenizer=tok,
        image_transform=vision_backbone.image_transform,
        prompt_max_len=PROMPT_MAX_LEN,
    )
    rlds_dataset = Gemma4RLDSDataset(
        data_root_dir=str(cfg.data_root_dir),
        dataset_name=cfg.dataset_name,
        batch_transform=batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
    )
    return DataLoader(
        rlds_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collate_gemma4,
        num_workers=0,   # RLDS 内部並列、外側 workers=0 必須
    )


def build_pretrain_dataloader(cfg: FinetuneConfig, tok, vision_backbone) -> DataLoader:
    """Stage 3 pretrain: MultiDatasetPretrainDataset (Taco + Fractal) を DataLoader に包む.

    Phase 3b-5 (2026-04-20 夜、Data pipeline 最適化、DDP throughput 問題対応):
      num_workers=4 で CPU 並列 data prep (image_transform + tokenize) を worker process 化、
      main process の GPU forward/backward と overlap して DDP scaling を改善。
      persistent_workers=True で worker 再 fork コスト排除、prefetch_factor=2 で
      batch buffer 保持。TF datasets は worker process 内で lazy init (fork 後安全)。
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "stage3"))
    from multi_dataset_loader import MultiDatasetPretrainDataset, collate_pretrain
    dataset = MultiDatasetPretrainDataset(
        tokenizer=tok,
        image_transform=vision_backbone.image_transform,
        num_actions_chunk=cfg.num_action_chunks,
    )
    # num_workers は cfg から override 可能、default 4 (Phase 3b-5 smoke 実測で決定)
    num_workers = getattr(cfg, "pretrain_num_workers", 4)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collate_pretrain,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def build_pretrain_optimizer(model_vla, cfg: FinetuneConfig, inner_model) -> AdamW:
    """Stage 3 pretrain 用 custom LR param groups (X-VLA 流、User plan §R21 準拠).

    4 group 構成:
      - vision_projector:      LR = base × learning_coef (= base × 0.25)
      - soft_prompt_library:   LR = base × learning_coef
      - action_head + proprio_projector + action_queries: LR = base
      - (LLM + vision_backbone は frozen、requires_grad=False で除外)

    各 group の name は train loop 側で LR scheduler (freeze_steps + warmup_steps) が参照。
    """
    base_lr = cfg.learning_rate
    coef = cfg.learning_coef
    wd = cfg.pretrain_weight_decay
    betas = (0.9, cfg.pretrain_betas_beta2)

    # inner_model (DDP unwrap 済 VLAAdapterGemma4) から直接 module 取得
    vproj = list(inner_model.vision_projector.parameters())
    pproj = list(inner_model.proprio_projector.parameters())
    aq = list(inner_model.action_queries.parameters())
    ah = list(inner_model.action_head.parameters())
    sp = list(inner_model.soft_prompt_library.parameters()) if inner_model.soft_prompt_library is not None else []

    # 全 trainable ID set、漏れ検知用
    expected_ids = set(id(p) for p in (vproj + pproj + aq + ah + sp))
    actual_trainable = set(id(p) for p in model_vla.parameters() if p.requires_grad)
    missing = actual_trainable - expected_ids
    assert not missing, f"trainable param not in any group: {len(missing)} params、設計漏れ"

    param_groups = [
        {"name": "vision_projector",    "params": vproj, "lr": base_lr * coef, "weight_decay": wd},
        {"name": "soft_prompt_library", "params": sp,    "lr": base_lr * coef, "weight_decay": wd},
        {"name": "action_head",         "params": ah,    "lr": base_lr,        "weight_decay": wd},
        {"name": "proprio_projector",   "params": pproj, "lr": base_lr,        "weight_decay": wd},
        {"name": "action_queries",      "params": aq,    "lr": base_lr,        "weight_decay": wd},
    ]
    # Phase 3c-0 T1: fused=True で forward/backward kernel overhead 削減 (CUDA graph 互換 kernel)
    opt = AdamW(param_groups, betas=betas, fused=cfg.optim_fused)
    return opt


def update_pretrain_lrs(optimizer: AdamW, step: int, cfg: FinetuneConfig) -> dict:
    """X-VLA 流 two-step adaptation LR スケジュール (X-VLA/train.py:164-172 準拠).

    X-VLA mapping → Gemma 4 mapping:
      X-VLA `vlm` / `transformer_core` (freeze 中 LR=0)  ≈  本実装 `vision_projector`
      X-VLA `soft_prompts` (coef 適用、常時 active)       ≈  本実装 `soft_prompt_library`
      X-VLA `action_heads` (常時 active、coef 非適用)      ≈  本実装 `action_head` + `proprio_projector` + `action_queries`

    Phase 1 (step < freeze_steps): prompt warmup phase
      vision_projector: 0 (frozen、X-VLA vlm/transformer_core 相当)
      soft_prompt_library: base × coef (active with coef)
      action_head / proprio_projector / action_queries: base (active)

    Phase 2 (freeze_steps ≤ step < freeze_steps + warmup_steps): post-freeze vision warmup
      vision_projector: 0 → base × coef linear warmup
      soft_prompt_library / action_head / proprio / action_queries: base maintaining (no discontinuity)

    Phase 3 (step ≥ freeze_steps + warmup_steps): joint full-train
      全 group: base (cosine decay は Plan §3c-2 errata で非採用、hard_stop_datetime で停止)

    戻り値: {group_name: current_lr} (log 用)
    """
    base_lr = cfg.learning_rate
    coef = cfg.learning_coef
    freeze = cfg.pretrain_freeze_steps
    warmup = cfg.pretrain_warmup_steps
    base_map = {
        "vision_projector":    base_lr * coef,   # X-VLA vlm 相当 (coef 適用)
        "soft_prompt_library": base_lr * coef,   # X-VLA soft_prompts (coef 適用)
        "action_head":         base_lr,          # X-VLA action_heads (coef 非適用)
        "proprio_projector":   base_lr,          # 同上
        "action_queries":      base_lr,          # 同上
    }
    # freeze 中 LR=0 にする group (X-VLA vlm/transformer_core 相当)
    VISION_ONLY_FROZEN = {"vision_projector"}
    current = {}
    for g in optimizer.param_groups:
        name = g["name"]
        target = base_map[name]
        if step < freeze:
            # Phase 1: vision_projector のみ freeze、他は active (X-VLA 準拠)
            new_lr = 0.0 if name in VISION_ONLY_FROZEN else target
        elif step < freeze + warmup:
            # Phase 2: vision_projector のみ linear warmup、他は base 維持
            if name in VISION_ONLY_FROZEN:
                progress = (step - freeze) / max(1, warmup)
                new_lr = target * progress
            else:
                new_lr = target
        else:
            # Phase 3: 全 group base
            new_lr = target
        g["lr"] = new_lr
        current[name] = new_lr
    return current


def move_batch_to_device(batch, device, dtype=torch.bfloat16):
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


# =============================================================================
# Checkpoint save / load
# =============================================================================
def save_checkpoint(
    checkpoint_path: Path,
    model_vla: VLAAdapterGemma4,
    optimizer: AdamW,
    scheduler: Optional[MultiStepLR],    # pretrain_mode は None (update_pretrain_lrs で per-step 制御)
    gradient_step_idx: int,
    current_lr: float,
    cfg: FinetuneConfig,
) -> None:
    """全 trainable state を単一 .pt に保存.

    保存内容:
      - model_state_dict: llm + vision_backbone 除く全モジュール (trainable のみ)
      - optimizer_state_dict
      - scheduler_state_dict: MultiStepLR state (Stage 1-2)、pretrain_mode は None
      - gradient_step_idx (warmup continuity、resume 時の lr 再計算 source)
      - current_lr (sanity check 用)
      - cfg (dict 化、resume 時の config drift 検出)
    """
    # LLM と vision_backbone は frozen、再 load 時は pretrained から初期化するため保存不要
    trainable_state = {
        k: v for k, v in model_vla.state_dict().items()
        if not k.startswith("llm.") and not k.startswith("vision_backbone.")
    }
    payload = {
        "trainable_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "gradient_step_idx": gradient_step_idx,
        "current_lr": current_lr,
        "cfg": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()},
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint_into(
    checkpoint_path: Path,
    model_vla: VLAAdapterGemma4,
    optimizer: AdamW,
    scheduler: Optional[MultiStepLR],    # pretrain_mode は None
    map_location: str = "cuda:0",
) -> dict:
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    missing, unexpected = model_vla.load_state_dict(payload["trainable_state_dict"], strict=False)
    # llm.* と vision_backbone.* が missing に入るのは frozen なので期待通り
    relevant_missing = [k for k in missing if not k.startswith("llm.") and not k.startswith("vision_backbone.")]
    assert not relevant_missing, f"unexpected missing keys: {relevant_missing[:5]}..."
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}..."
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return {
        "gradient_step_idx": payload["gradient_step_idx"],
        "current_lr": payload["current_lr"],
    }


# =============================================================================
# Resume 検証 (subprocess 方式、User 仕様の旧 Phase 2d 移管分)
# =============================================================================
def run_resume_verification(
    cfg: FinetuneConfig,
    run_dir: Path,
    checkpoint_path: Path,
    model_vla: VLAAdapterGemma4,
    optimizer: AdamW,
    current_lr_at_save: int,
    gradient_step_idx_at_save: int,
    next_batch,
) -> dict:
    """step 50 save 後、別 Python process で同じ batch を forward → loss 比較.

    procedure (User 仕様):
      1. parent (本 process) で next_batch を forward → loss_a 計算、grad は一切触らない (detach)
      2. batch を pickle で disk に書き出し
      3. subprocess spawn: test_12_resume_child.py が checkpoint + batch を load → loss_b 計算 + state diff
      4. 子 process stdout から JSON parse、4 criteria 評価
    """
    device = next(model_vla.parameters()).device

    # --- (1) loss_a ---
    pv, input_ids, proprio, actions, _ = move_batch_to_device(next_batch, device)
    with torch.no_grad():
        model_vla.eval()
        _, loss_a_tensor = model_vla(pv, input_ids, proprio, actions)
        model_vla.train()
    loss_a = float(loss_a_tensor.item())

    # --- (2) pickle batch ---
    batch_path = (run_dir / "_resume_verify_batch.pt").resolve()
    torch.save({
        "pixel_values_dino":   next_batch["pixel_values"]["dino"].cpu(),
        "pixel_values_siglip": next_batch["pixel_values"]["siglip"].cpu(),
        "input_ids":           next_batch["input_ids"].cpu(),
        "proprio":             next_batch["proprio"].cpu(),
        "actions":             next_batch["actions"].cpu(),
    }, batch_path)

    # --- (3) subprocess spawn (absolute paths for safety) ---
    child_script = (SCRIPTS_GEMMA4 / "test_12_resume_child.py").resolve()
    child_out_path = (run_dir / "_resume_verify_child_out.json").resolve()
    checkpoint_path_abs = Path(checkpoint_path).resolve()
    cmd = [
        sys.executable,
        str(child_script),
        "--checkpoint-path", str(checkpoint_path_abs),
        "--batch-path", str(batch_path),
        "--output-path", str(child_out_path),
        "--gemma-model-id", cfg.gemma_model_id,
        "--vision-backbone-id", cfg.vision_backbone_id,
    ]
    print(f"  [resume-verify] spawn: {' '.join(cmd[:2])} ...")
    t0 = time.time()
    # child の stderr はそのまま parent に流す、stdout はキャプチャ不要 (JSON は file に)
    proc = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    child_sec = time.time() - t0
    assert proc.returncode == 0, f"child process failed with exit code {proc.returncode}"

    # --- (4) child 結果 load + 4 criteria ---
    child_result = json.loads(child_out_path.read_text())
    loss_b = child_result["loss_b"]
    state_diff_max = child_result["state_diff_max"]
    optim_diff_max = child_result["optim_diff_max"]
    lr_after_resume = child_result["lr_after_resume"]
    gradient_step_idx_after_resume = child_result["gradient_step_idx_after_resume"]

    criteria = {
        "loss_diff":        {"value": abs(loss_a - loss_b),                  "tol": 1e-3,  "pass": abs(loss_a - loss_b) < 1e-3},
        "state_diff_max":   {"value": state_diff_max,                         "tol": 1e-6,  "pass": state_diff_max < 1e-6},
        "optim_diff_max":   {"value": optim_diff_max,                         "tol": 1e-6,  "pass": optim_diff_max < 1e-6},
        "lr_continuity":    {"value": abs(lr_after_resume - current_lr_at_save),
                             "tol": 1e-12, "pass": abs(lr_after_resume - current_lr_at_save) < 1e-12},
        "gradient_step_idx_continuity": {
            "value": abs(gradient_step_idx_after_resume - gradient_step_idx_at_save),
            "tol": 0, "pass": gradient_step_idx_after_resume == gradient_step_idx_at_save,
        },
    }
    all_pass = all(c["pass"] for c in criteria.values())

    return {
        "loss_a": loss_a,
        "loss_b": loss_b,
        "state_diff_max": state_diff_max,
        "optim_diff_max": optim_diff_max,
        "lr_at_save": current_lr_at_save,
        "lr_after_resume": lr_after_resume,
        "gradient_step_idx_at_save": gradient_step_idx_at_save,
        "gradient_step_idx_after_resume": gradient_step_idx_after_resume,
        "criteria": criteria,
        "all_pass": all_pass,
        "child_sec": round(child_sec, 2),
    }


# =============================================================================
# Main finetune
# =============================================================================
@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    assert torch.cuda.is_available()

    # --- DDP setup (R15 active、Stage 2 後半) ---
    if cfg.ddp_mode:
        assert "LOCAL_RANK" in os.environ, "ddp_mode=True requires torchrun (LOCAL_RANK env 未設定)"
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_rank = int(os.environ.get("RANK", "0"))
        # 注: `prismatic.overwatch.overwatch` が import 時に accelerate.PartialState() を作り、
        # torchrun 環境下では内部で init_process_group を既に実行する。
        # 再 init は "initialize twice" error になるため、既に init 済ならスキップ。
        if not dist.is_initialized():
            dist.init_process_group(backend=cfg.ddp_backend, init_method="env://")
        else:
            # 情報表示のみ
            print(f"[rank {global_rank}] dist already initialized by accelerate.PartialState, "
                  f"backend={dist.get_backend()}")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main_process = (global_rank == 0)
    else:
        local_rank = 0
        world_size = 1
        global_rank = 0
        device = torch.device("cuda:0")
        is_main_process = True

    # seed: 全 rank で 42 統一 → model 初期パラメータが rank 間で完全一致
    #   (DDP は grad のみ同期、初期 params が rank 間で揃っている前提。init 乱数が
    #    異なると initial weights が rank 毎に違う bug に落ちる。)
    # data variance は TF shuffle の非決定性に任せる。
    torch.manual_seed(42)

    # smoke override (before run_dir)
    if cfg.smoke_mode:
        cfg.max_steps = cfg.smoke_max_steps

    # --- Pretrain mode assertions (Stage 3、R18-R21) ---
    if cfg.pretrain_mode:
        assert cfg.num_pretrain_datasets > 0, \
            "pretrain_mode=True requires num_pretrain_datasets >= 1 (SoftPromptLibrary 構築用)"
        # X-VLA recipe: pretrain_weight_decay (= 0.0 default) を使用
        # Stage 2 fine-tune mode では cfg.weight_decay (= 0.01) を使用、両立

    run_id = build_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    if is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.ddp_mode:
        dist.barrier()   # rank 0 の mkdir 完了を待つ

    def rprint(*args, **kwargs):
        """rank 0 のみ print (DDP 時の log 洪水回避)、single GPU は常に print."""
        if is_main_process:
            print(*args, **kwargs)

    rprint(f"[finetune_gemma4] run_id:        {run_id}")
    rprint(f"[finetune_gemma4] run_dir:       {run_dir}")
    rprint(f"[finetune_gemma4] device:        {device} ({torch.cuda.get_device_name(local_rank)})")
    rprint(f"[finetune_gemma4] ddp_mode:      {cfg.ddp_mode}  (world_size={world_size}, rank={global_rank}, local_rank={local_rank})")
    rprint(f"[finetune_gemma4] smoke_mode:    {cfg.smoke_mode}")
    rprint(f"[finetune_gemma4] max_steps:     {cfg.max_steps}")
    rprint(f"[finetune_gemma4] warmup_steps:  {cfg.lr_warmup_steps}")
    rprint(f"[finetune_gemma4] lr:            {cfg.learning_rate}")
    rprint(f"[finetune_gemma4] batch_size (per-GPU): {cfg.batch_size}  effective={cfg.batch_size * world_size}")

    # --- WandB (rank 0 のみ、R11) ---
    use_wandb = bool(cfg.wandb_project) and is_main_process
    if use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            config=cfg.__dict__,
        )
        rprint(f"[finetune_gemma4] wandb.init OK (mode={os.environ.get('WANDB_MODE', 'online')}, run={run_id})")

    # --- Build model / data ---
    rprint("\n=== Loading model ===")
    t0 = time.time()
    model_vla, tok, vision_backbone = build_model(cfg, device)
    rprint(f"  model loaded in {time.time()-t0:.1f}s")

    # --- Optional: init trainable weights from prior checkpoint (Stage 3 → Stage 2 transfer) ---
    if cfg.init_weights_from:
        rprint(f"\n=== Init weights from: {cfg.init_weights_from} ===")
        payload = torch.load(cfg.init_weights_from, map_location=device, weights_only=False)
        src_state = payload["trainable_state_dict"]
        # Stage 3 ckpt は soft_prompt_library.* を含む、Stage 2 model は持たない → strict=False で無視
        missing, unexpected = model_vla.load_state_dict(src_state, strict=False)
        relevant_missing = [k for k in missing if not k.startswith("llm.") and not k.startswith("vision_backbone.")]
        rprint(f"  loaded keys: {len(src_state) - len(unexpected)}")
        rprint(f"  unexpected (ignored): {unexpected[:3]}{' ...' if len(unexpected) > 3 else ''} (total {len(unexpected)})")
        rprint(f"  relevant_missing: {relevant_missing[:3]}{' ...' if len(relevant_missing) > 3 else ''} (total {len(relevant_missing)})")
        src_step = payload.get("gradient_step_idx", "?")
        rprint(f"  source gradient_step_idx: {src_step} (new run starts from step 0)")

    rprint(f"\n=== Building data pipeline ===")
    t0 = time.time()
    if cfg.pretrain_mode:
        rprint(f"  mode: Stage 3 multi-dataset pretrain (Taco + Fractal)")
        loader = build_pretrain_dataloader(cfg, tok, vision_backbone)
    else:
        rprint(f"  mode: Stage 2 single-dataset ({cfg.data_root_dir})")
        loader = build_dataloader(cfg, tok, vision_backbone)
    rprint(f"  dataloader built in {time.time()-t0:.1f}s")

    # --- DDP wrap (R15): model_vla → DDP(model_vla)、frozen param は all-reduce 対象外 (requires_grad=False) ---
    if cfg.ddp_mode:
        # find_unused_parameters=True: L1RegressionActionHead Pro version の 24 block 内で
        #   phase="Training" 時に一部 block の weight が forward 経由せず grad 非受領となるケースあり
        #   (D1 初回 smoke で rank 0/1 とも parameter indices 36,37,59,... が grad 無しと判明)。
        #   True 指定で DDP が per-iteration で使用 param を検出、reduction を調整。overhead 5-10% の代償で
        #   architecture 互換性確保。Phase 2i C1 本番 DDP retrain でも同設定。
        model_vla = DDP(model_vla, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=True,
                        bucket_cap_mb=cfg.ddp_bucket_cap_mb)   # T3: default 25 → 100 で all-reduce overhead 削減
        # NCCL all-reduce scope (trainable のみ、frozen LLM/VB は excluded by DDP spec when requires_grad=False)
        all_reduce_params = sum(p.numel() for p in model_vla.parameters() if p.requires_grad)
        rprint(f"[ddp] all-reduce scope (trainable only): {all_reduce_params/1e6:.3f}M params "
               f"(frozen LLM+VB excluded by DDP spec when requires_grad=False)")

    # inner_model: leak check / save で使う DDP wrap を剥いた VLAAdapterGemma4
    inner_model = model_vla.module if cfg.ddp_mode else model_vla

    # --- Optimizer / Scheduler ---
    # DDP wrap 後でも parameters() は underlying module を返す、requires_grad=True のみ collect
    trainable_params = [p for p in model_vla.parameters() if p.requires_grad]
    if cfg.pretrain_mode:
        # Stage 3: X-VLA 流 custom LR param groups + betas=(0.9, 0.95) + wd=0.0
        optimizer = build_pretrain_optimizer(model_vla, cfg, inner_model)
        scheduler = None   # pretrain は update_pretrain_lrs で per-step LR 制御、MultiStepLR 不使用
        rprint(f"\nOptimizer: AdamW pretrain mode")
        rprint(f"  betas=(0.9, {cfg.pretrain_betas_beta2})、wd={cfg.pretrain_weight_decay}")
        rprint("  param groups (name, init lr, num_params, requires_grad=True):")
        total_trainable_M = 0.0
        for g in optimizer.param_groups:
            n_params = sum(p.numel() for p in g["params"])
            total_trainable_M += n_params / 1e6
            rprint(f"    {g['name']:<22s}: lr={g['lr']:.3e}  num_params={n_params/1e6:7.3f}M  wd={g.get('weight_decay', 0.0)}")
        rprint(f"  Total trainable (optimizer scope): {total_trainable_M:.3f}M "
               f"(Stage 2 baseline 675.138M + soft_prompt {cfg.num_pretrain_datasets*cfg.num_soft_prompt_tokens*1536/1e6:.3f}M)")
        rprint(f"  LR schedule X-VLA 準拠 (X-VLA/train.py:164-172):")
        rprint(f"    Phase 1 (step <{cfg.pretrain_freeze_steps}): vision_projector LR=0、他 base")
        rprint(f"    Phase 2 ({cfg.pretrain_freeze_steps}-{cfg.pretrain_freeze_steps+cfg.pretrain_warmup_steps}): vision_projector 0→base×coef linear warmup")
        rprint(f"    Phase 3 (≥{cfg.pretrain_freeze_steps+cfg.pretrain_warmup_steps}): 全 group base、cosine decay 非採用")
        original_lr = cfg.learning_rate
    else:
        # Stage 1-2: 従来 flat LR + manual warmup
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                          fused=cfg.optim_fused)
        original_lr = optimizer.param_groups[0]["lr"]
        scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)
        rprint(f"\nOptimizer: AdamW(lr={original_lr}, wd={cfg.weight_decay})")
        rprint(f"Scheduler: MultiStepLR(milestones=[{cfg.num_steps_before_decay}], gamma=0.1)")
        rprint(f"Warmup: 10% → 100% linear over {cfg.lr_warmup_steps} steps (原 finetune.py:1060-1065 準拠)")

    # --- 5/13 hard deadline parser (Stage 3 R19、pretrain mode のみ active) ---
    hard_stop_deadline = None
    if cfg.pretrain_mode and cfg.hard_stop_datetime:
        from datetime import datetime
        try:
            hard_stop_deadline = datetime.strptime(cfg.hard_stop_datetime, "%Y-%m-%d %H:%M:%S")
            rprint(f"  hard_stop_datetime: {cfg.hard_stop_datetime} (R19)")
        except ValueError as e:
            rprint(f"  [WARN] hard_stop_datetime parse 失敗: {e}、deadline check disabled")

    # --- Main train loop ---
    # Escalation #9 (Plan v2 update 2026-04-20):
    #   Warning:    1000-step rolling mean > 1.29 s/step → print [WARN]
    #   Escalation: 100-step consecutive mean > 1.72 s/step → raise RuntimeError
    #   step 0 (warmup kernel JIT 3.18 s in Phase 2b 実測) は deque に入れずに除外
    ESCALATION9_WARN_THRESHOLD = 1.29     # s/step、1000-step rolling mean
    ESCALATION9_HALT_THRESHOLD = 1.72     # s/step、100-step consecutive mean
    print(f"\n=== Training ({cfg.max_steps} steps) ===")
    step_results = []
    step_times = deque(maxlen=1000)       # Escalation #9 rolling window
    step_times_all = []                   # final median 計算用の全 step 履歴 (メモリ軽、float のみ)
    llm_leak_all_zero = True
    data_iter = iter(loader)
    resume_check_result = None
    escalation9_warn_emitted = False

    for step in range(cfg.max_steps):
        t_step = time.time()
        gradient_step_idx = step // cfg.grad_accumulation_steps

        optimizer.zero_grad(set_to_none=True)

        # --- Smoke: step 50 で save + resume 検証 ---
        do_resume_check_after_this_step = (
            cfg.smoke_mode
            and cfg.smoke_run_resume_check
            and step == cfg.smoke_save_step
        )

        # --- Hard stop check (Stage 3 R19、pretrain_mode) ---
        if hard_stop_deadline is not None:
            from datetime import datetime
            if datetime.now() >= hard_stop_deadline:
                rprint(f"[finetune_gemma4] Hard stop datetime reached: {cfg.hard_stop_datetime} (R19)")
                rprint(f"  stopping at step {step}/{cfg.max_steps}")
                break

        # batch load
        batch = next(data_iter)
        pv, input_ids, proprio, actions, languages = move_batch_to_device(batch, device)
        B, L = input_ids.shape

        # Stage 3: dataset_id を forward に渡す (pretrain mode、SoftPromptLibrary 入力)
        if cfg.pretrain_mode:
            dataset_id = batch["dataset_id"].to(device)   # (B,) long
        else:
            dataset_id = None

        # --- Forward ---
        if cfg.pretrain_mode:
            predicted, loss = model_vla(pv, input_ids, proprio, actions, dataset_id=dataset_id)
        else:
            predicted, loss = model_vla(pv, input_ids, proprio, actions)
        assert predicted.shape == (B, cfg.num_action_chunks, cfg.action_dim)
        assert loss.dim() == 0
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Loss non-finite at step {step}: {loss.item()}")
        loss_val = float(loss.item())

        # --- Per-dataset loss breakdown (C5、smoke + pretrain_mode) ---
        per_dataset_loss = {}
        if cfg.pretrain_mode and cfg.smoke_mode and is_main_process:
            with torch.no_grad():
                sample_loss = (predicted - actions).abs().mean(dim=[1, 2])   # (B,) per-sample L1
                for ds_name, d_id in (("taco_play", 0), ("fractal20220817_data", 1)):
                    mask = (dataset_id == d_id)
                    if mask.any():
                        per_dataset_loss[ds_name] = float(sample_loss[mask].mean().item())

        # --- Backward ---
        loss.backward()

        # --- Soft prompt grad norm (C1、smoke + pretrain_mode で log、freeze 中 non-zero 確認) ---
        soft_prompt_grad_norm = -1.0
        if cfg.pretrain_mode and cfg.smoke_mode and is_main_process:
            sp_weight = inner_model.soft_prompt_library.embedding.weight
            if sp_weight.grad is not None:
                soft_prompt_grad_norm = sp_weight.grad.detach().float().norm().item()

        # Grad leak check (毎 step、smoke 100 step なら overhead ~数 ms 許容)
        # R4 DDP 互換: DDP wrap 時は inner_model (= model_vla.module) 経由で llm/vision_backbone に access
        if cfg.smoke_mode:
            llm_leak = sum(
                1 for p in inner_model.llm.parameters()
                if p.grad is not None and p.grad.abs().sum().item() > 0
            )
            vb_leak = sum(
                1 for p in inner_model.vision_backbone.parameters()
                if p.grad is not None and p.grad.abs().sum().item() > 0
            )
            if llm_leak != 0 or vb_leak != 0:
                llm_leak_all_zero = False
                raise RuntimeError(f"step {step}: LLM leak {llm_leak}, VB leak {vb_leak} (R4 違反、DDP 時は model_vla.module 経由)")
        else:
            # prod は負荷回避のため step 0 + save_freq timing のみ
            llm_leak = vb_leak = -1
            if step == 0 or (step + 1) % cfg.save_freq == 0:
                llm_leak = sum(
                    1 for p in inner_model.llm.parameters()
                    if p.grad is not None and p.grad.abs().sum().item() > 0
                )
                vb_leak = sum(
                    1 for p in inner_model.vision_backbone.parameters()
                    if p.grad is not None and p.grad.abs().sum().item() > 0
                )
                assert llm_leak == 0 and vb_leak == 0, f"step {step}: leak (LLM {llm_leak}, VB {vb_leak})"

        # --- LR update ---
        if cfg.pretrain_mode:
            # Stage 3: X-VLA 流 two-step (freeze + warmup)、per-group LR
            lr_map = update_pretrain_lrs(optimizer, step, cfg)
            current_lr = lr_map.get("soft_prompt_library", 0.0)   # log 用に代表値 (soft_prompt)
            # C6: smoke mode で step 0/50/99 に LR state dump (train loop 内判定で全 group 状態可視化)
            if cfg.smoke_mode and is_main_process and step in (0, cfg.smoke_save_step, cfg.max_steps - 1):
                phase_desc = (
                    "freeze phase (step<freeze)" if step < cfg.pretrain_freeze_steps
                    else ("warmup phase (freeze≤step<freeze+warmup)" if step < cfg.pretrain_freeze_steps + cfg.pretrain_warmup_steps
                          else "joint full-train phase (≥freeze+warmup)")
                )
                print(f"  [step {step+1}] LR state ({phase_desc}):")
                for name, lr in lr_map.items():
                    print(f"    {name:<22s}: lr={lr:.3e}")
        else:
            # Stage 1-2: flat LR with manual warmup (原 finetune.py:1060-1065)
            current_lr = compute_warmup_lr(gradient_step_idx, original_lr, cfg.lr_warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        # --- Clip ---
        grad_norm_pre = torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm=cfg.clip_max_norm
        ).item()
        with torch.no_grad():
            sq_sum = 0.0
            for p in trainable_params:
                if p.grad is not None:
                    sq_sum += p.grad.detach().float().norm().item() ** 2
        grad_norm_post = sq_sum ** 0.5

        # --- Optimizer / Scheduler step (原 finetune.py:1078-1081) ---
        if (step + 1) % cfg.grad_accumulation_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()   # pretrain_mode は None (update_pretrain_lrs で制御済)

        step_sec = time.time() - t_step
        step_times_all.append(step_sec)
        # Escalation #9: step 0 は warmup kernel JIT overhead (Phase 2b 実測 3.18s) なので除外
        if step >= 1:
            step_times.append(step_sec)
            # Halt check: 100-step consecutive mean > 1.72 s/step
            if len(step_times) >= 100:
                last_100 = list(step_times)[-100:]
                mean_100 = sum(last_100) / 100
                if mean_100 > ESCALATION9_HALT_THRESHOLD:
                    raise RuntimeError(
                        f"Escalation #9 (Stage 2, 2026-04-20 閾値): "
                        f"last 100 step mean {mean_100:.3f} s/step > {ESCALATION9_HALT_THRESHOLD} s/step "
                        f"(at step {step+1}/{cfg.max_steps}). per-step time regression detected — halt for investigation."
                    )
            # Warning: 1000-step rolling mean > 1.29 s/step (fire-once で log 洪水回避)
            if len(step_times) == step_times.maxlen and not escalation9_warn_emitted:
                mean_1000 = sum(step_times) / len(step_times)
                if mean_1000 > ESCALATION9_WARN_THRESHOLD:
                    print(f"[WARN] Escalation #9 warning: rolling 1000-step mean {mean_1000:.3f} s/step "
                          f"> {ESCALATION9_WARN_THRESHOLD} s/step (at step {step+1}/{cfg.max_steps})")
                    escalation9_warn_emitted = True
        step_results.append({
            "step": step,
            "gradient_step_idx": gradient_step_idx,
            "lr": current_lr,
            "loss": round(loss_val, 6),
            "grad_norm_pre": round(grad_norm_pre, 4),
            "grad_norm_post": round(grad_norm_post, 6),
            "llm_leak": llm_leak,
            "vb_leak": vb_leak,
            "L": L,
            "step_sec": round(step_sec, 3),
        })

        # --- Log (rank 0 のみ) ---
        if is_main_process and (cfg.smoke_mode or step % cfg.wandb_log_freq == 0):
            # Pretrain mode では per-dataset loss + soft_prompt_grad_norm を append
            extra = ""
            if cfg.pretrain_mode and cfg.smoke_mode:
                if per_dataset_loss:
                    pdl = "  ".join(f"{k[:4]}={v:.4f}" for k, v in per_dataset_loss.items())
                    extra += f"  per-ds: {pdl}"
                if soft_prompt_grad_norm >= 0:
                    extra += f"  sp_gn={soft_prompt_grad_norm:.3f}"
            print(f"  [step {step+1:4d}/{cfg.max_steps}] lr={current_lr:.3e}  loss={loss_val:.4f}  "
                  f"gn_pre={grad_norm_pre:.2f}  gn_post={grad_norm_post:.4f}  "
                  f"({step_sec:.2f}s){extra}")
        if use_wandb and step % cfg.wandb_log_freq == 0:
            wandb.log({
                "train/loss": loss_val,
                "train/grad_norm_pre": grad_norm_pre,
                "train/grad_norm_post": grad_norm_post,
                "train/lr": current_lr,
                "train/step": step,
            }, step=step)

        # --- Checkpoint save ---
        save_this_step = False
        if cfg.smoke_mode and step == cfg.smoke_save_step:
            save_this_step = True
            checkpoint_path = run_dir / "smoke_step50_checkpoint.pt"
        elif not cfg.smoke_mode and (step + 1) % cfg.save_freq == 0:
            save_this_step = True
            if cfg.save_latest_checkpoint_only:
                checkpoint_path = run_dir / "latest_checkpoint.pt"
            else:
                checkpoint_path = run_dir / f"step{step+1:07d}_checkpoint.pt"

        if save_this_step and is_main_process:
            # DDP 時は inner_model の state_dict を保存 (module. prefix 無し)、load 側は single GPU でも
            # DDP wrap 後でも同じ形式で読めるため互換性確保 (R15 acceptance criterion 4)。
            print(f"  [step {step+1}] saving checkpoint → {checkpoint_path}")
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model_vla=inner_model,
                optimizer=optimizer,
                scheduler=scheduler,
                gradient_step_idx=gradient_step_idx,
                current_lr=current_lr,
                cfg=cfg,
            )

            # --- Resume 検証 (smoke + single GPU のみ、DDP 時は subprocess spawn 互換性未検証で skip) ---
            if do_resume_check_after_this_step and not cfg.ddp_mode:
                print(f"  [step {step+1}] running resume verification (subprocess)...")
                # 次 batch (step 51 用) を取得、これを loss_a/loss_b の input として使う
                next_batch = next(data_iter)
                resume_check_result = run_resume_verification(
                    cfg=cfg,
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    model_vla=inner_model,
                    optimizer=optimizer,
                    current_lr_at_save=current_lr,
                    gradient_step_idx_at_save=gradient_step_idx,
                    next_batch=next_batch,
                )
                print(f"  [step {step+1}] resume check: all_pass={resume_check_result['all_pass']}")
                for name, crit in resume_check_result["criteria"].items():
                    mark = "OK" if crit["pass"] else "FAIL"
                    print(f"    - {name:<32s} value={crit['value']:.3e}  tol={crit['tol']:.0e}  [{mark}]")
                assert resume_check_result["all_pass"], \
                    "resume verification FAILED — Escalation #3 (即停止)"

        # DDP 時は save/resume で rank 間同期を確保
        if cfg.ddp_mode and save_this_step:
            dist.barrier()

    # --- Final summary ---
    total_sec = sum(step_times_all)
    step_time_warmup_excluded = step_times_all[2:] if len(step_times_all) > 2 else step_times_all
    median_step_sec = sorted(step_time_warmup_excluded)[len(step_time_warmup_excluded) // 2] if step_time_warmup_excluded else 0.0

    loss_init = step_results[0]["loss"]
    loss_final = step_results[-1]["loss"]
    loss_max = max(r["loss"] for r in step_results)
    loss_min = min(r["loss"] for r in step_results)
    post_clip_vals = [r["grad_norm_post"] for r in step_results]

    if is_main_process:
        print("\n" + "=" * 80)
        print(f"Finished {len(step_results)} steps in {total_sec:.1f}s")
        print(f"  median step sec (step 3+): {median_step_sec:.3f}")
        print(f"  loss: init={loss_init:.4f}  min={loss_min:.4f}  max={loss_max:.4f}  final={loss_final:.4f}")
        print(f"  post-clip grad_norm range: [{min(post_clip_vals):.4f}, {max(post_clip_vals):.4f}]")
        if cfg.ddp_mode:
            # DDP 用の acceptance 指標 (criterion 3): throughput 計算用
            all_reduce_params = sum(p.numel() for p in model_vla.parameters() if p.requires_grad)
            print(f"  ddp throughput estimate: {world_size} × single-GPU baseline / median_step_sec (D1 acceptance criterion 3)")
            print(f"  ddp all-reduce scope: {all_reduce_params/1e6:.3f}M params (criterion 6)")

    # JSON dump (rank 0 のみ、smoke/prod どちらも)
    if is_main_process:
        phase_id = "2b" if cfg.smoke_mode else "2e"
        if cfg.ddp_mode and cfg.smoke_mode:
            phase_id = "d1"   # DDP pre-flight smoke
        summary = {
            "phase": phase_id,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()},
            "num_steps_executed": len(step_results),
            "total_sec": round(total_sec, 2),
            "median_step_sec_exc_warmup": round(median_step_sec, 4),
            "world_size": world_size,
            "ddp_mode": cfg.ddp_mode,
            "all_reduce_scope_params": (
                sum(p.numel() for p in model_vla.parameters() if p.requires_grad) if cfg.ddp_mode else None
            ),
            "steps": step_results,
            "loss_summary": {"init": loss_init, "min": loss_min, "max": loss_max, "final": loss_final},
            "post_clip_grad_norm_range": [min(post_clip_vals), max(post_clip_vals)],
            "llm_leak_all_zero": llm_leak_all_zero,
            "resume_check": resume_check_result,
        }
        out_fname = "d1_ddp_smoke_result.json" if (cfg.ddp_mode and cfg.smoke_mode) else \
                    ("smoke_result.json" if cfg.smoke_mode else "train_result.json")
        out_path = run_dir / out_fname
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary JSON: {out_path}")

    # --- DDP cleanup ---
    if cfg.ddp_mode:
        dist.destroy_process_group()


if __name__ == "__main__":
    finetune()
