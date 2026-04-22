"""Phase 0.5 dual-track smoke test (Task 15).

For each mode in {quality, speed}:
  1. Build VLAAdapterGemma4 with the correct config (including LoRA for quality mode).
  2. Construct a synthetic batch with correct input_ids placeholder layout
     (matching taco_solo_loader: 1 BOS + 20 prompt PAD + 256 VISION + 1 PROPRIO + 64 ACTION + 1 EOS = 343).
  3. Run forward + backward.
  4. Assert loss is finite (no NaN).
  5. Print forward+backward time and peak GPU memory.
  6. Mode-specific grad checks:
     - speed: action_queries.weight.grad is None (frozen, zero-init)
     - quality: action_queries.weight.grad is populated (trainable)

This is the gate before starting full pretrain runs (Phase 2).

Run:
  CUDA_VISIBLE_DEVICES=4 .venv-gemma4/bin/python scripts/gemma4/test_13_dual_track_smoke.py
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
from prismatic.vla.constants_gemma4 import (
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTION_TOKENS,
    PROPRIO_PLACEHOLDER_IDX,
    VISION_PLACEHOLDER_BEGIN_IDX,
)

PROMPT_MAX_LEN = 20
MODEL_ID = "google/gemma-4-E2B"


def build_input_ids(tok, B: int, device, num_vision_tokens: int = 256) -> torch.Tensor:
    """Construct input_ids matching the layout expected by VLAAdapterGemma4.forward.

    Layout (per taco_solo_loader):
      [BOS] + prompt(20 PAD) + VISION(num_vision_tokens) + [PROPRIO] + ACTION(64) + [EOS]

    Vision/action placeholder IDs are spread across their reserved ranges so that
    forward's range-based mask `(id >= BEGIN) & (id < BEGIN + N)` picks them up.
    """
    bos = tok.bos_token_id
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos

    L = 1 + PROMPT_MAX_LEN + num_vision_tokens + 1 + NUM_ACTION_TOKENS + 1
    ids = torch.full((B, L), pad, dtype=torch.long, device=device)
    off = 0
    ids[:, off] = bos
    off += 1
    # prompt: pad (no real text in smoke)
    off += PROMPT_MAX_LEN
    # vision placeholders (distinct IDs per position, all inside the VISION range)
    ids[:, off : off + num_vision_tokens] = torch.arange(
        VISION_PLACEHOLDER_BEGIN_IDX,
        VISION_PLACEHOLDER_BEGIN_IDX + num_vision_tokens,
        device=device,
        dtype=torch.long,
    )
    off += num_vision_tokens
    # proprio
    ids[:, off] = PROPRIO_PLACEHOLDER_IDX
    off += 1
    # action placeholders (64 consecutive IDs inside the ACTION range)
    ids[:, off : off + NUM_ACTION_TOKENS] = torch.arange(
        ACTION_TOKEN_BEGIN_IDX,
        ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS,
        device=device,
        dtype=torch.long,
    )
    off += NUM_ACTION_TOKENS
    ids[:, off] = eos
    return ids


def build_model(mode: str, device: str = "cuda", max_soft_tokens: int = 280):
    gemma = (
        Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )

    model = VLAAdapterGemma4(
        gemma_model=gemma,
        max_soft_tokens=max_soft_tokens,
        num_pretrain_datasets=1,
        num_soft_prompt_tokens=32,
        training_mode=mode,
    ).to(device, dtype=torch.bfloat16)

    if mode == "quality":
        # LoRA wrap first, then enable gradient checkpointing on the wrapped module.
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type=None,
        )
        model.llm.model.language_model = get_peft_model(
            model.llm.model.language_model, lora_cfg
        )
        # GC 有効化: PEFT 経由 wrapped base model に対しても動く。
        if hasattr(model.llm.model.language_model, "gradient_checkpointing_enable"):
            model.llm.model.language_model.gradient_checkpointing_enable()
        else:
            model.llm.model.language_model.base_model.gradient_checkpointing_enable()
    elif mode == "speed":
        # action_queries は zero init かつ frozen (LLM は no_grad wrap で grad 流れない)
        model.action_queries.weight.data.zero_()
        model.action_queries.weight.requires_grad = False

    model.train()
    return model


def build_dummy_batch(B: int, device, tok, num_vision_tokens: int = 256):
    return {
        "pixel_values": {
            # scene: CPU float [0, 255] — Gemma4ImageProcessor will rescale+resize internally.
            "scene": torch.rand(B, 3, 224, 224) * 255,
            # wrist: GPU bf16, roughly ImageNet-normalized — WristResNet18 input contract.
            "wrist": torch.randn(B, 3, 224, 224, device=device, dtype=torch.bfloat16),
        },
        "input_ids": build_input_ids(tok, B, device, num_vision_tokens=num_vision_tokens),
        "proprio": torch.randn(B, 8, device=device, dtype=torch.bfloat16),
        "actions": torch.randn(B, 8, 7, device=device, dtype=torch.bfloat16),
        "dataset_id": torch.zeros(B, device=device, dtype=torch.long),
    }


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    results = {}
    for mode in ("quality", "speed"):
        print(f"\n=== mode = {mode} ===", flush=True)
        model = build_model(mode)
        batch = build_dummy_batch(
            B=2, device="cuda", tok=tok, num_vision_tokens=model.num_vision_tokens
        )

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
        print(
            f"loss = {loss.item():.4f}, forward+backward = {t1 - t0:.3f} s, peak mem = {peak_gb:.2f} GB",
            flush=True,
        )
        results[mode] = {"loss": loss.item(), "time": t1 - t0, "peak_gb": peak_gb}

        if mode == "speed":
            assert model.action_queries.weight.grad is None, (
                f"speed mode: action_queries.weight.grad should be None, got "
                f"{None if model.action_queries.weight.grad is None else model.action_queries.weight.grad.shape}"
            )
            print("OK: action_queries grad is None (frozen)", flush=True)
        elif mode == "quality":
            assert model.action_queries.weight.grad is not None, (
                "quality mode: action_queries.weight.grad should be populated"
            )
            print("OK: action_queries grad is populated (trainable)", flush=True)

        del model
        torch.cuda.empty_cache()

    print("\n=== summary ===", flush=True)
    print(
        f"quality: loss={results['quality']['loss']:.4f}, "
        f"time={results['quality']['time']:.3f}s, "
        f"peak={results['quality']['peak_gb']:.2f}GB",
        flush=True,
    )
    print(
        f"speed:   loss={results['speed']['loss']:.4f}, "
        f"time={results['speed']['time']:.3f}s, "
        f"peak={results['speed']['peak_gb']:.2f}GB",
        flush=True,
    )
    mem_reduction = (
        1 - results["speed"]["peak_gb"] / results["quality"]["peak_gb"]
    ) * 100
    print(f"memory reduction (speed vs quality): {mem_reduction:.1f}%", flush=True)


if __name__ == "__main__":
    main()
