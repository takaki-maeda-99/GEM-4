"""Quick speed bench: max_soft_tokens × mode (synthetic batch).

Measures forward+backward time at different max_soft_tokens values for both modes.
No real data pipeline, no loss curve — just compute speed and memory.

Run (single GPU):
  CUDA_VISIBLE_DEVICES=4 .venv-gemma4/bin/python scripts/gemma4/bench_max_soft_tokens.py

Configs tested:
  max_soft_tokens ∈ {70, 140, 280} × mode ∈ {quality, speed} = 6 configs
  batch_size = 24 (per Medium phase)
  steps = 30 (first 5 = warmup, report mean of last 25)
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
B = 24   # batch_size (per Medium phase)
WARMUP_STEPS = 5
MEASURE_STEPS = 25
TOTAL_STEPS = WARMUP_STEPS + MEASURE_STEPS


def build_input_ids(tok, B, device, num_vision_tokens):
    bos = tok.bos_token_id
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    L = 1 + PROMPT_MAX_LEN + num_vision_tokens + 1 + NUM_ACTION_TOKENS + 1
    ids = torch.full((B, L), pad, dtype=torch.long, device=device)
    off = 0
    ids[:, off] = bos; off += 1
    off += PROMPT_MAX_LEN
    ids[:, off : off + num_vision_tokens] = torch.arange(
        VISION_PLACEHOLDER_BEGIN_IDX, VISION_PLACEHOLDER_BEGIN_IDX + num_vision_tokens,
        device=device, dtype=torch.long,
    )
    off += num_vision_tokens
    ids[:, off] = PROPRIO_PLACEHOLDER_IDX; off += 1
    ids[:, off : off + NUM_ACTION_TOKENS] = torch.arange(
        ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS,
        device=device, dtype=torch.long,
    )
    off += NUM_ACTION_TOKENS
    ids[:, off] = eos
    return ids


def build_model(mode, max_soft_tokens, device="cuda"):
    gemma = Gemma4ForConditionalGeneration.from_pretrained(
        "google/gemma-4-E2B",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    model = VLAAdapterGemma4(
        gemma_model=gemma,
        max_soft_tokens=max_soft_tokens,
        num_pretrain_datasets=1,
        num_soft_prompt_tokens=32,
        training_mode=mode,
    ).to(device, dtype=torch.bfloat16)
    if mode == "quality":
        model.llm.model.language_model.gradient_checkpointing_enable()
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none", task_type=None,
        )
        model.llm.model.language_model = get_peft_model(
            model.llm.model.language_model, lora_cfg
        )
    elif mode == "speed":
        model.action_queries.weight.data.zero_()
        model.action_queries.weight.requires_grad = False
    model.train()
    return model


def run_bench(mode, max_soft_tokens, tok, device="cuda"):
    print(f"\n--- mode={mode}, max_soft_tokens={max_soft_tokens} ---")
    model = build_model(mode, max_soft_tokens, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4, betas=(0.9, 0.95),
    )
    batch = {
        "pixel_values": {
            "scene": torch.rand(B, 3, 224, 224) * 255,
            "wrist": torch.randn(B, 3, 224, 224, device=device, dtype=torch.bfloat16),
        },
        "input_ids": build_input_ids(tok, B, device, model.num_vision_tokens),
        "proprio": torch.randn(B, 8, device=device, dtype=torch.bfloat16),
        "actions": torch.randn(B, 8, 7, device=device, dtype=torch.bfloat16),
        "dataset_id": torch.zeros(B, device=device, dtype=torch.long),
    }

    torch.cuda.reset_peak_memory_stats()
    step_times = []
    for step in range(TOTAL_STEPS):
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        predicted, loss = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            proprio=batch["proprio"],
            actions=batch["actions"],
            dataset_id=batch["dataset_id"],
        )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        if step >= WARMUP_STEPS:
            step_times.append(dt)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    mean_s = sum(step_times) / len(step_times)
    print(f"  num_vision_tokens: {model.num_vision_tokens}")
    print(f"  mean step time (last {len(step_times)}): {mean_s:.3f} s")
    print(f"  peak memory: {peak_gb:.2f} GB")

    del model, optimizer
    torch.cuda.empty_cache()
    return {
        "mode": mode,
        "max_soft_tokens": max_soft_tokens,
        "num_vision_tokens": model.num_vision_tokens if False else None,  # computed above
        "mean_s_per_step": mean_s,
        "peak_gb": peak_gb,
    }


def main():
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    results = []
    for mst in (70, 140, 280):
        for mode in ("speed", "quality"):
            r = run_bench(mode, mst, tok)
            r["max_soft_tokens"] = mst
            r["mode"] = mode
            results.append(r)

    print("\n" + "=" * 70)
    print(f"{'mode':<10} {'max_soft':<10} {'s/step':<10} {'peak_gb':<10} {'speedup_vs_280':<15}")
    print("=" * 70)
    base = {r["mode"]: r["mean_s_per_step"] for r in results if r["max_soft_tokens"] == 280}
    for r in results:
        sp = base[r["mode"]] / r["mean_s_per_step"]
        print(f"{r['mode']:<10} {r['max_soft_tokens']:<10} {r['mean_s_per_step']:<10.3f} {r['peak_gb']:<10.2f} {sp:.2f}x")


if __name__ == "__main__":
    main()
