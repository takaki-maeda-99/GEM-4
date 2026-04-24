"""DDP 2-way speed bench: single (mode, max_soft_tokens) config.

Measures forward+backward time under DDP 2-way (realistic training condition).
Synthetic batch — no data loader, no loss curve, just compute speed + memory.

Run (torchrun, 2 GPUs):
  CUDA_VISIBLE_DEVICES=2,3 .venv-gemma4/bin/torchrun --standalone --nproc_per_node=2 \
    --master_port=29510 scripts/gemma4/bench_max_soft_tokens_ddp.py \
    --mode speed --max_soft_tokens 70

Configurable via CLI:
  --mode {speed, quality}
  --max_soft_tokens {70, 140, 280, 560, 1120}
  --batch_size (default 24)
  --steps (default 30; first 5 warmup)
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
sys.path.insert(0, str(VLA_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
WARMUP_STEPS = 5


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["speed", "quality"], required=True)
    parser.add_argument("--max_soft_tokens", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if rank == 0:
        print(f"\n=== bench: mode={args.mode}, max_soft_tokens={args.max_soft_tokens}, "
              f"B={args.batch_size} per GPU, DDP {world_size}-way ===")

    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")

    gemma = Gemma4ForConditionalGeneration.from_pretrained(
        "google/gemma-4-E2B",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    model = VLAAdapterGemma4(
        gemma_model=gemma,
        max_soft_tokens=args.max_soft_tokens,
        num_pretrain_datasets=1,
        num_soft_prompt_tokens=32,
        training_mode=args.mode,
    ).to(device, dtype=torch.bfloat16)
    if args.mode == "quality":
        model.llm.model.language_model.gradient_checkpointing_enable()
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none", task_type=None,
        )
        model.llm.model.language_model = get_peft_model(
            model.llm.model.language_model, lora_cfg
        )
    elif args.mode == "speed":
        model.action_queries.weight.data.zero_()
        model.action_queries.weight.requires_grad = False
    model.train()

    # DDP wrap trainable params only
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4, betas=(0.9, 0.95),
    )

    B = args.batch_size
    num_vision_tokens = model.module.num_vision_tokens
    batch = {
        "pixel_values": {
            "scene": torch.rand(B, 3, 224, 224) * 255,
            "wrist": torch.randn(B, 3, 224, 224, device=device, dtype=torch.bfloat16),
        },
        "input_ids": build_input_ids(tok, B, device, num_vision_tokens),
        "proprio": torch.randn(B, 8, device=device, dtype=torch.bfloat16),
        "actions": torch.randn(B, 8, 7, device=device, dtype=torch.bfloat16),
        "dataset_id": torch.zeros(B, device=device, dtype=torch.long),
    }

    torch.cuda.reset_peak_memory_stats()
    step_times = []
    for step in range(args.steps):
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

    # Gather from all ranks
    mean_tensor = torch.tensor([mean_s], device=device)
    peak_tensor = torch.tensor([peak_gb], device=device)
    dist.all_reduce(mean_tensor, op=dist.ReduceOp.AVG)
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"  num_vision_tokens per image: {num_vision_tokens}")
        print(f"  mean step time (last {len(step_times)} steps, avg over {world_size} ranks): "
              f"{mean_tensor.item():.3f} s")
        print(f"  peak memory (max over ranks): {peak_tensor.item():.2f} GB")
        print(f"  effective batch: {B * world_size}")
        print(f"  throughput: {B * world_size / mean_tensor.item():.2f} samples/sec")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
