"""
HQQ で Gemma-4-26B-A4B を 4bit ロードできるか、
特に MoE expert (3D nn.Parameter) が量子化されるか確認する短いプローブ。
"""
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from transformers import Gemma4ForConditionalGeneration, HqqConfig

MODEL_ID = "google/gemma-4-26b-a4b-it"


def main():
    print("=== HQQ probe load ===")
    qcfg = HqqConfig(nbits=4, group_size=64, axis=1)
    print(f"HqqConfig: nbits=4, group_size=64")

    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=qcfg,
        device_map="auto",
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # tally by dtype
    from collections import Counter
    sz_per_dtype = Counter()
    cnt_per_dtype = Counter()

    def total_size(t):
        if hasattr(t, "numel"):
            return t.numel() * t.element_size()
        return 0

    for name, p in model.named_parameters():
        sz_per_dtype[str(p.dtype)] += total_size(p)
        cnt_per_dtype[str(p.dtype)] += 1
    for name, b in model.named_buffers():
        sz_per_dtype[str(b.dtype)] += total_size(b)
        cnt_per_dtype[str(b.dtype)] += 1

    print("\n=== Tensor totals by dtype (params + buffers) ===")
    for dt, sz in sorted(sz_per_dtype.items(), key=lambda x: -x[1]):
        print(f"  {dt}: {sz / 1e9:.2f} GB ({cnt_per_dtype[dt]} tensors)")

    # check expert state
    print("\n=== MoE expert tensor sample ===")
    for n, p in model.named_parameters():
        if "experts.gate_up_proj" in n or "experts.down_proj" in n:
            print(f"  {n}: dtype={p.dtype}, shape={tuple(p.shape)}")
            break

    # Linear vs HQQLinear count
    from torch.nn import Linear as TLinear
    try:
        from hqq.core.quantize import HQQLinear
    except Exception:
        HQQLinear = type("Dummy", (), {})
    n_linear = sum(1 for m in model.modules() if isinstance(m, TLinear))
    n_hqq = sum(1 for m in model.modules() if isinstance(m, HQQLinear))
    print(f"\n  nn.Linear remaining: {n_linear}")
    print(f"  HQQLinear placed:    {n_hqq}")

    print(f"\nVRAM allocated (cuda:0): {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    if torch.cuda.device_count() > 1:
        print(f"VRAM allocated (cuda:1): {torch.cuda.memory_allocated(1) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
