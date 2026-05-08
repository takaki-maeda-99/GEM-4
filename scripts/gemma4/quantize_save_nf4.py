"""
google/gemma-4-26b-a4b-it を NF4 で 1 度だけ quantize して disk に保存する。

後続: from_pretrained(quantized_path, quantization_config=bnb_cfg) で
steady-state (4bit, ~13GB) ロードのみとなり、bf16 transient peak が発生しない。
40GB 単機にも収まるようになる。

Run:
  CUDA_VISIBLE_DEVICES=3,4 .venv-gemma4/bin/python scripts/gemma4/quantize_save_nf4.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma4ForConditionalGeneration

SRC_MODEL_ID = "google/gemma-4-26b-a4b-it"
DST_DIR = Path("/misc/dl00/takaki/quantized_models/gemma-4-26b-a4b-it-nf4")


def main():
    assert torch.cuda.is_available()
    print(f"Source : {SRC_MODEL_ID}")
    print(f"Dest   : {DST_DIR}")
    print(f"Visible GPUs: {torch.cuda.device_count()}")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n=== Loading + on-the-fly NF4 quantization ===")
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        SRC_MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()
    load_t = time.time() - t0
    print(f"Loaded in {load_t:.1f}s")

    # device_map per layer
    print("\n=== Device map ===")
    if hasattr(model, "hf_device_map"):
        from collections import Counter
        c = Counter(model.hf_device_map.values())
        print(f"  layer→device counts: {dict(c)}")

    print("\n=== save_pretrained ===")
    DST_DIR.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    model.save_pretrained(DST_DIR, safe_serialization=True)
    save_t = time.time() - t1
    print(f"Saved in {save_t:.1f}s")

    tok = AutoTokenizer.from_pretrained(SRC_MODEL_ID)
    tok.save_pretrained(DST_DIR)

    sz = sum(f.stat().st_size for f in DST_DIR.rglob("*") if f.is_file())
    print(f"\nTotal on-disk size: {sz / 1e9:.2f} GB")
    print(f"Files:")
    for f in sorted(DST_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    print(f"\n[OK] NF4 ckpt at {DST_DIR}")


if __name__ == "__main__":
    main()
