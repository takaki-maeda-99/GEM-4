"""Task 6 smoke: VLAAdapterGemma4 が Gemma 4 native vision で動くか。
Run:
    CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_task6_native_vision.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "VLA-Adapter"))

import torch
from transformers import Gemma4ForConditionalGeneration
from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4


def main():
    device = torch.device("cuda")

    print("[1] Load Gemma4ForConditionalGeneration ...")
    gemma = Gemma4ForConditionalGeneration.from_pretrained(
        "google/gemma-4-E2B",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    print("[2] Instantiate VLAAdapterGemma4 (no vision_backbone arg) ...")
    model = VLAAdapterGemma4(
        gemma_model=gemma,
        max_soft_tokens=280,          # spec rev 3 default
        proprio_dim=8,
        action_dim=7,
        num_action_chunks=8,
        num_pretrain_datasets=1,
        num_soft_prompt_tokens=32,
    ).to(device)
    model.eval()

    # num_vision_tokens は max_soft_tokens=280 で 256 になる (実測値)
    assert model.num_vision_tokens == 256, \
        f"expected 256 vision tokens at max_soft_tokens=280, got {model.num_vision_tokens}"
    print(f"  model.num_vision_tokens = {model.num_vision_tokens}")

    print("[3] Run vision preprocess + get_image_features ...")
    # fake 2-image batch, 224x224 uint8-equivalent
    scene_imgs = torch.rand(2, 3, 224, 224) * 255   # CPU, [0, 255]
    h_v = model.encode_scene(scene_imgs)
    expected = (2, 256, 1536)
    assert tuple(h_v.shape) == expected, \
        f"expected scene vision shape {expected}, got {tuple(h_v.shape)}"
    print(f"  h_v shape = {tuple(h_v.shape)}")

    print("[4] Verify embed_vision presence (not multi_modal_projector) ...")
    assert hasattr(model.llm.model, "embed_vision"), \
        "Gemma 4 expects .embed_vision attribute"
    assert not hasattr(model.llm.model, "multi_modal_projector"), \
        "multi_modal_projector must not be present"
    print(f"  embed_vision OK, no multi_modal_projector")

    print("[5] Verify vision_tower + embed_vision frozen ...")
    for n, p in model.llm.model.vision_tower.named_parameters():
        assert not p.requires_grad, f"vision_tower param {n} must be frozen"
    for n, p in model.llm.model.embed_vision.named_parameters():
        assert not p.requires_grad, f"embed_vision param {n} must be frozen"
    print(f"  vision_tower + embed_vision all frozen")

    print("\nOK: Task 6 smoke passed")


if __name__ == "__main__":
    main()
