"""
AWQ 4bit (compressed-tensors) Gemma-4-26B-A4B-it backbone での 1-step forward smoke test.

確認:
  - cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit (17GB, MoE expert 含む 4bit) ロード
  - PLE 無効分岐 (per_layer_inputs=None) で Gemma4TextModel.forward が動く
  - VLAAdapterGemma4 wrapper が forward → loss を返す
  - peak VRAM, model load 時間, 1-step 時間
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from transformers.utils.quantization_config import CompressedTensorsConfig

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


MODEL_ID = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
B = 2
ACTION_DIM = 7
NUM_ACTION_CHUNKS = 8
PROPRIO_DIM = 8


def build_input_ids(tok, prompt: str, device):
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)[:, :50]
    bos = torch.tensor([[tok.bos_token_id]], dtype=torch.long, device=device)
    eos = torch.tensor([[tok.eos_token_id]], dtype=torch.long, device=device)
    vision_ids = torch.arange(
        VISION_PLACEHOLDER_BEGIN_IDX,
        VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS,
        device=device,
    ).unsqueeze(0)
    proprio_id = torch.tensor([[PROPRIO_PLACEHOLDER_IDX]], dtype=torch.long, device=device)
    action_ids = torch.arange(
        ACTION_TOKEN_BEGIN_IDX,
        ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS,
        device=device,
    ).unsqueeze(0)
    return torch.cat([bos, prompt_ids, vision_ids, proprio_id, action_ids, eos], dim=1)


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    torch.cuda.reset_peak_memory_stats()

    print(f"\n=== Loading {MODEL_ID} (AWQ 4bit / compressed-tensors, run_compressed=True) ===")
    t0 = time.time()
    # config.json の quantization_config を継承しつつ run_compressed=True を強制
    # (default False だと load 時に bf16 へ decompress されて 38GB+ になる)
    qcfg = CompressedTensorsConfig(run_compressed=True)
    gemma = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=qcfg,
        device_map={"": device},
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).eval()
    gemma.config.use_cache = True
    for p in gemma.parameters():
        p.requires_grad = False
    print(f"Backbone loaded in {time.time() - t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated() / 1e9:.2f} GB, "
          f"peak={torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  text_config.hidden_size={gemma.config.text_config.hidden_size}, "
          f"layers={gemma.config.text_config.num_hidden_layers}, "
          f"PLE={gemma.config.text_config.hidden_size_per_layer_input}, "
          f"MoE={gemma.config.text_config.enable_moe_block}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("\n=== Building VLAAdapterGemma4 wrapper (Mode B wrist_bridge) ===")
    model_vla = VLAAdapterGemma4(
        gemma_model=gemma,
        max_soft_tokens=280,
        feature_norm=torch.nn.Identity(),
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        num_action_chunks=NUM_ACTION_CHUNKS,
        num_pretrain_datasets=0,
        num_soft_prompt_tokens=32,
        training_mode="speed",  # Mode B
        vision_backbone_type="siglip",
        siglip_use_tensor_transform=True,
        use_xvla_style=False,
        use_wrist_bridge=True,
        use_proper_ffn=False,
        wrist_bridge_layer_mode="per_layer",
        num_action_head_blocks=24,
    )
    # NF4: skip moving llm child (bnb Linear4bit は dtype 変更不可)
    for name, child in model_vla.named_children():
        if name == "llm":
            continue
        child.to(device=device, dtype=torch.bfloat16)
    model_vla.train()

    n_train = sum(p.numel() for p in model_vla.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model_vla.parameters())
    print(f"Trainable: {n_train / 1e6:.1f}M / Total: {n_total / 1e6:.1f}M")
    print(f"VRAM after wrapper: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
          f"peak={torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\n=== Synthetic batch ===")
    prompt = "Pick up the alphabet soup and place it in the basket"
    ids_single = build_input_ids(tok, prompt, device)
    input_ids = ids_single.repeat(B, 1)
    pixel_values = {
        "scene": torch.randint(0, 256, (B, 3, 224, 224), device=device, dtype=torch.float32),
        "wrist": torch.randint(0, 256, (B, 3, 224, 224), device=device, dtype=torch.bfloat16),
    }
    proprio = torch.randn(B, PROPRIO_DIM, device=device, dtype=torch.bfloat16)
    actions = torch.randn(B, NUM_ACTION_CHUNKS, ACTION_DIM, device=device, dtype=torch.bfloat16)
    print(f"  input_ids: {tuple(input_ids.shape)}, scene: {tuple(pixel_values['scene'].shape)}, "
          f"wrist: {tuple(pixel_values['wrist'].shape)}, proprio: {tuple(proprio.shape)}, "
          f"actions: {tuple(actions.shape)}")

    print("\n=== Forward + loss (Mode B = LLM no_grad wrapped) ===")
    t1 = time.time()
    predicted, loss = model_vla(pixel_values, input_ids, proprio, actions)
    torch.cuda.synchronize()
    fwd_t = time.time() - t1
    print(f"  predicted.shape={tuple(predicted.shape)}, loss={loss.item():.4f}, fwd={fwd_t:.2f}s")
    print(f"  VRAM after fwd: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
          f"peak={torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\n=== Backward (adapter only) ===")
    t2 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    bwd_t = time.time() - t2
    print(f"  bwd={bwd_t:.2f}s, "
          f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    grad_ok = sum((p.grad is not None and torch.isfinite(p.grad).all().item())
                  for p in model_vla.parameters() if p.requires_grad)
    grad_total = sum(1 for p in model_vla.parameters() if p.requires_grad)
    print(f"  grads finite on {grad_ok}/{grad_total} trainable tensors")

    assert predicted.shape == (B, NUM_ACTION_CHUNKS, ACTION_DIM), predicted.shape
    assert torch.isfinite(loss).item()
    print("\n[PASS] NF4 + 26B-A4B forward/backward smoke OK")


if __name__ == "__main__":
    main()
