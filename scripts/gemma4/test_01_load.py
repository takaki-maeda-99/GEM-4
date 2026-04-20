"""
Phase 1b.1: Load + text forward + baseline 測定

目的:
  - Gemma 4 E2B を single GPU でロードし、text-only forward が走ることを確認
  - 36 entries (embedding + 35 layers) の hidden state std/mean を記録
    → 1b.5 の LayerNorm 配置判定の根拠データ
  - Peak memory を計測 (Exit criterion: < 15GB)

Attn implementation: sdpa (User 判断, 2026-04-19)
  理由: flash-attn は Gemma 4 global 層 (head_dim=512) で fallback 発生 +
        install コスト大。Phase 1c の batch 拡大時に改めて検討。

Run: .venv-gemma4/bin/python scripts/gemma4/test_01_load.py
"""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B"
LOG_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "gemma4_migration_log.md"


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Initial free memory: {torch.cuda.mem_get_info(0)[0] / 1024**3:.2f} GB")

    # -----------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------
    print("\n=== Loading tokenizer ===")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # R2b: pad_token_id 確認 (無ければ eos で fallback)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    assert tok.pad_token_id is not None, "pad_token_id still None after fallback"
    print(f"pad_token_id={tok.pad_token_id}, eos_token_id={tok.eos_token_id}, bos_token_id={tok.bos_token_id}")

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    print("\n=== Loading model (bf16, sdpa) ===")
    torch.cuda.reset_peak_memory_stats(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    load_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"Peak memory after load: {load_peak_gb:.2f} GB")

    # R1: 明示 config
    model.config.use_cache = True
    assert not model.is_gradient_checkpointing, "gradient_checkpointing should be off"

    # 凍結 (Stage 1 前提 / R4)
    for p in model.parameters():
        p.requires_grad = False
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen params:   {frozen_params / 1e6:.1f}M")
    print(f"Trainable params: {trainable_params / 1e6:.1f}M (expected 0)")
    assert trainable_params == 0

    # Module 階層確認 (R2)
    llm = model.model.language_model
    hidden_size = model.config.text_config.hidden_size
    n_layers = model.config.text_config.num_hidden_layers
    print(f"\nModule hierarchy check:")
    print(f"  model.model.language_model: {type(llm).__name__}")
    print(f"  embed_tokens:               {type(llm.embed_tokens).__name__}")
    print(f"  embed_tokens_per_layer:     {type(llm.embed_tokens_per_layer).__name__}")
    print(f"  hidden_size:                {hidden_size}")
    print(f"  num_hidden_layers:          {n_layers}")
    assert hidden_size == 1536
    assert n_layers == 35

    # -----------------------------------------------------------------
    # Text-only forward
    # -----------------------------------------------------------------
    print("\n=== Text-only forward (B=1, L=100) ===")
    prompt = "hello world " * 20
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)[:, :100]
    B, L = input_ids.shape
    print(f"input_ids.shape = {tuple(input_ids.shape)}")

    torch.cuda.reset_peak_memory_stats(0)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    fwd_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3
    total_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3  # same reading

    # -----------------------------------------------------------------
    # T3: 各層の std/mean 測定
    # -----------------------------------------------------------------
    print("\n=== Hidden state statistics per layer ===")
    print(f"{'layer':>6} {'std':>8} {'mean':>9} {'min':>9} {'max':>9} {'finite':>7}")
    stats = []
    for i, h in enumerate(out.hidden_states):
        h_std = h.float().std().item()
        h_mean = h.float().mean().item()
        h_min = h.float().min().item()
        h_max = h.float().max().item()
        h_finite = bool(torch.isfinite(h).all().item())
        stats.append(
            {"layer": i, "std": h_std, "mean": h_mean, "min": h_min, "max": h_max, "finite": h_finite}
        )
        print(f"{i:>6d} {h_std:>8.3f} {h_mean:>9.3f} {h_min:>9.3f} {h_max:>9.3f} {str(h_finite):>7s}")

    # -----------------------------------------------------------------
    # Exit criteria (assertions)
    # -----------------------------------------------------------------
    print("\n=== Exit criteria check ===")
    # (1) peak memory
    print(f"(1) peak_gb = {total_peak_gb:.2f} GB (< 15.0 ?)")
    assert total_peak_gb < 15.0, f"Peak memory {total_peak_gb:.2f} GB exceeds 15 GB budget"

    # (2) 全 36 entries finite
    all_finite = all(s["finite"] for s in stats)
    print(f"(2) all_finite = {all_finite}")
    assert all_finite, "Some hidden states contain NaN/Inf"
    assert len(stats) == n_layers + 1, f"Expected {n_layers + 1} entries, got {len(stats)}"

    # (3) 全層 std が [0.1, 10] 範囲
    stds = [s["std"] for s in stats]
    min_std, max_std = min(stds), max(stds)
    in_range = all(0.1 <= s <= 10.0 for s in stds)
    print(f"(3) std range = [{min_std:.3f}, {max_std:.3f}]  in_range = {in_range}")
    assert in_range, f"std out of [0.1, 10] range: min={min_std:.3f}, max={max_std:.3f}"

    # (4) LayerNorm 判定の事前情報
    n_over_2 = sum(1 for s in stds if s > 2.0)
    n_over_5 = sum(1 for s in stds if s > 5.0)
    n_boundary = sum(1 for s in stds if 1.8 <= s <= 2.2)
    print(f"(4) std > 2.0 : {n_over_2} entries")
    print(f"    std > 5.0 : {n_over_5} entries")
    print(f"    std ∈ [1.8, 2.2] (境界): {n_boundary} entries")

    # LayerNorm ルール (1b.5 で使用) ― Claude Code は境界判定を User に委ねる
    if n_boundary > 0:
        layernorm_decision = "BOUNDARY - User 判断要請"
    elif max_std <= 2.0:
        layernorm_decision = "不要 (全層 std ≤ 2.0)"
    elif (n_over_2 in (1, 2, 3)) and n_over_5 == 0:
        layernorm_decision = f"sparse — {n_over_2} 層のみ該当層に LayerNorm"
    elif n_over_2 >= 4 or n_over_5 >= 1:
        layernorm_decision = "action head 入力側に 1 枚 (v1 パターン)"
    else:
        layernorm_decision = "UNDETERMINED - User 判断要請"
    print(f"    LayerNorm 判定: {layernorm_decision}")

    # -----------------------------------------------------------------
    # Dump JSON (機械可読、後続 Phase で利用)
    # -----------------------------------------------------------------
    summary = {
        "phase": "1b.1",
        "model_id": MODEL_ID,
        "attn_implementation": "sdpa",
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "load_peak_gb": round(load_peak_gb, 3),
        "forward_peak_gb": round(total_peak_gb, 3),
        "batch_size": B,
        "seq_len": L,
        "hidden_size": hidden_size,
        "num_hidden_layers": n_layers,
        "stats": stats,
        "std_min": min_std,
        "std_max": max_std,
        "n_std_over_2": n_over_2,
        "n_std_over_5": n_over_5,
        "n_std_boundary": n_boundary,
        "layernorm_decision": layernorm_decision,
    }
    json_path = Path(__file__).resolve().parent / "test_01_result.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {json_path}")

    print("\n=== Phase 1b.1 OK ===")
    return summary


if __name__ == "__main__":
    main()
