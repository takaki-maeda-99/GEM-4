"""
Phase 1b.2: `inputs_embeds` PLE 罠の切り分け + 対策検証

目的:
  - v5.1 で解明された PLE 真因 (逆引き処理による (B, L, V, D) 巨大 tensor) の実機確認
  - 対策 (`per_layer_inputs` を一緒に渡す) の効果を数値で検証
  - 10 試行 (2 cases × 5 (B, L) configurations) の memory を記録
  - 結果に基づき 1b.3 の実装方針を決定

Case 1: inputs_embeds のみ (per_layer_inputs なし) — OOM 予想
Case 2: per_layer_inputs 付き (対策版) — 成功予想

Run: CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_02_inputs_embeds.py
"""
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B"

# plan v5.2 より
CASES = [
    (1, 100),   # baseline
    (1, 500),   # seq 線形?
    (1, 1000),  # seq 非線形?
    (4, 500),   # batch 線形?
    (4, 1000),  # batch × seq 相互作用 (OOM 覚悟)
]


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)


def run_case1(llm, B, L, hidden_size, dtype):
    """Case 1: inputs_embeds のみ (per_layer_inputs なし)."""
    dummy_embeds = torch.randn(B, L, hidden_size, dtype=dtype, device="cuda:0")
    with torch.no_grad():
        _ = llm(inputs_embeds=dummy_embeds)
    return torch.cuda.max_memory_allocated(0) / 1024**3


def run_case2(llm, B, L, hidden_size, dtype):
    """Case 2: per_layer_inputs 付き (対策版)."""
    # vocab_size = 262144 (Gemma 4 E2B)
    dummy_ids = torch.randint(0, 262144, (B, L), dtype=torch.long, device="cuda:0")
    dummy_embeds = torch.randn(B, L, hidden_size, dtype=dtype, device="cuda:0")
    with torch.no_grad():
        per_layer_inputs = llm.get_per_layer_inputs(dummy_ids, None)
    with torch.no_grad():
        _ = llm(inputs_embeds=dummy_embeds, per_layer_inputs=per_layer_inputs)
    return torch.cuda.max_memory_allocated(0) / 1024**3


def main():
    assert torch.cuda.is_available()
    print(f"Device: cuda:0 ({torch.cuda.get_device_name(0)})")

    # Model load (Phase 1b.1 と同じ条件)
    print("\n=== Loading model (bf16, sdpa) ===")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0").eval()
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad = False

    llm = model.model.language_model  # R2: 階層統一
    hidden_size = model.config.text_config.hidden_size
    dtype = torch.bfloat16

    baseline_peak = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"After load peak: {baseline_peak:.2f} GB")

    # -----------------------------------------------------------------
    # Case 1: inputs_embeds のみ
    # -----------------------------------------------------------------
    print("\n=== Case 1: inputs_embeds のみ (per_layer_inputs なし、OOM 予想) ===")
    case1_results = []
    for B, L in CASES:
        reset_memory()
        try:
            peak = run_case1(llm, B, L, hidden_size, dtype)
            status = "OK"
            reason = ""
            print(f"  B={B:>2d} L={L:>4d}: peak={peak:.2f}GB OK")
        except torch.cuda.OutOfMemoryError as e:
            peak = None
            status = "OOM"
            reason = "cuda.OutOfMemoryError (PLE 逆引き発動)"
            print(f"  B={B:>2d} L={L:>4d}: OOM (PLE 逆引き)")
        except Exception as e:
            peak = None
            status = "ERROR"
            reason = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"  B={B:>2d} L={L:>4d}: ERROR {reason}")
        case1_results.append(
            {"B": B, "L": L, "peak_gb": peak, "status": status, "reason": reason}
        )
        reset_memory()

    # -----------------------------------------------------------------
    # Case 2: per_layer_inputs 付き (対策版)
    # -----------------------------------------------------------------
    print("\n=== Case 2: per_layer_inputs 付き (対策版、成功予想) ===")
    case2_results = []
    for B, L in CASES:
        reset_memory()
        try:
            peak = run_case2(llm, B, L, hidden_size, dtype)
            status = "OK"
            reason = ""
            print(f"  B={B:>2d} L={L:>4d}: peak={peak:.2f}GB OK (対策有効)")
        except torch.cuda.OutOfMemoryError as e:
            peak = None
            status = "OOM"
            reason = "cuda.OutOfMemoryError (対策無効、要再調査)"
            print(f"  B={B:>2d} L={L:>4d}: OOM (対策無効)")
        except Exception as e:
            peak = None
            status = "ERROR"
            reason = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"  B={B:>2d} L={L:>4d}: ERROR {reason}")
        case2_results.append(
            {"B": B, "L": L, "peak_gb": peak, "status": status, "reason": reason}
        )
        reset_memory()

    # -----------------------------------------------------------------
    # 判定
    # -----------------------------------------------------------------
    print("\n=== Summary Table ===")
    print(f"{'B':>3} {'L':>5} | {'Case1 peak':>12} {'Case1 status':>13} | {'Case2 peak':>12} {'Case2 status':>13}")
    print("-" * 78)
    for c1, c2 in zip(case1_results, case2_results):
        p1 = f"{c1['peak_gb']:.2f} GB" if c1["peak_gb"] is not None else "-"
        p2 = f"{c2['peak_gb']:.2f} GB" if c2["peak_gb"] is not None else "-"
        print(f"{c1['B']:>3d} {c1['L']:>5d} | {p1:>12s} {c1['status']:>13s} | {p2:>12s} {c2['status']:>13s}")

    # 判定分岐 (plan v5.2 の表に従う)
    case1_has_oom = any(r["status"] == "OOM" for r in case1_results)
    case2_all_ok = all(r["status"] == "OK" for r in case2_results)

    print("\n=== Diagnosis ===")
    if case1_has_oom and case2_all_ok:
        diagnosis = "v5.1 の真因分析が正しい → 1b.3 は Option A (per_layer_inputs + clone + indexing)"
        verdict = "OPTION_A"
    elif not case1_has_oom:
        diagnosis = "Case 1 も全 pass (5.5.x で修正済み?) → 1b.3 は シンプル採用 (inputs_embeds 直渡し)"
        verdict = "SIMPLE"
    elif case1_has_oom and not case2_all_ok:
        diagnosis = "Case 1/Case 2 両方 OOM → 我々の理解が間違い。User 報告要"
        verdict = "ESCALATE"
    else:
        diagnosis = "想定外パターン"
        verdict = "UNKNOWN"
    print(f"Diagnosis: {diagnosis}")
    print(f"Verdict:   {verdict}")

    # -----------------------------------------------------------------
    # JSON dump
    # -----------------------------------------------------------------
    summary = {
        "phase": "1b.2",
        "case1_results": case1_results,
        "case2_results": case2_results,
        "case1_has_oom": case1_has_oom,
        "case2_all_ok": case2_all_ok,
        "diagnosis": diagnosis,
        "verdict": verdict,
    }
    json_path = Path(__file__).resolve().parent / "test_02_result.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {json_path}")
    print("\n=== Phase 1b.2 done ===")
    return summary


if __name__ == "__main__":
    main()
