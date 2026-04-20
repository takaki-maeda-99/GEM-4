"""
Phase 1b.3: Embedding 注入 + 勾配検証 (最重要)

目的:
  - 64 個の distinct action queries を input embedding に正しく挿入
  - 勾配と semantic の両方を機械的に検証
  - forward / backward peak memory を記録 (Phase 1b.4 以降のベースライン)

=== Overwrite strategy decision ===
Chose **Option A** (clone + advanced indexing) because:
  - Preserves gradient flow to action_queries.weight (__setitem__ is autograd-safe for RHS)
  - Each of 64 positions receives its corresponding distinct query vector
  - Readable; easy to verify correctness

Rejected:
  - torch.where + broadcast: all 64 positions get same value (mode collapse trap,
    assertion をパスしても後段で mode collapse、しかも表面的には動く)
  - Option B (scatter): correct but harder to read
  - Option C (index_put_): correct but less intuitive

This decision follows Phase 1b.2 verdict = OPTION_A (Case 1 all fail, Case 2 all pass).

Run: CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_03_action_queries.py
"""
import gc
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------
# Import constants_gemma4 (prismatic/__init__ が draccus を要求するため
# importlib で直接ロード、test script 時点では prismatic 全体を初期化しない)
# -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import importlib.util as _ilu  # noqa: E402

_cg_path = REPO_ROOT / "VLA-Adapter" / "prismatic" / "vla" / "constants_gemma4.py"
_spec = _ilu.spec_from_file_location("constants_gemma4", _cg_path)
_cg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cg)
ACTION_TOKEN_BEGIN_IDX = _cg.ACTION_TOKEN_BEGIN_IDX
NUM_ACTION_TOKENS = _cg.NUM_ACTION_TOKENS

MODEL_ID = "google/gemma-4-E2B"


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # -----------------------------------------------------------------
    # Model load (1b.1/1b.2 と同条件)
    # -----------------------------------------------------------------
    print("\n=== Loading model (bf16, sdpa) ===")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad = False
    assert not model.is_gradient_checkpointing

    llm = model.model.language_model               # R2: 階層統一
    hidden_size = model.config.text_config.hidden_size  # 1536
    assert hidden_size == 1536

    # 前倒し: LLM 凍結の機械 verify (User 提案、1b.6 予定を前倒し)
    llm_trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    assert llm_trainable == 0, f"LLM should be frozen, got {llm_trainable} trainable params"
    print(f"LLM trainable params: {llm_trainable} (expected 0) OK")

    # -----------------------------------------------------------------
    # Action queries module (Stage 1 で学習対象、zero init)
    # -----------------------------------------------------------------
    action_queries = torch.nn.Embedding(NUM_ACTION_TOKENS, hidden_size).to(device, dtype=torch.bfloat16)
    action_queries.weight.data.zero_()

    aq_trainable = sum(p.numel() for p in action_queries.parameters() if p.requires_grad)
    print(f"action_queries trainable params: {aq_trainable} (expected {NUM_ACTION_TOKENS * hidden_size})")

    # -----------------------------------------------------------------
    # Input 構築: L=100, 位置 36..99 に action placeholder ID (258885..258948)
    # -----------------------------------------------------------------
    B, L = 1, 100
    input_ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long, device=device)
    input_ids[:, 36:100] = torch.arange(
        ACTION_TOKEN_BEGIN_IDX,
        ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS,
        device=device,
    )
    # 健全性: 64 個 placeholder があること
    amask = (input_ids >= ACTION_TOKEN_BEGIN_IDX) & (
        input_ids < ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS
    )
    assert amask.sum().item() == NUM_ACTION_TOKENS, f"expected {NUM_ACTION_TOKENS} placeholders, got {amask.sum().item()}"

    # -----------------------------------------------------------------
    # PLE 事前計算 (Option A の核心: OOM 回避)
    # -----------------------------------------------------------------
    with torch.no_grad():
        per_layer_inputs = llm.get_per_layer_inputs(input_ids, None)
    print(f"per_layer_inputs.shape = {tuple(per_layer_inputs.shape)} (expected (B, L, 35, 256))")
    assert per_layer_inputs.shape == (B, L, 35, 256)

    # -----------------------------------------------------------------
    # Embedding lookup + Option A 上書き
    # -----------------------------------------------------------------
    with torch.no_grad():
        raw_embeddings = llm.embed_tokens(input_ids)                # (B, L, 1536)
    print(f"raw_embeddings.shape   = {tuple(raw_embeddings.shape)}")
    embeddings = raw_embeddings.clone()                             # new autograd branch

    for b in range(B):
        positions = amask[b].nonzero(as_tuple=True)[0]
        assert positions.numel() == NUM_ACTION_TOKENS
        embeddings[b, positions] = action_queries.weight            # (64, D)

    # -----------------------------------------------------------------
    # attention_mask / position_ids (R6; L=100 < sliding=512 だが統一のため明示)
    # -----------------------------------------------------------------
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats(0)
    try:
        out = llm(
            inputs_embeds=embeddings,
            per_layer_inputs=per_layer_inputs,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        guard_triggered = False
        fallback_used = False
    except RuntimeError as e:
        msg = str(e)
        if "inputs_embeds" in msg and "input_ids" in msg:
            # User 提案の fallback: input_ids も同時に渡して guard を通す
            print(f"[guard triggered] RuntimeError: {msg[:200]}")
            print("Retrying with input_ids passed alongside inputs_embeds ...")
            guard_triggered = True
            fallback_used = True
            torch.cuda.reset_peak_memory_stats(0)
            out = llm(
                input_ids=input_ids,
                inputs_embeds=embeddings,
                per_layer_inputs=per_layer_inputs,
                use_cache=True,
                output_hidden_states=True,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            raise

    fwd_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"\nForward peak: {fwd_peak_gb:.2f} GB")
    print(f"last_hidden_state.shape = {tuple(out.last_hidden_state.shape)} (expected (B, L, 1536))")
    print(f"len(out.hidden_states)  = {len(out.hidden_states)} (expected 36 = embedding + 35)")

    # -----------------------------------------------------------------
    # 2. Semantic 検証 (torch.where 罠検知) — BEFORE backward
    # -----------------------------------------------------------------
    print("\n=== Semantic verification (torch.where trap detection) ===")
    hidden = out.last_hidden_state                                    # (B, L, D)
    positions_b0 = amask[0].nonzero(as_tuple=True)[0]                # (64,)
    aq_hiddens = hidden[0, positions_b0].detach().float()            # (64, D)

    # (S1) 隣接 2 個が distinct
    pair_diff = (aq_hiddens[0] - aq_hiddens[1]).abs().max().item()
    print(f"  (S1) |h[0] - h[1]|_max = {pair_diff:.6e}")
    assert not torch.allclose(aq_hiddens[0], aq_hiddens[1], atol=1e-6), \
        "action position 0 and 1 produce identical hidden states (torch.where trap?)"

    # (S2) 64 個全体の std
    pos_std = aq_hiddens.std(dim=0).mean().item()
    print(f"  (S2) std across 64 action positions = {pos_std:.6e} (threshold > 1e-4)")
    assert pos_std > 1e-4, f"action position variance too small: {pos_std}"
    print("  Semantic verification: PASS")

    # -----------------------------------------------------------------
    # 1. 勾配検証 (backward)
    # -----------------------------------------------------------------
    print("\n=== Gradient verification ===")
    torch.cuda.reset_peak_memory_stats(0)
    loss = out.last_hidden_state.sum()
    loss.backward()
    bwd_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3

    # (G1) action_queries.weight.grad 存在
    aq_grad = action_queries.weight.grad
    print(f"  (G1) action_queries.weight.grad is None? {aq_grad is None}")
    assert aq_grad is not None, "action_queries.weight.grad is None"

    # (G2) grad が全部 0 ではない
    aq_grad_abs_sum = aq_grad.abs().sum().item()
    aq_grad_norm = aq_grad.norm().item()
    print(f"  (G2) grad.abs().sum() = {aq_grad_abs_sum:.6e}")
    print(f"       grad.norm()      = {aq_grad_norm:.6e}")
    assert aq_grad_abs_sum > 0, "action_queries.weight.grad is all-zero"
    print("  Gradient verification: PASS")

    # Backward memory
    print(f"\nBackward peak: {bwd_peak_gb:.2f} GB")
    print(f"Backward delta (bwd - fwd): {bwd_peak_gb - fwd_peak_gb:+.2f} GB")

    # -----------------------------------------------------------------
    # 3. 追加 sanity: LLM パラメータに grad が漏れていない (凍結維持)
    # -----------------------------------------------------------------
    leak = 0
    for p in llm.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            leak += 1
    print(f"\nLLM params with nonzero grad: {leak} (expected 0 since requires_grad=False)")
    # requires_grad=False のパラメータは p.grad が None のはずだが念のため

    # -----------------------------------------------------------------
    # JSON dump
    # -----------------------------------------------------------------
    summary = {
        "phase": "1b.3",
        "verdict": "OPTION_A",
        "B": B,
        "L": L,
        "num_action_tokens": NUM_ACTION_TOKENS,
        "forward_peak_gb": round(fwd_peak_gb, 3),
        "backward_peak_gb": round(bwd_peak_gb, 3),
        "backward_delta_gb": round(bwd_peak_gb - fwd_peak_gb, 3),
        "guard_triggered": guard_triggered,
        "fallback_used": fallback_used,
        "semantic_pair_diff_max": pair_diff,
        "semantic_pos_std": pos_std,
        "action_queries_grad_abs_sum": aq_grad_abs_sum,
        "action_queries_grad_norm": aq_grad_norm,
        "llm_grad_leak_count": leak,
        "llm_trainable_params": llm_trainable,
    }
    json_path = Path(__file__).resolve().parent / "test_03_result.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {json_path}")
    print("\n=== Phase 1b.3 OK: all gradient + semantic assertions pass ===")
    return summary


if __name__ == "__main__":
    main()
