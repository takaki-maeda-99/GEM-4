"""
Phase 1b.4: Vision integration (LIBERO = 2 カメラ × 256 patches = 512 tokens)

目的:
  - VisionProjector (2048 -> 8192 -> 1536 -> 1536) で vision features を 1536 に投影
  - action (64) + vision (512) の 2 種類の placeholder を clone + advanced indexing で両方上書き
  - L=628 (sliding_window=512 超過) → attention_mask / position_ids を明示構築 (R6)
  - vision_projector 3 層 + action_queries の grad + semantic を機械的に検証
  - (User 追加提案) Prompt 依存性テスト: 異なる prompt で最終 action hidden が distinct に変化
  - forward / backward peak memory 記録 (予算 25 GB)

注意: DINO+SigLIP 本体は dummy 実装 (random (B, 512, 2048) 返却) を使用。
  理由:
    - prismatic/__init__ が transformers 5.5 で壊れた Qwen2TokenizerFast を import → チェーン失敗
    - 1b.4 の Exit criteria は projector 統合・placeholder 挿入・attention_mask・grad flow の検証で、
      vision 特徴量の質には依存しない
    - 実 DINO+SigLIP は Phase 1b.6 (統合クラス化) で import 解決と同時に差し込む

Run: CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_04_vision.py
"""
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import importlib.util as _ilu  # noqa: E402

_cg_path = REPO_ROOT / "VLA-Adapter" / "prismatic" / "vla" / "constants_gemma4.py"
_spec = _ilu.spec_from_file_location("constants_gemma4", _cg_path)
_cg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cg)
ACTION_TOKEN_BEGIN_IDX = _cg.ACTION_TOKEN_BEGIN_IDX
NUM_ACTION_TOKENS = _cg.NUM_ACTION_TOKENS
VISION_PLACEHOLDER_BEGIN_IDX = _cg.VISION_PLACEHOLDER_BEGIN_IDX
NUM_VISION_TOKENS = _cg.NUM_VISION_TOKENS
PROPRIO_PLACEHOLDER_IDX = _cg.PROPRIO_PLACEHOLDER_IDX

MODEL_ID = "google/gemma-4-E2B"


# =====================================================================
# Modules
# =====================================================================

class VisionProjector(torch.nn.Module):
    """3 層 MLP (VLA-Adapter fused backbone 相当)."""

    def __init__(self, vision_dim=2048, llm_dim=1536, initial_projection_dim=8192):
        super().__init__()
        self.fc1 = torch.nn.Linear(vision_dim, initial_projection_dim, bias=True)
        self.fc2 = torch.nn.Linear(initial_projection_dim, llm_dim, bias=True)
        self.fc3 = torch.nn.Linear(llm_dim, llm_dim, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, x):  # (B, 512, 2048)
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


class DummyVisionBackbone(torch.nn.Module):
    """DINO+SigLIP の出力形状 (B, 512, 2048) を模した dummy.
    実 DINO+SigLIP は Phase 1b.6 で差し替え。"""

    def __init__(self, num_patches=512, vision_dim=2048, dtype=torch.bfloat16):
        super().__init__()
        self.num_patches = num_patches
        self.vision_dim = vision_dim
        self._dtype = dtype

    def forward(self, pixel_values):
        # pixel_values: (B, 12, 224, 224)  ← 2 cam × 6 ch
        B = pixel_values.shape[0]
        device = pixel_values.device
        # 画素に弱く依存する random features (同じ画像→同じ特徴、異なる画像→異なる特徴)
        gen = torch.Generator(device="cpu")
        # pixel_values.sum() を seed に使って deterministic
        seed_val = int(pixel_values.detach().float().sum().item() * 1e3) & 0xFFFFFFFF
        gen.manual_seed(seed_val)
        feats = torch.randn(
            B, self.num_patches, self.vision_dim,
            generator=gen, dtype=torch.float32,
        ).to(device=device, dtype=self._dtype)
        return feats


# =====================================================================
# Helper: build_input_ids
# =====================================================================

def build_input_ids(tok, prompt, device):
    """[BOS] + prompt + vision(512) + proprio(1) + action(64) + EOS"""
    prompt_ids_full = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_ids = prompt_ids_full[:, :50]  # plan: prompt ~50 tokens
    prompt_len = prompt_ids.shape[1]

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

    input_ids = torch.cat([bos, prompt_ids, vision_ids, proprio_id, action_ids, eos], dim=1)
    L = input_ids.shape[1]
    expected = 1 + prompt_len + NUM_VISION_TOKENS + 1 + NUM_ACTION_TOKENS + 1
    assert L == expected, f"L={L} expected={expected}"
    return input_ids, prompt_len


def forward_once(
    llm, input_ids, action_queries, vision_projector, vision_backbone, pixel_values
):
    """共通 forward path (Option A 上書き込み)."""
    B, L = input_ids.shape
    device = input_ids.device

    # Vision features
    vision_features = vision_backbone(pixel_values)                 # (B, 512, 2048)
    vision_projected = vision_projector(vision_features)            # (B, 512, 1536)

    # PLE 事前計算 (OOM 回避)
    with torch.no_grad():
        per_layer_inputs = llm.get_per_layer_inputs(input_ids, None)
        raw_embeddings = llm.embed_tokens(input_ids)

    embeddings = raw_embeddings.clone()

    # Option A: 2 種類の placeholder を両方上書き
    amask = (input_ids >= ACTION_TOKEN_BEGIN_IDX) & (
        input_ids < ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS
    )
    vmask = (input_ids >= VISION_PLACEHOLDER_BEGIN_IDX) & (
        input_ids < VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS
    )
    for b in range(B):
        apos = amask[b].nonzero(as_tuple=True)[0]
        vpos = vmask[b].nonzero(as_tuple=True)[0]
        assert apos.numel() == NUM_ACTION_TOKENS
        assert vpos.numel() == NUM_VISION_TOKENS
        embeddings[b, apos] = action_queries.weight                 # (64, 1536)
        embeddings[b, vpos] = vision_projected[b]                   # (512, 1536)

    # attention_mask / position_ids (R6: sliding_window=512 超過のため明示)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    position_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

    out = llm(
        inputs_embeds=embeddings,
        per_layer_inputs=per_layer_inputs,
        use_cache=True,
        output_hidden_states=True,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    return out, amask, vmask


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # -----------------------------------------------------------------
    # Model load
    # -----------------------------------------------------------------
    print("\n=== Loading Gemma 4 (bf16, sdpa) ===")
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

    llm = model.model.language_model
    hidden_size = model.config.text_config.hidden_size
    assert hidden_size == 1536

    llm_trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    assert llm_trainable == 0
    print(f"LLM frozen: trainable={llm_trainable}")

    # -----------------------------------------------------------------
    # Module 構築
    # -----------------------------------------------------------------
    vision_backbone = DummyVisionBackbone(
        num_patches=NUM_VISION_TOKENS, vision_dim=2048, dtype=torch.bfloat16,
    ).to(device).eval()
    for p in vision_backbone.parameters():
        p.requires_grad = False

    vision_projector = VisionProjector(
        vision_dim=2048, llm_dim=hidden_size, initial_projection_dim=8192,
    ).to(device, dtype=torch.bfloat16)
    action_queries = torch.nn.Embedding(NUM_ACTION_TOKENS, hidden_size).to(
        device, dtype=torch.bfloat16
    )
    action_queries.weight.data.zero_()

    vp_params = sum(p.numel() for p in vision_projector.parameters())
    aq_params = sum(p.numel() for p in action_queries.parameters())
    print(f"VisionProjector params: {vp_params / 1e6:.2f}M")
    print(f"action_queries params:  {aq_params / 1e6:.2f}M")

    # -----------------------------------------------------------------
    # Input 構築
    # -----------------------------------------------------------------
    prompt = "pick up the red cube and place it in the blue tray"
    input_ids, prompt_len = build_input_ids(tok, prompt, device)
    B, L = input_ids.shape
    print(f"\ninput_ids.shape = {tuple(input_ids.shape)} (prompt_len={prompt_len})")
    assert L == 1 + prompt_len + NUM_VISION_TOKENS + 1 + NUM_ACTION_TOKENS + 1

    pixel_values = torch.randn(B, 12, 224, 224, dtype=torch.bfloat16, device=device)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    print("\n=== Forward pass ===")
    torch.cuda.reset_peak_memory_stats(0)
    out, amask, vmask = forward_once(
        llm, input_ids, action_queries, vision_projector, vision_backbone, pixel_values
    )
    fwd_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"Forward peak: {fwd_peak_gb:.2f} GB")
    print(f"last_hidden_state.shape = {tuple(out.last_hidden_state.shape)}")
    print(f"len(hidden_states) = {len(out.hidden_states)}")

    apos = amask[0].nonzero(as_tuple=True)[0]
    vpos = vmask[0].nonzero(as_tuple=True)[0]

    # -----------------------------------------------------------------
    # Semantic 検証 (BEFORE backward)
    # -----------------------------------------------------------------
    print("\n=== Semantic verification ===")
    hidden = out.last_hidden_state.detach().float()

    # Vision: 隣接 2 位置 distinct
    v_pair_diff = (hidden[0, vpos[0]] - hidden[0, vpos[1]]).abs().max().item()
    v_pos_std = hidden[0, vpos].std(dim=0).mean().item()
    print(f"  Vision: |h[v0] - h[v1]|_max = {v_pair_diff:.4e}, std across 512 = {v_pos_std:.4e}")
    assert not torch.allclose(hidden[0, vpos[0]], hidden[0, vpos[1]], atol=1e-6), \
        "vision position 0 and 1 produce identical hidden states"

    # Action: 隣接 2 位置 distinct (1b.3 で確認済みだが統合後も維持確認)
    a_pair_diff = (hidden[0, apos[0]] - hidden[0, apos[1]]).abs().max().item()
    a_pos_std = hidden[0, apos].std(dim=0).mean().item()
    print(f"  Action: |h[a0] - h[a1]|_max = {a_pair_diff:.4e}, std across 64 = {a_pos_std:.4e}")
    assert not torch.allclose(hidden[0, apos[0]], hidden[0, apos[1]], atol=1e-6)
    assert a_pos_std > 1e-4
    print("  Semantic verification: PASS")

    # -----------------------------------------------------------------
    # Prompt 依存性テスト (User 推奨、global 層が言語 → action を運んでいるか)
    # -----------------------------------------------------------------
    print("\n=== Prompt dependency test (global attention の language-to-action path 検証) ===")
    prompt_a = "pick up the red cube"
    prompt_b = "pick up the blue cube"

    # 同じ prompt_len になるよう両方 tokenize してクランプ
    input_ids_a, plen_a = build_input_ids(tok, prompt_a, device)
    input_ids_b, plen_b = build_input_ids(tok, prompt_b, device)
    print(f"  prompt_a len={plen_a}, prompt_b len={plen_b}")

    # 同じ dummy vision features を使うため pixel_values は同じ seed になるよう固定
    pv = torch.zeros(B, 12, 224, 224, dtype=torch.bfloat16, device=device)  # 固定画素 → 同じ dummy 特徴

    with torch.no_grad():
        out_a, amask_a, _ = forward_once(
            llm, input_ids_a, action_queries, vision_projector, vision_backbone, pv
        )
        out_b, amask_b, _ = forward_once(
            llm, input_ids_b, action_queries, vision_projector, vision_backbone, pv
        )

    apos_a = amask_a[0].nonzero(as_tuple=True)[0]
    apos_b = amask_b[0].nonzero(as_tuple=True)[0]
    # 最後の action query
    h_a = out_a.last_hidden_state[0, apos_a[-1]].float()
    h_b = out_b.last_hidden_state[0, apos_b[-1]].float()
    prompt_diff_max = (h_a - h_b).abs().max().item()
    prompt_diff_mean = (h_a - h_b).abs().mean().item()
    print(f"  Last action hidden |h_a - h_b|_max  = {prompt_diff_max:.4e}")
    print(f"  Last action hidden |h_a - h_b|_mean = {prompt_diff_mean:.4e}")
    # global 層が 5 個 (layer 4, 9, 14, 19, 24) あるので差が出るはず
    assert not torch.allclose(h_a, h_b, atol=1e-3), \
        "last action hidden does not depend on prompt — global language-to-action path broken"
    print("  Prompt dependency: PASS (global attention が言語を action まで運んでいる)")

    # -----------------------------------------------------------------
    # Gradient 検証 (メイン forward の loss)
    # -----------------------------------------------------------------
    print("\n=== Gradient verification ===")
    torch.cuda.reset_peak_memory_stats(0)
    loss = out.last_hidden_state.sum()
    loss.backward()
    bwd_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"Backward peak: {bwd_peak_gb:.2f} GB")
    print(f"Backward delta: {bwd_peak_gb - fwd_peak_gb:+.2f} GB")

    # vision_projector 3 層
    vp_grads = {
        "fc1": vision_projector.fc1.weight.grad,
        "fc2": vision_projector.fc2.weight.grad,
        "fc3": vision_projector.fc3.weight.grad,
    }
    for name, g in vp_grads.items():
        assert g is not None, f"vision_projector.{name}.weight.grad is None"
        val = g.abs().sum().item()
        print(f"  vision_projector.{name}.grad.abs().sum() = {val:.4e}")
        assert val > 0, f"vision_projector.{name} grad is zero"

    # action_queries
    aq_grad = action_queries.weight.grad
    assert aq_grad is not None
    aq_abs = aq_grad.abs().sum().item()
    print(f"  action_queries.grad.abs().sum()        = {aq_abs:.4e}")
    assert aq_abs > 0

    # LLM 凍結 leak check
    llm_leak = sum(
        1 for p in llm.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    print(f"  LLM nonzero grad params: {llm_leak} (expected 0)")
    assert llm_leak == 0
    print("  Gradient verification: PASS")

    # -----------------------------------------------------------------
    # Exit criteria
    # -----------------------------------------------------------------
    print("\n=== Exit criteria ===")
    budget_gb = 25.0
    print(f"(1) Forward peak  = {fwd_peak_gb:.2f} GB (< {budget_gb:.1f}?)")
    print(f"    Backward peak = {bwd_peak_gb:.2f} GB (< {budget_gb:.1f}?)")
    assert fwd_peak_gb < budget_gb
    assert bwd_peak_gb < budget_gb

    # -----------------------------------------------------------------
    # JSON dump
    # -----------------------------------------------------------------
    summary = {
        "phase": "1b.4",
        "vision_backbone": "DummyVisionBackbone (DINO+SigLIP 実体は 1b.6 で差替)",
        "B": B,
        "L": L,
        "prompt_len": prompt_len,
        "forward_peak_gb": round(fwd_peak_gb, 3),
        "backward_peak_gb": round(bwd_peak_gb, 3),
        "backward_delta_gb": round(bwd_peak_gb - fwd_peak_gb, 3),
        "vision_projector_params_M": round(vp_params / 1e6, 3),
        "action_queries_params_M": round(aq_params / 1e6, 3),
        "vision_pair_diff_max": v_pair_diff,
        "vision_pos_std": v_pos_std,
        "action_pair_diff_max": a_pair_diff,
        "action_pos_std": a_pos_std,
        "prompt_diff_max": prompt_diff_max,
        "prompt_diff_mean": prompt_diff_mean,
        "vp_fc1_grad_abs_sum": vp_grads["fc1"].abs().sum().item(),
        "vp_fc2_grad_abs_sum": vp_grads["fc2"].abs().sum().item(),
        "vp_fc3_grad_abs_sum": vp_grads["fc3"].abs().sum().item(),
        "aq_grad_abs_sum": aq_abs,
        "llm_leak": llm_leak,
    }
    json_path = Path(__file__).resolve().parent / "test_04_result.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {json_path}")
    print("\n=== Phase 1b.4 OK ===")
    return summary


if __name__ == "__main__":
    main()
