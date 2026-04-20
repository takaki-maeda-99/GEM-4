"""
Phase 1b.5: Action head 接続 + LayerNorm=Identity (1b.1 判定)

目的:
  - Step 0: action_head.named_parameters() を print して具体層名を確定
  - L1RegressionActionHead(use_pro_version=True) を 1b.4 の出力と接続
  - entries 0-24 slice を action_head.predict_action に渡す
  - predicted.shape == (B, 8, 7) を確認
  - action_head / proprio_projector の具体層に grad が流れる
  - LayerNorm は 1b.1 User 判定により Identity (feature_norm=Identity) 固定

1b.1 判定根拠:
  entries 0-24 範囲の max std = 1.754 < 2.0、v5.2 ルール「全 25 entries で std ≤ 2.0 → 入れない」該当。

import 対応:
  action_heads.py は `from prismatic.vla.constants import ...` を top-level で実行するため、
  prismatic/__init__ (transformers 5.5 で壊れている) を通してしまう。
  sys.modules に空の stub module を注入することで __init__ の実行を skip し、
  サブパッケージ (prismatic.vla.constants 等) は __path__ 経由で直接ロードさせる。

Run: CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_05_action_head.py
"""
import gc
import json
import sys
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"

# -----------------------------------------------------------------
# sys.modules stubbing: prismatic/__init__.py の実行を skip、
# サブパッケージ (constants / action_heads) は __path__ 経由で個別ロード
# -----------------------------------------------------------------
for _mod_name, _sub_path in [
    ("prismatic", ""),
    ("prismatic.models", "prismatic/models"),
    ("prismatic.vla", "prismatic/vla"),
]:
    if _mod_name not in sys.modules:
        _m = types.ModuleType(_mod_name)
        if _sub_path:
            _m.__path__ = [str(VLA_ROOT / _sub_path)]
        else:
            _m.__path__ = [str(VLA_ROOT / "prismatic")]
        sys.modules[_mod_name] = _m

sys.path.insert(0, str(VLA_ROOT))

from prismatic.models.action_heads import L1RegressionActionHead  # noqa: E402

# constants_gemma4 も importlib で個別ロード (1b.3/1b.4 と同じ方法)
import importlib.util as _ilu  # noqa: E402

_cg_path = VLA_ROOT / "prismatic" / "vla" / "constants_gemma4.py"
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
# Modules (test_04 と同じ、dummy vision backbone / VisionProjector)
# =====================================================================

class VisionProjector(torch.nn.Module):
    def __init__(self, vision_dim=2048, llm_dim=1536, initial_projection_dim=8192):
        super().__init__()
        self.fc1 = torch.nn.Linear(vision_dim, initial_projection_dim, bias=True)
        self.fc2 = torch.nn.Linear(initial_projection_dim, llm_dim, bias=True)
        self.fc3 = torch.nn.Linear(llm_dim, llm_dim, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, x):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


class ProprioProjector(torch.nn.Module):
    def __init__(self, proprio_dim=8, llm_dim=1536):
        super().__init__()
        self.fc1 = torch.nn.Linear(proprio_dim, llm_dim)
        self.fc2 = torch.nn.Linear(llm_dim, llm_dim)
        self.act = torch.nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DummyVisionBackbone(torch.nn.Module):
    def __init__(self, num_patches=512, vision_dim=2048, dtype=torch.bfloat16):
        super().__init__()
        self.num_patches = num_patches
        self.vision_dim = vision_dim
        self._dtype = dtype

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        device = pixel_values.device
        gen = torch.Generator(device="cpu")
        seed_val = int(pixel_values.detach().float().sum().item() * 1e3) & 0xFFFFFFFF
        gen.manual_seed(seed_val)
        return torch.randn(
            B, self.num_patches, self.vision_dim,
            generator=gen, dtype=torch.float32,
        ).to(device=device, dtype=self._dtype)


def build_input_ids(tok, prompt, device):
    prompt_ids_full = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_ids = prompt_ids_full[:, :50]
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
    return input_ids, prompt_len


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # =================================================================
    # Step 0: action_head の構造確認 (model load 前に実行可能)
    # =================================================================
    print("\n=== Step 0: action_head.named_parameters() の実機確認 ===")
    hidden_size_for_step0 = 1536  # Gemma 4 E2B の値、model load せずに検証可能

    action_head_step0 = L1RegressionActionHead(
        input_dim=hidden_size_for_step0,
        hidden_dim=hidden_size_for_step0,
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=True,
    )

    # 全 named_parameters を記録 (head だけ 20 個表示、全体は JSON へ)
    all_named = list(action_head_step0.named_parameters())
    print(f"Total named_parameters: {len(all_named)}")
    print(f"Total trainable count (head only): {sum(p.numel() for _, p in all_named) / 1e6:.2f}M")
    print(f"\nFirst 10 params:")
    for name, p in all_named[:10]:
        print(f"  {name:60s} shape={tuple(p.shape)}")
    print("...")
    print(f"Last 10 params:")
    for name, p in all_named[-10:]:
        print(f"  {name:60s} shape={tuple(p.shape)}")

    # 学習可能な Linear の最初と最後 (fallback 検出)
    linear_params = [
        (name, p) for name, p in all_named
        if "weight" in name and p.ndim == 2 and p.requires_grad
    ]
    if not linear_params:
        raise RuntimeError("No trainable Linear found in action_head")
    first_linear_name, first_linear_param = linear_params[0]
    last_linear_name, last_linear_param = linear_params[-1]
    print(f"\nFirst trainable Linear: {first_linear_name} shape={tuple(first_linear_param.shape)}")
    print(f"Last  trainable Linear: {last_linear_name} shape={tuple(last_linear_param.shape)}")

    # 想定名 (plan v5.2) との一致確認
    plan_first_name = "model.fc1.weight"
    plan_last_name = "model.fc2.weight"
    matches_plan_first = first_linear_name == plan_first_name
    matches_plan_last = last_linear_name == plan_last_name
    print(f"\nPlan v5.2 想定の層名 '{plan_first_name}' と一致? {matches_plan_first}")
    print(f"Plan v5.2 想定の層名 '{plan_last_name}' と一致? {matches_plan_last}")

    # step0 用の head は解放 (以降は full run 用に新規インスタンス)
    del action_head_step0
    gc.collect()

    # =================================================================
    # Model load (1b.1 / 1b.4 と同じ)
    # =================================================================
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

    # =================================================================
    # Modules
    # =================================================================
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

    proprio_projector = ProprioProjector(proprio_dim=8, llm_dim=hidden_size).to(
        device, dtype=torch.bfloat16
    )

    # feature_norm = Identity (1b.1 判定より)
    feature_norm = torch.nn.Identity()

    action_head = L1RegressionActionHead(
        input_dim=hidden_size,
        hidden_dim=hidden_size,
        action_dim=7,
        num_task_tokens=NUM_VISION_TOKENS,  # 512
        use_pro_version=True,
    ).to(device, dtype=torch.bfloat16)

    ah_params = sum(p.numel() for p in action_head.parameters() if p.requires_grad)
    pp_params = sum(p.numel() for p in proprio_projector.parameters() if p.requires_grad)
    print(f"action_head trainable: {ah_params / 1e6:.2f}M (plan 想定 ~218M)")
    print(f"proprio_projector trainable: {pp_params / 1e6:.2f}M")

    # =================================================================
    # Input
    # =================================================================
    prompt = "pick up the red cube and place it in the blue tray"
    input_ids, prompt_len = build_input_ids(tok, prompt, device)
    B, L = input_ids.shape
    print(f"\ninput_ids.shape = {tuple(input_ids.shape)}")

    pixel_values = torch.randn(B, 12, 224, 224, dtype=torch.bfloat16, device=device)
    proprio_raw = torch.randn(B, 8, dtype=torch.bfloat16, device=device)

    # =================================================================
    # Forward (1b.4 と同じ pattern)
    # =================================================================
    print("\n=== Forward pass (LLM + action head) ===")
    torch.cuda.reset_peak_memory_stats(0)

    vision_features = vision_backbone(pixel_values)
    vision_projected = vision_projector(vision_features)

    with torch.no_grad():
        per_layer_inputs = llm.get_per_layer_inputs(input_ids, None)
        raw_embeddings = llm.embed_tokens(input_ids)
    embeddings = raw_embeddings.clone()

    amask = (input_ids >= ACTION_TOKEN_BEGIN_IDX) & (
        input_ids < ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS
    )
    vmask = (input_ids >= VISION_PLACEHOLDER_BEGIN_IDX) & (
        input_ids < VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS
    )
    for b in range(B):
        apos = amask[b].nonzero(as_tuple=True)[0]
        vpos = vmask[b].nonzero(as_tuple=True)[0]
        embeddings[b, apos] = action_queries.weight
        embeddings[b, vpos] = vision_projected[b]

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

    # entries 0-24 slice
    all_hidden = torch.stack(out.hidden_states, dim=1)  # (B, 36, L, 1536)
    hidden_subset = all_hidden[:, :25, :, :]            # (B, 25, L, 1536)
    print(f"all_hidden.shape     = {tuple(all_hidden.shape)}")
    print(f"hidden_subset.shape  = {tuple(hidden_subset.shape)} (期待: (B, 25, L, 1536))")

    apos_b0 = amask[0].nonzero(as_tuple=True)[0]
    vpos_b0 = vmask[0].nonzero(as_tuple=True)[0]
    assert apos_b0.numel() == NUM_ACTION_TOKENS
    assert vpos_b0.numel() == NUM_VISION_TOKENS

    vision_hidden = hidden_subset[:, :, vpos_b0, :]     # (B, 25, 512, 1536)
    action_hidden = hidden_subset[:, :, apos_b0, :]     # (B, 25, 64, 1536)
    # feature_norm = Identity → 変化なし
    vision_hidden = feature_norm(vision_hidden)
    action_hidden = feature_norm(action_hidden)
    combined = torch.cat([vision_hidden, action_hidden], dim=2)  # (B, 25, 576, 1536)
    print(f"combined.shape       = {tuple(combined.shape)} (期待: (B, 25, 576, 1536))")

    # action_head 推論
    predicted = action_head.predict_action(
        actions_hidden_states=combined,
        proprio=proprio_raw,
        proprio_projector=proprio_projector,
        phase="Training",
    )
    fwd_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"predicted.shape      = {tuple(predicted.shape)} (期待: (B, 8, 7))")
    print(f"Forward peak: {fwd_peak_gb:.2f} GB")

    assert predicted.shape == (B, 8, 7), f"predicted.shape {predicted.shape} != (B, 8, 7)"
    assert torch.isfinite(predicted).all(), "predicted contains NaN/Inf"

    # =================================================================
    # Backward + Gradient 検証
    # =================================================================
    print("\n=== Gradient verification ===")
    torch.cuda.reset_peak_memory_stats(0)
    actions_target = torch.zeros_like(predicted)
    loss = torch.nn.functional.l1_loss(predicted, actions_target)
    print(f"L1 loss value: {loss.item():.4f}")
    loss.backward()
    bwd_peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3

    # Step 0 で確定した層名で assertion
    grad_checks = {}
    if matches_plan_first:
        # 想定どおり action_head.model.fc1.weight が存在
        g1 = action_head.model.fc1.weight.grad
        grad_checks["action_head.model.fc1.weight"] = (g1, plan_first_name)
    g1_used_name = None
    if matches_plan_last:
        g2 = action_head.model.fc2.weight.grad
        grad_checks["action_head.model.fc2.weight"] = (g2, plan_last_name)

    # Fallback: 実際の first/last linear を直接参照
    # (Step 0 で確定した param object を使用)
    # named_parameters は deep copy しないので、action_head_step0 は別インスタンスだが
    # full run の action_head から改めて first/last を取る
    full_linears = [
        (name, p) for name, p in action_head.named_parameters()
        if "weight" in name and p.ndim == 2 and p.requires_grad
    ]
    full_first_name, full_first_p = full_linears[0]
    full_last_name, full_last_p = full_linears[-1]
    grad_checks[f"(first linear) {full_first_name}"] = (full_first_p.grad, full_first_name)
    grad_checks[f"(last linear) {full_last_name}"] = (full_last_p.grad, full_last_name)

    # proprio_projector
    pp_grads = {
        "proprio_projector.fc1.weight": proprio_projector.fc1.weight.grad,
        "proprio_projector.fc2.weight": proprio_projector.fc2.weight.grad,
    }

    # Print + assert
    for label, (g, actual_name) in grad_checks.items():
        if g is None:
            val = 0.0
        else:
            val = g.abs().sum().item()
        print(f"  {label}: grad.abs().sum() = {val:.4e}")
        assert g is not None, f"{label} grad is None"
        assert val > 0, f"{label} grad is zero"

    for name, g in pp_grads.items():
        val = g.abs().sum().item() if g is not None else 0.0
        print(f"  {name}: grad.abs().sum() = {val:.4e}")
        assert val > 0, f"{name} grad is zero"

    # LLM 凍結維持 leak check
    llm_leak = sum(
        1 for p in llm.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    print(f"  LLM leak count: {llm_leak} (expected 0)")
    assert llm_leak == 0

    # action_queries & vision_projector grad も維持されているか (1b.3/1b.4 から継続)
    aq_g = action_queries.weight.grad.abs().sum().item()
    vp_g = vision_projector.fc1.weight.grad.abs().sum().item()
    print(f"  action_queries grad: {aq_g:.4e}")
    print(f"  vision_projector.fc1 grad: {vp_g:.4e}")
    assert aq_g > 0
    assert vp_g > 0

    print(f"\nBackward peak: {bwd_peak_gb:.2f} GB")
    print(f"Backward delta: {bwd_peak_gb - fwd_peak_gb:+.2f} GB")

    # =================================================================
    # Exit criteria
    # =================================================================
    budget_gb = 30.0
    print(f"\n=== Exit criteria ===")
    print(f"peak_gb fwd={fwd_peak_gb:.2f} bwd={bwd_peak_gb:.2f} (< {budget_gb:.1f})")
    assert fwd_peak_gb < budget_gb
    assert bwd_peak_gb < budget_gb

    # =================================================================
    # JSON dump
    # =================================================================
    summary = {
        "phase": "1b.5",
        "feature_norm": "Identity (1b.1 判定)",
        "vision_backbone": "DummyVisionBackbone",
        "B": B,
        "L": L,
        "prompt_len": prompt_len,
        "hidden_subset_shape": list(hidden_subset.shape),
        "combined_shape": list(combined.shape),
        "predicted_shape": list(predicted.shape),
        "forward_peak_gb": round(fwd_peak_gb, 3),
        "backward_peak_gb": round(bwd_peak_gb, 3),
        "backward_delta_gb": round(bwd_peak_gb - fwd_peak_gb, 3),
        "action_head_params_M": round(ah_params / 1e6, 3),
        "proprio_projector_params_M": round(pp_params / 1e6, 3),
        "step0": {
            "total_named_parameters": len(all_named),
            "first_linear_name": first_linear_name,
            "first_linear_shape": list(first_linear_param.shape),
            "last_linear_name": last_linear_name,
            "last_linear_shape": list(last_linear_param.shape),
            "matches_plan_first (model.fc1.weight)": matches_plan_first,
            "matches_plan_last (model.fc2.weight)": matches_plan_last,
            "all_param_names": [n for n, _ in all_named],
        },
        "grad_abs_sums": {
            full_first_name: full_first_p.grad.abs().sum().item() if full_first_p.grad is not None else 0.0,
            full_last_name: full_last_p.grad.abs().sum().item() if full_last_p.grad is not None else 0.0,
            "proprio_projector.fc1.weight": proprio_projector.fc1.weight.grad.abs().sum().item(),
            "proprio_projector.fc2.weight": proprio_projector.fc2.weight.grad.abs().sum().item(),
            "action_queries.weight": aq_g,
            "vision_projector.fc1.weight": vp_g,
        },
        "llm_leak": llm_leak,
        "loss_value": float(loss.item()),
    }
    json_path = Path(__file__).resolve().parent / "test_05_result.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary JSON: {json_path}")
    print("\n=== Phase 1b.5 OK ===")
    return summary


if __name__ == "__main__":
    main()
