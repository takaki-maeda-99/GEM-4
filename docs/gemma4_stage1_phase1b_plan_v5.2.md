# Gemma 4 E2B 移植 Stage 1 / Phase 1b 実装計画 (v5.2 確定版)

**対象**: Phase 1b (Model 組み立て) のマイクロステップ分割。Phase 0 / 1a は完了済み。
**作成経緯**: User との 5 ラウンドのレビュー (泥沼化懸念 → v1 教訓反映 → 勾配検証 → semantic 罠 → PLE/action_head 詳細確定 → Phase 間 pattern 伝播) を経て確定。
**目的**: Gemma 4 E2B backbone + VLA-Adapter 設計で LIBERO 10-step smoke train を走らせる (性能は問わない)。

---

## v5.2 で確定した Gemma 4 実装詳細 (実機検証済み)

### 真の module 階層

v1 troubleshooting の `model.language_model.*` は **transformers 5.5 で 1 階層深くなった**:

```
Gemma4ForConditionalGeneration
├─ model: Gemma4Model                              ← multimodal wrapper
│  ├─ language_model: Gemma4TextModel              ← **真のテキスト LLM**
│  │  ├─ embed_tokens: Gemma4TextScaledWordEmbedding
│  │  ├─ layers: ModuleList (35 decoder layers)
│  │  ├─ norm, rotary_emb
│  │  ├─ embed_tokens_per_layer: Gemma4TextScaledWordEmbedding  ← **PLE 本体**
│  │  ├─ per_layer_model_projection: Linear
│  │  └─ per_layer_projection_norm: Gemma4RMSNorm
│  ├─ vision_tower, embed_vision (Stage 1 では未使用)
│  └─ audio_tower, embed_audio (未使用)
└─ lm_head: Linear
```

全てのアクセスは `model.model.language_model.*` 経由。**全 Phase で統一**。

### PLE 注入の正しい pattern (v1 OOM の真因解明版)

**v1 OOM の真因** (ソースコード精読で判明):

```python
# Gemma4TextModel.get_per_layer_inputs(input_ids, inputs_embeds):
if input_ids is None:
    # 逆引き: inputs_embeds[:, :, None, :] == embed_weights[None, None, :, :]
    # shape (B, L, V=262144, D=1536) → B=1, L=100 で 80GB 確保 → OOM
    input_ids = (...).nonzero()[:, 2]
return self.embed_tokens_per_layer(input_ids).reshape(*input_ids.shape, 35, 256)
```

**対策**: 常に `input_ids` を一緒に渡す。逆引きが発動しない。

### 正しい forward pattern (1b.3 以降すべてに適用)

```python
# ① input_ids に placeholder ID を含めて構築
#    action positions: 258885..258948 (64 個、distinct IDs)
#    vision positions: 258949..259460 (512 個、distinct IDs)
#    proprio position: 259461 (1 個)
#    prompt / BOS / EOS: 通常 token
input_ids = build_full_input_ids(...)  # shape (B, L)

# ② PLE を input_ids から事前計算 (OOM 回避)
per_layer_inputs = model.model.language_model.get_per_layer_inputs(input_ids, None)
# shape: (B, L, 35, 256)

# ③ 標準 embedding lookup
with torch.no_grad():
    raw_embeds = model.model.language_model.embed_tokens(input_ids)  # (B, L, 1536)

# ④ clone + advanced indexing で action / vision 位置を上書き (Option A)
embeds = raw_embeds.clone()
for b in range(B):
    apos = get_action_positions(input_ids[b])      # 64 positions
    vpos = get_vision_positions(input_ids[b])      # 512 positions
    embeds[b, apos] = action_queries.weight        # (64, 1536)
    embeds[b, vpos] = vision_projected[b]          # (512, 1536)

# ⑤ Gemma4TextModel を直接呼ぶ (ConditionalGeneration wrapper をスキップ)
#    attention_mask / position_ids は seq が sliding_window (512) を超える 1b.4 以降で必須
out = model.model.language_model(
    inputs_embeds=embeds,
    per_layer_inputs=per_layer_inputs,
    use_cache=True,                                 # G1 対策
    output_hidden_states=True,
    attention_mask=attention_mask,
    position_ids=position_ids,
)
```

### Action head 入力は vision+action 両方 (task token 混入)

`action_heads.py:43` の `predict_action(actions_hidden_states, ...)` は内部で split:

```python
task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]  # 前 512
actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]  # 後 64
```

さらに `MLPResNet.forward:118` で `h_t[:, i+1, :]` と `h_a[:, i+1, :]` を block i が参照する。24 blocks (Pro 版 default) なので **layer index 1-24 の hidden state が必要** (layer 0 は embedding)。

Gemma 4 は 35 layers → `output_hidden_states=True` で 36 entries (embedding + 35) 返る。**最初の 25 layer (index 0-24) を slice して action head に渡す**:

```python
all_hidden = torch.stack(out.hidden_states, dim=1)       # (B, 36, L, 1536)
hidden_subset = all_hidden[:, :25, :, :]                 # (B, 25, L, 1536)
# vision と action 位置を抽出
vision_hidden = hidden_subset[:, :, vpos_b0, :]          # (B, 25, 512, 1536)
action_hidden = hidden_subset[:, :, apos_b0, :]          # (B, 25, 64, 1536)
combined = torch.cat([vision_hidden, action_hidden], dim=2)  # (B, 25, 576, 1536)

predicted = action_head.predict_action(
    actions_hidden_states=combined,
    proprio=proprio_raw,                                 # raw (B, 8), 正規化不要
    proprio_projector=proprio_projector,                 # 別 module
    phase="Training",
)
# output shape: (B, 8, 7)
```

### Action head の学習可能 layer (仮置き、1b.5 Step 0 で実機確認)

`action_heads.py` の事前読み込みで **仮に** 以下と想定:

```python
# MLPResNet 内の学習可能な具体 layer (仮)
action_head.model.fc1.weight                          # 入力側 Linear (仮)
action_head.model.fc2.weight                          # 出力側 Linear (仮)
```

**1b.5 Step 0 で `action_head.named_parameters()` を print して実在の層名を確定する** (v5.2 で追加)。想定と違えば assertion コードをその場で書き換える。

---

## 前提コンテキスト

### Phase 0 完了 (config 検証)
- `hidden_size=1536`, `num_hidden_layers=35`, `num_kv_shared_layers=20`, `sliding_window=512`
- Attention: 4 sliding + 1 full (layer 4/9/14/19/24/29/34) × 7 cycle
- Dual head_dim: sliding=256, global=512
- `tie_word_embeddings=True`, `use_bidirectional_attention` field 存在 (現在 None)
- 詳細: [docs/gemma4_migration_log.md](gemma4_migration_log.md)

### Phase 1a 完了 (special token)
- Action token 配置: **ID 258885-258948** (`<unused2968>`〜`<unused3031>`)
- Vision placeholder: **ID 258949-259460** (512 個 distinct、v5.1 で確定)
- Proprio placeholder: **ID 259461** (v5.1 で確定)
- `resize_token_embeddings` **不要** (既存 unused token を流用)
- Action head: `L1RegressionActionHead(use_pro_version=True)` → **~640M params** (1b.5 Step 0 実測)
  v5.2 初版の「218M」は非 Pro 版の誤記。Pro 版は cross-attention + FiLM で 3 倍規模
- `constants_gemma4.py` 作成済み:
  ```python
  ACTION_TOKEN_BEGIN_IDX         = 258885
  VISION_PLACEHOLDER_BEGIN_IDX   = 258949
  PROPRIO_PLACEHOLDER_IDX        = 259461
  STOP_INDEX                     = 1
  NUM_ACTION_TOKENS              = 64
  NUM_VISION_TOKENS              = 512
  ```

---

## v1 から引き継ぐ重要教訓

| # | 教訓 | Phase 1b での対応 |
|---|---|---|
| T1 | Gemma 4 PLE は `inputs_embeds` 直渡しで **105GB OOM** | 真因は逆引き処理 → **`per_layer_inputs` を常に一緒に渡す** (v5.2 正しい pattern) |
| T2 | 特殊 token を **文字列で書くと BPE 分割**される | 終始 **ID 直挿入** で統一 (Phase 1a の方針) |
| T3 | LLM hidden state `std≈6` で **action head が mode collapse** | 1b.1 で測定 → 1b.5 で測定値に基づき LayerNorm 配置決定 |
| T4 | model 階層は `model.language_model.*` | v5.1 で訂正: transformers 5.5 では `model.model.language_model.*` |
| T5 | PEFT ラップで階層が変わる | Stage 1 は LoRA 不使用なので影響なし (Stage 2 で対応) |

---

## 横断ルール (全 test script に適用)

### R1. 明示的 config 設定

```python
model.config.use_cache = True                  # Gemma 4 KV 共有バグ (HF #45242) 回避
assert not model.is_gradient_checkpointing      # checkpointing は Stage 1 で使わない

attn_implementation="sdpa"   # Stage 1 は SDPA 統一 (2026-04-19 User 判断)
# flash_attention_2 は (a) Gemma 4 global 層 (head_dim=512) で未対応のため fallback 発生、
# (b) install コストが高い。Phase 1c で batch 拡大時にメモリ逼迫したら flash 導入検討。
```

### R2. model 階層アクセス規約

```python
# NG (v1 時代): model.language_model.embed_tokens
# NG: model.hidden_size
# OK (transformers 5.5):
config      = model.config.text_config
llm         = model.model.language_model              # Gemma4TextModel
embed       = llm.embed_tokens                        # Gemma4TextScaledWordEmbedding
ple_embed   = llm.embed_tokens_per_layer              # PLE 本体
hidden_size = config.hidden_size                      # 1536
```

### R2b. pad_token_id 確認 (test_01 冒頭)

```python
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
assert tok.pad_token_id is not None, "pad_token_id still None after fallback"
```

### R3. LLM 凍結の徹底

Phase 1b 全ステップで LLM 側パラメータは `requires_grad=False`。学習対象は action_head / action_queries / vision_projector / proprio_projector のみ。

### R4. Test script の形式

```
scripts/gemma4/test_01_load.py           # Phase 1b.1
scripts/gemma4/test_02_inputs_embeds.py  # Phase 1b.2
...
scripts/gemma4/test_06_full_forward.py   # Phase 1b.6
```

各スクリプトは独立に `.venv-gemma4/bin/python scripts/gemma4/test_XX.py` で走る。

### R5. ログ更新

各ステップ完了時に `docs/gemma4_migration_log.md` に結果 (memory, 速度, 発見) を追記。

### R6. attention_mask / position_ids (1b.4 以降で必須、v5.2 追加)

Seq が sliding window (512) を超える 1b.4 以降では、以下を明示的に構築して渡す:

```python
attention_mask = torch.ones_like(input_ids, dtype=torch.long)    # 全 token 可視
position_ids   = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, -1).cuda()
```

auto-generate に任せると action → prompt の attention path が実装依存になるリスクあり。

---

# Phase 1b.1: Load + text forward + baseline 測定

**目的**: Gemma 4 E2B を single GPU でロードし、text-only forward が正常に走ることを確認。PLE 込みの health check を実施し、以降の Phase の基準値を取る。

## 実装

```python
# scripts/gemma4/test_01_load.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B"
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# R2b: pad_token_id 確認
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
assert tok.pad_token_id is not None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # sliding 層は flash、global 層は SDPA fallback
).cuda().eval()

# R1: 明示 config
model.config.use_cache = True
assert not model.is_gradient_checkpointing

# 凍結 (Stage 1 前提)
for p in model.parameters():
    p.requires_grad = False

# Dummy input
prompt = "hello world " * 20
input_ids = tok(prompt, return_tensors="pt").input_ids.cuda()[:, :100]

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    out = model(input_ids, output_hidden_states=True)

# T3: 各層の std/mean 測定 (LayerNorm 要否の根拠データ)
print("=== Hidden state statistics per layer ===")
for i, h in enumerate(out.hidden_states):
    print(f"layer {i:2d}: std={h.std().item():.3f} mean={h.mean().item():.3f}")

peak_gb = torch.cuda.max_memory_allocated() / 1024**3
print(f"\nPeak memory: {peak_gb:.2f} GB")
```

## Exit criteria (数値 assertion)

- [ ] `peak_gb < 15.0` (15GB 上限)。超えたら **PLE 異常として停止・報告**
- [ ] 全 36 entries (embedding + 35 layers) の hidden state が finite (`not torch.isnan(h).any()`)
- [ ] 全層の std が [0.1, 10] 範囲
- [ ] ログに 36 entry 全部の std/mean が記録される

**→ User check-in point #1**: std 分布を見せて LayerNorm 判定 (1b.5 で使用)

---

# Phase 1b.2: `inputs_embeds` PLE 罠を段階的に切り分け + 対策検証

**目的**: v5.1 で解明された PLE 真因を実機で確認 + 対策 (per_layer_inputs 付き) の効果を数値で検証 (v5.2 で拡張)。

## 実装

```python
# scripts/gemma4/test_02_inputs_embeds.py
llm = model.model.language_model  # R2

cases = [
    (1,  100),  # baseline
    (1,  500),  # seq 線形?
    (1, 1000),  # seq 非線形?
    (4,  500),  # batch 線形?
    (4, 1000),  # batch × seq 相互作用 (OOM 覚悟)
]

print("=== Case 1: inputs_embeds のみ (per_layer_inputs なし、OOM 予想) ===")
for B, L in cases:
    try:
        dummy_embeds = torch.randn(B, L, 1536, dtype=torch.bfloat16).cuda()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out = llm(inputs_embeds=dummy_embeds)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  B={B} L={L}: peak={peak:.2f}GB OK")
    except torch.cuda.OutOfMemoryError:
        print(f"  B={B} L={L}: OOM (逆引き発動)")
    torch.cuda.empty_cache()

print("\n=== Case 2: per_layer_inputs 付き (対策版、成功予想) ===")
for B, L in cases:
    try:
        dummy_ids = torch.randint(0, 262144, (B, L), dtype=torch.long).cuda()
        dummy_embeds = torch.randn(B, L, 1536, dtype=torch.bfloat16).cuda()
        with torch.no_grad():
            per_layer_inputs = llm.get_per_layer_inputs(dummy_ids, None)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out = llm(inputs_embeds=dummy_embeds, per_layer_inputs=per_layer_inputs)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  B={B} L={L}: peak={peak:.2f}GB OK (対策有効)")
    except torch.cuda.OutOfMemoryError:
        print(f"  B={B} L={L}: OOM (対策無効、要再調査)")
    torch.cuda.empty_cache()
```

## 判定分岐 (→ 1b.3 の実装方針)

| 観測結果 | 診断 | 1b.3 方針 |
|---|---|---|
| Case 1 で OOM 発生, Case 2 で全 pass | v5.1 の真因分析が正しい | **Option A 採用**: per_layer_inputs + clone + indexing |
| Case 1 で全 pass (つまり OOM しない) | 5.5.4 以降で修正済み | **シンプル採用**: inputs_embeds 直渡しでも OK |
| Case 1 と Case 2 の両方で OOM | 我々の理解が間違い | **停止して User 報告** |

## Exit criteria

- [ ] 10 試行 (2 × 5) のメモリ消費をログ
- [ ] Case 1 / Case 2 の挙動差が明確に出る
- [ ] 1b.3 の実装方針が数値に基づいて決定される

**→ User check-in point #2**: 判定結果を見せて方針確定 (Option A or シンプル採用)

---

# Phase 1b.3: Embedding 注入 + **勾配検証** (最重要)

**目的**: 64 個の distinct action queries を input embedding に正しく挿入。勾配と semantic の両方を機械的に検証する。

## 冒頭宣言 (script に記載)

```python
# === Overwrite strategy decision ===
# Chose Option A (clone + advanced indexing) because:
#   - Preserves gradient flow to action_queries.weight (__setitem__ is autograd-safe for RHS)
#   - Each of 64 positions receives its corresponding distinct query vector
#   - Readable; easy to verify correctness
#
# Rejected:
#   - torch.where + broadcast: all 64 positions get same value (mode collapse trap)
#   - Option B (scatter): correct but harder to read
#   - Option C (index_put_): correct but less intuitive
```

1b.2 の結果が「シンプル採用」だった場合もこの冒頭は残すが、Option A のコードパスを skip して `inputs_embeds` 直渡しを選ぶ旨を記載。

## Option A 実装 (v5.1 確定版、PLE 対応込み)

```python
from prismatic.vla.constants_gemma4 import ACTION_TOKEN_BEGIN_IDX, NUM_ACTION_TOKENS

# module 階層 (R2)
llm = model.model.language_model
hidden_size = model.config.text_config.hidden_size  # 1536

action_queries = torch.nn.Embedding(NUM_ACTION_TOKENS, hidden_size).cuda().to(torch.bfloat16)
action_queries.weight.data.zero_()  # VLA-Adapter の慣習に従い zero init

# input_ids に 64 個の distinct action placeholder を埋める
B = 1
L = 100
input_ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long).cuda()
# 位置 36..99 に 258885..258948 を配置
input_ids[:, 36:100] = torch.arange(
    ACTION_TOKEN_BEGIN_IDX,
    ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS,
).cuda()

# === PLE を input_ids から事前計算 (105GB OOM 回避の核心) ===
with torch.no_grad():
    per_layer_inputs = llm.get_per_layer_inputs(input_ids, None)  # (B, L, 35, 256)

# === Option A: clone + advanced indexing で action 位置を上書き ===
with torch.no_grad():
    raw_embeddings = llm.embed_tokens(input_ids)  # (B, L, 1536)
embeddings = raw_embeddings.clone()                # 新枝 (autograd)

for b in range(B):
    mask = (input_ids[b] >= ACTION_TOKEN_BEGIN_IDX) & \
           (input_ids[b] <  ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS)
    positions = mask.nonzero(as_tuple=True)[0]
    assert positions.numel() == NUM_ACTION_TOKENS, f"expected 64, got {positions.numel()}"
    embeddings[b, positions] = action_queries.weight   # (64, D) 代入

# === Gemma4TextModel を直接呼ぶ ===
# seq=100 < sliding_window=512 なので attention_mask は auto でもよいが、
# 1b.4 以降の統一性のため明示
attention_mask = torch.ones_like(input_ids, dtype=torch.long)
position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, -1).cuda()

out = llm(
    inputs_embeds=embeddings,
    per_layer_inputs=per_layer_inputs,
    use_cache=True,
    output_hidden_states=True,
    attention_mask=attention_mask,
    position_ids=position_ids,
)
```

## 3 層検証

```python
# === 1. 勾配検証 ===
loss = out.last_hidden_state.sum()
loss.backward()
assert action_queries.weight.grad is not None,              "action_queries grad is None"
assert action_queries.weight.grad.abs().sum().item() > 0,   "action_queries grad is zero"

# === 2. Semantic 検証 (torch.where 罠の検知) ===
hidden = out.hidden_states[-1]                              # (B, L, D)
positions_b0 = ((input_ids[0] >= ACTION_TOKEN_BEGIN_IDX) &
                (input_ids[0] <  ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS)
               ).nonzero(as_tuple=True)[0]
aq_hiddens = hidden[0, positions_b0]                        # (64, D)
assert not torch.allclose(aq_hiddens[0], aq_hiddens[1], atol=1e-6), \
    "action position 0 and 1 produce identical hidden states — overwrite broken"
assert aq_hiddens.std(dim=0).mean().item() > 1e-4, \
    "action position variance across 64 queries too small"

# === 3. LLM 凍結検証 (1b.6 で再実行) ===
# requires_grad=False なら .grad は None (冗長 check 削除)
```

## Exit criteria

- [ ] Option A コードが走る
- [ ] 勾配検証 2 assertion が全て pass
- [ ] semantic 検証 2 assertion が全て pass
- [ ] log に peak memory, output shape を記録

**→ User check-in point #3**: 全 assertion pass を確認してから 1b.4 着手

---

# Phase 1b.4: Vision integration (LIBERO = 2 カメラ = **512 patches**)

**目的**: DINO+SigLIP (既存の VLA-Adapter 実装を流用) の出力を 1536 dim に projection して token 列に差し込む。**v5.2 で 1b.3 の pattern を完全継承**。

## Token layout (VLA-Adapter §2.3 準拠)

```
[BOS(1)] [prompt(~50)] [vision_placeholders(512)] [proprio_placeholder(1)] [action_placeholders(64)] [EOS(1)]
total ≈ 628 tokens  (sliding_window 512 を超える → attention_mask 必須)
```

- `vision_placeholders`: 2 カメラ × 256 patches (agentview + wrist)、**ID 258949..259460** (distinct)
- `proprio_placeholder`: 1 個、**ID 259461**
- `action_placeholders`: ID 258885..258948

## Projector 定義

VLA-Adapter fused backbone の 3 層 MLP:

```python
class VisionProjector(torch.nn.Module):
    def __init__(self, vision_dim=2048, llm_dim=1536, initial_projection_dim=8192):
        super().__init__()
        self.fc1 = torch.nn.Linear(vision_dim, initial_projection_dim, bias=True)
        self.fc2 = torch.nn.Linear(initial_projection_dim, llm_dim, bias=True)
        self.fc3 = torch.nn.Linear(llm_dim, llm_dim, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, x):  # (B, 512, 2048)
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))
```

## 実装 (v5.2: 1b.3 pattern を完全継承、**per_layer_inputs・階層・attention_mask すべて対応**)

```python
# scripts/gemma4/test_04_vision.py
from prismatic.vla.constants_gemma4 import (
    ACTION_TOKEN_BEGIN_IDX, VISION_PLACEHOLDER_BEGIN_IDX, PROPRIO_PLACEHOLDER_IDX,
    NUM_ACTION_TOKENS, NUM_VISION_TOKENS,
)

# R2: 階層統一
llm = model.model.language_model
hidden_size = model.config.text_config.hidden_size

# Modules (vision backbone は既存の VLA-Adapter 実装を import)
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone
dino_siglip_backbone = DinoSigLIPViTBackbone(...).cuda().to(torch.bfloat16).eval()
for p in dino_siglip_backbone.parameters():
    p.requires_grad = False

vision_projector = VisionProjector(2048, hidden_size, 8192).cuda().to(torch.bfloat16)
action_queries   = torch.nn.Embedding(NUM_ACTION_TOKENS, hidden_size).cuda().to(torch.bfloat16)
action_queries.weight.data.zero_()

# === Dummy input 構築 ===
B = 1
pixel_values = torch.randn(B, 12, 224, 224, dtype=torch.bfloat16).cuda()  # 2 cam × 6 ch

# Token 列を組み立て: [BOS] [prompt 50] [vision 512] [proprio 1] [action 64] [EOS]
prompt = "pick up the red cube and place it in the blue tray"
prompt_ids = tok(prompt, return_tensors="pt").input_ids.cuda()[:, :50]  # ~50
prompt_len = prompt_ids.shape[1]

bos = torch.tensor([[tok.bos_token_id]], dtype=torch.long).cuda()
vision_ids = torch.arange(
    VISION_PLACEHOLDER_BEGIN_IDX,
    VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS,
).unsqueeze(0).cuda()
proprio_id = torch.tensor([[PROPRIO_PLACEHOLDER_IDX]], dtype=torch.long).cuda()
action_ids = torch.arange(
    ACTION_TOKEN_BEGIN_IDX,
    ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS,
).unsqueeze(0).cuda()
eos = torch.tensor([[tok.eos_token_id]], dtype=torch.long).cuda()

input_ids = torch.cat([bos, prompt_ids, vision_ids, proprio_id, action_ids, eos], dim=1)
L = input_ids.shape[1]
assert L == 1 + prompt_len + NUM_VISION_TOKENS + 1 + NUM_ACTION_TOKENS + 1

# === Vision features ===
vision_features = dino_siglip_backbone(pixel_values)   # (B, 512, 2048)
vision_projected = vision_projector(vision_features)    # (B, 512, 1536)

# === PLE 事前計算 (1b.3 と同じ、OOM 回避) ===
with torch.no_grad():
    per_layer_inputs = llm.get_per_layer_inputs(input_ids, None)  # (B, L, 35, 256)
    raw_embeddings = llm.embed_tokens(input_ids)                   # (B, L, 1536)

# === Option A: clone + 2 種類の placeholder を両方上書き ===
embeddings = raw_embeddings.clone()
for b in range(B):
    # action positions
    amask = (input_ids[b] >= ACTION_TOKEN_BEGIN_IDX) & \
            (input_ids[b] <  ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS)
    apos = amask.nonzero(as_tuple=True)[0]
    assert apos.numel() == NUM_ACTION_TOKENS
    embeddings[b, apos] = action_queries.weight                    # (64, 1536)

    # vision positions
    vmask = (input_ids[b] >= VISION_PLACEHOLDER_BEGIN_IDX) & \
            (input_ids[b] <  VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS)
    vpos = vmask.nonzero(as_tuple=True)[0]
    assert vpos.numel() == NUM_VISION_TOKENS
    embeddings[b, vpos] = vision_projected[b]                      # (512, 1536)

# === R6: attention_mask / position_ids を明示構築 (sliding_window 超過対策) ===
attention_mask = torch.ones_like(input_ids, dtype=torch.long)
position_ids   = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, -1).cuda()

# === llm 直接呼び (wrapper スキップ) ===
out = llm(
    inputs_embeds=embeddings,
    per_layer_inputs=per_layer_inputs,
    use_cache=True,
    output_hidden_states=True,
    attention_mask=attention_mask,
    position_ids=position_ids,
)
```

## 検証 (1b.3 と同じパターン + vision grad + vision semantic)

```python
loss = out.last_hidden_state.sum()
loss.backward()

# (a) vision_projector grad (3 層 MLP 全部)
assert vision_projector.fc1.weight.grad is not None
assert vision_projector.fc1.weight.grad.abs().sum().item() > 0
assert vision_projector.fc2.weight.grad.abs().sum().item() > 0
assert vision_projector.fc3.weight.grad.abs().sum().item() > 0

# action_queries grad も維持されているか再確認
assert action_queries.weight.grad.abs().sum().item() > 0

# (F) Vision semantic check — 隣接 2 つの vision position が distinct
vision_hidden = out.last_hidden_state[0, vpos[:2]]    # (2, D)
assert not torch.allclose(vision_hidden[0], vision_hidden[1], atol=1e-6), \
    "vision position 0 and 1 produce identical hidden states"

# Action semantic もまだ生きているか (統合後も維持)
action_hidden = out.last_hidden_state[0, apos[:2]]
assert not torch.allclose(action_hidden[0], action_hidden[1], atol=1e-6)
```

## Exit criteria

- [ ] Forward が通る (L ≈ 628 token で OOM しない、peak memory ログ記録)
- [ ] `attention_mask`, `position_ids` が明示的に構築されている
- [ ] Vision projector 全 3 層に grad が流れる
- [ ] Action queries grad も保持
- [ ] semantic 検証が action position でも vision position でも破綻なし

---

# Phase 1b.5: Action head 接続 + LayerNorm 判断

**目的**: VLA-Adapter 既存の `action_heads.py` (原版 or Pro) をそのまま流用。1b.1 で測定した hidden state std に基づいて LayerNorm 配置を決定する。

## LayerNorm 決定ルール (1b.1 測定値ベース、数値明示)

| 1b.1 で観測された std 分布 | LayerNorm 配置 |
|---|---|
| 全 36 entries で std ≤ 2.0 | **入れない** (Gemma 4 RMSNorm で十分) |
| std > 2.0 の層が 1-3 層 (sparse) | **該当層だけ**に LayerNorm (sparse) |
| std > 2.0 の層が 4 層以上 **or** std > 5.0 の層が 1 層以上存在 | **Action head 入力側に 1 枚**のみ挿入 (v1 パターン) |

**境界値の扱い**: std ∈ [1.8, 2.2] の層が存在する場合、Claude Code は自動判定せず **User に確認を求めて停止**。

## Step 0 (v5.2 強化: code 書く前の必須準備)

`action_heads.py` を精読し、**実機で具体層名を確定する**。想定値を assertion に埋める前に以下を実行:

```python
# scripts/gemma4/test_05_action_head.py の最初のセクション
from prismatic.models.action_heads import L1RegressionActionHead

action_head = L1RegressionActionHead(
    input_dim=hidden_size,
    hidden_dim=hidden_size,
    action_dim=7,
    num_task_tokens=512,
    use_pro_version=True,
).cuda().to(torch.bfloat16)

# === 実機の構造を print ===
print("=== action_head structure ===")
for name, p in action_head.named_parameters():
    print(f"  {name:60s} shape={tuple(p.shape)} trainable={p.requires_grad}")

# 学習可能な最初と最後の Linear を自動検出 (fallback)
linear_params = [
    (name, p) for name, p in action_head.named_parameters()
    if 'weight' in name and p.ndim == 2 and p.requires_grad
]
if not linear_params:
    raise RuntimeError("No trainable Linear found in action_head")
first_linear_name, first_linear_param = linear_params[0]
last_linear_name,  last_linear_param  = linear_params[-1]
print(f"\nFirst trainable Linear: {first_linear_name}")
print(f"Last  trainable Linear: {last_linear_name}")

# ↓ 以降の assertion はこの実在名を使う
#    例: action_head.model.fc1.weight が存在すればそれを使う
#        存在しなければ first_linear_param を直接参照
```

**想定との比較**:
- 想定どおり `action_head.model.fc1.weight` / `action_head.model.fc2.weight` が出てくる → そのまま assertion
- 想定と違う層名 (`in_proj`, `layers[0]` 等) → **assertion コードを実在名で書き直す**、log に記録

## 実装

```python
# Step 0 の結果を受けて、以下を継続

# Proprio projector (別モジュール)
class ProprioProjector(torch.nn.Module):
    def __init__(self, proprio_dim=8, llm_dim=1536):
        super().__init__()
        self.fc1 = torch.nn.Linear(proprio_dim, llm_dim)
        self.fc2 = torch.nn.Linear(llm_dim, llm_dim)
        self.act = torch.nn.GELU()
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

proprio_projector = ProprioProjector(proprio_dim=8, llm_dim=1536).cuda().to(torch.bfloat16)

# 1b.1 の測定結果 (LayerNorm 採否) に基づいて設定
USE_LAYERNORM = <True or False — 1b.1 判定後に記入>
if USE_LAYERNORM:
    feature_norm = torch.nn.LayerNorm(hidden_size).cuda().to(torch.bfloat16)
else:
    feature_norm = torch.nn.Identity()

# ===== 1b.4 の forward 結果を継承 =====
# (1b.4 の test_04 で定義した out, vpos, apos を再利用する場合は、
#  test_05 でも同じ構築を繰り返す or test_04 の結果を pickle で渡す)

all_hidden = torch.stack(out.hidden_states, dim=1)    # (B, 36, L, 1536)
# action_head は 24 blocks で layer index 1-24 を参照するので前 25 layer を slice
hidden_subset = all_hidden[:, :25, :, :]              # (B, 25, L, 1536)

# vision (task) + action position を抽出
vision_hidden = hidden_subset[:, :, vpos, :]          # (B, 25, 512, 1536)
action_hidden = hidden_subset[:, :, apos, :]          # (B, 25, 64, 1536)

# LayerNorm 適用 (上で決定済み)
vision_hidden = feature_norm(vision_hidden)
action_hidden = feature_norm(action_hidden)

# concat (task が先、action が後)
combined = torch.cat([vision_hidden, action_hidden], dim=2)  # (B, 25, 576, 1536)

# proprio を raw で渡す (action_head が内部で projection する)
proprio_raw = torch.randn(B, 8, dtype=torch.bfloat16).cuda()

# 推論
predicted = action_head.predict_action(
    actions_hidden_states=combined,
    proprio=proprio_raw,
    proprio_projector=proprio_projector,
    phase="Training",
)
assert predicted.shape == (B, 8, 7)
```

## 検証 (Step 0 で確定した具体層名を使う)

```python
loss = torch.nn.L1Loss()(predicted, torch.zeros_like(predicted))
loss.backward()

# Step 0 で確定した層名で assertion
# 想定どおりの場合:
assert action_head.model.fc1.weight.grad is not None
assert action_head.model.fc1.weight.grad.abs().sum().item() > 0
assert action_head.model.fc2.weight.grad.abs().sum().item() > 0

# Step 0 で別名だった場合 (fallback):
assert first_linear_param.grad is not None
assert first_linear_param.grad.abs().sum().item() > 0
assert last_linear_param.grad.abs().sum().item() > 0

# proprio_projector も学習対象
assert proprio_projector.fc1.weight.grad.abs().sum().item() > 0

# LayerNorm 使用時のみ
if USE_LAYERNORM:
    assert feature_norm.weight.grad.abs().sum().item() > 0
```

## Exit criteria

- [ ] Step 0 で action_head の named_parameters が log に残っている
- [ ] `predicted.shape == (B, 8, 7)`
- [ ] 具体層 (実在確認済み) の grad が存在
- [ ] trainable param count を記録 (action_head 約 218M 期待)
- [ ] LayerNorm 配置が決定されログに記載

---

# Phase 1b.6: 統合 `VLAAdapterGemma4` クラス + frozen LLM 機械的 verify

**目的**: 1b.3-1b.5 の部品を `modeling_prismatic_gemma4.py` にまとめる。LLM 凍結が正しく守られているか機械的に確認。

## クラス雛形

```python
# VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py
class VLAAdapterGemma4(torch.nn.Module):
    def __init__(self, gemma_model, vision_backbone, feature_norm=None):
        super().__init__()
        self.llm = gemma_model  # Gemma4ForConditionalGeneration
        self.llm.config.use_cache = True
        for p in self.llm.parameters():
            p.requires_grad = False
        self.vision_backbone = vision_backbone  # DINO+SigLIP, frozen
        for p in self.vision_backbone.parameters():
            p.requires_grad = False
        self.vision_projector  = VisionProjector(
            vision_dim=2048, llm_dim=self.llm_dim, initial_projection_dim=8192,
        )
        self.proprio_projector = ProprioProjector(proprio_dim=8, llm_dim=self.llm_dim)
        self.action_queries    = torch.nn.Embedding(NUM_ACTION_TOKENS, self.llm_dim)
        self.action_queries.weight.data.zero_()
        self.feature_norm      = feature_norm or torch.nn.Identity()
        self.action_head       = L1RegressionActionHead(
            input_dim=self.llm_dim, hidden_dim=self.llm_dim, action_dim=7,
            num_task_tokens=512, use_pro_version=True,
        )

    @property
    def llm_dim(self):
        return self.llm.config.text_config.hidden_size

    @property
    def text_model(self):
        return self.llm.model.language_model

    def forward(self, pixel_values, input_ids, proprio, actions=None):
        """
        pixel_values: (B, 12, 224, 224)  DINO+SigLIP input
        input_ids:    (B, L)              placeholder 込みの系列
        proprio:      (B, 8)              raw proprio
        actions:      (B, 8, 7) or None   Training loss 計算用
        """
        B, L = input_ids.shape
        llm = self.text_model

        # Vision features
        vision_features = self.vision_backbone(pixel_values)
        vision_projected = self.vision_projector(vision_features)

        # PLE 事前計算
        with torch.no_grad():
            per_layer_inputs = llm.get_per_layer_inputs(input_ids, None)
            raw_embeddings = llm.embed_tokens(input_ids)
        embeddings = raw_embeddings.clone()

        # 2 種類の placeholder を上書き (apos, vpos は batch ごとに共通配置と仮定)
        amask = (input_ids >= ACTION_TOKEN_BEGIN_IDX) & \
                (input_ids <  ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS)
        vmask = (input_ids >= VISION_PLACEHOLDER_BEGIN_IDX) & \
                (input_ids <  VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS)
        for b in range(B):
            apos = amask[b].nonzero(as_tuple=True)[0]
            vpos = vmask[b].nonzero(as_tuple=True)[0]
            embeddings[b, apos] = self.action_queries.weight
            embeddings[b, vpos] = vision_projected[b]

        # attention_mask / position_ids
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        position_ids = torch.arange(L, dtype=torch.long, device=input_ids.device
                                   ).unsqueeze(0).expand(B, -1)

        # LLM forward
        out = llm(
            inputs_embeds=embeddings,
            per_layer_inputs=per_layer_inputs,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Action head 入力組み立て
        all_hidden = torch.stack(out.hidden_states, dim=1)
        hidden_subset = all_hidden[:, :25, :, :]
        # 全 batch で同じ apos/vpos と仮定 (LIBERO は固定 layout)
        apos0 = amask[0].nonzero(as_tuple=True)[0]
        vpos0 = vmask[0].nonzero(as_tuple=True)[0]
        vision_hidden = self.feature_norm(hidden_subset[:, :, vpos0, :])
        action_hidden = self.feature_norm(hidden_subset[:, :, apos0, :])
        combined = torch.cat([vision_hidden, action_hidden], dim=2)

        predicted = self.action_head.predict_action(
            actions_hidden_states=combined,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            phase="Training" if self.training else "Inference",
        )

        if actions is None:
            return predicted
        loss = torch.nn.functional.l1_loss(predicted, actions)
        return predicted, loss
```

## 統合 test (`scripts/gemma4/test_06_full_forward.py`)

```python
model_vla = VLAAdapterGemma4(gemma_model, vision_backbone, feature_norm)

# === LLM 凍結の機械的 verify ===
llm_trainable = sum(p.numel() for p in model_vla.llm.parameters() if p.requires_grad)
assert llm_trainable == 0, f"LLM should be frozen, got {llm_trainable} trainable"

vb_trainable = sum(p.numel() for p in model_vla.vision_backbone.parameters() if p.requires_grad)
assert vb_trainable == 0, f"Vision backbone should be frozen, got {vb_trainable}"

# 総 trainable が想定範囲
# (action_head Pro ~640M + vision_projector ~32M + proprio_projector ~2.4M + action_queries ~0.1M ≈ 675M)
# v5.2 初版は action_head を ~218M と誤推定 (非 Pro 版の値を Pro に適用したミス)。
# 1b.5 Step 0 で実測 639.89M と確定。Pro 版は cross-attention (q/k/v_task) + FiLM gen を含む。
total_trainable = sum(p.numel() for p in model_vla.parameters() if p.requires_grad)
assert 600e6 < total_trainable < 750e6, f"Unexpected trainable: {total_trainable/1e6:.1f}M (expected ~675M for Pro action head)"
print(f"Trainable: {total_trainable/1e6:.1f}M (expected ~675M, Pro版)")

# === Dummy forward + backward + 全 grad verify ===
B = 1
pixel_values = torch.randn(B, 12, 224, 224, dtype=torch.bfloat16).cuda()
# input_ids 構築は 1b.4 の手順を再利用
input_ids = build_full_input_ids(B, tok, prompt="pick up the red cube")
proprio = torch.randn(B, 8, dtype=torch.bfloat16).cuda()
actions = torch.randn(B, 8, 7, dtype=torch.bfloat16).cuda()

predicted, loss = model_vla(pixel_values, input_ids, proprio, actions)
assert predicted.shape == (B, 8, 7)
loss.backward()

# 各 trainable module に grad が流れるか (Step 0 で確定した層名を使用)
assert model_vla.action_queries.weight.grad.abs().sum() > 0
assert model_vla.vision_projector.fc1.weight.grad.abs().sum() > 0
assert model_vla.vision_projector.fc3.weight.grad.abs().sum() > 0
assert model_vla.proprio_projector.fc1.weight.grad.abs().sum() > 0
# action_head の実層名は Step 0 で確定 (想定 or fallback)
assert model_vla.action_head.model.fc1.weight.grad.abs().sum() > 0

# semantic 検証 (統合後も維持されているか)
# action position hidden が distinct であること
```

## Exit criteria

- [ ] Class インスタンス化 OK
- [ ] LLM 凍結 assertion (`llm_trainable == 0`)
- [ ] Vision backbone 凍結 assertion
- [ ] Total trainable 200M-300M
- [ ] 全 trainable module の grad > 0
- [ ] Output shape `(B, 8, 7)`
- [ ] Peak memory 記録
- [ ] Loss が finite

---

# Phase 1b → 1c → 1d への橋渡し

Phase 1b 完了後、1c (backward pass の batch scaling) と 1d (LIBERO データで 10-step train) は元プラン準拠。追加ガード:

- **1c**: `torch.cuda.max_memory_allocated()` を batch=1, 2, 4 で記録、A100-80GB で batch=4 が 70GB 未満なら OK
- **1d**: `finetune.py` を `finetune_gemma4.py` として複製、LoRA / gradient_checkpointing は**呼び出し箇所をコメントアウト**
- **1d** の smoke train で loss が NaN/Inf なら即停止

## 1d の精度についての注記 (v5.2 追加)

**Stage 1 は causal attention のまま進める**。Action queries は系列末尾に配置されるため、**後半の query ほど参照できる情報が減る** (自分より前の query しか見られない)。このため:

- 論文値 (LIBERO-Spatial 99.6%) の再現は **期待しない**
- 10 step で loss が finite かつ減少傾向なら smoke 合格
- 論文値近くの精度は Stage 2 の bidirectional attention patch 後に検証

loss が「減らない」ときの切り分け:
1. まず causal の限界か (この場合は Stage 2 に進む)
2. 勾配検証 / semantic 検証が壊れている (1b.3/1b.4 に戻る)
3. データパイプラインが壊れている (1d 内で調査)

---

# Check-in Points (v5.2 追加)

Claude Code は以下のポイントで**必ず停止し、User に結果を見せてから次に進む**:

| # | タイミング | User に見せるもの | 次ステップの判断材料 |
|---|---|---|---|
| 1 | **1b.1 完了後** | 36 entries の std/mean 分布、peak memory | LayerNorm 判定ルールの適用結果 (USE_LAYERNORM) |
| 2 | **1b.2 完了後** | 10 試行のメモリスケーリング表 | 1b.3 方針 (Option A or シンプル採用) |
| 3 | **1b.3 完了後** | 勾配 / semantic assertion の全 pass 確認、peak memory | 1b.4 着手許可 |
| 4 | **1b.5 Step 0 完了後** | `action_head.named_parameters()` の出力 | 具体層名が想定どおりか (fallback 要否) |
| 5 | **1b.6 完了後** | 統合 test の全 assertion 結果、peak memory | 1c 着手許可 |

境界ケース (例: std=2.05 で LayerNorm 要否が曖昧) も停止して User 判断を仰ぐ。**自動で進めない**。

---

# Deliverables (Stage 1 完了時)

| # | ファイル | 状態 |
|---|---|---|
| 1 | `requirements-gemma4.txt` | Phase 1b.1 前に生成 (uv pip freeze) |
| 2 | `VLA-Adapter/prismatic/models/backbones/llm/gemma4.py` | Phase 1b.6 |
| 3 | `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` | Phase 1b.6 |
| 4 | `VLA-Adapter/prismatic/vla/constants_gemma4.py` | Phase 1a **(完了)** |
| 5 | `VLA-Adapter/vla-scripts/finetune_gemma4.py` | Phase 1d |
| 6 | `scripts/gemma4/test_01_load.py` 〜 `test_06_full_forward.py` | Phase 1b 各ステップ |
| 7 | `scripts/gemma4/smoke_test_gemma4.py` | Phase 1c/1d 再現用 |
| 8 | `docs/gemma4_migration_log.md` | 全 Phase で随時追記 |

---

# Escalation Policy (即座停止して User 報告)

1. Phase 0 で config 値が想定と食い違う (既に検証済み、該当なし)
2. Phase 1a で resize が必要になる (既に不要確認済み)
3. Phase 1b.1 で peak memory が 15GB を超える
4. Phase 1b.2 で Case 1 も Case 2 も OOM (真因分析が間違い)
5. Phase 1b.3 の**勾配または semantic assertion が fail**
6. Phase 1b.5 Step 0 で action_head の層構造が**fallback でも特定不能**
7. Phase 1b.6 の LLM 凍結 assertion が fail
8. 80GB GPU で batch=1 すら OOM
9. `Gemma4ForConditionalGeneration` の API が想定と大幅に異なる
10. 単一の問題で 2 時間以上ハマる
11. Stage 1 スコープ外に踏み込まないと解けない問題
12. Check-in Point で境界判定が必要な場合

## やってはいけないこと

- Monkey patch で問題を隠蔽 (特に KV 共有バグ、PLE 逆引き)
- Workaround で「動いているように見せる」(assertion を潰す、warning を黙らせる)
- 設計判断を単独で行う (例: 勝手に DINO を外す、LayerNorm を入れるかの判断を勝手にする → 決定ルールに従う)
- **`torch.where` + broadcast で action queries を代入**する (64 個が全て同じになり mode collapse、しかも表面上は動く)
- **Check-in Point をスキップ**して次 Phase に自動進行
- `model.language_model.*` (v1 時代の階層) を使う → transformers 5.5 では `model.model.language_model.*`
- `inputs_embeds` だけ渡して `per_layer_inputs` を省略 → PLE 逆引きで OOM

---

# Out of Scope (Stage 2 以降)

- LoRA の導入
- 双方向 attention patch (`use_bidirectional_attention=True` の挙動確認は Stage 2)
- Vision backbone の差し替え (DINO+SigLIP を維持)
- 推論 (LIBERO eval) / rollout
- Multi-GPU DDP / FSDP
- 論文精度 (99.6%) の再現検証
- transformers 4.40 への後方互換性維持

---

# 変更履歴

| Date | Version | 変更内容 |
|---|---|---|
| 2026-04-19 | v1 | 初版 (User 提供の Stage 1 task spec) |
| 2026-04-19 | v2 | 泥沼化懸念受けてマイクロステップ化 |
| 2026-04-19 | v3 | v1 教訓 (PLE OOM, LayerNorm) 反映 |
| 2026-04-19 | v4 | 勾配検証, memory 数値 assertion, 2 カメラ修正, use_cache 明示 |
| 2026-04-19 | v5 | torch.where の mode collapse 罠を Option A に確定、semantic 検証 |
| 2026-04-19 | v5.1 | PLE 105GB OOM の真因解明、階層 `model.model.language_model.*` に訂正、action_head 入力は task+action 両方、placeholder ID 全種類確定、LayerNorm 数値化 |
| 2026-04-19 | **v5.2 (確定)** | **1b.4 のコードを 1b.3 pattern に完全統一 (per_layer_inputs, 階層, attention_mask)**、1b.5 Step 0 で named_parameters print + fallback 方針追加、1b.2 に per_layer_inputs 付きケース追加、Check-in Points セクション新設、1d に causal 注記追加、R6 (attention_mask 必須) 追加 |
