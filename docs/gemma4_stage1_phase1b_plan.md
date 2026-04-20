# Gemma 4 E2B 移植 Stage 1 / Phase 1b 実装計画 (v5.1 確定版)

**対象**: Phase 1b (Model 組み立て) のマイクロステップ分割。Phase 0 / 1a は完了済み。
**作成経緯**: User との 4 ラウンドのレビュー (泥沼化懸念 → v1 教訓反映 → 勾配検証 → semantic 罠 → PLE/action_head 詳細確定) を経て確定。
**目的**: Gemma 4 E2B backbone + VLA-Adapter 設計で LIBERO 10-step smoke train を走らせる (性能は問わない)。

---

## v5.1 で確定した Gemma 4 実装詳細 (実機検証済み)

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

全てのアクセスは `model.model.language_model.*` 経由。

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

### 正しい forward pattern (1b.3 以降)

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

### Action head の学習可能 layer (具体層名確定)

`action_heads.py` 読んで確定:

```python
# MLPResNet 内の学習可能な具体 layer
action_head.model.fc1.weight                          # 入力側 Linear
action_head.model.fc2.weight                          # 出力側 Linear
action_head.model.mlp_resnet_blocks[0].<block内層>    # 24 blocks 存在
```

1b.6 の assertion には `action_head.model.fc1.weight.grad.abs().sum() > 0` を使用。

---

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
- `resize_token_embeddings` **不要** (既存 unused token を流用)
- `constants_gemma4.py` 作成済み: `ACTION_TOKEN_BEGIN_IDX=258885, STOP_INDEX=1`

---

## v1 から引き継ぐ重要教訓

| # | 教訓 | Phase 1b での対応 |
|---|---|---|
| T1 | Gemma 4 PLE は `inputs_embeds` 直渡しで **105GB OOM** | 1b.2 で罠を段階的に再現 → 必要なら PAD+overwrite 採用 |
| T2 | 特殊 token を **文字列で書くと BPE 分割**される | 終始 **ID 直挿入** で統一 (Phase 1a の方針) |
| T3 | LLM hidden state `std≈6` で **action head が mode collapse** | 1b.1 で測定 → 1b.5 で測定値に基づき LayerNorm 配置決定 |
| T4 | model 階層は `model.language_model.*` | 全 test script で明示 |
| T5 | PEFT ラップで階層が変わる | Stage 1 は LoRA 不使用なので影響なし (Stage 2 で対応) |

---

## 横断ルール (全 test script に適用)

### R1. 明示的 config 設定

```python
model.config.use_cache = True                  # Gemma 4 KV 共有バグ (HF #45242) 回避
assert not model.is_gradient_checkpointing      # checkpointing は Stage 1 で使わない
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
- [ ] 全 35 層の hidden state が finite (`not torch.isnan(h).any()`)
- [ ] 全層の std が [0.1, 10] 範囲
- [ ] ログに 35 層全部の std/mean が記録される

---

# Phase 1b.2: `inputs_embeds` PLE 罠を段階的に切り分け

**目的**: VLA-Adapter 原設計の `inputs_embeds` 直渡しが Gemma 4 でどう振る舞うかを測定。二値判断ではなく、**スケーリング挙動で真因を絞る**。

## 実装

```python
# scripts/gemma4/test_02_inputs_embeds.py
# 同じ model をロード後
cases = [
    (1,  100),  # baseline
    (1,  500),  # seq 線形?
    (1, 1000),  # seq 非線形?
    (4,  500),  # batch 線形?
    (4, 1000),  # batch × seq 相互作用 (OOM 覚悟)
]
for B, L in cases:
    try:
        dummy_embeds = torch.randn(B, L, 1536, dtype=torch.bfloat16).cuda()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out = model(inputs_embeds=dummy_embeds)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"B={B} L={L}: peak={peak:.2f}GB, out.shape={out.logits.shape}")
    except torch.cuda.OutOfMemoryError:
        print(f"B={B} L={L}: OOM")
```

## 判定分岐 (→ 1b.3 の実装方針)

| 観測されたスケーリング | 診断 | 1b.3 選択肢 |
|---|---|---|
| memory ∝ `B × L` (線形) | (c) batch 過大のみ | **シンプル採用**: `inputs_embeds` 直渡し |
| memory ∝ `B × L²` か `layers × L` | PLE broadcast バグ確定 | **Option A**: PAD + clone + advanced indexing |
| 全 case で 15GB 以下 | 5.5.4 で修正済み | **シンプル採用** |

## Exit criteria

- [ ] 5 試行のメモリ消費をログ
- [ ] 上記テーブルで分類できるスケーリング挙動
- [ ] 1b.3 の実装方針が決定される

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
from prismatic.vla.constants_gemma4 import ACTION_TOKEN_BEGIN_IDX
NUM_TOKENS = 64

# module 階層 (R2)
llm = model.model.language_model
hidden_size = model.config.text_config.hidden_size  # 1536

action_queries = torch.nn.Embedding(NUM_TOKENS, hidden_size).cuda().to(torch.bfloat16)
action_queries.weight.data.zero_()  # VLA-Adapter の慣習に従い zero init

# input_ids に 64 個の distinct action placeholder を埋める
B = 1
L = 100
input_ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long).cuda()
# 位置 36..99 に 258885..258948 を配置
input_ids[:, 36:100] = torch.arange(ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_TOKENS).cuda()

# === PLE を input_ids から事前計算 (105GB OOM 回避の核心) ===
with torch.no_grad():
    per_layer_inputs = llm.get_per_layer_inputs(input_ids, None)  # shape (B, L, 35, 256)

# === Option A: clone + advanced indexing で action 位置を上書き ===
with torch.no_grad():
    raw_embeddings = llm.embed_tokens(input_ids)  # (B, L, 1536)
embeddings = raw_embeddings.clone()                # 新枝 (autograd)

for b in range(B):
    mask = (input_ids[b] >= ACTION_TOKEN_BEGIN_IDX) & \
           (input_ids[b] <  ACTION_TOKEN_BEGIN_IDX + NUM_TOKENS)
    positions = mask.nonzero(as_tuple=True)[0]
    assert positions.numel() == NUM_TOKENS, f"expected 64, got {positions.numel()}"
    embeddings[b, positions] = action_queries.weight   # (64, D) 代入

# === Gemma4TextModel を直接呼ぶ (ConditionalGeneration wrapper をスキップ) ===
out = llm(
    inputs_embeds=embeddings,
    per_layer_inputs=per_layer_inputs,   # PLE を明示的に供給
    use_cache=True,                       # G1 (KV 共有バグ) 対策
    output_hidden_states=True,
)
```

## 3 層検証

```python
# === 1. 勾配検証 ===
loss = out.logits.sum()  # Stage 1 は loss 形状問わず
loss.backward()
assert action_queries.weight.grad is not None,              "action_queries grad is None"
assert action_queries.weight.grad.abs().sum().item() > 0,   "action_queries grad is zero"

# === 2. Semantic 検証 (torch.where 罠の検知) ===
hidden = out.hidden_states[-1]                              # (B, L, D)
positions_b0 = ((input_ids[0] >= ACTION_TOKEN_BEGIN_IDX) &
                (input_ids[0] <  ACTION_TOKEN_BEGIN_IDX + NUM_TOKENS)).nonzero(as_tuple=True)[0]
aq_hiddens = hidden[0, positions_b0]                        # (64, D)
assert not torch.allclose(aq_hiddens[0], aq_hiddens[1], atol=1e-6), \
    "action position 0 and 1 produce identical hidden states — overwrite broken"
assert aq_hiddens.std(dim=0).mean().item() > 1e-4, \
    "action position variance across 64 queries too small"

# === 3. LLM 凍結検証 (1b.6 で再実行) ===
# 冗長な `.grad == 0` check は削除。requires_grad=False なら .grad は None
```

## Exit criteria

- [ ] Option A コードが走る
- [ ] 勾配検証 3 assertion が全て pass
- [ ] semantic 検証 2 assertion が全て pass
- [ ] log に peak memory, output shape を記録

---

# Phase 1b.4: Vision integration (LIBERO = 2 カメラ = **512 patches**)

**目的**: DINO+SigLIP (既存の VLA-Adapter 実装を流用) の出力を 1536 dim に projection して token 列に差し込む。

## Token layout (VLA-Adapter §2.3 準拠)

```
[BOS(1)] [prompt(~50)] [vision_placeholders(512)] [proprio_placeholder(1)] [action_placeholders(64)] [EOS(1)]
total ≈ 628 tokens
```

- `vision_placeholders`: 2 カメラ × 256 patches (agentview + wrist)、**ID 258949..259460** (`constants_gemma4.VISION_PLACEHOLDER_BEGIN_IDX`、512 個 distinct)
- `proprio_placeholder`: 1 個、**ID 259461** (`constants_gemma4.PROPRIO_PLACEHOLDER_IDX`)
- `action_placeholders`: ID 258885..258948 (Phase 1a で確定)

## Projector 定義

VLA-Adapter fused backbone の 3 層 MLP:
```python
class VisionProjector(nn.Module):
    def __init__(self, vision_dim=2048, llm_dim=1536, initial_projection_dim=8192):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, initial_projection_dim, bias=True)
        self.fc2 = nn.Linear(initial_projection_dim, llm_dim, bias=True)
        self.fc3 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act = nn.GELU()
    def forward(self, x):  # (B, 512, 2048)
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))
```

## 実装

```python
# scripts/gemma4/test_04_vision.py

# Existing DINO+SigLIP from VLA-Adapter (load via prismatic.models.backbones.vision)
# Dummy image 2 枚
pixel_values = torch.randn(B, 12, 224, 224, dtype=torch.bfloat16).cuda()  # 2 images × 6 ch
vision_features = dino_siglip_backbone(pixel_values)        # (B, 512, 2048)
vision_projected = vision_projector(vision_features)        # (B, 512, 1536)

# input_ids に vision placeholder (ID VISION_PLACEHOLDER_ID) を 512 個埋める
# その位置の embedding を vision_projected で上書き (Option A と同じパターン)
with torch.no_grad():
    raw_embeddings = model.language_model.embed_tokens(input_ids)
embeddings = raw_embeddings.clone()
for b in range(B):
    vmask = (input_ids[b] == VISION_PLACEHOLDER_ID)
    vpos  = vmask.nonzero(as_tuple=True)[0]
    assert vpos.numel() == 512
    embeddings[b, vpos] = vision_projected[b]
    # action も同様 (1b.3 のロジック)
    ...

out = model(inputs_embeds=embeddings, output_hidden_states=True)
```

## 検証 (1b.3 と同じパターン + vision grad + vision semantic)

```python
loss = out.last_hidden_state.sum()
loss.backward()

# (a) vision_projector grad (3 層 MLP 全部)
assert vision_projector.fc1.weight.grad is not None
assert vision_projector.fc1.weight.grad.abs().sum().item() > 0
assert vision_projector.fc3.weight.grad.abs().sum().item() > 0

# action_queries grad も維持されているか再確認
assert action_queries.weight.grad.abs().sum().item() > 0

# (F) Vision semantic check — 隣接 2 つの vision position が distinct
# projector 重みがゼロ初期化されてたり、overwrite が壊れていたら検出
vision_hidden = out.last_hidden_state[0, vpos[:2]]    # (2, D)
assert not torch.allclose(vision_hidden[0], vision_hidden[1], atol=1e-6), \
    "vision position 0 and 1 produce identical hidden states"
```

## Exit criteria

- [ ] Forward が通る
- [ ] Vision projector 全 3 層に grad が流れる
- [ ] Action queries grad も保持
- [ ] semantic 検証が action position でも vision position でも破綻なし

---

# Phase 1b.5: Action head 接続 + LayerNorm 判断

**目的**: VLA-Adapter 既存の `action_heads.py` (原版 or Pro) をそのまま流用。1b.1 で測定した hidden state std に基づいて LayerNorm 配置を決定する。

## LayerNorm 決定ルール (1b.1 測定値ベース、数値明示)

| 1b.1 で観測された std 分布 | LayerNorm 配置 |
|---|---|
| 全 35 層で std ≤ 2.0 | **入れない** (Gemma 4 RMSNorm で十分) |
| std > 2.0 の層が 1-3 層 (sparse) | **該当層だけ**に LayerNorm (sparse) |
| std > 2.0 の層が 4 層以上 **or** std > 5.0 の層が 1 層以上存在 | **Action head 入力側に 1 枚**のみ挿入 (v1 パターン) |

**ルール適用の注意**: 数値で判定できるが、境界ケース (std = 2.05 等) は User 確認ステップ (test_01 の結果を見せて判断仰ぐ)。

## 実装

**Step 0 (code 書く前の必須準備)**: `action_heads.py` を精読し、具体層名を確定する。
- `action_head.model.fc1.weight` / `fc2.weight` — MLPResNet の入出力 Linear
- `action_head.model.layer_norm1.weight` / `layer_norm2.weight` — MLPResNet 内の LayerNorm
- `action_head.model.mlp_resnet_blocks[i]` — MLPResNetBlock_Pro (24 個)

```python
# scripts/gemma4/test_05_action_head.py
import sys
sys.path.insert(0, 'VLA-Adapter')
from prismatic.models.action_heads import L1RegressionActionHead

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

# action_head 初期化 (input_dim = hidden_size、内部で * ACTION_DIM される)
action_head = L1RegressionActionHead(
    input_dim=hidden_size,       # 1536 (内部で × ACTION_DIM=7 される)
    hidden_dim=hidden_size,      # 1536
    action_dim=7,
    num_task_tokens=512,
    use_pro_version=True,
).cuda().to(torch.bfloat16)

# ===== 全 hidden state の shape 調整 =====
all_hidden = torch.stack(out.hidden_states, dim=1)    # (B, 36, L, 1536)
# action_head は 24 blocks で layer index 1-24 を参照するので前 25 layer を slice
hidden_subset = all_hidden[:, :25, :, :]              # (B, 25, L, 1536)

# vision (task) + action position を抽出
vision_hidden = hidden_subset[:, :, vpos_b0, :]       # (B, 25, 512, 1536)
action_hidden = hidden_subset[:, :, apos_b0, :]       # (B, 25, 64, 1536)

# LayerNorm 適用 (上で決定済み)
vision_hidden = feature_norm(vision_hidden)
action_hidden = feature_norm(action_hidden)

# concat (task が先、action が後)
combined = torch.cat([vision_hidden, action_hidden], dim=2)  # (B, 25, 576, 1536)

# 推論
predicted = action_head.predict_action(
    actions_hidden_states=combined,
    proprio=proprio_raw,                              # (B, 8) raw
    proprio_projector=proprio_projector,
    phase="Training",
)
assert predicted.shape == (B, 8, 7)
```

## 検証 (具体層名確定)

```python
loss = torch.nn.L1Loss()(predicted, torch.zeros_like(predicted))
loss.backward()

# action_head 内の具体層 (Step 0 で確定済み)
assert action_head.model.fc1.weight.grad is not None
assert action_head.model.fc1.weight.grad.abs().sum().item() > 0
assert action_head.model.fc2.weight.grad.abs().sum().item() > 0

# proprio_projector も学習対象
assert proprio_projector.fc1.weight.grad.abs().sum().item() > 0

# LayerNorm 使用時のみ
if USE_LAYERNORM:
    assert feature_norm.weight.grad.abs().sum().item() > 0
```

## Exit criteria

- [ ] `predicted_actions.shape == (B, 8, 7)`
- [ ] action head 内の具体層 (1b.6 で確定) の grad 存在
- [ ] trainable param count を記録 (action_head 約 218M 期待)
- [ ] LayerNorm 配置が決定されログに記載

---

# Phase 1b.6: 統合 `VLAAdapterGemma4` クラス + frozen LLM 機械的 verify

**目的**: 1b.3-1b.5 の部品を `modeling_prismatic_gemma4.py` にまとめる。LLM 凍結が正しく守られているか機械的に確認。

## クラス雛形

```python
# VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py
class VLAAdapterGemma4(nn.Module):
    def __init__(self, gemma_model, vision_backbone, ...):
        super().__init__()
        self.llm = gemma_model
        self.llm.config.use_cache = True
        for p in self.llm.parameters():
            p.requires_grad = False
        self.vision_backbone = vision_backbone  # DINO+SigLIP, frozen
        for p in self.vision_backbone.parameters():
            p.requires_grad = False
        self.vision_projector  = VisionProjector(...)   # trainable
        self.proprio_projector = ProprioProjector(...)  # trainable
        self.action_queries    = nn.Embedding(64, self.llm_dim)  # trainable
        self.action_queries.weight.data.zero_()
        self.feature_norm      = ...                    # 1b.5 で決定
        self.action_head       = L1RegressionActionHead(...)

    @property
    def llm_dim(self):
        return self.llm.config.text_config.hidden_size

    def forward(self, pixel_values, input_ids, proprio, actions=None):
        # 1b.4 のパターン: vision + action を embedding 上書き
        # 1b.5 のパターン: LLM forward → action hidden → action_head
        ...
```

## 統合 test (`scripts/gemma4/test_06_full_forward.py`)

```python
model = VLAAdapterGemma4(...)

# === LLM 凍結の機械的 verify (階層に注意) ===
# R2: model.llm が Gemma4ForConditionalGeneration (self.llm)
llm_trainable = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
assert llm_trainable == 0, f"LLM should be frozen, got {llm_trainable} trainable"

vb_trainable = sum(p.numel() for p in model.vision_backbone.parameters() if p.requires_grad)
assert vb_trainable == 0, f"Vision backbone should be frozen, got {vb_trainable}"

# 総 trainable が想定範囲 (action_head ~218M + projectors ~14M + action_queries ~0.1M ≈ 230M)
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert 200e6 < total_trainable < 300e6, f"Unexpected trainable: {total_trainable/1e6:.1f}M"
print(f"Trainable: {total_trainable/1e6:.1f}M (expected ~230M)")

# === Dummy forward + backward + 全 grad verify ===
out = model(pixel_values=..., input_ids=..., proprio=...)
assert out.shape == (B, 8, 7)

loss = torch.nn.L1Loss()(out, torch.zeros_like(out))
loss.backward()

# 各 trainable module に grad が流れるか (全て具体層名)
assert model.action_queries.weight.grad.abs().sum() > 0
assert model.vision_projector.fc1.weight.grad.abs().sum() > 0
assert model.vision_projector.fc3.weight.grad.abs().sum() > 0
assert model.proprio_projector.fc1.weight.grad.abs().sum() > 0
assert model.action_head.model.fc1.weight.grad.abs().sum() > 0
assert model.action_head.model.fc2.weight.grad.abs().sum() > 0

# === 1b.3 の semantic 検証を再実行 (統合後も維持されているか) ===
# action position hidden state が distinct
```

## Exit criteria

- [ ] Class インスタンス化 OK
- [ ] LLM 凍結 assertion (`llm_trainable == 0`)
- [ ] Vision backbone 凍結 assertion
- [ ] Total trainable 200M-300M
- [ ] 全 trainable module の grad > 0
- [ ] Output shape `(B, 8, 7)`
- [ ] Peak memory 記録

---

# Phase 1b → 1c → 1d への橋渡し

Phase 1b 完了後、1c (backward pass の batch scaling) と 1d (LIBERO データで 10-step train) は元プラン準拠。追加ガード:

- 1c で `torch.cuda.max_memory_allocated()` を batch=1,2,4 で記録、A100-80GB で batch=4 が 70GB 未満なら OK
- 1d で finetune.py を `finetune_gemma4.py` として複製、LoRA / gradient_checkpointing は**呼び出し箇所をコメントアウト**
- 1d の smoke train で loss が NaN/Inf なら即停止

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
4. Phase 1b.2 で scaling が `B × L²` または `layers × L` 依存
5. Phase 1b.3 の**勾配または semantic assertion が fail**
6. Phase 1b.6 の LLM 凍結 assertion が fail
7. 80GB GPU で batch=1 すら OOM
8. `Gemma4ForConditionalGeneration` の API が想定と大幅に異なる
9. 単一の問題で 2 時間以上ハマる
10. Stage 1 スコープ外に踏み込まないと解けない問題

## やってはいけないこと

- Monkey patch で問題を隠蔽 (特に KV 共有バグ)
- Workaround で「動いているように見せる」(assertion を潰す、warning を黙らせる)
- 設計判断を単独で行う (例: 勝手に DINO を外す、LayerNorm を入れるかの判断を勝手にする → 決定ルールに従う)
- **`torch.where` + broadcast で action queries を代入**する (64 個が全て同じになり mode collapse、しかも表面上は動く)

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
| 2026-04-19 | v5 | torch.where の mode collapse 罠を Option A に確定、semantic 検証、軽微調整 |
| 2026-04-19 | **v5.1 (確定)** | **PLE 105GB OOM の真因解明** (get_per_layer_inputs の逆引き) + 対策パターン明示 / 階層を `model.model.language_model.*` に訂正 / **action_head 入力は task+action 両方 (576 tokens)** / vision placeholder ID 範囲確定 (258949-259460) / action_head 具体層名確定 (`model.fc1/fc2`) / LayerNorm 判定ルール数値化 / pad_token_id 確認追加 |
