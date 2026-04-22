# VLA Redesign: Scene / Wrist Path Split + Dual-Track (Quality vs Speed) Strategy

**Status**: Draft — pending user review (rev 3: Gemma 4 vision API 実測反映)
**Date**: 2026-04-22 (rev 1: initial → rev 2: dual-track → rev 3: Gemma 4 vision findings)
**Target**: Hackathon 2026-05-18 (4 weeks)
**Checkpoint disposition**: 60k step Stage 3c-0 checkpoint will be discarded, fresh pretrain from step 0

**Rev 3 変更点 (2026-04-22)**:
- `multi_modal_projector` → `embed_vision` (`Gemma4MultimodalEmbedder` = RMSNorm + Linear) に訂正
- Vision 入力は raw 画像ではなく **patchified `(B, max_patches, 768)` + `pixel_position_ids (B, max_patches, 2)`**
- Vision 出力は **padding-stripped flat `(N_valid_total, 1536)`**、バッチ reshape 要
- `max_soft_tokens` supported 値: `{70, 140, 280, 560, 1120}` (`_SUPPORTED_SOFT_TOKENS` 実測)、default 280
- Preprocess は `Gemma4ImageProcessor.preprocess()` を直接使用、`do_normalize=False`
- `attn_implementation = "sdpa"` (torch 2.11 用 flash-attn 2.x wheel 存在せず、torch 2.11 native SDPA に Flash backend 組込あり)
- venv は `.venv-gemma4` (torch 2.11.0, transformers 5.5.4, torchvision 0.26)

---

## 1. Motivation

現 VLA-Gemma4 の学習は GPU util 90%+ の compute bound 状態で memory bound、`batch_size=8` から拡大不可。原因分析により 3 つの構造的損失が判明:

1. **手元視点 (wrist camera) を frozen LLM に通している** — Gemma 4 は自然画像で pretrain、wrist POV は OOD。LLM は完全 frozen で OOD 入力に適応する術がない。
2. **DinoSigLIP + `vision_projector` が冗長** — Gemma 4 E2B は native multimodal で自前の vision encoder (`Gemma4VisionModel`, 151M) を持つ。外付け DinoSigLIP (400M) は後付けで feature alignment を学習で獲得せねばならず loss。
3. **SoftPromptLibrary 配置が X-VLA 論文と乖離** — 案 B (LLM input 前段 concat) は X-VLA 原実装の action generation transformer 入力 concat と deviation。

これら 3 つは全 track 共通で解消。**frozen LLM + Bridge Attention (25 層 hidden 抽出) という VLA-Adapter 核心思想は死守**。

### 戦略的二項対立: Quality vs Speed

`action_queries` (VLAAdapterGemma4 保持の trainable `nn.Embedding(64, 1536)`) が LLM input に書き込まれる以上、通常 autograd は LLM backward を要求する。これを `torch.no_grad()` で skip するには action_queries を frozen 化する必要があり、**quality** と **speed** の間に明確な trade-off が生じる:

| 観点 | Quality 寄り (action_queries trainable + LoRA) | Speed 寄り (action_queries frozen + LLM no_grad) |
|---|---|---|
| LLM per-step | forward + backward = 2F | **forward only = F** (50% 減) |
| activation memory | ~10-15 GB (GC で圧縮可) | **~0 GB** (no_grad) |
| batch_size 上限 | 16 (GC あり) | **24-32+** |
| LLM 適応性 | LoRA で薄く適応 | 完全 frozen |
| h_a (Bridge action hidden) 品質 | trainable action_queries で learning | frozen、"null query" から LLM 生成 |
| 4 週間で回せる full run | 1-2 回 (一発勝負) | **3-4 回** (iteration 可能) |

**実測 data**: 110k step ckpt の action_queries.weight は init=zero → std=0.025 の **modest** な learning (§A1 参照)。"そこそこしか成長していない" = 捨てても fatal でない可能性。Bitter Lesson 的観点では **scale / iteration で勝負する価値あり**。

### 結論: Dual-Track で data-driven に決着

事前議論で優劣判定できないため、**Mode A (Quality) と Mode B (Speed) を並行実装・並行学習させ、LIBERO eval で勝った方を deliverable とする** 戦略を採用。flag で切替可能な単一コードベースで両 mode をサポート。

## 2. Goals

### 全 track 共通
- **G1**: Wrist 視点を LLM 経路から外し、action head の入力 token として x に concat (X-VLA auxiliary view 設計準拠)
- **G2**: Scene 視点を Gemma 4 native vision に切り替え、feature alignment を pretrained 状態で獲得
- **G3**: SoftPromptLibrary を action head の入力 token として x に concat (X-VLA 原実装準拠)、LLM input から除去
- **G4**: Bridge Attention (25 層 hidden 抽出) を完全温存
- **G5**: Mode A/B の切替を YAML flag 1 個で完全に制御可能な実装
- **G6**: Hackathon 期限 (2026-05-18) までに LIBERO eval を Mode A と Mode B の両方で完了、勝者を deliverable

### Mode A (Quality-first)
- **GA1**: `action_queries.requires_grad=True` で trainable 維持
- **GA2**: LLM backward を通常通り実行、**GC 必須**
- **GA3**: Gemma 4 LM の attention q/k/v/o_proj に **LoRA r=16** 挿入
- **GA4**: batch_size 16 目標 (GC で memory 圧縮)

### Mode B (Speed-first)
- **GB1**: `action_queries.requires_grad=False` で frozen (§5.10.1 で init 戦略)
- **GB2**: `with torch.no_grad():` で LLM forward wrap、**backward 丸ごと skip**
- **GB3**: LoRA 不採用 (LLM backward なしで LoRA 学習不可)
- **GB4**: batch_size 24+ 目標 (activation memory 解放)
- **GB5**: GC 不要 (LLM backward なし)

## 3. Non-Goals (今回スコープ外)

- **Flow Matching / DiT action head 化** — Hackathon 期間に収束検証不可。現行 `MLPResNetBlock_Pro` + L1 regression を継続。
- **AdaLN-Zero / timestep conditioning** — FM を入れないので不要。
- **Cross-embodiment multi-dataset pretrain** — Fractal 1cam 問題 + `multi_dataset_loader` 看板倒れ。Taco Play 単独 pretrain に縮退。
- **Gemma 4 E2B 以外の model variant 探索** — E2B で継続。
- **proprio zero input (X-VLA cross-embodiment pattern)** — 単アーム単 embodiment でメリットゼロ、proprio 有効化。
- **Fractal 等 1cam dataset の活用** — 保留、Hackathon 後。
- **action head の self-attn transformer 全面置換** — Bridge Attention 構造と不整合、MLPResNetBlock_Pro を残しつつ concat-to-x で self-attn pool 参加に近づける hybrid。

## 4. Architecture Overview

### 4.1 Module inventory

全 track 共通モジュール:

| Module | 役割 | State | Params |
|---|---|---|---|
| `Gemma4VisionModel` | Scene 画像 → N soft tokens (768-dim)、N は `max_soft_tokens` 設定で決定 | **frozen** | ~151M |
| `embed_vision` (`Gemma4MultimodalEmbedder`, RMSNorm + Linear) | vision hidden 768 → llm 1536 | **frozen** | 小 |
| `Gemma 4 LM` | multi-layer hidden 抽出 (Bridge 供給源) | **frozen** (params update なし) | ~2.0B |
| `ResNet18` (new) | Wrist 画像 → 7×7×512 feature map | **trainable** (ImageNet init) | 11.7M |
| `WristProjector` (new) | 512 → 1536 | **trainable** | 1.2M |
| `SoftPromptLibrary` (relocated) | Dataset 条件付け 32 tokens × 1536 | **trainable** | ~0.05M (num_datasets=1) |
| `ProprioProjector` (existing) | proprio 8 → 1536 | **trainable** | ~2.4M |
| `ActionHead` (`MLPResNetBlock_Pro` × 24) | x = [action_latent, wrist, soft_prompt] concat で self-attn pool 拡張 + Bridge cross-attn | **trainable** | 562M (film_gen 削除後) |

Mode 固有:

| Module | Mode A (Quality) | Mode B (Speed) |
|---|---|---|
| `action_queries` (64×1536) | **trainable** (既存通り) | **frozen** (§5.10.1) |
| LLM forward context | 通常 autograd 有効 | **`torch.no_grad()` wrap** |
| LLM GC | **有効化必須** | 不要 (backward なし) |
| LoRA (q/k/v/o_proj r=16) | **挿入、trainable** | 不採用 |

合計 trainable:
- **Mode A**: 582M (共通) + 0.1M (action_queries) + 4M (LoRA) = **~586M**
- **Mode B**: 582M (共通) = **~582M** (action_queries frozen のため不算入)

### 4.2 Data flow (両 Mode 共通)

```
  scene_img (raw, any HxW)
        ▼
  [Gemma4ImageProcessor.preprocess]
    - aspect-ratio-preserving resize (target: max_soft_tokens 由来)
    - rescale ÷255 (do_normalize=False、ImageNet mean/std は適用しない)
    - patchify: 16×16 タイルで (B, 3, H, W) → (B, max_patches, 768)
    - position_ids 生成 (B, max_patches, 2)、padding 箇所は (-1, -1)
        ▼
  pixel_values (B, max_patches, 768)  + pixel_position_ids (B, max_patches, 2)
        ▼
  [Gemma4VisionModel (FROZEN)]      ──▶ vision hidden (padding-stripped flat)
        ▼
  [embed_vision (FROZEN)]           ──▶ pooler_output (N_valid_total, 1536)
        ▼
  reshape (B, N_valid_per_image, 1536)  # fixed input size なら batch 内で均一
        ▼
        │
        │  N_valid_per_image は max_soft_tokens 設定次第:
        │    max_soft_tokens=70:  N=64 tokens/image
        │    max_soft_tokens=140: N=121 tokens/image
        │    max_soft_tokens=280: N=256 tokens/image (default)
        │                                              │
  text_tokens (20) ───────────────────────────────────┤
  [Gemma 4 embed (FROZEN)]                            ├── concat ──▶ embeddings
                                                      │
  action_placeholders (64 positions) ─────────────────┤
  [action_queries (Mode A: trainable,                 │
                   Mode B: frozen)]                   │
                                              ▼
                        ┌──────────────────────────────────────┐
                        │   Gemma 4 LM                         │
                        │                                      │
                        │   Mode A: with autograd enabled      │
                        │           + GC + LoRA on attn proj   │
                        │   Mode B: with torch.no_grad():      │
                        │           (no LoRA)                  │
                        │                                      │
                        │   → out.hidden_states[0:25] (25 層)  │
                        └──────────────────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
   h_t @ scene_vision positions               h_a @ action_placeholder positions
     (B, 25, N_valid_per_image, 1536)           (B, 25, 64, 1536)
     (N = 64 / 121 / 256, max_soft_tokens 次第)


  wrist_img (224×224×3) ──▶ [ResNet18 (TRAINABLE)] ──▶ (B, 512, 7, 7)
        ▼ flatten + project
  h_w (B, 49, 1536)

  proprio (8 dim) ──▶ [ProprioProjector] ──▶ p (B, 1, 1536)

  dataset_id ──▶ [SoftPromptLibrary] ──▶ h_sp (B, 32, 1536)


  ┌──────── Action Head (MLPResNet + MLPResNetBlock_Pro × 24) ────────┐
  │  x1 = fc1(layer_norm1(x0))                            (B, 8, 1536) │
  │  x_ext = torch.cat([x1, h_w, h_sp], dim=1)            (B, 89, 1536) │
  │                                                                    │
  │  for i in range(24):                                               │
  │    block_i:                                                        │
  │      Q = q_proj(x_ext)                                             │
  │      Stream 1 (self): k_self(x_ext), v_self(x_ext)                 │
  │      Stream 2 (adapter): k_adapter([h_a[:,i+1], p])                │
  │      Stream 3 (task): k_task(h_t[:,i+1]) × ratio_g                 │
  │      concat softmax → output + FFN + residual                      │
  │                                                                    │
  │  x_final = x_ext[:, :8, :]     (action 位置のみ trim)              │
  │  action = fc2(layer_norm2(x_final))                    (B, 8, 7)   │
  └────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            L1 loss vs action target
```

### 4.3 Dual-Track Mode Configuration

単一 YAML config flag `training_mode ∈ {"quality", "speed"}` で切替:

```yaml
# config/pretrain_taco_quality.yaml
training_mode: "quality"        # Mode A
# automatically sets:
#   action_queries_trainable: True
#   llm_no_grad: False
#   gradient_checkpointing: True
#   lora:
#     enabled: True
#     r: 16
#     alpha: 32
#     target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

# config/pretrain_taco_speed.yaml
training_mode: "speed"          # Mode B
# automatically sets:
#   action_queries_trainable: False
#   action_queries_init: "zero"     # §5.10.1、or "reserved_token"
#   llm_no_grad: True
#   gradient_checkpointing: False
#   lora:
#     enabled: False
```

実装側は `finetune_gemma4.py` で training_mode を読んで分岐:

```python
if cfg.training_mode == "quality":
    # Mode A setup
    model.action_queries.weight.requires_grad = True
    model.llm.gradient_checkpointing_enable()
    apply_lora(model.llm, r=16, alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    # forward: 通常
elif cfg.training_mode == "speed":
    # Mode B setup
    model.action_queries.weight.requires_grad = False  # frozen
    # forward: LLM 部分を torch.no_grad() で wrap (model.forward 内で分岐)
```

`VLAAdapterGemma4.forward` 内で training_mode を参照し LLM call を切替:

```python
if self.training_mode == "speed":
    with torch.no_grad():
        out = llm(inputs_embeds=embeddings, output_hidden_states=True, ...)
else:  # quality
    out = llm(inputs_embeds=embeddings, output_hidden_states=True, ...)
```

## 5. Detailed Design

### 5.1 Scene Path (Gemma 4 Native Vision)

**Model load:**

```python
from transformers import Gemma4ForConditionalGeneration
llm = Gemma4ForConditionalGeneration.from_pretrained(
    "google/gemma-4-E2B",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",   # torch 2.11 native SDPA + Flash backend
).eval()
# llm.model.vision_tower, llm.model.embed_vision, llm.model.language_model が使える
# (multi_modal_projector は存在しない、Gemma 4 では embed_vision が projector 役)
```

**Image preprocessing (`Gemma4ImageProcessor` を直接使用、自作 patchify 禁止):**

```python
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
image_proc = Gemma4ImageProcessor(max_soft_tokens=cfg.max_soft_tokens)  # default 280
out = image_proc.preprocess(scene_imgs, return_tensors="pt")
pixel_values = out["pixel_values"]             # (B, max_patches, 768)
pixel_position_ids = out["image_position_ids"] # (B, max_patches, 2)、padding は (-1, -1)
# do_rescale=True (÷255)、do_normalize=False (mean=0, std=1)
#   → VLA 側 ImageNet 正規化を外す必要あり
```

**Vision forward (padding-stripped output → reshape):**

```python
feats = llm.model.get_image_features(pixel_values, pixel_position_ids)
# feats.last_hidden_state: (N_valid_total, 768) padding-stripped flat
# feats.pooler_output:     (N_valid_total, 1536) embed_vision 適用後

# fixed input size なら batch 内 N_valid_per_image は均一、reshape 可
n_valid_per_image = out["num_soft_tokens_per_image"][0].item()
h_v = feats.pooler_output.view(B, n_valid_per_image, 1536)
```

**`max_soft_tokens` 設定 (YAML config で切替可、supported 値のみ):**
- `_SUPPORTED_SOFT_TOKENS = {70, 140, 280, 560, 1120}` (transformers 5.5.4 実測)
- `max_soft_tokens=70` → 64 tokens/image、max_patches=630 (padding 8.6%)
- `max_soft_tokens=140` → 121 tokens/image、max_patches=1260 (padding 13.6%)
- `max_soft_tokens=280` (default) → 256 tokens/image、max_patches=2520 (padding 8.6%)
- compute 見積り比較: §9.1

**全 submodule frozen:**

```python
for p in llm.parameters():
    p.requires_grad = False  # vision_tower + embed_vision + language_model 全部
# LoRA 使う場合 (Mode A only) は LM の target_modules のみ trainable 化
```

### 5.2 Wrist Path (ResNet18)

- `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`
- 最終 FC を捨て、conv 出力 `(B, 512, 7, 7)` を採取、GAP なし
- `rearrange("b c h w -> b (h w) c")` → `(B, 49, 512)`
- `nn.Linear(512, 1536)` → `h_w (B, 49, 1536)`
- `requires_grad=True`、bf16、ImageNet init

### 5.3 Action Head: X-VLA Style Concat-to-x

**MLPResNetBlock_Pro 内部構造は不変**、`MLPResNet.forward` の token 流れのみ変更:

```python
def forward(self, x, h_a, h_t, p, h_w=None, h_sp=None):
    x = self.layer_norm1(x)
    x = self.fc1(x)                          # (B, 8, 1536)
    x = self.relu(x)

    if h_w is not None:
        x = torch.cat([x, h_w], dim=1)       # (B, 57, 1536)
    if h_sp is not None:
        x = torch.cat([x, h_sp], dim=1)      # (B, 89, 1536)

    for i, block in enumerate(self.mlp_resnet_blocks):
        x = block(x, h_t=h_t[:, i+1], h_a=h_a[:, i+1], p=p)

    x = x[:, :NUM_ACTIONS_CHUNK, :]          # trim (B, 8, 1536)
    x = self.layer_norm2(x)
    x = self.fc2(x)                          # (B, 8, 7)
    return x
```

### 5.4 Dead Code Removal

`film_gen` + `apply_film` 削除 (`action_heads.py:320-334, 403-406`):
- forward で全コメントアウト済 dead code
- 113M 死重除去
- 60k ckpt 破棄で chkpt 互換縛りなし

### 5.5 SoftPromptLibrary Relocation

- クラス本体は不変 (`modeling_prismatic_gemma4.py:41-63`)
- 挿入位置: LLM input → action head の `h_sp` 引数
- LLM input 側の soft_prompt concat / offset 計算を全削除

### 5.6 Proprio (Enabled, 両 Mode 共通)

- Taco pretrain と LIBERO fine-tune で proprio 使用
- `ProprioProjector` 現状維持
- format alignment は §9.3 で確認

### 5.7 Bridge Attention (Unchanged, 両 Mode 共通)

- `modeling_prismatic_gemma4.py:246-256` の処理維持、25 層 hidden 抽出
- h_t: (B, 25, N_valid_per_image, 1536) (N=64/121/256, max_soft_tokens 次第)、h_a: (B, 25, 64, 1536)
- action head 側 `predict_action` の h_t / h_a 分離ロジック維持

### 5.8 LoRA on Gemma 4 LM (Mode A only)

Mode A で LLM backward を払う以上、LoRA の追加コストはほぼゼロ:
- 対象 module: Gemma 4 LM の `self_attn.q_proj / k_proj / v_proj / o_proj`
- rank: r=16
- scaling α: 32
- 実装: `peft` library の `LoraConfig` + `get_peft_model`、or 手動 `LoRALinear` 実装
- Phase 1 で Mode A 内の LoRA あり/なしの ablation を小さく実施

### 5.9 RoPE Positions (両 Mode 共通)

concat 後 seq=89 に一様 RoPE 適用:
- position 0-7: action latent
- position 8-56: wrist
- position 57-88: soft_prompt

X-VLA paper は PE 具体を指定しておらず、一様適用は paper spec 違反ではない (§9.6 open question)。

### 5.10 Mode B: action_queries Frozen Strategy

#### 5.10.1 Init strategy

以下 2 option のうち Phase 0 smoke で決定:

- **FC (zero init + frozen)**: 最小変更、`self.action_queries.weight.data.zero_(); self.action_queries.weight.requires_grad = False`
  - 利点: 実装 1 行
  - 懸念: LLM は "null query" に対する meaningful response を pretrain で学んでいない、hidden state 品質不明
- **FA (reserved token embed init + frozen)**: Gemma vocab の reserved/unused token の embedding を採用
  - 利点: LLM は既存 token に対して学習済の応答パターンを持つ、h_a 品質向上期待
  - 実装: `with torch.no_grad(): action_queries.weight.copy_(llm.embed_tokens(ACTION_QUERY_TOKEN_IDS))`
  - 懸念: 適切な token ID 選定要 (reserved range or semantically related 単語 tokens)

**デフォルト**: FC (zero init)、smoke の loss curve 品質次第で FA に切替。

#### 5.10.2 torch.no_grad() wrap

LLM forward 完全に autograd graph から切り離す:

```python
# VLAAdapterGemma4.forward 内
if self.training_mode == "speed":
    with torch.no_grad():
        out = llm(
            inputs_embeds=embeddings,
            output_hidden_states=True,
            ...
        )
    # out.hidden_states は自動的に detached
else:  # quality
    out = llm(...)  # 通常 autograd
```

注意: Mode B でも action_head 側の backward は通常通り走る (action_head の params は trainable)。out.hidden_states は no_grad 下で生成された "値" として action head に流れ込み、そこから backward は action_head / wrist / soft_prompt / proprio 側のみ伝播。

#### 5.10.3 Mode B での LLM activation memory

`torch.no_grad()` は autograd graph を作らないため、forward 時に中間 activation を保存しない (通常 forward は backward 用に保存する)。これで **LLM 側 activation memory がほぼゼロに** なり、batch_size 大幅拡大の原資となる。

## 6. Training Pipeline (Dual-Track)

### 6.1 Pretrain — Mode A (Quality track)

- **Config**: `config/pretrain_taco_quality.yaml`
- **Dataset**: Taco Play 単独 (2cam: rgb_static + rgb_gripper)
- **Data loader**: `taco_solo_loader.py` (新規、Fractal 削除、proprio 有効化)
- **Batch size**: 16 目標 (GC 有効で memory 圧縮)
- **Steps**: 60k-100k (Taco 収束点を smoke で確認)
- **Optimizer**: AdamW、X-VLA 準拠 LR schedule (freeze + warmup)
- **LoRA**: r=16 on q/k/v/o_proj
- **LLM GC**: `gradient_checkpointing_enable()` 呼び出し
- **action_queries**: trainable

### 6.2 Pretrain — Mode B (Speed track)

- **Config**: `config/pretrain_taco_speed.yaml`
- **Dataset**: Mode A と同じ Taco Play
- **Data loader**: 同じ `taco_solo_loader.py`
- **Batch size**: 24-32 目標 (LLM activation memory 解放分で拡大)
- **Steps**: 60k-100k (wall-time ベースで Mode A と同時刻終了目標、3-4x throughput なので step 数は大幅増)
- **Optimizer**: AdamW、同じ LR schedule
- **LoRA**: 不採用
- **LLM GC**: 不要
- **action_queries**: zero init + frozen (Phase 0 で品質確認、必要なら FA init に切替)

### 6.3 Parallel execution (hardware 要件)

- 2 GPU 以上あれば Mode A と Mode B を **並行 pretrain**
- 1 GPU しかない場合は Mode B を先行 (3-4x 速く完走) → Mode A を続けて回す sequential fallback
- wandb で両 run を同じ project に logging、sweep で比較可能に

### 6.4 Fine-tune — Mode A / Mode B 両方

- **Dataset**: LIBERO-90
- **Data loader**: 既存 `Gemma4RLDSDataset` を新 signature に合わせ改修
- **SoftPromptLibrary**: §9.2 (disable or Taco 重み初期値で継続)
- **Proprio**: 有効化
- **LoRA (Mode A のみ)**: pretrain の LoRA 重み初期値
- **ResNet18**: pretrain 重み初期値 (両 Mode)
- **Steps**: LIBERO 標準プロトコル (2k-10k step)
- **action_queries (Mode A)**: pretrain 重み初期値、fine-tune で継続学習
- **action_queries (Mode B)**: pretrain と同じ frozen init のまま fine-tune でも frozen

### 6.5 Deliverable Selection Criteria

LIBERO eval 完了後、以下の順で deliverable 決定:

1. **LIBERO success rate が高い方** を deliverable (最重要)
2. 差が 3% 以内 → **Mode B 採用** (シンプル、iteration 容易、将来の scale 勝負に有利)
3. 両方失敗 (baseline 以下) → **Mode A 採用** (quality 優先で rescue run)、Hackathon 期限ギリギリまで run 延長

## 7. Optimization Integration

### 7.1 Phase 0: Architecture Implementation (両 Mode 対応)

Week 1 前半:

1. 新アーキ実装 (両 Mode の共通部分):
   - scene = Gemma4 native vision、`Gemma4ForConditionalGeneration` ロード
   - wrist = ResNet18 + projector、action head concat
   - soft_prompt = action head concat、LLM input から削除
   - action head: film_gen 削除 + concat-to-x
   - dual-track config flag + forward 分岐
2. `attn_implementation="sdpa"` (torch 2.11 native SDPA + Flash backend、flash-attn 2.x は torch 2.11 用 wheel なし)
3. Mode A 固有: GC 有効化、LoRA 挿入
4. Mode B 固有: action_queries frozen、torch.no_grad wrap

### 7.2 Phase 0.5: Dual-Track Smoke

Week 1 後半:

1. Mode A smoke (1k step、batch 16): loss 減少 / NaN なし / gradient flow 確認
2. Mode B smoke (1k step、batch 24): loss 減少 / NaN なし / action_queries frozen 確認
3. Mode A/B の samples/sec, memory peak, loss @ 1k を wandb で比較
4. **Mode B の loss が Mode A の 20% 劣化以上**なら action_queries init を FC → FA に切替試行
5. 両 mode が smoke 通過 = Phase 1 進行可

### 7.3 Phase 1: Batch Size Expansion (両 Mode)

Week 1 末 - Week 2 初:

- Mode A: batch 16 → 20 → 24 (GC で memory 余裕次第)
- Mode B: batch 24 → 32 → 48 (LLM activation memory 解放分)
- 各 batch size で 100 step throughput 測定、OOM 手前で決定
- `grad_accumulation_steps` で effective batch 調整

### 7.4 Phase 2-3: Parallel Pretrain (Taco)

Week 2-3:

- Mode A と Mode B を並行 pretrain
- 3-4 日ごとに中間 ckpt save
- wandb で両 run を dashboard 比較

### 7.5 Phase 4: LIBERO Fine-tune + Eval

Week 3 末 - Week 4:

- Mode A / Mode B の pretrain ckpt を LIBERO fine-tune
- LIBERO eval 実行
- §6.5 の基準で deliverable 決定

## 8. Parameters Summary

### 8.1 Trainable

| Group | Mode A | Mode B | 備考 |
|---|---|---|---|
| `action_head` base (3-stream, MLPResNetBlock_Pro × 24) | 675M | 675M | assertion 値 675.138M |
| `action_head` film_gen 削除 | −113M | −113M | dead code 除去 |
| `action_head` concat 改修 | +0M | +0M | K/V projection 追加なし |
| `action_queries` | +0.1M (trainable) | — (frozen、非 trainable) | Mode B では 学習対象外 |
| `soft_prompt_library` | 0.05M | 0.05M | num_datasets=1 |
| `wrist_resnet18` (new) | 11.7M | 11.7M | ImageNet init |
| `wrist_projector` (new) | 1.2M | 1.2M | |
| `proprio_projector` | 2.4M | 2.4M | |
| LoRA (r=16 on q/k/v/o_proj) | +4M | — | Mode A only |
| **Net action head + extras** | **~586M** | **~582M** | |

### 8.2 Frozen

| Group | Params |
|---|---|
| `Gemma 4 LM (text_model)` | ~2.0B |
| `Gemma 4 VisionModel` | ~151M |
| `Gemma 4 embed_vision` (Gemma4MultimodalEmbedder) | 小 |
| `action_queries` (Mode B のみ) | 0.1M |

## 9. Open Questions

### 9.1 Soft tokens: 70 / 140 / 280 比較

Supported 値 (実測): `{70, 140, 280, 560, 1120}`。各 max_soft_tokens の compute 見積り (1 sample):

| max_soft_tokens | LLM 入力 tokens | Vision attn (patches²) | LLM forward (seq~N+α) | Mode A 合計 (LLM backward 込み) | Mode B 合計 (no_grad LLM) |
|---|---|---|---|---|---|
| 70 | 64 | 630²/層 × 16 層 = ~24G | ~40G (seq ~170) | ~185G (LLM 支配) | ~64G (Vision/LLM 同等) |
| 140 | 121 | 1260²/層 × 16 層 = ~38G | ~70G (seq ~230) | ~320G | ~110G |
| 280 (default) | 256 | 2520²/層 × 16 層 = ~100G | ~125G (seq ~360) | ~600G | ~225G |

**選定方針**:
- **default 280**: Gemma 4 pretrain natural default、alignment 最大
- **YAML config `max_soft_tokens` で切替可**、ablation 用
- Phase 0 smoke で 70 / 140 / 280 loss curve 比較、実測で決定
- Mode B は vision compute も効くので 70 / 140 が効果大、Mode A は LLM 支配なので 280 でも差小

### 9.2 LIBERO fine-tune での soft_prompt_library
選択肢 A (disable) / B (Taco init で継続学習)。Phase 0 smoke 後決定。

### 9.3 Proprio format alignment (Taco ↔ LIBERO)
実装時に tfds episode 1 個 dump で確認、不整合なら canonical 8 dim 正規化。

### 9.4 LIBERO fine-tune の ResNet18 初期化
pretrain 重み採用推奨。

### 9.5 LoRA (Mode A 内) 効果の ablation
Phase 0.5 smoke で LoRA あり/なし比較、効果薄ければ Mode A からも削除検討。

### 9.6 RoPE positional encoding
concat 後 seq に一様適用が spec。気になれば soft_prompt/wrist 除外条件分岐を ablation。

### 9.7 Mode B: action_queries init strategy (FC vs FA)
- FC (zero init): Phase 0 default
- FA (reserved token init): FC が Mode A から >20% loss 劣化時に切替
- 両方 smoke で先行評価も可

### 9.8 Hardware allocation (確認済み: 8 GPUs)

**利用可能 GPU** (`nvidia-smi` 実測):
- GPU 0, 1: A100 **80GB** PCIe × 2
- GPU 2-7: A100 **40GB** PCIe × 6

**推奨配分** (Mode A は memory tight なので 80GB に優先的に):
- **Mode A (Quality pretrain)**: GPU 0, 1, 2, 3 (= 80GB × 2 + 40GB × 2)、**DDP 4-way**
- **Mode B (Speed pretrain)**: GPU 4, 5, 6, 7 (= 40GB × 4)、**DDP 4-way**
- 両 Mode を **同時並行で DDP pretrain**、effective batch = per_gpu × 4 で各 mode 独立 run

**Effective batch 見積り**:
- Mode A: per_gpu=16 × 4 gpus = 64 (grad_accumulation 不要で足りる)
- Mode B: per_gpu=24-32 × 4 gpus = 96-128

Phase 0 smoke (Week 1 後半) は single GPU で両 mode 切替し基本動作確認、DDP は Phase 2 (Week 2) から。

## 10. Implementation Phasing (Dual-Track Timeline)

| Week | Phase | 成果物 |
|---|---|---|
| Week 1 前半 (-04-25) | **Phase 0**: 新アーキ実装 (両 Mode 共通 + dual-track flag) | 新 model class、data loader、config 2 本、switch logic |
| Week 1 後半 (-04-29) | **Phase 0.5**: Mode A/B 両 smoke + batch size 探索 | smoke pass、baseline throughput / loss @ 1k 記録 |
| Week 2 (-05-06) | **Phase 2**: Mode A (GPU 0-3 DDP) / Mode B (GPU 4-7 DDP) **並行 pretrain 開始** (Taco) | 両 mode 20k+ step、3 日後に中間 diff 確認 |
| Week 3 (-05-13) | **Phase 3**: Pretrain 継続 + LIBERO fine-tune 開始 | pretrain 80k+、LIBERO fine-tune 開始 |
| Week 4 (-05-18) | **Phase 4**: LIBERO eval + deliverable selection + 提出 | §6.5 基準で勝者決定、Hackathon 提出 |

## 11. Risk Register

| Risk | Probability | Mitigation |
|---|---|---|
| Gemma 4 E2B の native vision が weights 未配布 | Low | config load 済、実装時 full load で verify |
| torch 2.11 native SDPA が Flash backend 起動しない | Low | `torch.backends.cuda.sdp_kernel` で明示、smoke で確認 |
| Gemma 4 image preprocess の upsample で semantic 破綻 | Med | 処理前後 PNG round-trip 確認、feature cosine sim > 0.99 smoke |
| Mode A: GC が transformers 5.5.4 で動かない | Low-Med | PR #45312 含 version 想定、Phase 0 で確認 |
| Mode B: action_queries frozen で h_a が無意味化、loss が大幅劣化 | Med | FC init で smoke、劣化 >20% なら FA init に切替 (§5.10.1) |
| Mode B: LLM no_grad で Bridge の `output_hidden_states` が正しく動かない | Low | torch.no_grad は forward 値には影響しない、標準動作 |
| concat 後 seq=89 の self-attn が compute bottleneck | Med | T² O(89²)=7921、block compute ~10x 増、LLM forward より軽く許容範囲 |
| 並行 pretrain に必要な GPU が足りない | Med | §9.8 sequential fallback に切替、総 wall time 1.5-2x 増 |
| Taco Play proprio format 想定違い | Med | §9.3 で canonical 正規化 |
| Mode A / Mode B 両方が baseline 以下 | Low-Med | §6.5 の rescue run 運用、Hackathon 期限ギリギリまで run 延長 |
| 60k ckpt 破棄で学習やり直しが間に合わない | Med | Week 1 Phase 0 で問題発生あれば P3 (pretrain skip, LIBERO 直行) にフォールバック検討 |

## 11.5 Environment (Rev 3 addendum)

- **venv**: `.venv-gemma4` (not `.venv`、両方存在するが前者が project env)
  - torch **2.11.0** + cu128
  - transformers **5.5.4**
  - torchvision 0.26.0
  - Python 3.11.15
  - uv 管理 (pip 直接利用不可、`uv pip install ...` 必要)
- **Flash Attention**:
  - `flash_attn` package 未インストール、**インストール不要**
  - 理由: torch 2.11 用 flash-attn 2.x wheel は HF release に存在せず (最新 v2.8.3 も torch2.10 まで)、ソースビルドは依存解決が不安定
  - torch 2.11 の native SDPA に Flash Attention backend が組み込み済、Gemma 4 での実効性能は flash-attn 2.x と同等レンジ
- **smoke test 実行**: 必ず `.venv-gemma4/bin/python` を明示呼び出し (PATH の system python は 2.7)
- **Gemma 4 AutoProcessor 不可**: `Gemma4VideoProcessor` が torchvision extras 不足で import 失敗する。**Image だけ使うなら `Gemma4ImageProcessor` を直接 import** して回避

## 12. Files to Modify

- `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py`: model wrapper 全面改修 (scene native vision、`Gemma4ImageProcessor` 統合、`embed_vision` 呼び出し、reshape back、soft_prompt relocation、wrist 注入、**training_mode flag に応じた forward 分岐**、LoRA wrap (Mode A)、action_queries requires_grad 制御)
- `VLA-Adapter/prismatic/models/action_heads.py`:
  - `MLPResNet.forward` に `h_w`, `h_sp` 引数追加、concat + trim
  - `L1RegressionActionHead.predict_action` signature 更新
  - `film_gen` / `apply_film` 削除
- `VLA-Adapter/vla-scripts/finetune_gemma4.py`: model build / dataloader / forward / trainable param 列挙 を新構造 + dual-track 対応、LoRA setup (Mode A)
- `scripts/stage3/multi_dataset_loader.py` → `taco_solo_loader.py` 新規 (Fractal 削除、proprio 正規化、2cam、**scene 画像は raw のまま返し、正規化はモデル側で `Gemma4ImageProcessor` 経由で実施** = ImageNet 正規化は削除)
- `scripts/gemma4/test_08_data_pipeline.py`: LIBERO loader 更新
- `scripts/gemma4/test_06_full_forward.py` 等 smoke test: dual-track 両方の smoke を追加
- 新規: `VLA-Adapter/prismatic/models/backbones/vision/wrist_resnet18.py`
- 新規: `config/pretrain_taco_quality.yaml`, `config/pretrain_taco_speed.yaml` (両方に `max_soft_tokens` 項目、default 280)
- (optional) LoRA: `peft` 経由 or 手動 `LoRALinear`

## 13. Summary of Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Scene encoder: Gemma 4 native vision (frozen) | ✅ |
| 2 | Wrist encoder: ResNet18 (ImageNet init, trainable), 7×7×512 = 49 tokens | ✅ |
| 3 | Wrist injection: action head の concat-to-x (self-attn pool 参加) | ✅ |
| 4 | SoftPromptLibrary: action head の concat-to-x (LLM input から削除) | ✅ |
| 5 | action head base: `MLPResNetBlock_Pro` × 24 構造不変 (concat で対応) | ✅ |
| 6 | `film_gen` / `apply_film` 削除 (113M dead code) | ✅ |
| 7 | Bridge Attention (25 層): 維持 | ✅ |
| 8 | Pretrain strategy: Taco Play 単独 (P1) | ✅ |
| 9 | Fine-tune: LIBERO (Stage 2 protocol) | ✅ |
| 10 | Proprio: 有効化 | ✅ |
| 11 | 60k checkpoint: 破棄 | ✅ |
| 12 | `attn_implementation = "sdpa"` (torch 2.11 native SDPA + Flash backend、flash-attn 2.x は torch 2.11 用 wheel なし) | ✅ (rev 3) |
| 13 | RoPE: concat 後 seq に一様適用 | ✅ |
| 14 | **Dual-Track 戦略: Mode A (Quality) / Mode B (Speed) を並行実装・並行学習** | ✅ |
| 15 | **Mode A: action_queries trainable + GC + LoRA r=16** | ✅ |
| 16 | **Mode B: action_queries frozen + LLM no_grad + no LoRA** | ✅ |
| 17 | **Mode B action_queries init: デフォルト FC (zero)、劣化時 FA (reserved token)** | ✅ |
| 18 | **Deliverable selection: LIBERO eval 勝者、差 3% 以内なら Mode B 採用** | ✅ |
| 19 | Scene `max_soft_tokens` 70 / 140 / 280 比較 (YAML config 切替、default 280) | Open (§9.1) |
| 20 | LIBERO での soft_prompt_library 扱い | Open (§9.2) |
| 21 | Hardware: 8 GPU (A100 80GB×2 + 40GB×6)、Mode A に 80GB 優先割当、両 mode DDP 4-way 並行 | ✅ (§9.8) |
| 22 | **Vision preprocess: `Gemma4ImageProcessor` 直接使用、自作 patchify 禁止** | ✅ (rev 3) |
| 23 | **VLA 側 ImageNet 正規化を外す (processor は do_normalize=False)** | ✅ (rev 3) |
| 24 | **embed_vision 出力は padding-stripped flat、batch reshape に `num_soft_tokens_per_image` 使用** | ✅ (rev 3) |
| 25 | **venv: `.venv-gemma4` (torch 2.11.0, transformers 5.5.4, torchvision 0.26)** | ✅ (rev 3) |

---

## Appendix A: Empirical Data Supporting Dual-Track Rationale

### A1. 110k step pretrain ckpt inspection

`runs/gemma4/.../pretrain_baseline-20260420/latest_checkpoint.pt` (step 109999、lr=5e-5):

```
action_queries.weight:
  shape = (64, 1536)
  init: zero
  110k step 後: L2 norm = 7.94, mean abs = 0.022, std = 0.025
  → zero から learning が進んでいるが、std=0.025 は modest (1536 dim 中)

soft_prompt_library.embedding.weight (案 B 配置):
  shape = (2, 49152)
  init: std=0.02 Gaussian
  110k step 後: L2 norm = 6.29, std = 0.020
  → ほぼ init のまま、frozen LLM 経由の gradient signal が弱い裏付け
```

**解釈**: action_queries は機能しているが learning 幅は modest。Mode B で frozen 化しても fatal loss ではない可能性が高い — 実証で確認する価値あり。
