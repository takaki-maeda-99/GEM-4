# VLA-Adapter アーキテクチャ & 仕様レポート

**対象**: `VLA-Adapter/` (commit 時点) を LIBERO 用に再実装する際のリファレンス。
将来、Gemma 4 E2B を backbone に差し替える前提で「何を保存すべき設計要素か」「何を backbone 固有の値として扱うか」を分離して記述。

---

## 1. 全体アーキテクチャ

### 1.1 データフロー

```
Image(s) ──→ Vision Backbone (DINO+SigLIP) ──→ Vision Projector ──┐
                                                                   │
Language (text) ──→ Qwen Tokenizer ──→ LLM Embedding ──────────────┤
                                                                   ├─→ LLM forward ─→ hidden states ─→ Action Head ─→ 7D × 8 chunk
Proprio (8D) ──→ Proprio Projector ────────────────────────────────┤                                        ↑
                                                                   │                                        │
Action Queries (learnable nn.Embedding, 64 tokens) ─────────────────┘                                        │
                                                                                                             │
                                                                                                   L1 Loss ←─┘
```

### 1.2 主要モジュールと責務

| モジュール | ファイル:行 | 役割 |
|---|---|---|
| Vision Backbone | `prismatic/models/backbones/vision/dinosiglip_vit.py` | DINO + SigLIP の patch 特徴を channel 方向で連結 |
| Vision Projector | `prismatic/extern/hf/modeling_prismatic.py:242-273` | Fused vision 特徴 → LLM hidden space |
| LLM (Qwen2.5-0.5B) | `prismatic/models/backbones/llm/qwen25.py` | マルチモーダルトークン処理 |
| Proprio Projector | `prismatic/models/projectors.py:6-24` | proprio (8D) → LLM hidden dim の 2層 MLP |
| **Action Queries** | `prismatic/extern/hf/modeling_prismatic.py:375` | `nn.Embedding(64, llm_dim)` の学習可能クエリ。**zero-init** |
| Action Head | `prismatic/models/action_heads.py` | Cross-attention 付き MLP-ResNet、連続アクション回帰 |

### 1.3 テンソル次元

**backbone 非依存の設計値 (LIBERO)**

| 量 | 値 | 定義場所 |
|---|---|---|
| `NUM_TOKENS` (action query 総数) | **64** | `prismatic/vla/constants.py:15` |
| `NUM_ACTIONS_CHUNK` | 8 | `constants.py:29` |
| `ACTION_DIM` | 7 | `constants.py:30` |
| `PROPRIO_DIM` | 8 | `constants.py:31` |
| Image resolution | 224×224 | |
| Vision patches / image | 256 (=16×16) | ViT patch14 |
| 画像数 (LIBERO 推奨) | 2 (agentview + wrist) | `--num_images_in_input 2` |

> **NUM_TOKENS = 64 の意味**: 現在のアクション 8 + 「next action」予測 56 = 64。`_process_action_masks` で `current_action_mask | next_actions_mask` を合成 (`finetune.py:395-408`)。

**backbone 依存の値 (Qwen2.5-0.5B の場合)**

| 量 | 値 | 備考 |
|---|---|---|
| LLM hidden dim (`llm_dim`) | **896** | `config.text_config.hidden_size`。コード中は `self.llm_dim` で参照される (`modeling_prismatic.py:372`)。Gemma 4 E2B に差し替えたら変わる |
| LLM 最大長 | 2048 | `llm_max_length` (`qwen25.py:46`) |
| 拡張トークン | `<|extra_0|>` 〜 `<|extra_63|>` | 64個追加 (action query 用) |
| `ACTION_TOKEN_BEGIN_IDX` | 151386 | Qwen 固有。backbone を変えたら要変更 (`constants.py:13`) |

---

## 2. VLM Backbone: Prismatic-VLM

### 2.1 Vision (DINO + SigLIP Fusion)

- **DINO**: `vit_large_patch14_reg4_dinov2.lvd142m` (timm)
- **SigLIP**: `vit_so400m_patch14_siglip_224` (timm)
- **Fusion 方法** (`dinosiglip_vit.py:171`): `torch.cat([dino_patches, siglip_patches], dim=2)`
  - 各々の **second-to-last layer** の patch 特徴を抽出
  - 結果: `(B, 256, 1024+1024) = (B, 256, 2048)`
- **pixel 入力形状**:
  - `num_images_in_input=1` → `(B, 6, 224, 224)` (DINO RGB + SigLIP RGB, channel 方向にスタック)
  - `num_images_in_input=2` → `(B, 12, 224, 224)`

### 2.2 Vision Projector

**Fused backbone の projector** (3層 MLP, `modeling_prismatic.py:253-271`):

```
fused_vision (2048) → fc1 (→ 8192) → GELU → fc2 (→ llm_dim) → GELU → fc3 (→ llm_dim)
```

非 fused backbone (2層 MLP) より 1 層深い。

### 2.3 LLM 入力トークン列

```
[BOS] <language_tokens> <vision_patches (2*256=512)> <proprio (1)> <action_queries (64)> [EOS]
```

- proprio は `use_proprio=True` のときだけ 1 トークンとして挿入
- action token 位置は dataset 側で `<|extra_0..63|>` が埋め込まれ、forward 時に `action_queries.weight` で上書き

### 2.4 LoRA (`finetune.py:832-844`)

```python
LoraConfig(
    r = cfg.lora_rank,           # 論文は 64
    lora_alpha = 2 * lora_rank,  # 128
    lora_dropout = cfg.lora_dropout,  # 通常 0.0
    target_modules = "all-linear",
    init_lora_weights = "gaussian",
)
```

- **LoRA 対象**: LLM + projector + action head 内の全 Linear
- **LoRA 対象外 (完全学習)**: `action_queries`, proprio projector, action head の非 Linear 部

---

## 3. Action Head / Policy Network

### 3.1 型: Transformer-free な MLP-ResNet + Cross-Attention

**loss: L1 regression on continuous actions** (`finetune.py:418`)
```python
loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)
```

- **Diffusion / Flow matching は使用していない** (原版も Pro 版も共に回帰)

### 3.2 構成 (`action_heads.py`)

- **入力**: LLM 全層の隠れ状態を action token 位置で抽出
  - `(B, num_layers, 512 + 64, llm_dim)` 形状
  - 512 = task (vision) tokens, 64 = action tokens
- **ブロック**: `MLPResNetBlock` を N 層 (デフォルト 24)
- **出力**: `ACTION_DIM (7)` の連続値

### 3.3 MLPResNetBlock の attention

各ブロックで 3 種類の attention を混合:

```
attn_scores = concat([
    Q @ K_self^T,                   # action tokens 内部
    Q @ K_task^T,                   # vision (task) tokens
    Q @ K_adapter^T * ratio_g,      # proprio/adapter tokens (gated)
]) / sqrt(head_dim)
attn_weights = softmax(attn_scores)
out = attn_weights @ concat([V_self, V_task, V_adapter])
```

- `ratio_g = tanh(g)` で adapter 経路を学習可能にゲート

### 3.4 Parallel Decoding

- **全 8 チャンク同時予測** (non-autoregressive)
- 各 chunk 位置の hidden state から独立にアクションを射出
- 推論時のスループット: H100 で 8 chunks 生成に 0.036s (README 記載, `run_libero_eval.py:334-345` 周辺を計測)

### 3.5 Action 正規化

- **タイプ**: `BOUNDS_Q99` (LIBERO)
  - `action_norm = 2 * (a - q01) / (q99 - q01) - 1` → [-1, 1]
- **stats 保存**: checkpoint 同梱の `dataset_statistics.json`
- **逆正規化**: `modeling_prismatic.py:786-805`

---

## 4. Adapter / Bridge Paradigm (論文の核)

**定義** (`modeling_prismatic.py:375-376`):

```python
self.action_queries = nn.Embedding(NUM_TOKENS, self.llm_dim)
self.action_queries.weight.data.zero_()   # zero 初期化が重要
```

**挙動**:
1. Dataset 側で入力系列に `<|extra_0..63|>` がプレースホルダとして入る
2. Forward 時、action token 位置の埋め込みを `action_queries.weight` で置き換える (`modeling_prismatic.py:620-633`)
3. LLM は vision + language + action_queries を通常の self-attention で処理
4. Action head が action token 位置の hidden state を使って最終的にアクションを出力

**役割**: 言語理解 (LLM) とアクション生成 (action head) を分離しつつ、LLM 内で両者を同時に処理させるための "bridge" 役。

### 4.1 Pro 版との違い (`use_pro_version=True`)

| 項目 | 原版 | Pro 版 |
|---|---|---|
| K, V projection | 共有 (`k_proj`, `v_proj`) | 分離 (`k_self/adapter/task`, `v_self/adapter/task`) |
| 位置エンコード | なし | **RoPE** を Q/K に適用 |
| FiLM modulation | 実装済み (未使用) | 実装済み (未使用, コメントアウト) |
| Policy size | ~1GB | 207MB |
| LIBERO-Spatial | 97.8% | **99.6%** |
| LIBERO-Long | 95.0% | **96.4%** |

論文推奨: Pro 版を使う。

---

## 5. Inputs (詳細)

### 5.1 Image

| 項目 | 値 |
|---|---|
| 解像度 | 224×224 |
| チャンネル構成 | DINO (3ch) + SigLIP (3ch) = 6ch × num_images |
| 画像数 | 1 または 2 (`--num_images_in_input`) |
| augmentation | `--image_aug True` で random crop/rotation/color jitter |
| 推論時 | center crop |

### 5.2 Proprio

- `PROPRIO_DIM = 8` (LIBERO): joint angles 7 + gripper 1
- Projector: `Linear(8, llm_dim) → GELU → Linear(llm_dim, llm_dim)`
- `use_proprio=False` にすると Projector と入力自体が省略される

### 5.3 Language

- Qwen2.5 の tokenizer
- `QwenPromptBuilder` (シンプルな平文。chat template なし)
- 典型例: `"pick up the red cube and place it in the bowl"`

### 5.4 FiLM (`--use_film`)

- 実装: `prismatic/models/film_vit_wrapper.py:36-77`
- 式: `x_out = (1 + γ) * x + β` (γ, β は language embedding から学習)
- **論文本文の実験では使っていない** (README のコマンド例も `--use_film False`)

---

## 6. Training

### 6.1 凍結 / 学習 / LoRA

| コンポーネント | 状態 |
|---|---|
| DINO + SigLIP (vision backbone) | **Frozen** |
| Qwen2.5 LLM | LoRA (r=64) |
| Vision Projector | LoRA (all-linear) |
| Proprio Projector | **Full train** |
| Action Queries (`nn.Embedding`) | **Full train** (LoRA 対象外) |
| Action Head (MLPResNet) | **Full train** |

### 6.2 Optimizer & LR

- Optimizer: AdamW (HF Accelerate 経由)
- `learning_rate`: 2e-4 (論文推奨)
- Warmup: 全ステップの 10% (`lr_warmup_steps 0.1`)
- Decay: `num_steps_before_decay` に到達したら 10 倍減衰

### 6.3 GPU 別設定 (READMEより)

| VRAM | batch | grad_accum | lora_rank | max_steps |
|---|---|---|---|---|
| 10-12GB | 1 | 8 | 64 | 400k |
| 24GB | 4 | 4 | 64 | 200k |
| **40-48GB** (我々) | **8** | **2** | **64** | **200k** |
| ≥80GB (4GPU) | 16 | 1 | 64 | 150k |

### 6.4 データパイプライン

- フォーマット: **RLDS** (TFRecord)
- データセット: `modified_libero_rlds` (HF, ~10GB 4 suite 合計)
- Loader: `prismatic/vla/datasets/rlds/dataset.py`
- Action chunk を sliding window で抽出

---

## 7. Inference

### 7.1 Checkpoint 読み込み

```python
vla = AutoModelForVision2Seq.from_pretrained(
    pretrained_checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True)
```

別途、以下を個別ロード (`experiments/robot/openvla_utils.py`):
- `action_head--checkpoint.pt`
- `proprio_projector--checkpoint.pt`
- `dataset_statistics.json` (逆正規化用)

### 7.2 Rollout 戦略

- `num_open_loop_steps = 8` で **8 ステップごと replan**
- 1回の forward で chunk 8 個を parallel 生成 → queue
- 1 アクションずつ env.step()、queue が切れたら再推論

### 7.3 Checkpoint Directory 構造 (推定)

```
outputs/LIBERO-Spatial-Pro/
├── config.json
├── configuration_prismatic.py
├── modeling_prismatic.py
├── processing_prismatic.py
├── preprocessor_config.json
├── processor_config.json
├── pytorch_model.bin           # LoRA merged 済みならここ
├── action_head--checkpoint.pt  # action head の state_dict 単独
├── proprio_projector--checkpoint.pt
├── dataset_statistics.json
└── (lora_adapter/)             # merge していない場合のみ
```

---

## 8. 設計不変量 (Gemma 4 移植時に守るべき点)

Gemma 4 E2B に置き換える際、**backbone 以外は原則そのまま** でいい。

**保存すべき設計**:
- `nn.Embedding(64, llm_dim)` による action_queries (zero-init)
- LLM 内で vision + language + action_queries を並列処理させる token layout
- Action head: MLPResNetBlock (Pro 版の separate KV projection + RoPE 推奨)
- L1 loss on normalized continuous actions
- Parallel decoding (8 chunks 同時)
- LoRA target: "all-linear" (Gemma 4 も Linear 層ばかりなので流用可)
- `BOUNDS_Q99` 正規化
- `num_open_loop_steps = 8` での replan

**Gemma 4 に合わせて変更が必要な箇所**:
- `ACTION_TOKEN_BEGIN_IDX` (Qwen=151386 → Gemma の vocab に合わせる)
- 拡張 special token (`<|extra_0..63|>` → Gemma 用の相当する unused token)
- `self.llm_dim = config.text_config.hidden_size` は自動追従するが、projector の `initial_projection_dim` (fused 版の fc1 先の 8192) は要調整
- Prompt builder (Qwen 用 → Gemma 用のテンプレート)
- Vision backbone の組み合わせ: Gemma 4 は native multimodal なので、DINO+SigLIP ではなく **Gemma 4 の vision encoder を使う** 方向も検討 (ただし比較実験として DINO+SigLIP を保持する選択肢もあり)

---

## 9. 主要ファイル索引

| 内容 | ファイル:行 |
|---|---|
| 全体 VLA クラス | `prismatic/extern/hf/modeling_prismatic.py:70-686` |
| Action queries 定義 | `modeling_prismatic.py:375-376` |
| Action head (原版) | `prismatic/models/action_heads.py:21-283` |
| Action head (Pro) | `prismatic/models/action_heads.py:287-410` |
| Vision fusion | `prismatic/models/backbones/vision/dinosiglip_vit.py:49-171` |
| Vision projector | `modeling_prismatic.py:242-273` |
| Proprio projector | `prismatic/models/projectors.py:6-24` |
| LoRA 設定 | `vla-scripts/finetune.py:832-844` |
| Loss (L1) | `vla-scripts/finetune.py:418` |
| Training loop | `vla-scripts/finetune.py:689-1100` |
| LIBERO eval | `experiments/robot/libero/run_libero_eval.py` |
| 定数 | `prismatic/vla/constants.py` |
| Checkpoint save | `vla-scripts/finetune.py:494-602` |
| Inference (latency 計測箇所) | `run_libero_eval.py:334-345` |
