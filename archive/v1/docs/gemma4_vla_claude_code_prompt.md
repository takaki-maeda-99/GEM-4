# Gemma 4 ベース VLA モデル実装要件 (Claude Code 向け)

## 0. プロジェクトコンテキスト

Gemma 4 Good Hackathon (2026年5月18日締切) 向けに、Gemma 4 を VLM バックボーンとした Vision-Language-Action (VLA) モデルをスクラッチから実装する。事前学習済み VLA チェックポイント (π0, OpenVLA, X-VLA-Pt 等) は **使用不可** という制約のもと、計算リソース A100 80GB × 2 枚で学習可能なアーキテクチャを構築する。

研究的新規性は **「robot pretraining なしの vanilla Gemma 4 を frozen で運用しても、multi-layer feature aggregation + soft prompt による modality injection によって action-aligned な表現を引き出せる」** ことの実証に置く。最終的に force/torque modality を soft prompt 経由で注入する研究展開を見据えた、modular な実装が必要。

---

## 1. 高レベル要件

### 必須要件
- [ ] Gemma 4 (HuggingFace transformers 経由) を VLM として使用
- [ ] VLM は frozen (もしくは QLoRA で「ほぼ frozen」の状態)
- [ ] LIBERO ベンチマーク (LIBERO-Spatial / LIBERO-Object / LIBERO-Goal / LIBERO-Long のいずれか1つ以上) で評価
- [ ] A100 80GB × 2 枚で学習が完走すること (OOM を起こさない)
- [ ] modality injection 用の soft prompt スロットを設計段階で確保
- [ ] LeRobot dataset format との互換性 (もしくは LIBERO native format)

### 推奨要件
- [ ] VLA-Adapter スタイルの multi-layer feature aggregation
- [ ] Flow matching ベースの action head (π0 / X-VLA 流)
- [ ] Action chunking (chunk size 8〜16)
- [ ] bfloat16 mixed precision
- [ ] gradient checkpointing (action head 側のみ)
- [ ] Wandb 統合
- [ ] Hydra もしくは類似の config 管理

---

## 2. アーキテクチャ仕様

### 2.1 全体構成図

```
                    ┌─────────────────────────────────────┐
                    │         Frozen Gemma 4 VLM          │
   [RGB images]────▶│  ┌──────────────────────────────┐   │
   [Instruction]───▶│  │ Vision Encoder (ViT, frozen) │   │
   [Soft Prompts]──▶│  │ + Language Model (frozen)    │   │
                    │  │ output_hidden_states=True    │   │
                    │  └──────────────────────────────┘   │
                    └────────┬────────────┬───────┬───────┘
                             │            │       │
                       hidden_layer_1  ... layer_N (multi-layer features)
                             │            │       │
                             ▼            ▼       ▼
                    ┌─────────────────────────────────────┐
                    │      Bridge Attention Module        │
                    │   (ActionQuery × multi-layer KV)    │
                    └────────────────┬────────────────────┘
                                     │
                            [N_q × D] action latents
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │   Action Head (Flow Matching DiT)   │
                    │   conditioning: action latents      │
                    │              + proprioception       │
                    │              + flow timestep        │
                    └────────────────┬────────────────────┘
                                     │
                                     ▼
                       [chunk_size × action_dim]
```

### 2.2 各コンポーネント仕様

#### (a) VLM Backbone
- モデル: `google/gemma-4-e4b-it` をデフォルト (プロトタイプ用)、production 切替可能に `google/gemma-4-31b-it` を選べる構成
- 全パラメータを `requires_grad = False` に設定
- Forward は `torch.no_grad()` ラッパー内で実行 (activation memory 削減)
- `output_hidden_states=True` で全層の hidden states を取得
- vision token budget は config で指定可能に。デフォルト 280 (manipulation 想定)
- 入力 prompt 構造:
  ```
  [SOFT_PROMPT_TOKENS] [IMAGE_TOKENS] [TEXT_INSTRUCTION] [ACTION_QUERY_PLACEHOLDER]
  ```
  ただし ACTION_QUERY は VLM 内で処理せず、Bridge Attention 側で別管理する設計も可 (実装シンプルさを優先)

#### (b) Soft Prompt Module
- 学習可能な embedding: `nn.Embedding(num_soft_prompts, vlm_hidden_dim)`
- `num_soft_prompts`: デフォルト 32 (config で 8〜64 を可変に)
- 初期化: VLM の word embedding 平均を中心とした小さな gaussian noise
- 配置: VLM の input embedding stream の先頭に concat
- **将来拡張ポイント**: force/torque proprioception を MLP で `vlm_hidden_dim` に射影し、soft prompt の一部スロットに加算する経路を予約しておく (実装は stub でよい、interface だけ確保)

#### (c) Bridge Attention (VLA-Adapter スタイル)
- ActionQuery: 学習可能な `[N_q, D]` テンソル。`N_q = chunk_size` (デフォルト 8〜16)
- 抽出する VLM 層: 全層の中から等間隔 8 層をサンプル (例: 31B が 60 層なら層番号 [7, 14, 22, 30, 37, 45, 52, 60] 程度)。config で指定可
- 抽出した hidden states を **layer ごとに別々の K, V** として用いる (もしくは concat してから cross-attention)
- Bridge Attention block:
  - 4〜6 層の transformer decoder layer
  - 各層: cross-attention (Q=ActionQuery, KV=VLM features) → self-attention (ActionQuery 内部) → FFN
  - hidden_dim: VLM hidden dim と揃える or 別途プロジェクション
- 出力: `[chunk_size, D]` の action latent

#### (d) Action Head (Flow Matching)
- DiT (Diffusion Transformer) アーキテクチャ
- 6〜8 層程度の transformer block
- conditioning:
  - Bridge Attention 出力 (action latent)
  - Proprioception (MLP で射影)
  - Flow matching timestep (sinusoidal embedding)
- 学習目標: flow matching loss (π0 / X-VLA 流)
  ```
  L = E_{t, x_0, x_1} || v_θ(x_t, c, t) - (x_1 - x_0) ||^2
  where x_t = (1-t) x_0 + t x_1, x_0 ~ N(0, I), x_1 = ground truth action
  ```
- 推論: Euler 法で 5〜10 ステップで chunk を生成
- action_dim: LIBERO は 7 (xyz + rpy + gripper) 想定。config 化必須

#### (e) Proprioception Encoder
- MLP: `[proprio_dim] → [hidden] → [D]`
- LIBERO の場合 proprio_dim は通常 9 (joint position 7 + gripper 2) 程度

---

## 3. 学習レシピ

### 3.1 デフォルトハイパーパラメータ

| 項目 | 値 | 備考 |
|---|---|---|
| Optimizer | AdamW | β = (0.9, 0.95), weight_decay = 1e-4 |
| Learning rate | 1e-4 | Bridge / Soft Prompt / Action Head 全部共通 |
| LR schedule | Linear warmup (1k steps) → cosine decay | |
| Batch size | 32 (per GPU 16 × 2 GPU) | OOM 時は 16 に下げる |
| Gradient accumulation | 2 | 実効 batch 64 |
| Precision | bf16 mixed precision | VLM forward は no_grad + bf16 |
| Action chunk size | 8 | LIBERO で実績ある値 |
| Total steps | 50,000 | early stopping with eval success rate |
| Eval frequency | 5,000 steps | LIBERO success rate |

### 3.2 Frozen 戦略の実装詳細
```python
# VLM 全パラメータを freeze
for p in vlm.parameters():
    p.requires_grad = False

# VLM forward は no_grad
with torch.no_grad():
    outputs = vlm(input_ids=ids, pixel_values=imgs, output_hidden_states=True)
hidden_states = outputs.hidden_states  # tuple of [B, T, D]

# Bridge / Soft Prompt / Action Head のみ optimizer に渡す
trainable_params = [
    *bridge_attention.parameters(),
    *soft_prompt.parameters(),
    *action_head.parameters(),
    *proprio_encoder.parameters(),
]
optimizer = AdamW(trainable_params, lr=1e-4)
```

### 3.3 オプション: QLoRA モード
完全 frozen で性能が出ない場合のフォールバック。
- bitsandbytes で VLM を NF4 量子化
- PEFT で LoRA r=8, alpha=16 を q_proj, k_proj, v_proj, o_proj に挿入
- Action expert からの gradient が VLM に伝わるが、低 rank なので影響限定的
- config flag `use_lora_on_vlm: bool = False` で切替可能に

---

## 4. ディレクトリ構造

```
gemma4_vla/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── model/
│   │   ├── gemma4_e4b.yaml
│   │   └── gemma4_31b.yaml
│   ├── data/
│   │   └── libero_spatial.yaml
│   └── training/
│       └── default.yaml
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vlm_backbone.py        # Gemma 4 wrapper
│   │   ├── soft_prompt.py
│   │   ├── bridge_attention.py
│   │   ├── action_head.py         # Flow matching DiT
│   │   ├── proprio_encoder.py
│   │   └── vla_model.py           # 全体を組み立てる top-level model
│   ├── data/
│   │   ├── libero_dataset.py
│   │   └── transforms.py
│   ├── training/
│   │   ├── train.py
│   │   ├── losses.py              # flow matching loss
│   │   └── scheduler.py
│   ├── eval/
│   │   ├── libero_eval.py
│   │   └── inference.py
│   └── utils/
│       ├── logging.py
│       └── checkpoint.py
├── scripts/
│   ├── train_e4b.sh
│   ├── train_31b.sh
│   └── eval_libero.sh
└── tests/
    ├── test_vlm_forward.py
    ├── test_bridge.py
    └── test_action_head.py
```

---

## 5. 実装フェーズ (推奨順序)

### Phase 1: Skeleton & Smoke Test (Day 1〜2)
- リポジトリ初期化、依存関係セットアップ
- Gemma 4 E4B のロード、frozen 化、ダミー画像でforward が通ることを確認
- multi-layer hidden states の shape を print して構造把握
- **完了基準**: ダミー入力で end-to-end forward が通り、loss が計算できる

### Phase 2: Bridge + Action Head (Day 3〜5)
- Bridge Attention 実装、ActionQuery と多層 KV の cross attention
- Flow matching action head (DiT) 実装
- ランダム action データで overfitting テスト (1 batch を 1000 step 学習し loss → 0)
- **完了基準**: synthetic data での過学習が確認できる

### Phase 3: LIBERO データ統合 (Day 6〜8)
- LIBERO dataset loader (LeRobot 経由か native)
- 画像前処理、proprio 正規化、action chunking
- 小規模 (1000 step) 学習でloss が下がることを確認
- **完了基準**: 学習 loss が下がり、評価 pipeline が動く

### Phase 4: 本格学習 + 評価 (Day 9〜12)
- 50k step フル学習
- LIBERO success rate 計測
- Soft Prompt の有無で ablation
- **完了基準**: LIBERO-Spatial で success rate を取得

### Phase 5: Force/Torque modality 注入の stub (Day 13〜)
- proprio encoder を force/torque 対応に拡張する interface 設計
- Soft Prompt スロットへの inject 経路 (今は dummy zero でよい)
- 将来の研究展開のための土台

---

## 6. 評価

### 6.1 ベンチマーク
- LIBERO-Spatial (10 task) を最低限。余裕あれば LIBERO-Object, Goal も
- metric: success rate (各 task 50 episode 程度)

### 6.2 Ablation 計画
| 設定 | 期待値 |
|---|---|
| Frozen Gemma 4 + Bridge + No Soft Prompt | baseline |
| + Soft Prompt (32 tokens) | わずかに向上を期待 |
| + QLoRA (r=8) | 数% 向上、ただし学習時間 +30% |
| 単層 (last only) feature vs multi-layer | multi-layer が勝つことを期待 |

### 6.3 Sanity Check
- VLM が完全に frozen であること: optimizer.param_groups の総パラメータ数を assert
- Soft Prompt の gradient が non-zero であること
- Action head の出力分布が学習中に変化していること

---

## 7. アンチパターン (やらないこと)

- ❌ VLM の最終層 hidden state だけ使う (action-aligned でないので情報損失大)
- ❌ Action expert からの gradient を VLM に伝播 (Knowledge Insulation 論文の警告)
- ❌ Discrete action token の autoregressive 生成 (frozen VLM では tokenizer の埋め込みが action 用に最適化されてないので破綻しやすい)
- ❌ Vision encoder と LLM を別々にロード (Gemma 4 は native multimodal なので transformers の AutoModel に任せる)
- ❌ batch_size を盲目的に大きく (frozen でも VLM activation がでかい)
- ❌ FP32 学習 (bf16 で十分、メモリ的に必須)

---

## 8. 参考文献 / 参照すべき実装

| 論文 / 実装 | 参照ポイント |
|---|---|
| VLA-Adapter (arXiv:2509.09372) | Bridge Attention 設計、frozen VLM での性能 |
| X-VLA (arXiv:2510.10274) | Soft Prompt 機構、Phase II adaptation 戦略 |
| π0 / π0.5 (Physical Intelligence) | Flow matching action head、action expert 設計 |
| Knowledge Insulation (pi.website/research/knowledge_insulation) | Frozen 戦略の理論的根拠、避けるべき構造 |
| LeRobot (huggingface/lerobot) | Dataset format、PEFT integration の参考 |
| OpenVLA-OFT | 比較対象。frozen で動かない例として |
| Gemma 4 documentation (ai.google.dev/gemma/docs/core) | Vision token budget 仕様、入力フォーマット |

---

## 9. 質問・確認事項 (実装開始前に Takaki に確認すべき項目)

1. Gemma 4 のサイズ: E4B プロトタイプ → 31B 本番 でよいか? それとも最初から 31B 一本?
2. データセット: LIBERO のどの subset を優先するか?
3. 評価環境: LIBERO simulator のセットアップは別途行うか、Claude Code 側で含めるか?
4. force/torque データ: 現時点で実データはあるか? なければ stub 実装で十分か?
5. ロギング: wandb のプロジェクト名、entity 指定はあるか?
