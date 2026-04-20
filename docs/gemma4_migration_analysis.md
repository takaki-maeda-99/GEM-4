# Gemma 4 E2B 移植: 学習規模・環境要件・課題点 詳細分析

**目的**: VLA-Adapter (Qwen2.5-0.5B backbone) を Gemma 4 E2B backbone に移植する際の、学習コスト・必要環境・技術的課題を具体的に見積もるリファレンス。

作成日: 2026-04-19
対象環境: 2×A100-80GB

---

# Part 1: 学習規模と環境要件

## 1.1 Smoke test 実測データ (ベースライン)

2026-04-19 に実施した smoke test の測定値:

| 項目 | 値 |
|---|---|
| 環境 | 1×A100-80GB, batch=4, grad_accum=1 |
| モデル | VLA-Adapter Pro (LIBERO-Spatial, Qwen2.5-0.5B backbone) |
| 全パラメータ | 1,356,347,968 (~1.36B) |
| 学習可能 | 322,527,519 (**7.66%**) |
| - action_head | 217,864,223 (~67% of trainable) |
| - LoRA on LLM | 103,851,520 (~32%) |
| - proprio_projector | 811,776 (~0.25%) |
| Warmup 後速度 | **~1.3 iter/s** (batch=4) |
| Step あたり秒数 | ~0.77s |
| 初期 L1 loss | 0.535 (ランダム初期化の action_head) |

**注意**: 初回 step は初期化・compile overhead で 14.9 秒。安定後は 1 秒未満。

## 1.2 タスクスイート別の学習時間予想

VLA-Adapter 論文のトレーニング step 数と、我々の環境でのスケール予想:

| Task Suite | max_steps (論文) | 論文環境 (4×H100) | 1×A100-80GB | 2×A100-80GB (DDP) | 4×A100-80GB |
|---|---|---|---|---|---|
| LIBERO-Spatial | 200,000 | 5h | **42h** | **22h** | 12h |
| LIBERO-Object | 200,000 | <1h (※) | 42h | 22h | 12h |
| LIBERO-Goal | 200,000 | 3h | 42h | 22h | 12h |
| LIBERO-Long (10) | 400,000+ | ~12h | **84h** | 44h | 24h |
| CALVIN ABC→D | 150,000+ | 8h | ~32h | ~17h | ~9h |

(※) Object の論文環境 <1h は異常に速い。batch/GPU スケールによる差か、他タスクと別レシピか要確認。

**Gemma 4 E2B への移植時の速度補正**: 後述の **Hidden dim 1536** (Qwen の 1.7x) により、1 step あたりの FLOPs が 2-3x になる可能性。上記時間に **×2〜×3 の倍率** を掛けるのが安全見積もり。

→ Spatial を 2×A100 で実行時、**~44-66h (2-3日)** を想定。

## 1.3 メモリ内訳とbatch size最適化

### 1.3.1 Qwen2.5-0.5B backbone, batch=8 (VLA-Adapter Pro) — 実測ベース推定

| 項目 | 推定メモリ (GB) | 根拠 |
|---|---|---|
| モデル本体 (bf16) | 2.7 | 1.36B × 2 bytes |
| 勾配 (bf16) | 0.6 | 322M × 2 bytes |
| Optimizer state (AdamW, fp32) | 2.6 | 322M × 4 × 2 (m, v) |
| Activations (batch=8) | ~20 | vision (DINO+SigLIP)が重い |
| その他 (dataloader, cache, overhead) | ~3 | |
| **合計** | **~29 GB** | |

### 1.3.2 Gemma 4 E2B backbone, batch=8 推定

Gemma 4 E2B は **effective 2.3B / 実パラメータ 5.1B** (Per-Layer Embedding 込み)。

| 項目 | 推定メモリ (GB) | 備考 |
|---|---|---|
| モデル本体 (bf16) | ~10 | 5.1B × 2 bytes (PLE含む) |
| 勾配 (bf16) | 1.5-2 | LoRA + action_head + PLE の学習範囲依存 |
| Optimizer state | 6-8 | |
| Activations (batch=8) | ~35-45 | hidden 1536 × layers 35 |
| その他 | ~4 | |
| **合計** | **~55-70 GB** | |

→ **A100-80GB で batch=8 はギリギリ収まる**。batch=16 は OOM のリスク、batch=4 ~ 6 が安全圏。

### 1.3.3 batch size 推奨

| GPU | Qwen backbone | Gemma 4 E2B backbone |
|---|---|---|
| RTX 3090 (24GB) | 1 (grad_accum=8) | 不可能、40GB 以上要 |
| RTX 4090 (24GB) | 2 (grad_accum=4) | 不可能 |
| A100-40GB | 4 (grad_accum=2) | 2 (grad_accum=4) |
| **A100-80GB (我々)** | **8** | **4-6** (grad_accum=2) |
| H100-80GB | 16 | 8 |

実効 batch size 32 を維持するなら grad_accum で調整。

## 1.4 マルチGPU戦略

### 1.4.1 VLA-Adapter のデフォルト実装

`vla-scripts/finetune.py:19-227` を見ると:

- `accelerate.PartialState()` で distributed 環境管理
- `torch.nn.parallel.DistributedDataParallel` (DDP) で wrap
- `wrap_ddp(module, device_id, find_unused=False)` ヘルパー (line 215)
- **FSDP は使っていない**。純粋な DDP のみ

### 1.4.2 GPU 間通信コスト

DDP はバックワード時に勾配 all-reduce が入る。Gemma 4 E2B (5.1B params) で勾配 5.1B × 2 bytes = ~10GB の通信が毎 step 発生。

- **NVLink 接続 (A100/H100)**: 帯域 600GB/s → 17ms 程度 / step
- **PCIe のみ (消費者GPU)**: 32GB/s → 300ms+ / step、計算より通信がボトルネック

我々の A100-80GB x 2 は **NVLink 接続されているか要確認** (`nvidia-smi topo -m`)。接続されていれば DDP で効率的、そうでなければ単 GPU で batch 大きめの方が良いかも。

### 1.4.3 FSDP への移行検討

Gemma 4 E2B は 5.1B で DDP のメモリ要件が厳しくなる可能性。FSDP (ZeRO-3相当) で optimizer state を GPU 間で sharding すれば 3-4x のメモリ削減。ただし VLA-Adapter の現実装は DDP 前提なので、移植工数がかかる。

**推奨**: まず DDP で動作確認、メモリがカツなら FSDP 導入を検討。

---

# Part 2: Gemma 4 E2B のアーキテクチャ

## 2.1 基本スペック

(`google/gemma-4-E2B` config.json ベース。2026-04-02 リリース)

| 項目 | 値 | Qwen2.5-0.5B 比 |
|---|---|---|
| モデルタイプ | `gemma4` | — |
| アーキテクチャ | `Gemma4ForConditionalGeneration` | — |
| `hidden_size` | **1536** | 1.7x (vs 896) |
| `num_hidden_layers` | **35** | 1.5x (vs 24) |
| `num_attention_heads` | 8 | 0.6x (vs ~14) |
| `num_key_value_heads` | **1** | 極端な MQA |
| `head_dim` | 256 | (vs 64) |
| `intermediate_size` | 6144 | (vs 4864) |
| `vocab_size` | **262,144** | **1.73x** (vs ~151,643) |
| `max_position_embeddings` | 131,072 | 64x (vs 2048) |
| Effective params | 2.3B | 4.6x (vs 0.5B) |
| 実パラメータ (PLE込) | 5.1B | 10.2x |
| ライセンス | Apache 2.0 (クリックスルーgate) | Apache 2.0 |

## 2.2 特殊機能 (VLA移植で問題になる点)

### A. Per-Layer Embeddings (PLE)

- 埋め込みテーブルが通常の `nn.Embedding` ではなく **`Gemma4TextScaledWordEmbedding`**
- `sqrt(hidden_size_per_layer_input) = sqrt(256) = 16.0` でスケール
- **実際の embedding 次元は `num_hidden_layers * 256 = 8960`** (config の 256 は per-layer 分)
- 各 decoder layer に residual 注入される `per_layer_model_projection` / `per_layer_projection_norm` が存在
- ドキュメント不足 (HF issue #45206)

→ VLA-Adapter の action_queries 埋め込み (`nn.Embedding(64, llm_dim)`) を Gemma 4 に合わせるのが **非自明**。`llm_dim = 1536` で作っても PLE と整合が取れない可能性。

### B. Hybrid Attention (Sliding + Global)

- 35 層のうち一部は **sliding window (512 tokens, rope_theta=10000)**、一部は **global full-attention (rope_theta=1000000)**
- `partial_rotary_factor=0.25` で RoPE が proportional 適用
- これが交互に配置

→ VLA-Adapter の parallel decoding は全 action tokens (64個) が全 vision tokens (512個) を見る必要。sliding window 層で 512 tokens を超えると **attention が切れる**。token layout の再設計が必要。

### C. KV Cache Sharing

- 35 層で **unique KV は 15 組のみ** (KV が layer 間で共有)
- `num_key_value_heads=1` (MQA の極端版) と組み合わさる

→ hidden state 抽出時、どの layer の出力が独立情報を持っているか不明瞭。VLA-Adapter の action_head は全 layer の hidden states を使うが、Gemma 4 では冗長 or 無効な情報になる可能性。

### D. Native Multimodal

- **ViT vision encoder (16 layers, 768 dim, 280 soft tokens/image, 可変aspect ratio)**
- **USM audio encoder (12 layers, 1024 dim)** (今回は不要)
- 専用 special token: image=258880, audio=258881, video=258884

→ DINO+SigLIP vs Gemma 4 native vision の選択が重要な設計判断に。

---

# Part 3: 移植課題の詳細分析

## 3.1 🔴 最重要: transformers API 互換性

### 現状

VLA-Adapter pyproject.toml:
```python
"transformers==4.40.1",
#"transformers @ git+https://github.com/moojink/transformers-openvla-oft.git",
# IMPORTANT: Use this fork for bidirectional attn (for parallel decoding)
```

### 課題

Gemma 4 E2B は `transformers >= 5.5` を要求。VLA-Adapter の `modeling_prismatic.py` は transformers 4.40 前提で書かれており、5.5 では以下が壊れる:

1. **`GenerationMixin` の API 変更**: 4.x の `_prepare_model_inputs`, `prepare_inputs_for_generation` が 5.x でシグネチャ変更
2. **attention_mask の扱い変更**: 5.x では position_ids が明示要求、mask shape の制約が厳格化
3. **Cache クラスの刷新**: `DynamicCache`, `HybridCache` 等のクラス化で、`past_key_values` の dict アクセスが使えない
4. **moojink fork の消失**: moojink/transformers-openvla-oft は 4.40 base のため、Gemma 4 では使えない
5. **`AutoModelForVision2Seq` の挙動変更**: 5.x では multimodal model 前提の新しい generation pipeline

### 対処工数

- **最小改修 (動作させるだけ)**: `modeling_prismatic.py` (~690 行) を transformers 5.5 API に書き直し → **1-2 週間**
- **bidirectional attention patch の再実装**: Gemma 4 の attention layer (sliding/global interleaved) に非 causal モードを挿入 → **3-5 日**
- **テスト**: 論文スコア再現確認 → 1-2 週間

## 3.2 🔴 双方向 attention パッチ (Parallel decoding の核心)

### 仕組み

VLA-Adapter の parallel decoding は、64個の action token が *互いに* 参照できる必要がある (通常の causal LM では token_i は token_{i+1..} を見れない)。moojink fork はこれを実現するため Qwen2 attention に patch を入れている。

### 実装の目安 (Qwen→Gemma 4 移植時)

1. Gemma 4 の `Gemma4Attention.forward` (transformers 5.5) をフォーク
2. `attention_mask` の action token 範囲を特定
3. その範囲の causal mask を `-inf` → `0` で上書き (双方向化)
4. sliding window 層では window を超える距離で attention が切れないよう調整

### 課題

- Gemma 4 の sliding window attention は実装が複雑 (flex_attention or chunked kernel)
- 64 action tokens 全体を sliding window (512) 内に収めるため、token 配置の設計が必要
  - Language (~50) + Vision (~512) + Action (64) = **626 tokens** → 512 window を超える
  - → sliding window 層では vision と action が別 chunk になる可能性
  - → VLA-Adapter の「action tokens が全 vision tokens を見る」前提が崩れる

**対応案**:
- (a) すべて **global attention 層で処理** (sliding window 層を回避/hidden state 抽出対象から除外)
- (b) token layout を工夫 (action tokens を最後にまとめる、vision tokens を圧縮)
- (c) flash-attention 2 の block-sparse mask を直接制御

## 3.3 🟠 Action token の配置と special token

### 現状

- Qwen tokenizer: `<|extra_0|>` 〜 `<|extra_63|>` (合計 64 tokens、`ACTION_TOKEN_BEGIN_IDX=151386`)
- Action query Embedding: `nn.Embedding(64, 896)` zero-init
- Dataset 側で action 位置に `<|extra_X|>` を挿入 → forward 時に embedding を action_queries で上書き

### Gemma 4 での対応

**候補 A: 既存の未使用 token を流用**
- Gemma 4 vocab_size=262,144 は巨大、使われていない ID 範囲がある可能性
- `<unused_0..63>` のような予約トークンがあれば使う
- 要確認: `AutoTokenizer.from_pretrained('google/gemma-4-E2B')` で tokenizer を確認

**候補 B: 新規 token を追加**
- `tokenizer.add_tokens(['<action_0>', ..., '<action_63>'])`
- `model.resize_token_embeddings(len(tokenizer))` で埋め込み拡張
- **問題**: PLE 構造で `resize_token_embeddings` が正しく動くか不明

**候補 C: image/audio special token の流用**
- Gemma 4 は既に image=258880, audio=258881, video=258884 を持つ
- 「action」も同様の専用 special token 領域として追加が自然

→ **候補 C が論理的に綺麗。Google が用意した special token 拡張パターンに従う**

### 実装工数: 2-3 日 (tokenizer / embedding 層のチェック含む)

## 3.4 🟠 Vision backbone の選択

### 選択肢

| オプション | Pros | Cons |
|---|---|---|
| **(A) DINO+SigLIP を維持** | VLA-Adapter 論文と直接比較可能、projector pretraining 戦略流用 | Prismatic-VLM の vision projector を Gemma 4 用に事前学習必要 (~数日〜週) |
| **(B) Gemma 4 native vision (ViT 16層) を使う** | 既に multimodal pretrain 済みでゼロ追加学習、楽 | DINO の semantic + SigLIP の contrastive 特徴が失われる、VLA-Adapter のレシピと直接比較できない |
| **(C) SigLIP のみ維持、DINO 除外** | Gemma 4 native に近い、VLA との共通性もある | 中途半端、pretraining 問題は残る |

### 推奨: まず (B) で動作確認、その後 (A) を比較実験

理由:
- まず「Gemma 4 + VLA-Adapter 設計 = LIBERO で動くか」を確認するのが先
- Native vision なら vision projector の事前学習が不要 → 早く iterate できる
- 性能が論文値 (97.8-99.6%) に届かない場合、(A) の pretraining を追加

## 3.5 🟡 Hidden dim 変更の波及

### 具体的影響

| コンポーネント | Qwen (896) | Gemma 4 (1536) | 変更点 |
|---|---|---|---|
| `action_queries` | `Embedding(64, 896)` | `Embedding(64, 1536)` | 自動追従 (`config.text_config.hidden_size`) |
| Vision projector fc1 | `Linear(2048, 8192)` | `Linear(2048, ?)` | `initial_projection_dim` の再調整必要 (推奨: 4×hidden=6144 or 8192維持) |
| Action head input | MLPResNet input 896×64 = 57,344 | 1536×64 = 98,304 | 1.7x、パラメータも比例増 |
| MLPResNetBlock hidden | ? | ? | `action_heads.py` 実装を読んで、hidden_dim を 1536 に合わせる |
| Proprio projector | `Linear(8, 896)→GELU→Linear(896,896)` | `Linear(8, 1536)→GELU→Linear(1536,1536)` | 自動追従 |

### LR スケーリング

- モデルが大きくなる → LR を下げるのが定石 (linear scaling rule の逆)
- Qwen (0.5B) で LR=2e-4 → Gemma 4 E2B (5.1B) で **LR=1e-4** から試す
- LoRA rank: 64 → 128 に上げて表現力補強を検討

## 3.6 🟡 LoRA target_modules の再設定

### Gemma 4 の Linear 層の存在箇所

Gemma 4 は GQA/MQA + PLE という特殊構造。`target_modules="all-linear"` で拾われる層:

```
- q_proj, k_proj, v_proj, o_proj  (attention)
- gate_proj, up_proj, down_proj    (FFN)
- per_layer_model_projection        (PLE residual injection) ← これをLoRA対象にするか要検討
- lm_head                          (通常は除外)
```

**注意**: `per_layer_model_projection` に LoRA を当てると PLE の挙動が壊れる可能性。**attention と FFN だけに限定する target_modules 指定** のほうが安全:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### rank/alpha

- Qwen 0.5B で rank=64 → Gemma 4 E2B (5.1B) では **rank=128** or **256** が推奨
- alpha = 2 × rank (VLA-Adapter の慣習を踏襲)

## 3.7 🟢 変更不要な部分

以下は backbone 非依存で、そのまま使える:

1. **Action head (MLPResNet)** — `prismatic/models/action_heads.py`
   - `llm_dim` を config から読むだけ
   - Pro 版の RoPE 実装は独立、Gemma 4 の RoPE とは無関係
2. **Normalization (`BOUNDS_Q99`)** — `constants.py`
3. **Data pipeline (RLDS loader)** — `prismatic/vla/datasets/`
4. **LIBERO 評価 loop** — `experiments/robot/libero/run_libero_eval.py`
5. **L1 loss** — `finetune.py:418`
6. **Parallel decoding strategy (8 chunks/forward)** — inference 側の設計

---

# Part 4: 推奨移植戦略

## 4.1 フェーズ分割と工数見積もり

### Phase 0: 環境準備 (半日)
- `.venv` に transformers 5.5+ の新しい uv env を別途作成 (VLA-Adapter 用と併存)
- `google/gemma-4-E2B` を HF から取得 (gate accept 必要)
- Gemma 4 の Python での load テスト (PLE の挙動確認)

### Phase 1: 基盤移植 (2-3 週間)
- `modeling_prismatic.py` を transformers 5.5 API + Gemma 4 構造に移植
  - Gemma 4 の `forward` signature に合わせる
  - PLE の処理を追加
  - `_build_multimodal_attention` で Gemma 4 の hybrid attention に対応
- Action token 追加 (special token 戦略 C 採用)
- Action queries embedding の組み込み
- 単純な forward/backward が動くことを確認 (1 batch で loss 計算できる)

### Phase 2: 双方向 attention patch (1 週間)
- Gemma 4 の `Gemma4Attention` を拡張
- action token 範囲で causal mask 無効化
- sliding window 層と global attention 層で動作確認
- Token layout 検討 (sliding window 512 制約下でどう収めるか)

### Phase 3: 単タスク smoke test (3-5 日)
- LIBERO-Spatial 1 タスクで 10-step 学習
- Memory profile, 速度測定
- OOM / 精度劣化チェック

### Phase 4: フル学習と性能検証 (3-5 日 計算 + 2-3 日 デバッグ)
- LIBERO-Spatial でフル学習 (~2 日, 2×A100)
- 推論で 成功率測定、論文値 (99.6%) と比較
- 届かない場合は LR, batch, LoRA rank, vision backbone の組み合わせ調整

### 合計工数見積もり: **5-7 週間**

## 4.2 リスクと代替プラン

### 主要リスク

| リスク | 確率 | 影響 | 対策 |
|---|---|---|---|
| PLE を扱うコードが書けない | 中 | 致命的 | HF のGemma4実装を丁寧に読む、Google公式 example を参照 |
| Hybrid attention で parallel decoding が動かない | 中 | 致命的 | Token layout 工夫 or global attention のみの層を探して使う |
| transformers 5.5 が他依存と衝突 | 低 | 中 | 別 env で隔離 |
| 性能が Qwen 版に及ばない | 中 | 中 | vision backbone を DINO+SigLIP に差し替えて再試行 |
| Gemma 4 の巨大 vocab (262k) がメモリ消費 | 低 | 中 | `lm_head` を別GPUに、or LoRA のみ学習で frozen |

### 代替プラン: Gemma 3n E2B (conservative stepping stone)

Gemma 4 が難航した場合の代替:

- `google/gemma-3n-E2B-it`: transformers >= 4.53 (まだハードルあるが Gemma 4 より低い)
- MatFormer アーキテクチャは PLE より単純で文書化済み
- 32K context (Gemma 4 の 128K より小だが十分)
- 既に native multimodal
- 移植工数を **半分以下** に圧縮可能

**判断ポイント**: Phase 1 の最初の 1 週間で Gemma 4 の PLE ハンドリングが見えてこない場合、Gemma 3n に切り替え。

## 4.3 着手順序の推奨

1. Gemma 4 E2B を HF からロードし、**text-only の forward pass** が動くことをまず確認 (config 把握)
2. Gemma 4 の **vision tower** を text LLM と組み合わせた inference を試す
3. Action head を接続する **prototype** を書く (dummy data でも可)
4. はじめて RLDS dataset と繋げて smoke train
5. Bidirectional attention を **最後に** 最適化として追加 (causal のままでも訓練は進む、精度だけ劣化)

---

# まとめ

- **学習コスト**: 我々の 2×A100-80GB で Spatial 2-3 日、Long 3-4 日見込み (Gemma 4 に移植後)
- **メモリ**: Gemma 4 E2B (5.1B total) は 80GB GPU でも batch=4-6 がベスト、FSDP は検討の価値あり
- **主な技術障壁**: (1) transformers 4.40→5.5 の書き換え、(2) PLE の扱い、(3) bidirectional attention の Gemma 4 向け再実装
- **推奨戦略**: Phase 1 で PLE / transformers 5.5 の互換層を書き切る。それができれば残りは比較的素直
- **代替**: 詰まったら Gemma 3n E2B に降りる選択肢を Phase 1 の段階で評価

## 主要参考リンク

- [Welcome Gemma 4 (HF blog)](https://huggingface.co/blog/gemma4)
- [google/gemma-4-E2B-it config.json](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json)
- [HF transformers issue #45206 — Gemma4 PLE underdocumented](https://github.com/huggingface/transformers/issues/45206)
- [moojink/transformers-openvla-oft (4.40 base の bidirectional attn fork)](https://github.com/moojink/transformers-openvla-oft)
- [VLA-Adapter 論文](https://arxiv.org/abs/2509.09372)
- [docs/vla_adapter_architecture.md](vla_adapter_architecture.md) (先行レポート)
