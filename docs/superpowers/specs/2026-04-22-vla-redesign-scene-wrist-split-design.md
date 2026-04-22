# VLA Redesign: Scene / Wrist Path Split + LLM Backward Skip

**Status**: Draft — pending user review
**Date**: 2026-04-22
**Target**: Hackathon 2026-05-18 (4 weeks)
**Checkpoint disposition**: 60k step Stage 3c-0 checkpoint will be discarded, fresh pretrain from step 0

---

## 1. Motivation

現 VLA-Gemma4 の学習は GPU util 90%+ の compute bound 状態で速度が出ていない (memory bound で `batch_size=8` から拡大不可)。原因分析により、以下 3 つの構造的損失が判明した:

1. **手元視点 (wrist camera) を frozen LLM に通している** — Gemma 4 は自然画像で pretrain されており、手元 POV は OOD。しかも LLM は LoRA もかけず完全 frozen なので、この OOD 入力に適応する術がない。計算コストを払っているのに semantic 情報をほぼ取り出せていない疑いが強い。
2. **DinoSigLIP + `vision_projector` が冗長** — Gemma 4 E2B は native multimodal 設計で自前の vision encoder (`Gemma4VisionModel`, 151M params) を持ち、LM と同時 pretrain されている。外付け DinoSigLIP (400M) + 3 層 MLP projector は後付け、feature space alignment を学習で獲得せねばならず本質的に loss。
3. **frozen LLM なのに backward を毎 step 計算している** — `SoftPromptLibrary` が LLM input 前段に配置 (VLA-Adapter "案 B" deviation) されており、soft_prompt (trainable) 経由で LLM backward が autograd から要求される。forward + backward = 2F の計算を毎 step 消費、加えて backward のため LLM activation を全保持 → memory の主犯。

これら 3 つを同時に解消する設計へリプレース。**frozen LLM + Bridge Attention という VLA-Adapter の核心思想は死守する** (論文再現性とアーキ哲学の根幹のため)。

## 2. Goals

- **G1**: Wrist 視点を LLM 経路から外し、専用 encoder で action head に直接供給
- **G2**: Scene 視点を Gemma 4 native vision に切り替え、feature space alignment を pretrained 状態で獲得
- **G3**: SoftPromptLibrary を X-VLA 原実装位置 (action head 内) に戻し、**LLM input を完全に frozen source のみ**にすることで LLM backward を `torch.no_grad()` で丸ごと skip
- **G4**: Bridge Attention (25 層 hidden 抽出) を完全温存し、VLA-Adapter 設計思想を継承
- **G5**: `batch_size` を現 8 → 16+ に拡大可能にする throughput/memory 改善
- **G6**: Hackathon 期限 (2026-05-18) までに LIBERO eval 1 run を完了

## 3. Non-Goals (今回スコープ外)

- **Flow Matching / DiT action head 化** — 魅力的だが Hackathon 期間に収束検証不可。後続 Phase で検討。現行 `MLPResNetBlock_Pro` + L1 regression を継続。
- **AdaLN-Zero / timestep conditioning** — FM を入れないので不要。
- **Cross-embodiment multi-dataset pretrain** — Fractal 1cam 問題 + `multi_dataset_loader` が看板倒れ状態。Taco Play 単独 pretrain に縮退。
- **Gemma 4 E2B 以外の model variant 探索** — E2B で継続。
- **proprio zero input (X-VLA cross-embodiment pattern)** — 単アーム単 embodiment で proprio 不使用はメリットゼロ、proprio 有効化。
- **Fractal 等 1cam dataset の活用** — 保留、Hackathon 後に戦略検討。

## 4. Architecture Overview

### 4.1 Module inventory

| Module | 役割 | State | Params |
|---|---|---|---|
| `Gemma4VisionModel` | Scene 画像 → 140 or 280 soft tokens | **frozen** | 151M |
| `multi_modal_projector` (Gemma 4 内蔵) | vision hidden 768 → llm 1536 | **frozen** | 小 |
| `Gemma 4 LM (Gemma4TextModel)` | multi-layer hidden 抽出 (Bridge 供給源) | **frozen** + `torch.no_grad()` wrap | 2.0B |
| `ResNet18` | Wrist 画像 → 7×7×512 feature map | **trainable** (ImageNet init) | 11.7M |
| `WristProjector` | 512 → 1536 (llm_dim) | **trainable** | 1.2M |
| `SoftPromptLibrary` | Dataset 条件付け 32 tokens × 1536 | **trainable**、**action head 5 本目に接続** (X-VLA 原実装位置) | 約 0.1M / 2 datasets |
| `ProprioProjector` | proprio 8 dim → 1536 | **trainable** | 小 |
| `ActionHead` (`MLPResNetBlock_Pro` × 24) | 5 stream cross-attn + L1 regression | **trainable** | 現状 675M ベース |

合計 trainable: **~807M** (現状 675M − film_gen dead code 113M + 新 K/V projections 227M + wrist 系 13M + soft prompt 0.1M)。詳細は §8.1。

### 4.2 Data flow

```
  scene_img (224×224×3)
        ▼
  [Gemma4VisionModel (FROZEN)] ──▶ 140 or 280 soft tokens × 768
        ▼
  [multi_modal_projector (FROZEN)] ──▶ 140 or 280 tokens × 1536
                                              │
  text_tokens (20) ───────────────────────────┤
  [Gemma 4 embed (FROZEN)]                    │
                                              ├── concat ──▶ embeddings (input to LLM)
  action_placeholders (64)  ──────────────────┤
  [Gemma 4 embed (FROZEN)]                    │
                                              │
  (soft_prompt は LLM に入れない、action head 直接へ)
                                              ▼
                        ┌──────────────────────────────────────┐
                        │   Gemma 4 LM (FROZEN)                │
                        │   with torch.no_grad():              │
                        │     out = llm(inputs_embeds=...)     │
                        │                                      │
                        │   → out.hidden_states[0:25] (25 層)  │
                        │   → forward activation 保存なし      │
                        │   → backward 丸ごと skip             │
                        └──────────────────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
   hidden @ vision positions                  hidden @ action_placeholder positions
     → h_t (B, 25, 140 or 280, 1536)            → h_a (B, 25, 64, 1536)
     Bridge: scene semantic 階層                 Bridge: action placeholder 階層


  wrist_img (224×224×3)
        ▼
  [ResNet18 (TRAINABLE)] ──▶ (B, 512, 7, 7)
        ▼
  reshape ──▶ (B, 49, 512)
        ▼
  [WristProjector: Linear 512→1536] ──▶ h_w (B, 49, 1536)


  proprio (8 dim)
        ▼
  [ProprioProjector] ──▶ p (B, 1, 1536)


  dataset_id (0 or 1) ──▶ [SoftPromptLibrary] ──▶ h_sp (B, 32, 1536)


  ┌──────── Action Head (MLPResNetBlock_Pro × 24) ────────┐
  │  x (action latent: zero-init + random perturbation)    │
  │                                                        │
  │  各 block で Q = q_proj(x)                             │
  │                                                        │
  │  5-stream cross-attention (concat → softmax):         │
  │    stream 1: self       K/V = k_self(x), v_self(x)    │
  │    stream 2: adapter    K/V = k_adapter([h_a, p])     │
  │    stream 3: task       K/V = k_task(h_t), × ratio_g  │
  │    stream 4: wrist ★    K/V = k_wrist(h_w), × gate_w  │
  │    stream 5: soft_prompt ★ K/V = k_sp(h_sp), × gate_sp │
  │                                                        │
  │  output = o_proj(concat-softmax attention)             │
  │  x = FFN(output + x)   (residual)                      │
  │                                                        │
  │  (24 blocks 繰り返し)                                  │
  └────────────────────────────────────────────────────────┘
                    │
                    ▼
        predicted action chunk (8 steps × 7 dim)
                    │
                    ▼
        L1 loss vs action target
```

### 4.3 LLM Forward: torch.no_grad() 化の前提

G3 (LLM backward skip) は **LLM input source がすべて frozen** であることが前提。この前提は新アーキで成立する:

- `scene_vision_tokens`: `Gemma4VisionModel` (frozen) + `multi_modal_projector` (frozen) の出力 → frozen source
- `text_tokens`: Gemma 4 の embedding layer (frozen) 出力 → frozen source
- `action_placeholder_embeddings`: 同上 → frozen source
- `soft_prompt`: **LLM input に入れない** (action head 内へ移動) → LLM 経路に trainable source なし

したがって `torch.no_grad()` で LLM forward を wrap して問題ない:

```python
with torch.no_grad():
    out = llm(inputs_embeds=embeddings, output_hidden_states=True, ...)
hidden_states = out.hidden_states  # 自動的に detached
```

副次効果として forward 時に activation を保存しない (`torch.no_grad()` は autograd グラフを作らない) ため、activation memory (batch size 制約の主犯) が大幅に解放される。

## 5. Detailed Design

### 5.1 Scene Path (Gemma 4 Native Vision)

- Model load: `Gemma4ForConditionalGeneration.from_pretrained("google/gemma-4-E2B", dtype=torch.bfloat16)` に変更 (現行 `AutoModelForCausalLM` からの変更)
- `vision_tower` + `multi_modal_projector` + `language_model` が取り出せる
- 全 submodule を `requires_grad=False`
- Soft tokens: processor で `max_soft_tokens ∈ {140, 280}` を選択 (A/B 項目、§9.1 参照)
- LLM 入力形式: Gemma 4 multimodal の標準 chat template または独自の `inputs_embeds` 合成 (既存 `modeling_prismatic_gemma4.py` パイプラインを踏襲しつつ DinoSigLIP 部分を native vision 出力で置換)

### 5.2 Wrist Path (ResNet18 → Action Head)

- Model: `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`
- 最終 FC (分類 head) を捨て、**最後の conv block の出力 `(B, 512, 7, 7)` を採取**
- Global Average Pool **をかけない** (空間情報保持)
- Flatten: `(B, 512, 7, 7) → (B, 49, 512)` (`rearrange("b c h w -> b (h w) c")`)
- Projection: `nn.Linear(512, 1536)` → `h_w (B, 49, 1536)`
- 学習: `requires_grad=True`, bf16, ImageNet init からスタート

### 5.3 Action Head: 5-Stream Cross-Attention

現 `MLPResNetBlock_Pro` (`VLA-Adapter/prismatic/models/action_heads.py:287-410`) を 3 stream → 5 stream に拡張。

追加する stream ごとに独立の K/V projection を block 内に新設:

```python
# 既存: self.k_self, self.v_self, self.k_adapter, self.v_adapter, self.k_task, self.v_task
# 追加:
self.k_wrist = nn.Linear(dim, dim)
self.v_wrist = nn.Linear(dim, dim)
self.k_sp    = nn.Linear(dim, dim)
self.v_sp    = nn.Linear(dim, dim)

# gating parameters (zero init, LoRA 流)
self.gating_wrist      = nn.Parameter(torch.zeros(1))  # 新規
self.gating_soft_prompt = nn.Parameter(torch.zeros(1)) # 新規
# 既存の gating_factor (= task stream の ratio_g) は維持
```

`forward` での attention score 合成を 3 → 5 に拡張。併せて **現存する dead code `film_gen` (`nn.Linear(dim, dim*2)`) と `apply_film` メソッドを削除** (forward で既に全コメントアウト済、外部参照なし、60k ckpt 破棄で chkpt 互換縛りもなし、削除で block あたり 4.72M × 24 = 113M params の死重を除去):

```python
attn_scores = [
    torch.matmul(q_1, k_self.transpose(-2, -1)),
    torch.matmul(q_1, k_adapter.transpose(-2, -1)),
    torch.matmul(q_1, k_task.transpose(-2, -1))   * torch.tanh(self.gating_factor),
    torch.matmul(q_1, k_wrist.transpose(-2, -1))  * torch.tanh(self.gating_wrist),        # NEW
    torch.matmul(q_1, k_sp.transpose(-2, -1))     * torch.tanh(self.gating_soft_prompt),  # NEW
]
attn_weights = torch.softmax(torch.cat(attn_scores, dim=-1) / math.sqrt(head_dim), dim=-1)
v_combined = torch.cat([v_self, v_adapter, v_task, v_wrist, v_sp], dim=2)
output = torch.matmul(attn_weights, v_combined)
```

- gating は zero init (`tanh(0) = 0`) で初期は完全ミュート、学習で徐々に有効化
- wrist / soft_prompt 各 block で独立 scalar、block ごとに違う重要度を学べる

`predict_action` (`action_heads.py:43-80`) のシグネチャに `h_w`, `h_sp` を追加し、各 block の forward 呼び出しで渡す。

### 5.4 SoftPromptLibrary Relocation

- **クラス本体**: `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py:41-63` の `SoftPromptLibrary` はそのまま (`nn.Embedding(num_datasets, 32 × 1536)`)
- **呼び出し位置変更**:
  - Before: LLM `inputs_embeds` の先頭に concat (`modeling_prismatic_gemma4.py:222-229`)
  - After: `VLAAdapterGemma4.forward()` 内で `soft_prompt = self.soft_prompt_library(dataset_id)` を計算 → action head の `predict_action` に `h_sp=soft_prompt` 引数で渡す
- LLM input 側では soft_prompt を完全に削除 (attention_mask / position_ids の L_total から `num_sp` を除く)
- Bridge 抽出位置 (`vmask`, `amask`) の soft_prompt offset 計算 (`:251-252`) は削除

### 5.5 Proprio (Enabled)

- Stage 3 pretrain (Taco) と Stage 2 fine-tune (LIBERO) で **両方とも proprio を使用**
- `ProprioProjector` は現行のまま (`modeling_prismatic_gemma4.py:80-90`)
- 単アーム Franka で Taco / LIBERO の proprio format 互換性は実装時に確認:
  - Taco Play: OXE spec の `observation.robot_obs` など (8 dim or 15 dim、episode 1 個抜いて確認要)
  - LIBERO: `observation.robot_state` (8 dim 想定)
- 不整合があれば canonical 8 dim (xyz + axis-angle 3D + gripper 1 + optional 1) に正規化、`multi_dataset_loader.py` の `proprio_zeros` 置換

### 5.6 Bridge Attention (Unchanged)

- `modeling_prismatic_gemma4.py:246-256` の処理をそのまま維持
- ただし:
  - `all_hidden` は `torch.no_grad()` 下で取得 (detached)
  - 位置マスク (`vpos0`) は native vision tokens 位置 (L_total = 140/280 + 20 + 64)
  - `vision_hidden`: (B, 25, 140/280, 1536)
  - `action_hidden`: (B, 25, 64, 1536)
- action head 側 (`L1RegressionActionHead.predict_action`) は既存の `h_t / h_a` 分離 (`action_heads.py:57-58`) をそのまま利用

## 6. Training Pipeline

### 6.1 Pretrain (Stage 3 redesign)

- **Dataset**: Taco Play のみ (OXE 2cam: `rgb_static` + `rgb_gripper`)
- **Data loader**: `scripts/stage3/multi_dataset_loader.py` を改修 or 新規 `scripts/stage3/taco_solo_loader.py` (Fractal 関連削除)
- **Batch size**: 新アーキで最大化 (§7 参照)
- **Steps**: 60k-100k (経験的に Taco で saturate する step 数を smoke で確認)
- **Optimizer**: AdamW (現状維持)
- **LR schedule**: X-VLA 準拠 (freeze_steps + warmup、現 `finetune_gemma4.py` を継続)
- **SoftPromptLibrary**: `num_pretrain_datasets=1` (Taco のみ) で有効、`dataset_id=0` 固定
- **Proprio**: 有効化 (Taco の proprio を正規化して使用)

### 6.2 Fine-tune (Stage 2 redesign)

- **Dataset**: LIBERO-90 (standard)
- **Data loader**: 既存 `Gemma4RLDSDataset` を新 model シグネチャに合わせ改修
- **SoftPromptLibrary**: disable (`num_pretrain_datasets=0`)、または Taco pretrain の soft_prompt weights を初期値とし fine-tune で更新する選択肢 (未決、§9.2 参照)
- **Proprio**: 有効化
- **Steps**: LIBERO 標準プロトコル (2k-10k step)

### 6.3 Checkpoint disposition

- 60k step Stage 3c-0 checkpoint は **破棄**
- 新アーキで step 0 からやり直し (不可避)

## 7. Throughput Optimization Integration

§2 ロードマップ "B案" (アーキ先、sanity 後に最適化) を踏襲。

### 7.1 Phase 0: Architecture Implementation

1. 新アーキ実装 (scene = Gemma4 native, wrist = ResNet18, 5-stream action head, soft prompt relocation, LLM `torch.no_grad()`)
2. Existing `batch_size=8`, `attn_implementation="sdpa"`, `GC=off` で smoke test (1k step)
3. Loss 減少確認、NaN なし確認、parameter count / gradient flow sanity 確認

### 7.2 Phase 1: FA-2 Enable

1. `pip install flash-attn --no-build-isolation`
2. `AutoModelForCausalLM → Gemma4ForConditionalGeneration` load 時に `attn_implementation="flash_attention_2"`
3. 1 step forward + backward が通るか smoke 確認
4. Loss curve が Phase 0 と同等か diff 確認 (FA-2 の数値差で regression ないか)

### 7.3 Phase 2: GC Verification

1. `transformers 5.5.4` で Gemma 4 GC が動くか確認 (HF PR #45312 含まれる version。R6 の "GC 永久禁止" comment は outdated の可能性)
2. `gemma.gradient_checkpointing_enable()` を有効化して 10 step 動作確認
3. loss curve が一致するか diff 確認
4. **注意**: LLM は `torch.no_grad()` 化により activation memory を既に解放しているため、GC は本来不要な可能性。action head 側のみ GC を検討 (action head は 24 block あり activation memory 非零)

### 7.4 Phase 3: Batch Size Expansion

1. Phase 0-2 で確保された memory 余裕を元に `batch_size` を 8 → 16 → 24 と拡大
2. 各 batch size で 100 step throughput (samples/sec) と memory peak を測定
3. OOM 手前で決定、`grad_accumulation_steps` を合わせて effective batch を所望値に

### 7.5 Phase 4: (Optional) torch.compile

- Phase 0-3 で throughput 不足なら `torch.compile` を action head に適用 (LLM は no_grad 化済なので compile 恩恵小)
- graph break が多発する構造 (5-stream cross-attn 内の動的 concat 等) でベンチ結果次第、deferred

## 8. Parameters Summary

### 8.1 Trainable (≈ 807M)

| Group | Params | 備考 |
|---|---|---|
| `action_head` base (MLPResNetBlock_Pro × 24, 3-stream) | ~675M | 現状 `L1RegressionActionHead(input_dim=1536, hidden_dim=1536, use_pro_version=True)` 実測 (assertion 値 675.138M) |
| `action_head` **film_gen 削除分** | **−113M** | `Linear(1536, 3072)` × 24 blocks、forward で全コメントアウト済の dead code |
| `action_head` 5-stream 拡張分 (new K/V projections) | **+227M** | 4 new `Linear(1536,1536)` per block × 24 blocks = 4 × 2.36M × 24 ≈ 226.7M |
| `soft_prompt_library` | 0.1M | num_datasets=1 (Taco) × 32 × 1536 |
| `wrist_resnet18` | 11.7M | ImageNet init |
| `wrist_projector` | 1.2M | 512 → 1536 |
| `proprio_projector` | ~2.4M | 現状維持 (2 層 MLP: 8 → 1536 → 1536) |

**Net action head**: 675 − 113 + 227 = **789M**
**Total trainable**: **~807M**

**Note on +227M − 113M = +114M 純増**: action head の 4 new Linear (k_wrist, v_wrist, k_sp, v_sp) × 24 blocks が追加コスト、同時に死重 film_gen を除去して相殺。最適化対象として妥当なバランス。最適化時の optimizer state memory (AdamW だと params × 8 byte × 2 ≈ 1.8 GB 増) を意味する。さらに削減したい場合は low-rank 化 (bottleneck 128 dim 経由で new K/V を 227→38M) や block 共有 K/V (encoder-decoder cross-attn 流、227→9M) が選択肢だが、Hackathon では full dim で実装、必要なら後続 phase で ablation。

### 8.2 Frozen (≈ 2.15B、概算)

| Group | Params |
|---|---|
| `Gemma 4 LM (text_model)` | ~2.0B (E2B 表記、PLE / matryoshka 含む) |
| `Gemma 4 VisionModel` | ~151M (hidden 768, 16 layers) |
| `Gemma 4 multi_modal_projector` | 小 |

## 9. Open Questions (Implementation で決定)

### 9.1 Soft tokens: 140 vs 280 (A/B test)

- **判定方法**: Phase 0 smoke 後に 2 run 並列、5k step ずつ loss / validation action error を比較
- **候補**:
  - 140: より aggressive 圧縮、seq 長 224 → attention 負荷小
  - 280 (default): 標準設定、semantic 情報量 multi
- **デフォルト**: 280 (選定根拠が出るまで)

### 9.2 LIBERO fine-tune での soft_prompt_library 扱い

- **選択肢 A**: fine-tune で disable (`num_pretrain_datasets=0`)、Taco で学んだ soft prompt は使わない
- **選択肢 B**: Taco の soft prompt (dataset_id=0) を fine-tune 開始時の初期値として採用、fine-tune 中に更新
- **保留**: Phase 0 実装時に Taco pretrain の soft prompt が意味のある重みになっているか確認してから決定

### 9.3 Proprio format alignment (Taco ↔ LIBERO)

- 実装時に両 dataset の proprio 仕様を確認 (tfds episode 1 個を手元で dump)
- 不整合があれば canonical 8 dim (xyz + axis-angle + gripper) に正規化
- `multi_dataset_loader.py` → `taco_solo_loader.py` でこの正規化を実装

### 9.4 LIBERO fine-tune における ResNet18 初期化

- Taco pretrain 終了時の ResNet18 weights を LIBERO fine-tune の初期値として採用 (推奨)
- Or: fine-tune で ImageNet init からやり直し (pretrain 成果活かせない、非推奨)

## 10. Implementation Phasing

Hackathon 2026-05-18 までの 4 週間で以下の順で進める:

| Week | Phase | 成果物 |
|---|---|---|
| Week 1 (-04-29) | **Phase 0**: 新アーキ実装 | 新 model class、data loader 改修、smoke test pass |
| Week 2 (-05-06) | **Phase 1-2**: FA-2 + GC 検証、 **Phase 3**: batch 拡大、**Taco pretrain 開始** | FA-2 有効、batch_size=16+、Taco 20k step 到達 |
| Week 3 (-05-13) | **Taco pretrain 継続 + LIBERO fine-tune 開始**、soft tokens A/B | pretrain 80k+、LIBERO run 1 開始 |
| Week 4 (-05-18) | **LIBERO eval + 成果まとめ** | Hackathon 提出 |

## 11. Risk Register

| Risk | Probability | Mitigation |
|---|---|---|
| Gemma 4 E2B の native vision が実は weights 未配布 | Low | `Gemma4ForConditionalGeneration.from_pretrained` で vision_config 確認済 (§2 調査で vision hidden 768/16 layers 確認、pretrained weights ロード可能を仮定) — 実装時に full load で verify |
| Gemma 4 FA-2 互換性バグ | Low | HF 5.5.4 で `_supports_flash_attn=True` 宣言済、smoke で確認 |
| `torch.no_grad()` 下 LLM forward が `output_hidden_states` と干渉 | Low | hidden_states は普通の return tensor、`no_grad` は autograd 非追跡のみ、forward 値に影響なし |
| Taco Play の proprio format が想定と違う | Medium | §9.3 対応で canonical 正規化 |
| 60k ckpt 破棄で学習やり直しが Hackathon に間に合わない | Medium | Week 1 Phase 0 smoke までに問題発生あれば P3 (pretrain skip, LIBERO 直行) にフォールバック検討 |
| Wrist ResNet18 の学習が収束しない (gating zero-init のまま動かない) | Low | gating は tanh なので grad は 0 位置でも流れる、初期 step で学習開始される。念のため smoke で gating_wrist の値推移を monitor |
| action head の 5-stream 拡張で既存 L1 regression の収束性が劣化 | Low | gating zero-init で wrist/soft_prompt の影響は初期ゼロ、従来 3-stream と数値一致で起動可能 |

## 12. Files to Modify (Implementation Surface)

- `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py`: model wrapper 全面改修 (scene path native vision 化、soft prompt relocation、LLM no_grad、wrist 注入)
- `VLA-Adapter/prismatic/models/action_heads.py`: `MLPResNetBlock_Pro` 3 → 5 stream 拡張、gating 追加、`predict_action` signature 変更、**併せて dead code `film_gen` / `apply_film` を削除** (113M dead weight 除去)
- `VLA-Adapter/vla-scripts/finetune_gemma4.py`: model build / dataloader / forward 呼び出し / trainable param 列挙 を新構造に対応
- `scripts/stage3/multi_dataset_loader.py`: → `taco_solo_loader.py` 新規 (Fractal 削除、proprio 正規化、2cam で rgb_static + rgb_gripper)
- `scripts/gemma4/test_08_data_pipeline.py`: LIBERO loader を新 signature に合わせ調整
- `scripts/gemma4/test_06_full_forward.py` 等 smoke test: 新アーキ対応
- 新規: wrist 用 ResNet18 wrapper class (e.g. `VLA-Adapter/prismatic/models/backbones/vision/wrist_resnet18.py`)

## 13. Summary of Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Scene encoder: Gemma 4 native vision (frozen) | ✅ |
| 2 | Wrist encoder: ResNet18 (ImageNet init, trainable), 7×7×512 = 49 tokens | ✅ |
| 3 | Wrist injection: action head 4-th cross-attn stream, gating with zero-init | ✅ |
| 4 | SoftPromptLibrary: relocate from LLM input to action head 5-th cross-attn stream | ✅ |
| 5 | LLM forward: `torch.no_grad()` で backward skip | ✅ |
| 6 | Bridge Attention (25 層): 維持 | ✅ |
| 7 | action_placeholders (64 tokens): 維持 | ✅ |
| 8 | action head base: `MLPResNetBlock_Pro` × 24, L1 regression (no FM/AdaLN) | ✅ |
| 9 | Pretrain strategy: Taco Play 単独 (P1) | ✅ |
| 10 | Fine-tune: LIBERO (Stage 2 protocol) | ✅ |
| 11 | Proprio: 有効化 | ✅ |
| 12 | Optimization rollout: B案 (アーキ先 → FA-2 → GC → batch 拡大) | ✅ |
| 13 | 60k checkpoint: 破棄 | ✅ |
| 14 | Scene soft tokens 140 vs 280: A/B test (Phase 0 後) | Open |
| 15 | LIBERO での soft_prompt_library 扱い | Open |
| 16 | `MLPResNetBlock_Pro` の dead code (`film_gen` / `apply_film`) 削除 | ✅ |
