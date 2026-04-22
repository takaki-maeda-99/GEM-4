# VLA Redesign: Scene / Wrist Path Split + X-VLA Self-Attn Faithful Action Head

**Status**: Draft — pending user review (revised)
**Date**: 2026-04-22 (revised 2026-04-22)
**Target**: Hackathon 2026-05-18 (4 weeks)
**Checkpoint disposition**: 60k step Stage 3c-0 checkpoint will be discarded, fresh pretrain from step 0

---

## 1. Motivation

現 VLA-Gemma4 の学習は GPU util 90%+ の compute bound 状態で memory bound、`batch_size=8` から拡大不可。原因分析により 3 つの構造的損失が判明した:

1. **手元視点 (wrist camera) を frozen LLM に通している** — Gemma 4 は自然画像で pretrain、wrist POV は OOD。LLM は LoRA もかけず完全 frozen なので OOD 入力に適応する術がない。計算コストを払って semantic が取り出せていない。
2. **DinoSigLIP + `vision_projector` が冗長** — Gemma 4 E2B は native multimodal で自前の vision encoder (`Gemma4VisionModel`, 151M, LM と同時 pretrain 済) を持つ。外付け DinoSigLIP (400M) + 3 層 MLP projector は後付けで feature alignment を学習で獲得せねばならず本質的に loss。
3. **SoftPromptLibrary 配置が X-VLA 論文と乖離** — VLA-Adapter "案 B" で LLM input 前段に concat しているが、X-VLA 原実装コード (`2toinf/X-VLA/models/transformer.py`) では SoftPromptedTransformer の入力に concat、VLM は通さない。論文再現性と LLM forward コストの両面で deviation。

これら 3 つを同時に解消。**frozen LLM + Bridge Attention (25 層 hidden 抽出) という VLA-Adapter 核心思想は死守**、その上で X-VLA の soft_prompt 配置に戻す。

## 2. Goals

- **G1**: Wrist 視点を LLM 経路から外し、**action head の入力 token として x に concat** (X-VLA の auxiliary view 設計準拠)
- **G2**: Scene 視点を Gemma 4 native vision に切り替え、feature space alignment を pretrained 状態で獲得
- **G3**: SoftPromptLibrary を **action head の入力 token として x に concat** (X-VLA 原実装準拠)、LLM input からは除去
- **G4**: Bridge Attention (25 層 hidden 抽出) を完全温存し、VLA-Adapter 設計思想を継承
- **G5**: `batch_size` を現 8 → 16+ に拡大可能にする throughput/memory 改善 (**GC 必須**)
- **G6**: Hackathon 期限 (2026-05-18) までに LIBERO eval 1 run を完了

### LLM backward について (注記)

初案では「LLM input を全 frozen source にして `torch.no_grad()` で backward 丸ごと skip」を G3 に含めていたが、**`action_queries` (VLAAdapterGemma4 が保持する `nn.Embedding(64, 1536)`) が trainable で LLM input に書き込まれる** ため autograd が LLM backward を要求する。action_queries は Bridge の `action_hidden` 抽出位置を意味のあるものにするため必須 (zero 初期のまま freeze すると Bridge h_a が無意味な noise になる)。

よって **LLM backward は毎 step 実行、GC は必須**。加えて backward を払うなら LoRA の周辺コストがほぼ無料なので **LoRA 導入も現実的選択肢** (§5.8 参照)。

## 3. Non-Goals (今回スコープ外)

- **Flow Matching / DiT action head 化** — Hackathon 期間に収束検証不可。現行 `MLPResNetBlock_Pro` + L1 regression を継続。
- **AdaLN-Zero / timestep conditioning** — FM を入れないので不要。
- **Cross-embodiment multi-dataset pretrain** — Fractal 1cam 問題 + `multi_dataset_loader` が看板倒れ状態。Taco Play 単独 pretrain に縮退。
- **Gemma 4 E2B 以外の model variant 探索** — E2B で継続。
- **proprio zero input (X-VLA cross-embodiment pattern)** — 単アーム単 embodiment でメリットゼロ、proprio 有効化。
- **Fractal 等 1cam dataset の活用** — 保留、Hackathon 後。
- **action_queries の zero init + freeze** — LLM backward skip は不可、action_queries 学習維持。
- **action head の self-attn transformer 全面置換 (X-VLA 完全忠実 backbone)** — Bridge Attention (cross-attn to 25 層 LLM hidden) の構造と不整合。本 spec では MLPResNetBlock_Pro を残しつつ、soft_prompt と wrist を concat-to-x で self-attn pool 参加に近づける hybrid。

## 4. Architecture Overview

### 4.1 Module inventory

| Module | 役割 | State | Params |
|---|---|---|---|
| `Gemma4VisionModel` | Scene 画像 → 140 or 280 soft tokens (768-dim) | **frozen** | ~151M |
| `multi_modal_projector` (Gemma 4 内蔵) | vision hidden 768 → llm 1536 | **frozen** | 小 |
| `Gemma 4 LM (Gemma4TextModel)` | multi-layer hidden 抽出 (Bridge 供給源) | **frozen** | ~2.0B |
| `action_queries` (既存) | LLM input の action placeholder 位置に書き込む learnable embedding 64×1536 | **trainable** | ~0.1M |
| `ResNet18` (new) | Wrist 画像 → 7×7×512 feature map | **trainable** (ImageNet init) | 11.7M |
| `WristProjector` (new) | 512 → 1536 (llm_dim) | **trainable** | 1.2M |
| `SoftPromptLibrary` (配置変更) | Dataset 条件付け 32 tokens × 1536 | **trainable**、action head 入力に concat | ~0.05M (num_datasets=1) |
| `ProprioProjector` (既存) | proprio 8 dim → 1536 | **trainable** | ~2.4M |
| `ActionHead` (`MLPResNetBlock_Pro` × 24、**構造不変**) | x = [action_latent, wrist, soft_prompt] concat で self-attn pool 拡張 + 既存 cross-attn to Bridge | **trainable** | 現状 675M - film_gen 113M = **562M** |
| LoRA on Gemma 4 LM (optional、§5.8) | q/k/v/o_proj に r=16 adapter | **trainable** | ~4M |

合計 trainable: **~582M** (LoRA なし) / **~586M** (LoRA あり)。詳細は §8.1。

### 4.2 Data flow

```
  scene_img (224×224×3)
        ▼
  [Gemma4VisionModel (FROZEN)] ──▶ 140 or 280 soft tokens × 768
        ▼
  [multi_modal_projector (FROZEN)] ──▶ 140 or 280 tokens × 1536
                                              │
  text_tokens (20) ───────────────────────────┤
  [Gemma 4 embed (FROZEN)]                    ├── concat ──▶ embeddings (input to LLM)
                                              │
  action_placeholders (64 positions) ─────────┤
  [action_queries.weight (TRAINABLE)]         │
  を apos 位置に書き込み                      │
                                              ▼
                        ┌──────────────────────────────────────┐
                        │   Gemma 4 LM (FROZEN)                │
                        │   out = llm(inputs_embeds=...,       │
                        │             output_hidden_states=True)│
                        │   + GC 有効化で activation memory 削減 │
                        │                                      │
                        │   → out.hidden_states[0:25] (25 層)  │
                        └──────────────────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
   hidden @ scene_vision positions             hidden @ action_placeholder positions
     → h_t (B, 25, 140 or 280, 1536)            → h_a (B, 25, 64, 1536)


  wrist_img (224×224×3)
        ▼
  [ResNet18 (TRAINABLE)] ──▶ (B, 512, 7, 7)
        ▼
  reshape ──▶ (B, 49, 512)
        ▼
  [WristProjector: Linear 512→1536] ──▶ h_w (B, 49, 1536)


  proprio (8 dim) ──▶ [ProprioProjector] ──▶ p (B, 1, 1536)


  dataset_id (0 for Taco 単独) ──▶ [SoftPromptLibrary] ──▶ h_sp (B, 32, 1536)


  ┌──────── Action Head (MLPResNet + MLPResNetBlock_Pro × 24) ────────┐
  │                                                                    │
  │  action_latent x0 = zero-init + perturbation                       │
  │  x1 = fc1(layer_norm1(x0))                              (B, 8, 1536) │
  │                                                                    │
  │  x_ext = torch.cat([x1, h_w, h_sp], dim=1)             (B, 89, 1536) │
  │    ★ [action 8 | wrist 49 | soft_prompt 32]                         │
  │                                                                    │
  │  for i in range(24):                                               │
  │     block_i inputs:                                                │
  │       Q = q_proj(x_ext)                                            │
  │       Stream 1 (self): K,V = k_self(x_ext), v_self(x_ext)          │
  │         ↑ action ↔ wrist ↔ soft_prompt 全相互作用 (X-VLA-like)     │
  │       Stream 2 (adapter): K,V = k_adapter([h_a[:,i+1], p])         │
  │         ↑ x_ext 全 tokens が Bridge action hidden を cross-attn    │
  │       Stream 3 (task): K,V = k_task(h_t[:,i+1]) × ratio_g          │
  │         ↑ x_ext 全 tokens が Bridge scene hidden を cross-attn     │
  │                                                                    │
  │     concat softmax → output (B, 89, 1536)                           │
  │     + FFN + residual                                               │
  │                                                                    │
  │  x_final = x_ext[:, :8, :]   ★ action 位置だけ trim、他は捨てる    │
  │  x_final = fc2(layer_norm2(x_final))                    (B, 8, 7)   │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            predicted action chunk (8 × 7)
                                    │
                                    ▼
                            L1 loss vs action target
```

## 5. Detailed Design

### 5.1 Scene Path (Gemma 4 Native Vision)

- Model load: `Gemma4ForConditionalGeneration.from_pretrained("google/gemma-4-E2B", dtype=torch.bfloat16, attn_implementation="flash_attention_2")` に変更 (現行 `AutoModelForCausalLM` → multimodal クラス化)
- `vision_tower` + `multi_modal_projector` + `language_model` が取り出せる
- 全 submodule を `requires_grad=False` (LoRA 使う場合は LM の LoRA 挿入箇所だけ trainable)
- Soft tokens: processor で `max_soft_tokens ∈ {140, 280}` を選択 (§9.1、A/B test)
- LLM 入力形式: 既存 `modeling_prismatic_gemma4.py:205-216` の placeholder 上書きパイプラインを踏襲しつつ DinoSigLIP 部分を native vision 出力で置換

### 5.2 Wrist Path (ResNet18)

- Model: `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`
- 最終 FC (分類 head) を捨て、**最後の conv block の出力 `(B, 512, 7, 7)` を採取**
- Global Average Pool **をかけない** (空間情報保持)
- Flatten: `(B, 512, 7, 7) → (B, 49, 512)` (`rearrange("b c h w -> b (h w) c")`)
- Projection: `nn.Linear(512, 1536)` → `h_w (B, 49, 1536)`
- 学習: `requires_grad=True`, bf16, ImageNet init からスタート

### 5.3 Action Head: X-VLA Style Concat-to-x (Key Design Change)

**MLPResNetBlock_Pro の内部構造は現状維持 (K/V projection 追加なし)。変更は `MLPResNet.forward` の token 流れのみ**。

**実装スケッチ** (`action_heads.py` 相当の位置):

```python
# MLPResNet.forward (概略):
def forward(self, x, h_a, h_t, p, h_w=None, h_sp=None):  # h_w, h_sp を引数追加
    x = self.layer_norm1(x)
    x = self.fc1(x)                          # (B, 8, 1536)
    x = self.relu(x)

    # ★ concat: action | wrist | soft_prompt
    if h_w is not None:
        x = torch.cat([x, h_w], dim=1)       # (B, 8+49=57, 1536)
    if h_sp is not None:
        x = torch.cat([x, h_sp], dim=1)      # (B, 89, 1536)

    for i, block in enumerate(self.mlp_resnet_blocks):
        x = block(x, h_t=h_t[:, i+1], h_a=h_a[:, i+1], p=p)

    # ★ trim: action 位置だけ取り出す
    x = x[:, :NUM_ACTIONS_CHUNK, :]          # (B, 8, 1536)
    x = self.layer_norm2(x)
    x = self.fc2(x)                          # (B, 8, 7)
    return x
```

**MLPResNetBlock_Pro は現構造のまま** — q_proj, k_self, v_self, k_adapter, v_adapter, k_task, v_task, o_proj, FFN, gating_factor, RoPE。追加 K/V projection なし。

### 5.4 Action Head: Dead Code 削除

**`film_gen` + `apply_film` を削除** (`action_heads.py:320-334, 403-406`):

- forward で既に全コメントアウト済、dead code
- `Linear(1536, 3072)` × 24 blocks = **113M の dead weight を除去**
- 60k ckpt 破棄で state_dict 互換縛りもなし、安全に削除可能
- 外部参照なし (`FiLMedPrismaticVisionBackbone` は別系統、関係なし)

### 5.5 SoftPromptLibrary Relocation

- **クラス本体**: `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py:41-63` の `SoftPromptLibrary` はそのまま (`nn.Embedding(num_datasets, 32 × 1536)`)
- **呼び出し位置変更**:
  - Before: LLM `inputs_embeds` の先頭に concat (`modeling_prismatic_gemma4.py:222-229`)
  - After: `VLAAdapterGemma4.forward()` 内で `h_sp = self.soft_prompt_library(dataset_id)` を計算 → action head の `predict_action` に `h_sp=h_sp` 引数で渡す
- LLM input 側では soft_prompt を完全に削除 (attention_mask / position_ids の L_total から `num_sp` を除く)
- Bridge 抽出位置 (`vmask`, `amask`) の soft_prompt offset 計算 (`:251-252`) は削除

### 5.6 Proprio (Enabled)

- Stage 3 pretrain (Taco) と Stage 2 fine-tune (LIBERO) で **両方とも proprio 使用**
- `ProprioProjector` は現行のまま
- 単アーム Franka で Taco / LIBERO の proprio format 互換性は実装時に確認:
  - Taco Play: OXE spec の proprio (8 dim or 15 dim、episode 1 個抜いて確認要)
  - LIBERO: `observation.robot_state` (8 dim 想定)
- 不整合あれば canonical 8 dim (xyz + axis-angle 3D + gripper) に正規化、`multi_dataset_loader.py` の `proprio_zeros` 置換

### 5.7 Bridge Attention (Unchanged)

- `modeling_prismatic_gemma4.py:246-256` の処理維持、25 層 hidden 抽出
- 位置マスク (`vpos0`) は native vision tokens 位置 (soft_prompt を LLM 入力から除いたので offset 計算簡略化)
- `vision_hidden`: (B, 25, 140 or 280, 1536)
- `action_hidden`: (B, 25, 64, 1536)
- action head 側 (`L1RegressionActionHead.predict_action`) の既存 `h_t / h_a` 分離ロジックはそのまま

### 5.8 LoRA on Gemma 4 LM (optional、推奨)

LLM backward が必須になった以上、LoRA 追加の周辺コストはほぼゼロ:

| | LoRA なし | LoRA あり (r=16 on q/k/v/o_proj) |
|---|---|---|
| LLM forward | F | F + ε |
| LLM backward | F (action_queries grad 用) | F + ε (LoRA grad も同じ path) |
| trainable params | 582M | 582M + ~4M (LoRA) |
| optimizer state 増 | — | +50 MB 程度 |
| LLM 適応性 | 完全固定 | **robot 撮影分布に薄く適応、scene hidden quality 向上期待** |

- 対象 module: Gemma 4 LM の `self_attn.q_proj / k_proj / v_proj / o_proj` (MLP 層は Phase 0 では除外、シンプル優先)
- rank: r=16 (後で r=32 / MLP 追加の ablation)
- scaling α: 32 (r × 2 が標準)
- 実装: `peft` library の `LoraConfig` + `get_peft_model` で wrap、または手動で `LoRALinear` 実装
- Phase 0 smoke の時点で対照実験 (LoRA あり / なし) を回して loss diff を測る

### 5.9 RoPE Position for Concat Tokens (実装判断)

`MLPResNetBlock_Pro:381-386` の RoPE は `seq_len=T` で apply される。concat 後 T=89 になる:

- position 0-7: action latent tokens
- position 8-56: wrist tokens (49 個)
- position 57-88: soft_prompt tokens (32 個)

**選択**: **RoPE を一様に適用** (concat 後の 89 tokens 全部に position 0-88 を assign)。

**根拠**:
- X-VLA 原論文 (Section 4.1, Figure 10) は "Standard Self-attention Transformer Block" とのみ記載、positional encoding の扱いに specific な指定なし
- X-VLA 公式コード (`2toinf/X-VLA/models/transformer.py`) は PE を soft_prompt concat の前に加える実装判断をしているが、これは paper mandate ではなく実装選択
- 我々の RoPE 一様適用は X-VLA 実装と厳密には違うが **paper spec 違反ではない**
- 最小侵襲、action head block の RoPE ロジックを書き換えずに済む
- 気になる場合は ablation で positional encoding 条件分岐を追加可能 (§9 open question)

## 6. Training Pipeline

### 6.1 Pretrain (Stage 3 redesign)

- **Dataset**: Taco Play のみ (OXE 2cam: `rgb_static` + `rgb_gripper`)
- **Data loader**: `scripts/stage3/multi_dataset_loader.py` を改修 or 新規 `scripts/stage3/taco_solo_loader.py` (Fractal 関連削除、proprio 有効化)
- **Batch size**: §7 Phase 3 で Phase 0 実測 memory から決定 (16 目標、GC 前提)
- **Steps**: 60k-100k (Taco で saturate する step 数を smoke で確認)
- **Optimizer**: AdamW、LR schedule は X-VLA 準拠 (freeze_steps + warmup)
- **SoftPromptLibrary**: `num_pretrain_datasets=1` (Taco のみ) で有効、`dataset_id=0` 固定
- **Proprio**: Taco の proprio を正規化して使用
- **LoRA**: §5.8 で導入、r=16

### 6.2 Fine-tune (Stage 2 redesign)

- **Dataset**: LIBERO-90 (standard)
- **Data loader**: 既存 `Gemma4RLDSDataset` を新 model signature に合わせ改修
- **SoftPromptLibrary**: §9.2 open question (disable or Taco 重み初期値で継続学習)
- **Proprio**: 有効化
- **LoRA**: pretrain の LoRA 重みを初期値、fine-tune で継続更新
- **ResNet18**: pretrain 重みを初期値、fine-tune で継続更新
- **Steps**: LIBERO 標準プロトコル (2k-10k step)

### 6.3 Checkpoint disposition

60k step Stage 3c-0 checkpoint は **破棄**、新アーキで step 0 からやり直し。

## 7. Throughput Optimization Integration

**B案 (アーキ先、sanity 後に最適化) を踏襲**。ただし GC は Phase 0 から必須化 (LLM backward 確定のため):

### 7.1 Phase 0: Architecture Implementation

1. 新アーキ実装:
   - scene = Gemma4 native vision、`Gemma4ForConditionalGeneration` ロード
   - wrist = ResNet18 + projector、action head concat 経路
   - soft_prompt = action head concat、LLM input から削除
   - action head: film_gen 削除 + concat-to-x
   - LLM GC 有効化 (`model.text_model.gradient_checkpointing_enable()`)
2. `attn_implementation="flash_attention_2"` 有効化 (`pip install flash-attn`)
3. Existing `batch_size=8` で smoke test (1k step) → loss 減少確認、NaN なし、parameter count / gradient flow sanity

### 7.2 Phase 1: LoRA 導入 (optional、推奨)

1. Gemma 4 LM の q/k/v/o_proj に r=16 LoRA 挿入 (peft or 手動)
2. Phase 0 smoke 結果と diff 比較 (loss curve、wall time)
3. LoRA 有効時の効果を定量評価、採用判断

### 7.3 Phase 2: Batch Size Expansion

1. GC + FA-2 で確保された memory 余裕で `batch_size` を 8 → 16 → 24 拡大
2. 各 batch size で 100 step throughput (samples/sec) と memory peak 測定
3. OOM 手前で決定、`grad_accumulation_steps` で effective batch 調整

### 7.4 Phase 3: (Optional) torch.compile

- Phase 0-2 で throughput 不足なら `torch.compile` を action head に適用
- graph break 頻発する構造 (concat token length 変動、gating) で効果薄い可能性、ベンチ結果次第
- deferred (Hackathon 間に合わなくても致命的ではない)

## 8. Parameters Summary

### 8.1 Trainable (≈ 582M — LoRA なし基準)

| Group | Params | 備考 |
|---|---|---|
| `action_head` base (MLPResNetBlock_Pro × 24, 3-stream) | ~675M | 現状 `L1RegressionActionHead(input_dim=1536, hidden_dim=1536, use_pro_version=True)` 実測 (assertion 値 675.138M) |
| `action_head` **film_gen 削除分** | **−113M** | `Linear(1536, 3072)` × 24 blocks、forward で全コメントアウト済 dead code |
| `action_head` X-VLA-style concat 改修 | **+0M** | 新 K/V projection 追加なし、`MLPResNet.forward` の token 流れのみ変更 |
| `action_queries` (既存、保持) | 0.1M | `nn.Embedding(64, 1536)` |
| `soft_prompt_library` (配置変更、params 不変) | ~0.05M | num_datasets=1 (Taco) × 32 × 1536 |
| `wrist_resnet18` (new) | 11.7M | ImageNet init |
| `wrist_projector` (new) | 1.2M | Linear 512 → 1536 |
| `proprio_projector` (現状維持) | ~2.4M | 2 層 MLP: 8 → 1536 → 1536 |

**Net action head**: 675 − 113 = **562M** (concat 改修で +0M が肝)
**Total trainable (LoRA なし)**: **~582M**
**Total trainable (LoRA r=16, q/k/v/o)**: **~586M** (+4M)

**旧提案 (5 本目 cross-attn KV) からの削減**: 807M → 582M、**-225M** (新 K/V projection +227M を concat で回避)

### 8.2 Frozen (≈ 2.15B、概算)

| Group | Params |
|---|---|
| `Gemma 4 LM (text_model)` | ~2.0B (E2B 表記、PLE / matryoshka 含む) |
| `Gemma 4 VisionModel` | ~151M (hidden 768, 16 layers) |
| `Gemma 4 multi_modal_projector` | 小 |

## 9. Open Questions (Implementation で決定)

### 9.1 Soft tokens: 140 vs 280 (A/B test)

- **判定方法**: Phase 0 smoke 後に 2 run 並列、5k step ずつ loss / validation action error 比較
- **候補**: 140 (aggressive、seq 短縮) / 280 (default、標準品質)
- **デフォルト**: 280 (選定根拠が出るまで)

### 9.2 LIBERO fine-tune での soft_prompt_library 扱い

- **選択肢 A**: fine-tune で disable (`num_pretrain_datasets=0`)
- **選択肢 B**: Taco の soft prompt (dataset_id=0) を fine-tune 開始時の初期値として採用、fine-tune 中に更新
- **保留**: Phase 0 実装時に Taco pretrain の soft prompt が意味のある重みになっているか確認してから決定

### 9.3 Proprio format alignment (Taco ↔ LIBERO)

- 実装時に両 dataset の proprio 仕様を確認 (tfds episode 1 個を手元で dump)
- 不整合があれば canonical 8 dim (xyz + axis-angle + gripper) に正規化
- `multi_dataset_loader.py` → `taco_solo_loader.py` でこの正規化を実装

### 9.4 LIBERO fine-tune における ResNet18 初期化

- Taco pretrain 終了時の ResNet18 weights を LIBERO fine-tune 初期値として採用 (推奨)
- Or: fine-tune で ImageNet init からやり直し (pretrain 成果活かせない、非推奨)

### 9.5 LoRA 導入判断

- Phase 1 で LoRA あり / なしを smoke で比較
- LoRA あり側が明確に良ければ本番採用
- 差が不明瞭なら LoRA なしでシンプルに進行

### 9.6 RoPE positional encoding 扱い

- 現 spec: concat 後 seq に一様 RoPE 適用 (§5.9)
- 気になる場合の ablation: soft_prompt / wrist tokens に RoPE 適用しない条件分岐
- Phase 0 smoke で loss curve に異常なければそのまま進行

## 10. Implementation Phasing

Hackathon 2026-05-18 までの 4 週間:

| Week | Phase | 成果物 |
|---|---|---|
| Week 1 (-04-29) | **Phase 0**: 新アーキ実装 (concat-to-x、GC、FA-2、film_gen 削除) | 新 model class、data loader 改修、smoke test pass |
| Week 2 (-05-06) | **Phase 1-2**: LoRA 検証 + batch 拡大、**Taco pretrain 開始** | LoRA 判断、batch_size=16+、Taco 20k step 到達 |
| Week 3 (-05-13) | **Taco pretrain 継続 + LIBERO fine-tune 開始**、soft tokens A/B | pretrain 80k+、LIBERO run 1 開始 |
| Week 4 (-05-18) | **LIBERO eval + 成果まとめ** | Hackathon 提出 |

## 11. Risk Register

| Risk | Probability | Mitigation |
|---|---|---|
| Gemma 4 E2B の native vision が weights 未配布 | Low | config load 済 (vision_config 確認、hidden 768 / 16 layers) — 実装時に full load で verify |
| Gemma 4 FA-2 互換性バグ | Low | HF 5.5.4 で `_supports_flash_attn=True` 宣言済、smoke で確認 |
| GC が transformers 5.5.4 + Gemma 4 で動かない | Low-Med | PR #45312 含まれる version の想定、Phase 0 で確実に検証、NG なら `use_cache` 調整で対応 |
| LoRA 導入で既存学習済 weight と干渉 | Low | ImageNet init ResNet18 + Gemma 4 LoRA、互いに影響しない設計 |
| concat 後 seq=89 の self-attn が compute bottleneck | Med | T² O(89²)=7921 の self-attn、FFN は T=89 linear。block 全体で ~10x compute 増見込み。LLM forward (seq 300+) の方がまだ重い、許容範囲 |
| concat 後の RoPE 一様適用で soft_prompt/wrist が無意味な positional signal を受ける | Low | paper spec 違反ではない、モデルが学習で吸収する。気になれば §9.6 で ablation |
| Taco Play の proprio format が想定と違う | Med | §9.3 対応で canonical 正規化 |
| 60k ckpt 破棄で学習やり直しが Hackathon に間に合わない | Med | Week 1 Phase 0 smoke までに問題発生あれば P3 (pretrain skip, LIBERO 直行) にフォールバック検討 |
| action head の concat で既存 L1 regression の収束性が劣化 | Low | concat tokens は self-attn pool 参加、既存 action token の学習を破壊しない設計。Bridge cross-attn は block 単位で unchanged |

## 12. Files to Modify (Implementation Surface)

- `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py`: model wrapper 全面改修 (scene path native vision 化、soft_prompt relocation (LLM input から action head へ)、wrist 注入、LoRA wrap optional)
- `VLA-Adapter/prismatic/models/action_heads.py`:
  - `MLPResNet.forward` に `h_w`, `h_sp` 引数追加、fc1 直後で concat、fc2 直前で trim
  - `L1RegressionActionHead.predict_action` に `h_w`, `h_sp` 引数追加、MLPResNet 呼び出しに pass
  - **`film_gen` / `apply_film` 削除** (`MLPResNetBlock_Pro` 内 dead code)
  - `MLPResNetBlock_Pro` 本体は構造不変 (K/V projection 追加なし、gating 追加なし)
- `VLA-Adapter/vla-scripts/finetune_gemma4.py`: model build / dataloader / forward 呼び出し / trainable param 列挙 を新構造に対応、LoRA setup (optional)
- `scripts/stage3/multi_dataset_loader.py`: → `taco_solo_loader.py` 新規 (Fractal 削除、proprio 正規化、2cam で rgb_static + rgb_gripper)
- `scripts/gemma4/test_08_data_pipeline.py`: LIBERO loader を新 signature に合わせ調整
- `scripts/gemma4/test_06_full_forward.py` 等 smoke test: 新アーキ対応
- 新規: wrist 用 ResNet18 wrapper class (e.g. `VLA-Adapter/prismatic/models/backbones/vision/wrist_resnet18.py`)
- (optional) LoRA 設定: `peft` 経由 or 手動 `LoRALinear` 実装

## 13. Summary of Decisions

| # | Decision | Status |
|---|---|---|
| 1 | Scene encoder: Gemma 4 native vision (frozen) | ✅ |
| 2 | Wrist encoder: ResNet18 (ImageNet init, trainable), 7×7×512 = 49 tokens | ✅ |
| 3 | **Wrist injection: action head の concat-to-x** (X-VLA auxiliary view 設計準拠、自己 attention pool 参加) | ✅ |
| 4 | **SoftPromptLibrary: action head の concat-to-x** (X-VLA 原実装準拠、LLM input から削除) | ✅ |
| 5 | LLM forward/backward: **GC 必須で通常 backward** (torch.no_grad skip は action_queries 存在のため不可) | ✅ |
| 6 | Bridge Attention (25 層 cross-attn): 維持 | ✅ |
| 7 | action_queries (既存): trainable 保持 | ✅ |
| 8 | action head base: `MLPResNetBlock_Pro` × 24 構造不変、L1 regression (no FM/AdaLN) | ✅ |
| 9 | **action head の dead code `film_gen` / `apply_film` 削除** | ✅ |
| 10 | **action head の cross-attn stream 追加なし** (4本目 / 5本目 KV 案は撤回、concat で代替) | ✅ |
| 11 | Pretrain strategy: Taco Play 単独 (P1) | ✅ |
| 12 | Fine-tune: LIBERO (Stage 2 protocol) | ✅ |
| 13 | Proprio: 有効化 | ✅ |
| 14 | Optimization rollout: B案 (アーキ + GC 先 → LoRA → batch 拡大) | ✅ |
| 15 | 60k checkpoint: 破棄 | ✅ |
| 16 | Flash Attention 2: Phase 0 から有効化 | ✅ |
| 17 | Gradient Checkpointing: Phase 0 から必須化 (LLM backward 確定のため) | ✅ |
| 18 | RoPE: concat 後 seq に一様適用 (§5.9、paper spec 違反なし) | ✅ |
| 19 | LoRA (Gemma 4 LM の q/k/v/o_proj r=16): Phase 1 で検証、効果あれば採用 | Open (§9.5) |
| 20 | Scene soft tokens 140 vs 280: A/B test (Phase 0 後) | Open (§9.1) |
| 21 | LIBERO での soft_prompt_library 扱い | Open (§9.2) |
