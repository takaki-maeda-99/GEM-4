# Gemma 4 E2B 移植 Stage 1 / Phase 1c + 1d 実装計画 (v1.0)

**対象**: Phase 1c (Backward batch scaling) + Phase 1d (LIBERO 10-step smoke train)
**前提**: Phase 1b 完了済み (全 6 step + Scope Investigation 達成、Check-in Point #5 確認済み)
**成功基準**: 10-step smoke train が走り、loss が finite かつ初期値から減少傾向 or 安定。精度は問わない。

---

## v1.0 で確定した前提 (Phase 1b から継承)

### 確定済み数値

```
Gemma 4 E2B 構造:
  hidden_size=1536, num_hidden_layers=35, num_kv_shared_layers=20
  sliding_window=512 (4 local + 1 global pattern, E2B-specific)
  global layers: 4, 9, 14, 19, 24, 29, 34 (7 個)
  action head 受容野 = entries 1-24 (= layer 0-23 の出力)
  → 受容野内の global 層は 4 個 (layer 4, 9, 14, 19)
  → layer 24 の出力 (entry 25) は受容野外

メモリ (Phase 1b.6 実測, B=1, L=591):
  forward peak: 14.97 GB
  backward peak: 15.78 GB
  trainable: 675.14M (action_head 639.89M + vision_proj 32.78M + proprio_proj 2.37M + action_queries 0.10M)

学習対象 (Stage 1 凍結前提):
  frozen: Gemma 4 LLM (5104M) + DINO+SigLIP (731M) = 合計 5835M
  trainable: 675.14M
```

### 確定済み実装 pattern (v5.2 から)

- Module 階層: `model.model.language_model.*`
- PLE: `get_per_layer_inputs(input_ids, None)` 事前計算 → `per_layer_inputs` と `inputs_embeds` を一緒に渡す
- Placeholder IDs: action 258885-258948、vision 258949-259460、proprio 259461
- Overwrite: Option A (clone + advanced indexing)
- `use_cache=True` 必須、`gradient_checkpointing` 禁止 (KV 共有バグ HF #45242)
- `attn_implementation="sdpa"` (flash-attn は 1c で要否判定)
- `feature_norm=Identity` (1b.1 std 測定結果)
- Action head 入力: entries 0-24 × (vision 512 + action 64) = (B, 25, 576, 1536)

### 既存成果物 (1b 完了時)

- `VLA-Adapter/prismatic/vla/constants_gemma4.py`
- `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` (VLAAdapterGemma4)
- `scripts/gemma4/test_01_load.py` 〜 `test_06_full_forward.py`
- `requirements-gemma4.txt` (127 packages、1c/1d で数個追加予定)

### 4 つの罠と対処 (Scope Investigation 済)

| # | 罠 | 対処 | 1c/1d で再発可能性 |
|---|---|---|---|
| 1 | `Qwen2TokenizerFast` import path 変更 | `from transformers import Qwen2TokenizerFast` に置換 (3 ファイル完了) | 🟡 `finetune.py` にも同じ import があれば追加対応 |
| 2 | `dlimp` 未 install | git+ install 済 | 🟢 解決済み |
| 3 | `tensorflow_graphics` 未 install | pip install 済 | 🟢 解決済み |
| 4 | `unpack_tuple` が list 非対応 | `isinstance(..., (tuple, list))` に変更 | 🟢 解決済み |

---

## 環境 gap (1c 着手前の blocker、User install 済前提)

Check-in Point #5 後の環境確認で判明した追加 blocker:

| # | Blocker | 対処 | 状態 |
|---|---|---|---|
| B1 | `accelerate` 未 install | `uv pip install accelerate` | User install 済 |
| B2 | `peft` 未 install (1d で LoRA 使わないが import 経路に存在) | `uv pip install peft` | User install 済 |
| B3 | LIBERO データ `modified_libero_rlds/` 未 download | HF hub から snapshot_download | User download 中 |
| B4 | `dataset_statistics.json` | VLA-Adapter/outputs/LIBERO-Spatial-Pro/ から再利用 | 存在確認済 |

---

## GPU 運用方針 (1c/1d 共通)

Phase 1b 環境確認の `nvidia-smi topo -m` から:

```
GPU0 <-NV12-> GPU1     (A100 80GB PCIe、両方 0 MiB 空き)    ← Stage 1 使用
GPU2 <-NV12-> GPU3     (A100 40GB、空きあり)                ← 予備
GPU4-7                  (A100 40GB、22 GB 使用中 × 他ユーザー) ← 使用しない
```

**1c: 単 GPU (GPU 0) で実施**。batch=2/4/8 のメモリ線形性を確認。
**1d: 単 GPU (GPU 0) で開始**。まず smoke 完走を目指す。2 GPU DDP は Stage 2 以降。

`CUDA_VISIBLE_DEVICES=0` を全 test script で明示。

---

# Phase 1c: Backward batch scaling + Optimizer state 測定

**目的**: Phase 1b.6 (B=1) の memory を起点に、batch=2, 4, 8 で forward/backward/optimizer step のメモリスケーリングを測定。1d で使う batch size を**数値ベースで**決定する。

## 実装 (`scripts/gemma4/test_07_batch_scaling.py`)

```python
# CUDA_VISIBLE_DEVICES=0 で実行
import torch
from torch.optim import AdamW
# VLAAdapterGemma4 は Phase 1b.6 で作成済み
from prismatic.extern.hf.modeling_prismatic_gemma4 import VLAAdapterGemma4

# Model 構築 (1b.6 と同じ)
model_vla = VLAAdapterGemma4(
    gemma_model=gemma_model,
    vision_backbone=vision_backbone,
    feature_norm=torch.nn.Identity(),
).cuda()

# Optimizer (AdamW fp32 state、1d と揃える)
optimizer = AdamW(
    [p for p in model_vla.parameters() if p.requires_grad],
    lr=2e-4,
    weight_decay=0.01,
)

cases = [1, 2, 4, 8]  # batch sizes
results = []

for B in cases:
    # Dummy batch 構築
    pixel_values = torch.randn(B, 12, 224, 224, dtype=torch.bfloat16).cuda()
    # 1b.6 の build_input_ids を流用 (B に対応)
    input_ids = build_full_input_ids_batched(B=B, tok=tok, prompt="pick up the red cube")
    proprio = torch.randn(B, 8, dtype=torch.bfloat16).cuda()
    actions = torch.randn(B, 8, 7, dtype=torch.bfloat16).cuda()

    # Forward peak
    torch.cuda.reset_peak_memory_stats()
    predicted, loss = model_vla(pixel_values, input_ids, proprio, actions)
    fwd_peak = torch.cuda.max_memory_allocated() / 1024**3

    # Backward peak (optimizer step 前)
    loss.backward()
    bwd_peak = torch.cuda.max_memory_allocated() / 1024**3

    # Optimizer step (AdamW state が初回で allocate される)
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()
    optimizer.zero_grad()
    opt_peak = torch.cuda.max_memory_allocated() / 1024**3

    # 2 回目 step (state 再利用で安定値)
    predicted, loss = model_vla(pixel_values, input_ids, proprio, actions)
    loss.backward()
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()
    optimizer.zero_grad()
    step2_peak = torch.cuda.max_memory_allocated() / 1024**3

    results.append({
        "B": B, "fwd": fwd_peak, "bwd": bwd_peak,
        "opt_first": opt_peak, "opt_steady": step2_peak,
        "loss": loss.item(),
    })
    torch.cuda.empty_cache()

# Table print
print(f"{'B':>3} | {'fwd':>8} | {'bwd':>8} | {'opt_1st':>8} | {'opt_steady':>10} | loss")
for r in results:
    print(f"{r['B']:>3} | {r['fwd']:>6.2f}GB | {r['bwd']:>6.2f}GB | {r['opt_first']:>6.2f}GB | {r['opt_steady']:>8.2f}GB | {r['loss']:.4f}")
```

## 期待値と判定基準

| B | fwd 予想 | bwd 予想 | opt_steady 予想 | 合格条件 |
|---|---|---|---|---|
| 1 | ~15 GB (1b.6 再現) | ~16 GB | ~22 GB (+AdamW 5.4GB) | OOM なし |
| 2 | ~23 GB | ~25 GB | ~30 GB | OOM なし |
| 4 | ~38 GB | ~42 GB | ~48 GB | OOM なし |
| 8 | ~65 GB | ~72 GB | ~78 GB | 境界、OOM 可能性 |

**batch=4 まで OOM なしなら 1d は B=4** で開始。batch=8 が通るなら grad_accum=1、通らなければ grad_accum=2 で effective batch=8 を維持。

## 境界ケースと停止条件

- `opt_steady > 75 GB` (80 GB 枠の 94%) → 残り余裕なし、**そのバッチは不採用**
- OOM が特定 B で発生 → それ以上の B は不採用、その場で停止
- `opt_first > opt_steady + 3 GB` → AdamW state の一時増加が予想以上、要調査

## Exit Criteria

- [ ] B=1, 2, 4 で OOM なし (最低条件)
- [ ] 全 cases で loss が finite
- [ ] 1d で使用する batch size が数値に基づいて決定される
- [ ] `gemma4_migration_log.md` にスケーリング表を追記

## Check-in Point #6 (1c 完了後)

見せるもの:
- 4 cases × (fwd, bwd, opt_1st, opt_steady) のメモリ表
- 各 B での loss 値 (全 finite 確認)
- 1d で採用する batch size と grad_accum の提案

**→ User 確認後に 1d 着手許可**

---

# Phase 1d: LIBERO 10-step smoke train

**目的**: 実 LIBERO-Spatial データで `VLAAdapterGemma4` を 10-step 学習。Loss が finite かつ減少傾向 or 安定を確認。

3 サブフェーズに分割:
- **1d.a**: データパイプライン検証 (1 batch だけ load、shape と内容の sanity check)
- **1d.b**: 単 step 学習 (1 optimizer step、loss finite + LLM leak なし)
- **1d.c**: 10-step smoke train (最終形、loss トレンド記録)

---

## Phase 1d.a: データパイプライン検証

**目的**: RLDS dataloader から batch を取り、VLAAdapterGemma4 の forward signature に適合することを確認。1b までは dummy data、1d.a で初めて実データとの接続検証。

### 実装 (`scripts/gemma4/test_08_data_pipeline.py`)

```python
from prismatic.vla.datasets.rlds.dataset import make_interleaved_dataset
from prismatic.vla.datasets.rlds.oxe import make_oxe_dataset_kwargs_and_weights

# VLA-Adapter のパターンを流用、LIBERO-Spatial 単独で構築
# (finetune.py の該当箇所を参照)
dataset_kwargs = make_oxe_dataset_kwargs_and_weights(
    "libero_spatial_no_noops",
    data_root_dir="modified_libero_rlds",
    load_camera_views=("primary", "wrist"),  # 2 カメラ
    load_depth=False,
    load_proprio=True,
    load_language=True,
    action_proprio_normalization_type="bounds_q99",
)

dataset = make_interleaved_dataset(
    dataset_kwargs_list=[dataset_kwargs],
    ...
    batch_size=1,  # 1d.a は 1 batch だけ
)

# Iterate 1 回だけ
for batch in dataset.iterator():
    print("=== batch keys ===")
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
    break

# 想定 keys (VLA-Adapter の BatchTransform 出力):
#   observation.image_primary: (B, 224, 224, 3)
#   observation.image_wrist:   (B, 224, 224, 3)
#   observation.proprio:       (B, 8)
#   action:                    (B, 8, 7)   ← 8 chunk × 7 dim
#   task.language_instruction: (B,) の string list
```

### VLAAdapterGemma4 への forward 適合

Batch dict から forward 引数を構築:

```python
# 1. pixel_values: 2 カメラ分を stack → (B, 12, 224, 224)
pixel_primary = batch["observation.image_primary"].permute(0, 3, 1, 2)  # (B, 3, 224, 224)
pixel_wrist = batch["observation.image_wrist"].permute(0, 3, 1, 2)
# DINO+SigLIP は DINO RGB + SigLIP RGB の 6 ch × 2 camera = 12 ch
pixel_values = build_dino_siglip_input(pixel_primary, pixel_wrist)  # (B, 12, 224, 224)

# 2. input_ids: 言語 instruction + placeholders を組み立て (1b.6 の build_full_input_ids を拡張)
language = batch["task.language_instruction"]  # list of strings
input_ids = build_full_input_ids_batched(
    B=B, tok=tok, prompts=language
)  # (B, L)  ← L は batch 内 max、padding 必要

# 3. proprio
proprio = batch["observation.proprio"]  # (B, 8)

# 4. actions (target)
actions = batch["action"]  # (B, 8, 7)

# Forward
predicted, loss = model_vla(pixel_values, input_ids, proprio, actions)
print(f"loss: {loss.item():.4f}")
print(f"predicted shape: {predicted.shape}")
```

### 注意点

- `build_full_input_ids_batched` は Phase 1b.6 の hardcoded prompt から **batch 内可変 prompt** に拡張必要
- Padding token: `tok.pad_token_id` (Gemma 4 は 0)
- `attention_mask` で padding 部分を mask
- Prompt 長が batch 内で違う場合、**各行で placeholder の絶対位置が変わる** → `amask`, `vmask` の検出ロジックを batch 対応に (既に 1b.6 コードは batch loop 形式なので OK のはず)

### Exit Criteria

- [ ] 1 batch を load できる
- [ ] Batch dict のキーが想定どおり (`observation.image_primary`, `action` 等)
- [ ] `build_full_input_ids_batched` が batch 対応で動く
- [ ] VLAAdapterGemma4.forward が通る (loss finite)
- [ ] LLM に勾配 leak なし
- [ ] Peak memory が Phase 1c B=1 の測定値と一致 (±1 GB)

---

## Phase 1d.b: 単 step 学習

**目的**: 1d.a の forward に加えて **1 回の backward + optimizer.step**。AdamW state が初期化され、次 step でも loss が finite であることを確認。

### 実装 (`scripts/gemma4/test_09_single_step.py`)

1d.a のコードを流用 + optimizer:

```python
optimizer = AdamW(
    [p for p in model_vla.parameters() if p.requires_grad],
    lr=2e-4,
    weight_decay=0.01,
)

# 2 batch 連続処理
for step, batch in enumerate(dataset.iterator()):
    pixel_values, input_ids, proprio, actions = prepare_batch(batch, tok)

    # Forward
    predicted, loss = model_vla(pixel_values, input_ids, proprio, actions)
    print(f"Step {step}: loss={loss.item():.4f}")

    # Backward + step
    loss.backward()

    # LLM leak check (毎 step)
    llm_grad_leak = sum(
        1 for p in model_vla.llm.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert llm_grad_leak == 0, f"LLM grad leak: {llm_grad_leak} params have non-zero grad"

    optimizer.step()
    optimizer.zero_grad()

    if step >= 1:  # 2 step で止める
        break

assert torch.isfinite(loss).all(), "Loss is not finite"
```

### Exit Criteria

- [ ] 2 step 連続で loss が finite
- [ ] LLM に grad leak なし (2 step とも)
- [ ] AdamW state allocated 後の peak memory 記録
- [ ] Step 0 loss と Step 1 loss の差を記録 (減少 or 変動幅)

---

## Phase 1d.c: 10-step smoke train

**目的**: 1d.b を 10 step まで伸ばし、loss トレンドを記録。これが**Stage 1 Phase 1d の最終形**。

### 実装戦略: `finetune.py` からの差分で `finetune_gemma4.py`

VLA-Adapter 本家の `finetune.py` をコピーして `finetune_gemma4.py` を作成。以下を変更/コメントアウト:

| 箇所 | 変更内容 |
|---|---|
| Model 構築 | `OpenVLA` → `VLAAdapterGemma4` |
| Tokenizer | Qwen tokenizer → Gemma 4 tokenizer |
| Constants | `constants.py` → `constants_gemma4.py` |
| LoRA 関連 | 全部コメントアウト (Stage 1 は凍結のみ) |
| `gradient_checkpointing_enable()` | コメントアウト (KV 共有バグ回避) |
| Action stats | `dataset_statistics.json` は既存再利用 (`outputs/LIBERO-Spatial-Pro/`) |
| BatchTransform | 言語 instruction → placeholder 付き input_ids の構築ロジック追加 |
| `max_steps` | 10 に設定 |
| `save_steps` | 100 (= smoke では save しない) |

### 実装 (`scripts/gemma4/smoke_train_gemma4.py` または `finetune_gemma4.py` 経由)

```python
# pseudocode
for step in range(10):
    batch = next(dataloader)
    pixel_values, input_ids, proprio, actions = prepare_batch(batch, tok)

    predicted, loss = model_vla(pixel_values, input_ids, proprio, actions)
    loss.backward()

    # Grad norm clip (VLA-Adapter 慣習、発散対策)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model_vla.parameters() if p.requires_grad],
        max_norm=1.0,
    )

    optimizer.step()
    optimizer.zero_grad()

    # Logging
    print(f"Step {step}: loss={loss.item():.4f} grad_norm={grad_norm.item():.4f}")

    # NaN/Inf check
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"Loss became non-finite at step {step}")

# Final check: loss が初期値から 5% 以上減少、または安定 (±10% 範囲)
```

### 合格基準 (Stage 1 smoke の最終判定)

| 条件 | 合格 | 要調査 | 不合格 |
|---|---|---|---|
| Loss finite | 全 10 step | — | 1 step でも NaN/Inf |
| Loss トレンド | 初期比 5% 以上減少 | ±10% 変動 (停滞) | 発散 (2x 以上) |
| Grad norm | < 10 安定 | 10-100 | > 100 or 0 |
| LLM grad leak | 全 step 0 | — | 1 step でも leak |

**停滞 (要調査) の場合の切り分け**:
1. `feature_norm = LayerNorm(hidden_size)` に差し替えて再実行 (1b.1 の安全ネット)
2. LR を 2e-4 → 5e-5 に下げて再実行
3. それでも停滞 → causal attention 前提の限界 (Stage 2 で bidirectional 追加)

---

## Check-in Points (Phase 1c/1d 全体で 4 箇所)

| # | タイミング | 見せるもの | 次ステップの判断 |
|---|---|---|---|
| 6 | 1c 完了後 | batch scaling 表 (B=1,2,4,8 × fwd/bwd/opt) | 1d で使う batch size 確定 |
| 7 | 1d.a 完了後 | batch dict 構造、forward shape、peak memory | 1d.b 着手許可 |
| 8 | 1d.b 完了後 | 2 step loss + LLM leak check 結果 | 1d.c 着手許可 |
| 9 | 1d.c 完了後 | 10 step loss ログ、grad norm、合格判定 | **Stage 1 完了認定** |

境界ケース (loss 停滞、grad norm 異常等) は自動判断せず、User に報告して指示を仰ぐ。

---

## Escalation Policy

以下に該当したら**即停止して User 報告**:

1. Phase 1c で batch=1 でも OOM
2. Phase 1c で batch=2 が OOM (1d での実行不能)
3. Phase 1d.a でデータパイプラインが動かない (RLDS が壊れている / 依存 package 不足)
4. Phase 1d.a で batch dict のキーが想定と大幅に異なる
5. Phase 1d.b で 2 step 以内に loss が NaN/Inf
6. Phase 1d.b で LLM grad leak が発生
7. Phase 1d.c で 10 step 中に loss が発散 (初期値の 2 倍以上)
8. Phase 1d.c で grad norm が 100 超 or 0 に張り付く
9. 単一の問題で 2 時間以上ハマる
10. Stage 1 スコープ外に踏み込まないと解けない問題

---

## やってはいけないこと

- `gradient_checkpointing` を有効化 (KV 共有バグ直撃)
- LoRA を有効化 (Stage 1 スコープ外、Stage 2 で扱う)
- `dataset_statistics.json` を新規計算 (既存 `outputs/LIBERO-Spatial-Pro/` を再利用)
- DDP / FSDP を導入 (単 GPU で smoke 完走を優先)
- Loss が NaN で出たときの workaround (scaling tricks 等) を独断で入れる
- Check-in Point をスキップ
- `finetune.py` を大幅書き換え (差分で進めて、原本は保持)
- `build_full_input_ids_batched` の placeholder ID を勝手に変える (constants_gemma4.py 定義を厳守)

---

## Out of Scope (Stage 2 以降)

- LoRA 導入 / optimization
- Bidirectional attention patch (`use_bidirectional_attention=True` の検証)
- 論文値 (99.6%) への精度追求
- Multi-GPU DDP
- 学習 step の拡大 (> 10 step)
- LIBERO eval / rollout
- 実ロボット deployment

---

## Deliverables (Phase 1c/1d 完了時)

| # | ファイル | 生成タイミング |
|---|---|---|
| 1 | `scripts/gemma4/test_07_batch_scaling.py` | Phase 1c |
| 2 | `scripts/gemma4/test_08_data_pipeline.py` | Phase 1d.a |
| 3 | `scripts/gemma4/test_09_single_step.py` | Phase 1d.b |
| 4 | `scripts/gemma4/smoke_train_gemma4.py` | Phase 1d.c (or finetune_gemma4.py) |
| 5 | `VLA-Adapter/vla-scripts/finetune_gemma4.py` | Phase 1d.c (finetune.py からの差分) |
| 6 | 更新版 `requirements-gemma4.txt` | B1-B3 対応で accelerate, peft 追加 |
| 7 | `docs/gemma4_migration_log.md` に Phase 1c/1d の追記 | 各 Phase 完了時 |

---

## Stage 1 完了後の引き継ぎ事項

Phase 1d.c 合格 → Stage 1 完了。以下の成果が確定:

1. **Gemma 4 E2B backbone + VLA-Adapter 設計で LIBERO smoke train 成立**
2. **trainable 675M でメモリ X GB** (1c 測定値) → Stage 2 での LoRA 導入余地確認
3. **causal attention 制約下での loss トレンド** → Stage 2 bidirectional 導入効果の baseline

Stage 2 着手時の優先順位 (ハッカソン 5/18 残 ~3 週間):
1. LIBERO-Spatial で本番学習 (max_steps=200,000 or 時間制約で短縮版)
2. 自前データ収集 (iPhone teleop、1-2 タスク、30-50 demo)
3. 自前データで fine-tune
4. Demo 動画作成

---

## 変更履歴

| Date | Version | 変更内容 |
|---|---|---|
| 2026-04-19 | v1.0 | 初版。Phase 1b 完了後の 1c/1d 実装計画。環境 blocker (accelerate/peft/LIBERO data) 対応を前提条件に含む |
