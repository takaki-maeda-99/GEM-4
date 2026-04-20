# Claude Code 着手プロンプト: Gemma 4 E2B 移植 Stage 2 (v2)

## このプロンプトの扱い方

- Claude Code の**新規セッション冒頭**にこのファイル全文を貼る
- 実装計画の詳細は `gemma4_stage2_plan.md` (v2) に記載、これを**必ず最初に精読**
- Stage 1 の全履歴は `docs/gemma4_migration_log.md`、Stage 1 plan は `docs/gemma4_stage1_phase1c1d_plan.md`
- 作業は Check-in Point 駆動、着手前宣言・完了後報告、User 明示許可なき自動進行禁止

---

## Mission

Stage 1 完了 (4/4 軸 PASS) を起点に、以下 4 軸を達成してハッカソン提出 (2026-05-18) まで到達する:

1. LIBERO-Spatial で本番学習を完走、success rate を測定 (paper 99.6% との gap 分析付き)
2. 自前データ (teleop) で 1 タスク以上 fine-tune 成立
3. Demo 動画作成、ハッカソン提出
4. (条件付) Bidirectional attention 等の精度 ablation

論文精度の完全再現は scope 外。70-90% 帯で達成可、gap が analytical に説明できれば Mission 1 達成。

---

## 最初にやること (セッション開始時)

1. 作業ディレクトリ確認 (`/misc/dl00/takaki/vla-gemma-4/` を想定)
2. 以下のファイルを順に read:
   - `gemma4_stage2_plan.md` — 本タスクの詳細計画 (v2、必読)
   - `docs/gemma4_migration_log.md` — Stage 1 全結果、4 罠、確定設計
   - `docs/gemma4_stage1_phase1c1d_plan.md` — Stage 1 plan (R1-R9 定義の原典)
   - `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` — `VLAAdapterGemma4`
   - `scripts/gemma4/test_08_data_pipeline.py` — `Gemma4BatchTransform` / `Gemma4RLDSDataset`
   - `scripts/gemma4/test_09_single_step.py` — Stage 1d.b 単 step 学習
   - `scripts/gemma4/test_10_smoke_train.py` — Stage 1d.c 10-step smoke
   - `VLA-Adapter/vla-scripts/finetune.py` — `finetune_gemma4.py` 差分元 (line 915-921 scheduler、1060-1065 warmup)
   - `VLA-Adapter/experiments/robot/libero/run_libero_eval.py` — `eval_libero_gemma4.py` 差分元
3. 環境確認:
   - `nvidia-smi` で GPU 状態
   - `ls VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json` (R8 流用元)
   - `.venv-gemma4/bin/python -c "import wandb, accelerate, libero; print('OK')"` (Stage 2 で WandB / DDP / LIBERO simulator 利用)
4. 利用可能環境の User 確認:
   - **Env A**: A100 × 1 (80 GB) — 開発、smoke、debug、eval pipeline 構築
   - **Env B**: A100 × 8 (40 GB × 8) — 本番 train、本番 eval (8 task 並列)
5. 準備完了を User に報告、**Phase 2a 着手許可を待つ**

---

## Stage 1 引継ぎ (4/4 軸 PASS, 2026-04-20 認定)

```
Model:         VLAAdapterGemma4  (trainable 675.138M、frozen 6566M)
Data:          data/modified_libero_rlds/libero_spatial_no_noops/1.0.0/  (52,970 steps)
Memory (B=8, single 80GB):  fwd/bwd 37.05 GB, opt 16.40 GB
1d.c smoke loss:  0.4886 (init) → 2.125 (step 2 peak) → 0.396 (step 8 min, init 以下)
Post-clip grad_norm:  0.995 - 1.004 (10/10 step で 1.0 飽和)
LLM grad leak:  0/10 step
```

**重要: Stage 1 で未検証 (Stage 2 で初検証する経路):**

- Action sampling (推論経路): training loss は確認、しかし action_queries → action chunk 推論はゼロ実行
- LIBERO simulator + checkpoint load + denormalize logic
- Multi-GPU (DDP)
- Checkpoint resume (save/load + 'module.' prefix handling)

これらは Phase 2c / 2d で**本番 train の前に**初検証する。

---

## 絶対ルール (R1-R17)

### R1-R9 (Stage 1 から継承、変更なし)

`docs/gemma4_stage1_phase1c1d_plan.md` line 42-113 参照。要約:

- R1: `model.model.language_model.*` 階層
- R2: `get_per_layer_inputs(input_ids, None)` 事前計算 + 明示渡し
- R3: Overwrite は Option A (clone + advanced indexing) のみ
- R4: LLM 完全凍結、毎 step grad leak 検査 (DDP wrap 後は `model.module.llm.parameters()`)
- R5: `attention_mask` / `position_ids` 明示構築
- R6: `use_cache=True` / `gradient_checkpointing` **永久禁止** (HF #45242: Gemma 4 `num_kv_shared_layers=20` と GC の interaction バグ)
- R7: LoRA 全コメントアウト維持 (R14 で永続化)
- R8: `dataset_statistics.json` は `outputs/LIBERO-Spatial-Pro/` から再利用 (Phase 2m 自前データは R17 例外)
- R9: ~~max_steps=10 厳守~~ → Stage 2 で revoke、本番 max_steps を新規定義

### R10. Linear warmup 10% → 100%

VLA-Adapter 原実装 `finetune.py:1060-1065` 準拠:

```python
def _warmup_lambda(step, warmup_steps):
    if warmup_steps <= 0:
        return 1.0
    return 0.1 + 0.9 * min((step + 1) / warmup_steps, 1.0)
```

確定値: Env A `warmup_steps=500`, `target_lr=2e-4`。Env B 用は Phase 2d 実測後に再算定 (Plan §7)。

### R11. WandB は本番 run のみ

Smoke (2a/2b/2c/2d/2f) は stdout + JSON。Production kick-off (2e) で WandB on。事前に `WANDB_MODE=offline` 動作確認 → online 切替。

### R12. Checkpoint 保存 + resume 検証

`save_freq=10000`、`save_latest_checkpoint_only=True`。**Phase 2d で save → load → resume を強制検証** (DDP 'module.' prefix、optimizer state resume 含む)。

### R13. Scope 拡大は User check-in 必須

Bidirectional、DDP 切替、本番 kick-off、hyperparam sweep 等は Check-in Point 経由。

### R14. LoRA は Stage 2 でも永久 Out of Scope

**三重不可:**

1. 設計: VLA-Adapter は backbone frozen + Adapter 学習が architecture の核心
2. HF bug: R6 で GC 禁止、LoRA の memory 効率化に必須の GC 封じ
3. Memory: GC なしの LoRA は 5104M backbone activation で OOM 必至

`finetune.py` の LoRA コードはコメントアウトのまま、import のみ keep。

### R15. Multi-GPU は DDP のみ、FSDP/ZeRO 禁止

backbone frozen で shard 効果なし、trainable 675M は単 GPU 楽勝。OOM fallback としてのみ keep。

### R16 (新規). TF + DDP CUDA initialization order

DDP subprocess で TF が CUDA を grab しないよう、各 worker process 冒頭で:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import torch
```

逆順だと TF が GPU memory を先取り → DDP NCCL 初期化が OOM の典型 bug。

### R17 (新規). 自前データ statistics の R8 例外

R8 は **LIBERO-Spatial-Pro pre-train 時のみ適用**。Phase 2m (自前データ) では User judgment:

- (a) LIBERO 統計を流用 (R8 維持、suboptimal リスク)
- (b) 自前用に `dataset_statistics.json` を新規計算 (R8 例外)
- (c) 収集時に action range を LIBERO に合わせる前処理

**Default = (b) 推奨**。Phase 2j で User と確定。

---

## マイクロフェーズ

### Part 1: Pipeline build (training + eval 双方を本番前に検証)

| # | Phase | 内容 | 環境 | 所要 |
|---|---|---|---|---|
| 1 | **2a** | warmup smoke (`test_11_warmup_smoke.py`、30 step、lr/loss trajectory) | Env A | ~10 分 |
| 2 | **2b** | `finetune_gemma4.py` 作成 (原 `finetune.py` から fork、Qwen→Gemma 4、LoRA 全コメントアウト維持、100 step smoke) | Env A | ~3 時間 |
| 3 | **2c** | `eval_libero_gemma4.py` 作成 + **rollout dry run** (random-init checkpoint で 1 task × 1 episode、completion + action sampling + denormalize 全経路通過確認) | Env A | ~3-4 時間 |

**Phase 2c は本番 train の前に必ず実施**。eval pipeline 未構築のまま 6 時間 train を回すリスク排除。

### Part 2: Production train + eval

| # | Phase | 内容 | 環境 | 所要 |
|---|---|---|---|---|
| 4 | **2d** | DDP smoke (100 step + checkpoint save/load/resume 検証、per-GPU memory 実測、grad leak DDP 版) | Env B | ~1 時間 |
| 5 | **2e** | LIBERO-Spatial 本番学習 kick-off (per-GPU B=4、effective B=32、max_steps Plan §7、WandB on、save_freq=10000) | Env B | kick-off 30 分 + run 1-2 日 |
| 6 | **2f** | 10k checkpoint で **rollout sanity check** (Phase 2c の eval pipeline で 1 task × 1 episode、success/fail 不問、random-init からの regression なし確認) | Env A 1 GPU | ~30 分 |
| 7 | **2g** | LIBERO-Spatial full eval (num_trials=10) | Env B (8 task 並列) | ~6 時間 |
| 8 | **2h** | Gap 分析 → Bidirectional 投入判断 Check-in | — | ~1 時間 |
| 9 | **2i** | (条件付) Bidirectional retrain + eval (`use_bidirectional_attention=True`、**baseline checkpoint から fine-tune** で実験 design 統一) | Env A → Env B | ~1.5 日 |

### Part 3: 自前データ + Demo

| # | Phase | 内容 | 環境 | 所要 |
|---|---|---|---|---|
| 10 | **2j** | Teleop 環境準備、ツール選定、タスク決定、**R17 (a)/(b)/(c) 確定** | — | ~1-2 日 |
| 11 | **2k** | **Single demo + replay sanity** (1 demo 録音 → replay で fps/sampling/timestamp/observation 経路確認) | — | ~1-2 時間 |
| 12 | **2l** | データ収集 (30-50 demo per task) | — | ~1-2 日 |
| 13 | **2m** | RLDS 変換 + (R17 (b) なら) `dataset_statistics.json` 新規計算 + 自前データ fine-tune | Env A or B | ~6-12 時間 |
| 14 | **2n** | Demo 動画 + ハッカソン提出物 | — | ~2-3 日 |

詳細は `gemma4_stage2_plan.md` §5 参照。

---

## Check-in Points (自動進行禁止)

| # | タイミング | 提示物 |
|---|---|---|
| 10 | 2a 完了後 | lr/loss trajectory 30 step、warmup 数式が原実装通りか |
| 11 | 2b 完了後 | 100 step smoke 結果、loss/grad/leak、Stage 1 regression 比較 |
| 12 | **2c 完了後** | **eval dry run 結果**: simulator init log、checkpoint load 成否、action chunk shape `(8, 7)` 範囲、denormalize 後 value range、1 episode 完走 |
| 13 | 2d 完了後 | DDP per-GPU memory、effective B=32 per-step time、grad leak (DDP 版)、checkpoint save/load/resume 成否 |
| 14 | 2e kick-off 前 | WandB run url、checkpoint dir、config dump |
| 15 | 2e 進行中 (10k step) | loss/grad/lr trajectory、ETA 再算定 |
| 16 | **2f 完了後** | **10k checkpoint rollout 結果**: 1 episode 完走、action 出力 saturate なし、Phase 2c との分布変化 |
| 17 | 2g 完了後 | LIBERO eval success rate per task、論文値 gap |
| 18 | 2h 完了後 | Bidirectional 投入 YES/NO 判断 |
| 19 | 2i 完了後 (実施時のみ) | Bidirectional 前後の eval 比較 |
| 20 | **2j 完了後** | Teleop ツール選定、R17 確定、自前データ task 仕様 |
| 21 | **2k 完了後** | **Single demo replay 結果**: 録画 fps、action sampling rate、image res、timestamp sync、observation 経路の train 一致 |
| 22 | 2l 完了後 | データ収集統計 (demo 数、task 別 success rate、平均 episode length) |
| 23 | 2m 完了後 | 自前 fine-tune loss + 軽量 rollout (1-3 episode) |
| 24 | 2n 完了後 | Demo 動画 + 提出物 review |

境界ケース (loss 不安定、eval 異常低、データ不正等) は自動進行禁止。

---

## Escalation Policy (即停止 + 報告)

1. 2a で warmup lr が期待 trajectory に乗らない
2. 2b で 100 step smoke が Stage 1 regression を破壊
3. **2c で simulator 初期化失敗、checkpoint load 失敗、action 常時 saturate**
4. **2c で 1 episode rollout 完走しない**
5. 2d で DDP per-GPU memory 38 GB 超 (40GB margin 不足)
6. 2d で grad leak check が DDP wrap 後動かない
7. **2d で checkpoint save → load → resume で loss/optimizer state 不一致**
8. 2e で 1000 step 以内に loss NaN/Inf
9. 2e で post-clip grad_norm > 1.1
10. 2e で LLM grad leak 発生 (R4 違反)
11. 2e で per-step time が estimate 2x 超
12. **2f で 10k checkpoint が random-init より明らかに性能劣化**
13. 2g eval success rate < 50%
14. **2k で single demo replay 失敗 (録画欠損、timestamp 不整合等)**
15. **2m で R17 (b) 新規 statistics が train データ統計と劇的不整合**
16. 自前データ collection で teleop 不具合
17. 単一問題で 3 時間以上ハマる
18. Stage 2 scope 外項目に踏み込まないと解決不能

---

## やってはいけないこと (違反即時停止)

- LoRA / PEFT / `get_peft_model` 有効化 (R14 三重不可)
- `gradient_checkpointing` 有効化 (R6, HF #45242 永続)
- FSDP / ZeRO 独断導入 (R15)
- warmup なしで kick-off (Stage 1 step 2 peak 2.125 が本番で拡大)
- VLA-Adapter 原実装と異なる warmup 数式 (R10)
- checkpoint なしで長時間 run (R12)
- **eval pipeline 未構築のまま本番 train kick-off** (Phase 2c skip 禁止)
- **Single demo sanity 未実施で 30-50 demo 一気収集** (Phase 2k skip 禁止)
- `dataset_statistics.json` 勝手な扱い (R8 / R17 に従う)
- Stage 1 確定事項の勝手な変更 (placeholder ID、feature_norm、use_cache 等)
- Check-in Point スキップ
- Loss NaN workaround の独断導入

---

## Out of Scope (Stage 2 でも除外)

- LoRA 導入 (R14)
- FSDP / ZeRO (R15)
- Gemma 3n fallback
- 論文値 (99.6%) 完全再現
- LIBERO 4 suite 全部 (Spatial 単独 baseline)
- 学術論文執筆

---

## Deliverables

| # | ファイル | Phase |
|---|---|---|
| 1 | `scripts/gemma4/test_11_warmup_smoke.py` | 2a |
| 2 | `VLA-Adapter/vla-scripts/finetune_gemma4.py` | 2b |
| 3 | **`scripts/gemma4/eval_libero_gemma4.py`** | 2c |
| 4 | **`scripts/gemma4/test_12_rollout_dryrun.py`** | 2c |
| 5 | `scripts/gemma4/test_13_ddp_smoke.py` | 2d |
| 6 | `configs/gemma4_libero_spatial.yaml` (or argparse defaults) | 2e |
| 7 | **`scripts/gemma4/test_14_rollout_sanity.py`** | 2f |
| 8 | **`scripts/teleop/single_demo_sanity.py`** | 2k |
| 9 | `scripts/teleop/collect_demos.py` | 2l |
| 10 | `scripts/data/rlds_converter.py` + (R17 (b) なら) `compute_statistics.py` | 2m 前段 |
| 11 | `docs/gemma4_migration_log.md` Stage 2 セクション | 各 Phase |
| 12 | `README.md` 更新 | Stage 2 完了時 |
| 13 | Demo 動画 + 提出物 | 2n |

---

## 報告フォーマット

```
=== Phase 2X 完了報告 ===

Status: [OK / FAIL / BLOCKED]
変更ファイル: ...
実行コマンド: ...
環境: [Env A / Env B]
主要数値: ...
Escalation 該当: [YES / NO、該当の場合は番号]
Check-in Point #NN 判断要請: ...
```

Check-in では数値表 / loss/lr trajectory / stdout の生データ添付必須。要約のみ不可。

---

## Compute budget allocation (28 日、4/20 起点)

| Day | Phase | 環境 |
|---|---|---|
| 1 (4/20) | 2a 着手許可待ち | — |
| 2 | 2a + 2b | Env A |
| 3 | 2b 完了 + 2c 着手 | Env A |
| 4 | 2c 完了 (eval pipeline 構築 + dry run) | Env A |
| 5 | 2d (DDP smoke + checkpoint resume) | Env B |
| 6 | 2e kick-off (本番 train) | Env B |
| 7 | 2e 監視 + 2f rollout sanity | Env B + Env A |
| 8 | 2e 完了 + 2g eval kick-off | Env B |
| 9 | 2g 完了 + 2h gap 分析 | — |
| 10-11 | 2i Bidirectional (条件付) | Env A → Env B |
| 12 | 2j (teleop 準備、R17 確定) | — |
| 13 | 2k (single demo sanity) | — |
| 14-15 | 2l (データ収集 30-50 demo) | — |
| 16 | 2m (RLDS 変換 + 統計計算 + fine-tune) | Env A or B |
| 17-19 | 2n (Demo 動画 + 提出物) | — |
| 20-28 | Buffer (debug、retraining、調整) | — |

詳細は `gemma4_stage2_plan.md` §6 参照。

---

## 開始合図

準備完了したら User に報告し、**Phase 2a 着手許可**を得てから作業開始。勝手に 2a から始めない。

References:
- 実装計画詳細: `gemma4_stage2_plan.md`
- Stage 1 全履歴: `docs/gemma4_migration_log.md`
- Stage 1 plan: `docs/gemma4_stage1_phase1c1d_plan.md`
- 原実装 warmup: `VLA-Adapter/vla-scripts/finetune.py:1060-1065`
- 原実装 scheduler: `VLA-Adapter/vla-scripts/finetune.py:915-921`
- 原実装 eval: `VLA-Adapter/experiments/robot/libero/run_libero_eval.py`

---

## Stage 2 完了認定

4 軸すべて達成で Stage 2 完了:

1. 本番 train 完走 (max_steps 到達、loss finite、checkpoint 保存)
2. LIBERO-Spatial eval 結果 (success rate 数値化、論文値 gap が analytical に説明可能)
3. 自前データ fine-tune 成立 (1 タスク以上で loss 減少 + 軽量 rollout 確認)
4. Demo 動画 + ハッカソン提出

達成不能項目があれば Stage 2 中盤 (Day 14 前後) で User と再 scope 化。
