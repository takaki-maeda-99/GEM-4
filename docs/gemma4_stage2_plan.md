# Gemma 4 E2B 移植 Stage 2 Plan (v2)

**作成**: 2026-04-20 (Stage 1 完了認定後)
**v2 改訂**: sanity-check gap 補完 (eval pipeline / rollout dry run / teleop single-demo / DDP resume / TF+CUDA init / 自前データ統計)
**対象**: VLA-Adapter (backbone Gemma 4 E2B、LLM/Vision 凍結) の本番学習 + LIBERO eval + 自前データ + ハッカソン提出
**期限**: 2026-05-18 (Gemma 4 Good Hackathon)、残 28 日

---

## 1. Stage 1 引継ぎ (4/4 軸 PASS, 2026-04-20 認定)

```
Model:         VLAAdapterGemma4
Trainable:     675.138M  (Adapter + Pro action head + vision_projector + proprio_projector + action_queries)
Frozen:        6566M     (Gemma 4 E2B 5104M + DINO+SigLIP 731M + その他)
Data:          data/modified_libero_rlds/libero_spatial_no_noops/1.0.0/  (52,970 steps)
```

**Stage 1 で確立した数値 (single A100 80GB, B=8, L=599):**

| 項目 | 値 |
|---|---|
| fwd/bwd peak | 37.05 GB |
| optimizer steady | 16.40 GB |
| 1d.c smoke loss | 0.4886 (init) → 2.125 (step 2 peak) → 0.396 (step 8 min, init 以下) |
| post-clip grad_norm | 0.995 - 1.004 (10/10 step で 1.0 飽和) |
| pre-clip grad_norm | 61952 (step 0) → 47 (step 1) → 12 (step 2) → 16 (step 9) |
| LLM grad leak | 0/10 step |

**Stage 1 で未検証 (Stage 2 で初検証する経路):**

- **Action sampling (推論経路)**: training loss は確認、しかし action_queries embedding → action head の iterative denoise → 7-dim action chunk の推論経路は**ゼロ実行**。Phase 2c で初検証
- **LIBERO simulator + checkpoint load + denormalize logic**: 全部 Stage 1 では touch していない、Phase 2c で初検証
- **Multi-GPU (DDP)**: Phase 2d で初検証
- **Checkpoint resume**: Phase 2d で初検証

**Stage 1 で確立した制約 (R1-R9):**

R1-R9 は Stage 1 prompt (`claude_code_prompt_phase1c1d.md`) line 42-113 に定義。Stage 2 全期間で**継承**。特に:

- **R4** (LLM 完全凍結 + 毎 step grad leak 検査): VLA-Adapter architecture の核心、Stage 2 で revoke しない
- **R6** (`use_cache=True` / `gradient_checkpointing` 禁止): **HF transformers issue #45242** (Gemma 4 の `num_kv_shared_layers=20` と GC の interaction バグ) が患部。Stage 2 でも patch されない限り永続制約

---

## 2. Stage 2 Mission

```
Mission 1 (必須): LIBERO-Spatial で本番学習を回し、success rate を測定 (paper 99.6% との gap 分析付き)
Mission 2 (必須): 自前データ (teleop) で 1 タスク以上 fine-tune 成立
Mission 3 (必須): Demo 動画作成 → ハッカソン提出
Mission 4 (optional): Bidirectional attention 等の精度向上 ablation
```

論文精度の完全再現 (99.6%) は scope 外。70-90% 帯で達成可、gap が analytical に説明できれば Mission 1 達成。

---

## 3. 計算環境

| 環境 | GPU | VRAM | 用途 |
|---|---|---|---|
| **Env A** | A100 × 1 | 80 GB | 開発、smoke、debug、eval pipeline 構築、rollout sanity |
| **Env B** | A100 × 8 | 40 GB × 8 | 本番 train、本番 eval (8 task 並列) |

### 3.1. Env A (single 80GB) Memory profile

Stage 1 実測通り `B=8` で fwd/bwd 37.05 GB peak、margin 40+ GB あり安全。**Stage 2 開発は Env A、本番は Env B** のワークフロー。

### 3.2. Env B (8×40GB) Memory profile (推定 + Phase 2d で実測必須)

Per-GPU 40 GB - CUDA context (~1 GB) - NCCL buffer + grad bucket (~1-2 GB) = 実効 ~36-37 GB。

Stage 1 の B=8 数値 (37.05 GB) を per-GPU 適用すると margin 0、危険。

**Per-GPU batch の暫定推奨**: `B=4` (Stage 1 の B=8 線形外挿で fwd/bwd ~18-20 GB、margin ~20 GB)。

Effective batch = `B=4 × 8 GPU = 32`。Phase 2d で実測 → margin OK なら B=6 拡大も検討可 (要 User check-in)。

### 3.3. 並列化 strategy: DDP 一択

| 戦略 | 適性 | 根拠 |
|---|---|---|
| **DDP** | ◎ | backbone frozen で grad なし、trainable 675M のみ all-reduce |
| FSDP | × | backbone shard しても frozen で節約効果なし |
| ZeRO-1 | △ | optimizer state ~5.4 GB を 1/8 にする効果は B=4 で margin 取れば不要 |

**FSDP/ZeRO は Stage 2 では検討しない**。OOM fallback としてのみ keep。

### 3.4. R4 (LLM grad leak check) の DDP 互換性

DDP wrap 後は `model.module.llm.parameters()` でアクセス。Phase 2b の `finetune_gemma4.py` 作成時に check 文の path 修正が必要 (R4 自体は維持)。

---

## 4. アーキテクチャ制約 (R1-R17)

### R1-R9 (Stage 1 から継承、変更なし)

`docs/gemma4_stage1_phase1c1d_plan.md` line 42-113 参照。

### R10 (新規). Linear warmup 10% → 100%

VLA-Adapter 原実装 `finetune.py:1060-1065` 準拠:

```python
def _warmup_lambda(step, warmup_steps):
    if warmup_steps <= 0:
        return 1.0
    return 0.1 + 0.9 * min((step + 1) / warmup_steps, 1.0)
```

確定値: Env A で `warmup_steps=500`, `target_lr=2e-4`。Env B で effective batch を 4x にする場合は要再算定 (Plan §7)。

### R11 (新規). WandB は本番 run のみ

Smoke (2a/2b/2c/2d/2f) は stdout + JSON のみ。Production kick-off (2e) で WandB on。事前に `WANDB_MODE=offline` で動作確認 → online 切替。

### R12 (新規). Checkpoint 保存は本番 run で必須

`save_freq=10000`、`save_latest_checkpoint_only=True` (VLA-Adapter 原実装と同値)。**Phase 2d で save → load → resume の経路を強制検証** (DDP の 'module.' prefix handling、optimizer state の resume を含む)。

### R13 (新規). Scope 拡大は User check-in 必須

Bidirectional、DDP 切替、本番 kick-off、hyperparam sweep 等は Check-in Point 経由で許可取得。

### R14 (新規). LoRA は Stage 2 でも永久 Out of Scope

**三重不可:**

1. 設計: VLA-Adapter は backbone frozen + Adapter 学習が architecture の核心 (paper 主張)
2. HF bug: R6 で GC 禁止 (HF #45242)、LoRA の memory 効率化に必須の GC が封じられている
3. Memory: GC なしの LoRA は 5104M backbone activation を全層保持、本番 batch で OOM 必至

`finetune.py` の LoRA コードはコメントアウトのまま、import のみ keep。

### R15 (新規). Multi-GPU は DDP のみ、FSDP/ZeRO 禁止

3.3 の通り。OOM fallback としてのみ keep、独断導入禁止。

### R16 (新規). TF + DDP CUDA initialization order

RLDS dataloader は TF backend (`tensorflow_graphics` 含む)。DDP subprocess で TF が CUDA を grab しないよう、**各 worker process の冒頭で**:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)  # DDP launcher が設定する場合は不要
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # 強制無効化
import torch  # その後 torch import
```

順序を逆にすると TF が GPU memory を先取りし、DDP NCCL 初期化が OOM で落ちる典型 bug。

### R17 (新規). 自前データ statistics の R8 例外

R8 (`dataset_statistics.json` 新規計算禁止) は **LIBERO-Spatial-Pro pre-train 時にのみ適用**。Phase 2m (自前データ fine-tune) では以下の選択肢を **User judgment** で決定:

- **(a)** 自前データの action range が LIBERO と互換ならば LIBERO 統計を流用 (R8 維持、suboptimal の可能性あり)
- **(b)** 自前データ用に `dataset_statistics.json` を新規計算 (R8 例外、Phase 2m の前段で `compute_statistics.py` を実行)
- **(c)** 自前データ収集時に action range を LIBERO に合わせる前処理を組む

Default = **(b)** を推奨 (action range の不一致は denormalize logic を直撃、(a) は debug 困難)。Phase 2j (teleop 環境準備) で User と確認。

---

## 5. マイクロフェーズ

### Part 1: Pipeline build (training + eval 双方を本番前に検証)

| # | Phase | 内容 | 環境 | 所要 |
|---|---|---|---|---|
| 1 | **2a** | warmup smoke (`test_11_warmup_smoke.py`、30 step、lr/loss trajectory 記録) | Env A | ~10 分 |
| 2 | **2b** | `finetune_gemma4.py` 作成 (原 `finetune.py` から fork、Qwen→Gemma 4、LoRA 全コメントアウト維持、100 step smoke) | Env A | ~3 時間 |
| 3 | **2c** | `eval_libero_gemma4.py` 作成 + **rollout dry run** (random-init checkpoint で 1 task × 1 episode、success/fail 不問、completion + action サンプリング + denormalize の全経路通過確認) | Env A | ~3-4 時間 |

**Phase 2c は本番 train の前に必ず実施**。eval pipeline が動かない状態で 6 時間 train を回すリスクを排除。

### Part 2: Production train + eval

| # | Phase | 内容 | 環境 | 所要 |
|---|---|---|---|---|
| 4 | **2d** | DDP 動作確認 (100 step + checkpoint save/load/resume 検証、per-GPU memory 実測、grad leak DDP 版確認) | Env B | ~1 時間 |
| 5 | **2e** | LIBERO-Spatial 本番学習 kick-off (per-GPU B=4、effective B=32、max_steps Plan §7 参照、WandB on、save_freq=10000) | Env B | kick-off 30 分 + run 1-2 日 |
| 6 | **2f** | 10k step checkpoint で **rollout sanity check** (Phase 2c で構築した eval pipeline を 1 task × 1 episode で実行、success/fail 不問、random-init からの regression なしを確認) | Env A 1 GPU | ~30 分 |
| 7 | **2g** | LIBERO-Spatial full eval (num_trials=10、論文値との比較) | Env B (8 task 並列) | ~6 時間 |
| 8 | **2h** | Gap 分析 → Bidirectional 投入判断 Check-in | — | ~1 時間 |
| 9 | **2i** | (条件付) Bidirectional retrain + eval (`use_bidirectional_attention=True`、**baseline checkpoint から fine-tune** で実験 design 統一、smoke 100 step → 50k step retrain → eval) | Env A → Env B | ~1.5 日 |

### Part 3: 自前データ + Demo

| # | Phase | 内容 | 環境 | 所要 |
|---|---|---|---|---|
| 10 | **2j** | Teleop 環境準備、ツール選定 (iPhone teleop 等)、タスク決定、**R17 (自前データ統計の (a)/(b)/(c) 選択)** を User と確定 | — | ~1-2 日 |
| 11 | **2k** | **Single demo + replay sanity** (1 demo 録音 → 録画/録音/timestamp/action sampling rate を replay で確認、recording pipeline の全経路検証) | — | ~1-2 時間 |
| 12 | **2l** | データ収集 (30-50 demo per task) | — | ~1-2 日 |
| 13 | **2m** | RLDS 変換 + (R17 (b) なら) `dataset_statistics.json` 新規計算 + 自前データ fine-tune (LIBERO 学習済 checkpoint から) | Env A or B | ~6-12 時間 |
| 14 | **2n** | Demo 動画撮影 + 編集 + ハッカソン提出物作成 | — | ~2-3 日 |

### Part 4 (Out of Scope): その他 ablation

LoRA、FSDP、Gemma 3n fallback、論文値完全再現は Stage 2 でも実施しない。

---

## 6. Compute budget allocation (28 日、4/20 起点)

| Day | Phase | 環境 |
|---|---|---|
| 1 (4/20) | 2a 着手許可待ち、Plan/Prompt 確定 | — |
| 2 | 2a + 2b | Env A |
| 3 | 2b 完了 + 2c 着手 | Env A |
| 4 | 2c 完了 (eval pipeline 構築 + dry run) | Env A |
| 5 | 2d (DDP smoke + checkpoint resume 検証) | Env B |
| 6 | 2e kick-off (本番 train 開始) | Env B |
| 7 | 2e 監視 (10k step 経過時点で 2f rollout sanity) | Env B + Env A |
| 8 | 2e 完了 + 2g eval kick-off | Env B |
| 9 | 2g 完了 + 2h gap 分析 | — |
| 10-11 | 2i Bidirectional (条件付、不要なら飛ばす) | Env A → Env B |
| 12 | 2j (teleop 環境準備、R17 確定) | — |
| 13 | 2k (single demo + replay sanity) | — |
| 14-15 | 2l (データ収集 30-50 demo) | — |
| 16 | 2m (RLDS 変換 + 統計計算 + fine-tune) | Env A or B |
| 17-19 | 2n (Demo 動画 + 提出物作成) | — |
| 20-28 | **Buffer** (debug、retraining、ハッカソン直前調整) | — |

Buffer 9 日。本番 train が 2 日で済む前提だが、loss 不安定 → warmup 増やし → retrain 等の手戻りで 4-5 日に拡大する余地あり。

---

## 7. Time estimate (本番 train)

Stage 1 smoke (10 step ~10 秒) からの素朴外挿は危険。real production では:

| 要素 | overhead |
|---|---|
| RLDS dataloader I/O | +10-30% (sequence length 変動、prefetch hit ratio 次第) |
| WandB logging (毎 step) | +2-5% |
| Checkpoint save (10k step 毎、save_latest_only) | +0.1% (償却後) |
| DDP all-reduce (Env B) | +5-15% (NCCL backend と inter-GPU bandwidth 次第) |

**現実的 estimate:**

- **Env A** (single 80GB, B=8): 1.0-1.5 sec/step → 200k step = **2.3 - 3.5 日**
- **Env B** (8×40GB DDP, per-GPU B=4 = effective B=32): 1.2-1.8 sec/step、effective batch 4x → 同等 data 量到達は 50k step = **0.7-1.0 日**

**暫定方針 (Env B)**: paper の effective batch 維持を優先し `max_steps=50000`、`warmup_steps=125` で kick-off。Phase 2d で per-step time 実測後に再算定。

別案 (large batch で攻める): `max_steps=200000` のまま回せば effective サンプル 4x、収束加速期待だが paper 比較性低下。Phase 2h の gap 分析時に判断。

---

## 8. Check-in Points

| # | タイミング | 提示物 |
|---|---|---|
| 10 | 2a 完了後 | lr/loss trajectory 30 step、warmup 数式が原実装通りか |
| 11 | 2b 完了後 | 100 step smoke 結果、loss/grad/leak、Stage 1 regression 比較 |
| 12 | **2c 完了後** | **eval dry run 結果**: simulator init log、checkpoint load 成否、action chunk shape `(8, 7)` 範囲、denormalize 後の value range、1 episode 完走 (success 不問) |
| 13 | 2d 完了後 | DDP per-GPU memory、effective B=32 per-step time、grad leak (DDP 版)、checkpoint save/load/resume 成否 |
| 14 | 2e kick-off 前 | WandB run url、checkpoint dir、config dump |
| 15 | 2e 進行中 (10k step) | loss/grad/lr trajectory、ETA 再算定 |
| 16 | **2f 完了後** | **10k checkpoint rollout 結果**: 1 episode 完走、action 出力が saturate していない、Phase 2c の random-init からの分布変化 |
| 17 | 2g 完了後 | LIBERO eval success rate per task、論文値 gap |
| 18 | 2h 完了後 | Bidirectional 投入 YES/NO 判断 |
| 19 | 2i 完了後 (実施時のみ) | Bidirectional 前後の eval 比較 |
| 20 | **2j 完了後** | Teleop ツール選定結果、R17 (a)/(b)/(c) の確定、自前データ task 仕様 |
| 21 | **2k 完了後** | **Single demo replay 結果**: 録画 fps、action sampling rate、image resolution、timestamp sync、replay で観測経路が train と一致 |
| 22 | 2l 完了後 | データ収集統計 (demo 数、task 別 success rate、平均 episode length) |
| 23 | 2m 完了後 | 自前 fine-tune loss 推移 + 軽量 rollout (1-3 episode) |
| 24 | 2n 完了後 | Demo 動画 + 提出物 review |

境界ケース (loss 不安定、eval 異常低、データ不正等) は自動進行禁止、User judgment。

---

## 9. Escalation Policy (即停止 + 報告)

1. 2a で warmup lr が期待 trajectory に乗らない (param_group 更新漏れ、float ミス)
2. 2b で 100 step smoke が Stage 1 regression を破壊 (loss、leak、memory のいずれか)
3. **2c で simulator (libero / MuJoCo) 初期化失敗、checkpoint load 失敗、または action 出力が常時 saturate (max/min 等)**
4. **2c で 1 episode rollout が完走しない (環境エラー、infinite loop 等)**
5. 2d で DDP per-GPU memory が 38 GB 超 (40GB margin 不足)
6. 2d で grad leak check が DDP wrap 後に動かない (path 修正失敗)
7. **2d で checkpoint save → load → resume で loss/optimizer state が一致しない**
8. 2e で 1000 step 以内に loss NaN/Inf
9. 2e で post-clip grad_norm > 1.1 (clip 効いていない)
10. 2e で LLM grad leak 発生 (R4 違反)
11. 2e で per-step time が estimate の 2x 超 (I/O bottleneck 等)
12. **2f で 10k checkpoint が random-init より明らかに性能劣化** (action 分布、loss 等)
13. 2g eval success rate が 50% 未満 (本番 train が壊れた可能性)
14. **2k で single demo replay が失敗 (録画欠損、timestamp 不整合、action 不正等)** → 30-50 demo 収集の前に修正
15. **2m で R17 (b) を選択して新規 statistics 計算が train データの統計と整合しない** (action range が LIBERO と劇的に異なる、distribution shift 巨大等)
16. 自前データ collection で teleop 不具合
17. 単一問題で 3 時間以上ハマる
18. Stage 2 scope 外項目に踏み込まないと解決不能

---

## 10. やってはいけないこと

- LoRA / PEFT / `get_peft_model` 有効化 (R14 三重不可)
- `gradient_checkpointing` 有効化 (R6, HF #45242 永続)
- FSDP / ZeRO 独断導入 (R15)
- warmup なしで kick-off (Stage 1 step 2 peak 2.125 が本番で拡大する)
- VLA-Adapter 原実装と異なる warmup 数式 (R10)
- checkpoint なしで長時間 run (R12)
- **eval pipeline 未構築のまま本番 train kick-off** (R12 趣旨、Phase 2c を skip しない)
- **Single demo sanity 未実施で 30-50 demo 一気に collection** (Phase 2k を skip しない)
- `dataset_statistics.json` 勝手な扱い (R8 / R17 に従う、独断で新規計算 or 流用しない)
- Stage 1 確定事項の勝手な変更 (placeholder ID、feature_norm、use_cache 等)
- Check-in Point スキップ
- Loss NaN workaround の独断導入

---

## 11. Out of Scope

| 項目 | 理由 |
|---|---|
| LoRA 導入 | R14 三重不可 |
| FSDP / ZeRO | R15、現構成で不要 |
| Gemma 3n fallback | User 方針 |
| 論文値 (99.6%) 完全再現 | ハッカソン優先、70-90% で完了認定 |
| LIBERO 4 suite 全部 (Object/Goal/Long) | Spatial 単独で baseline 確立、余裕あれば追加 |
| 学術論文執筆 | ハッカソン後検討 |
| 10 step 制限 (R9) | Stage 2 で revoke、本番 max_steps を新規定義 |

---

## 12. Deliverables

| # | ファイル | Phase |
|---|---|---|
| 1 | `scripts/gemma4/test_11_warmup_smoke.py` | 2a |
| 2 | `VLA-Adapter/vla-scripts/finetune_gemma4.py` | 2b |
| 3 | **`scripts/gemma4/eval_libero_gemma4.py`** (eval pipeline、必須) | 2c |
| 4 | **`scripts/gemma4/test_12_rollout_dryrun.py`** (random-init rollout、Phase 2c の wrapper) | 2c |
| 5 | `scripts/gemma4/test_13_ddp_smoke.py` (DDP + checkpoint resume) | 2d |
| 6 | `configs/gemma4_libero_spatial.yaml` (or argparse defaults) | 2e |
| 7 | **`scripts/gemma4/test_14_rollout_sanity.py`** (10k checkpoint rollout) | 2f |
| 8 | **`scripts/teleop/single_demo_sanity.py`** (1 demo 録音 + replay 検証) | 2k |
| 9 | `scripts/teleop/collect_demos.py` (or 流用ツール) | 2l |
| 10 | `scripts/data/rlds_converter.py` (自前 → RLDS) + (R17 (b) なら) `compute_statistics.py` | 2m 前段 |
| 11 | `docs/gemma4_migration_log.md` に Stage 2 セクション追記 | 各 Phase |
| 12 | `README.md` 更新 (Stage 2 成果反映) | Stage 2 完了時 |
| 13 | Demo 動画 + ハッカソン提出物 | 2n |

---

## 13. Stage 2 完了認定

以下 4 軸すべて満たしたら Stage 2 完了:

1. **本番 train 完走**: max_steps 到達、loss finite、checkpoint 保存
2. **LIBERO-Spatial eval 結果**: success rate 数値化、論文値との gap が analytical に説明可能 (絶対値の閾値は設定しない)
3. **自前データ fine-tune 成立**: 1 タスク以上で loss 減少 + 軽量 rollout 確認
4. **Demo 動画 + ハッカソン提出**: 2n 完了

達成不能項目があれば Stage 2 中盤 (Day 14 前後) で User と再 scope 化。

---

## 14. References

- Stage 1 全履歴: `docs/gemma4_migration_log.md`
- Stage 1 plan: `docs/gemma4_stage1_phase1c1d_plan.md`
- 原実装 warmup: `VLA-Adapter/vla-scripts/finetune.py:1060-1065`
- 原実装 scheduler: `VLA-Adapter/vla-scripts/finetune.py:915-921`
- 原実装 eval (差分元): `VLA-Adapter/experiments/robot/libero/run_libero_eval.py`
- HF #45242 (Gemma 4 KV共有 + GC bug): R6 の根拠、Stage 2 期間中も patch 状況を監視

---

## 15. v2 改訂サマリ (v1 → v2)

| 種類 | 追加/変更内容 |
|---|---|
| 新規 Phase | 2c (eval pipeline + dry run)、2f (10k rollout sanity)、2k (single demo sanity) |
| 新規 R | R16 (TF + DDP CUDA init)、R17 (自前データ統計の R8 例外) |
| 拡張 Phase | 2d (DDP smoke に checkpoint resume 検証追加)、2i (Bidirectional の experimental design 明記) |
| 新規 Check-in | #12, #16, #20, #21 (sanity check 結果の review) |
| 新規 Escalation | #3, #4, #7, #12, #14, #15 (sanity 失敗、resume 失敗、自前データ統計問題) |
| Day allocation 再構成 | sanity check phase 反映、Buffer 9 日維持 |
| 新規 Deliverables | #3, #4, #7, #8, #10 (eval、rollout、teleop、RLDS converter、statistics) |
| 既知の未検証経路を明示 | §1 末尾に Stage 1 で touch していない経路 (action sampling、simulator、DDP、checkpoint resume) を列挙 |
