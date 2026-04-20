# Gemma 4 E2B 移植 Stage 2 後半 Plan (Latter Half)

**作成**: 2026-04-20 夜 (Phase 2e 進行中、~26k/200k step、ETA 4/22 早朝)
**前提**: Stage 2 Plan v3 (single GPU version) を基盤、Multi-GPU scope 再拡張を反映
**期限**: 2026-05-18 (Gemma 4 Good Hackathon)、残 ~28 日

---

## 1. 現状 Summary

### Phase 進捗 (2026-04-20 夜時点)

| Phase | Status | 備考 |
|---|---|---|
| 2a (warmup smoke) | PASS | loss ratio 5.19 → 1.17 |
| 2b (finetune_gemma4.py + 100 step smoke + checkpoint resume 5/5 bit-exact) | PASS | per-step 0.858 s 確定 |
| 2c (eval_libero_gemma4.py + random-init dry run) | PASS | bug 3 件即 fix |
| 2d | **dropped** (旧 DDP smoke、Stage 1 期の scope 変更で drop、本 plan で restore 対象) | — |
| **2e (LIBERO-Spatial 本番 train)** | **進行中** | 現 ~26k/200k、ETA 4/22 早朝 |
| 2f (20k checkpoint rollout sanity) | PASS | 5/10 task success、想定大幅超過 |
| 2g (full eval) | 2e 完走後 | ~34 分想定 |
| 2h-2n | 未着手 | gap 分析 / Bidirectional / 自前データ / Demo |

### サーバー環境 (現在)

| GPU | VRAM | 状態 |
|---|---|---|
| GPU 0 | A100 80GB | **Phase 2e 占有** (~26k/200k) |
| GPU 1 | A100 80GB | 空き |
| GPU 2-7 | A100 40GB × 6 | **全て空き** (Stage 1 期に占有していた他ユーザー解放済) |

### Buffer 状況

Plan v3 想定 Buffer 9 日 → 実 **~13-14 日** (Phase 2a-2c が Day 1 で完了 + Phase 2g 34 分 + Phase 2e 2.0 日)。Mission 4 (Bidirectional) 投入余裕 + Mission 2 (自前データ) + Mission 3 (Demo) に大幅余剰。

### Phase 2f で確認された key 数値 (本 plan の根拠)

- **Trained model の action 品質**: 20k step (10% trained) で 5/10 task success
- **proprio OOD 消失**: random-init の eef_x 範囲 [-0.844, -0.211] → trained 20k で [-0.211, +0.048]、fully in-distribution
- **Action std task-specific pattern**: Δx 1.81x / Δz 3.52x / gripper 2.35x 増、rotation 0.51-0.72x 減 (task-oriented converge)
- **Phase 2e 完走時の予測**: 対数的 convergence なら 85-95% success rate

### 推論速度 (Phase 2c/2f 実測、A100 80GB bf16)

- 1 query = 137 ms (median)
- Query rate 7.3 Hz / Amortized 58 Hz / Env 律速で 20 Hz
- Breakdown: vision ~25 ms (18%) / LLM ~70 ms (51%) / action head ~40 ms (29%)、**ただし実測未取得 estimate のみ**

---

## 2. Scope 変更の正式宣言

### Multi-GPU scope 再拡張の経緯

**Stage 1 期の判断** (2026-04-20 午前): User 提示「GPU 4-7 他ユーザー占有中 + 単 GPU での学習成立が最優先」→ Env B (8×40GB) を Stage 2 active scope 外、Plan v3 として R15/R16 を dormant 化、旧 Phase 2d drop

**Stage 2 後半の User 再認識** (2026-04-20 夜): 「複数 GPU scope 外は語弊があった。Stage 1 時点の GPU 占有 + 単 GPU 学習成立最優先という一時判断。単 GPU 学習成立確認済の現在 (Phase 2e 進行中で loss 健全) は Multi-GPU scope 内として再評価すべき」

### R15 / R16 の active 化

- **R15 (Multi-GPU は DDP のみ、FSDP/ZeRO 禁止)**: dormant → **Active**。FSDP/ZeRO 禁止は維持、DDP は available
- **R16 (TF + DDP CUDA initialization 順序)**: idle → **Active**。R15 と連動、各 worker 冒頭で `tf.config.set_visible_devices([], 'GPU')` 強制

### Plan v3 revise の扱い

**正式 revise は不要**、in-context 認識で Claude Code に伝え、`docs/gemma4_migration_log.md` に以下を 1 行記録:

```
### 2026-04-20 夜: Scope 変更記録
Multi-GPU (DDP) を Stage 2 active scope に再拡張。R15 / R16 を active rule として活性化 (Stage 1 期は GPU 占有のため dormant、現在は GPU 2-7 解放済)。旧 Phase 2d (DDP smoke) は D1 (DDP pre-flight smoke) として latter half で restore。詳細は gemma4_stage2_latter_half_plan.md を参照。
```

### 永続 Out of Scope 項目 (変更なし、念押し)

- **R14 (LoRA)**: 三重不可で永久除外、変更なし
- **R6 (`gradient_checkpointing`)**: HF #45242 永久禁止、変更なし
- **FSDP / ZeRO**: R15 下の禁止継続

---

## 3. 余剰 GPU 活用プラン

### Tier A: Phase 2e 進行中の並列実施 (今)

**全て Phase 2e (GPU 0) と独立 process、干渉なし**。Tier A の 4 item を同時並列で実行、wall clock ~3-4 時間。

#### A1. VLA-Adapter 原実装 (Qwen 0.5B) query time 実測

- **目的**: 「Gemma 4 移行で速度どれだけ劣化したか」確定、Mission 1 paper gap 分析の data point
- **環境**: GPU 1 (A100 80GB、Phase 2f と同 card で apples-to-apples 比較)
- **仕様**:
  - 別 venv `.venv-qwen-orig` 構築 (torch 2.x 系、transformers 4.x 系、Qwen tokenizer、VLA-Adapter 原実装の他依存)
  - VLA-Adapter 原 `finetune.py` / `run_libero_eval.py` (Qwen 0.5B backbone) をそのまま使用
  - 100 query の median query time 計測 (Phase 2c と同 protocol)
  - 副産物: LIBERO-Spatial eval success rate も同時取得 (paper 99.6% 再現性 check)
- **所要**: ~2-3 時間 (venv 構築 1 時間 + run 1-2 時間)
- **Deliverable**: `scripts/gemma4/test_15_qwen_baseline.py`、`runs/gemma4/eval/qwen_baseline/` (query time JSON、eval success rate)

#### A2. Component-level breakdown 実測

- **目的**: `torch.compile` 投入の bottleneck 特定、vision / LLM / action head の実測 break
- **環境**: GPU 1 (A100 80GB、A1 後 or 並行) or GPU 2 (40GB)
- **仕様**:
  - Phase 2c の `eval_libero_gemma4.py` に `torch.cuda.Event` insert
  - Vision encode (`vision_backbone.forward`) / LLM forward (`llm(...)`) / action head forward (`action_head.forward`) / denormalize / simulator step の 5 区間を計測
  - 28 query × 3 episode = 84 measurements の median
- **所要**: ~30 分
- **Deliverable**: `scripts/gemma4/test_16_latency_breakdown.py`、`runs/gemma4/latency_breakdown/result.json`

#### A3. `torch.compile` smoke test

- **目的**: R2 の `get_per_layer_inputs` 明示渡しが graph break を起こすか確認、動けば gain 実機 deploy 用
- **環境**: GPU 1 or 2
- **仕様**:
  - Phase 2c の random-init model に `torch.compile(model, mode="reduce-overhead")` を適用
  - 10 query warmup + 100 query measure
  - Graph break log を `TORCH_COMPILE_DEBUG=1` で取得、R2/R3 由来の break を列挙
  - 非 compile baseline との gain ratio 算定
- **所要**: ~1-2 時間 (compile 時間 5-10 分 + debug 調整含む)
- **Deliverable**: `scripts/gemma4/test_17_torch_compile_smoke.py`、graph break 一覧、gain ratio JSON
- **Escalation**: compile 失敗 or no-op gain (<1.05x) は Stage 2 scope 内での torch.compile 投入を断念、Stage 3 送り

#### D1 (新規、Multi-GPU scope 再拡張). DDP pre-flight smoke

- **目的**: R15/R16 active 化の動作確認、Stage 2 後半 / Stage 3 での DDP 経路使用前検証
- **環境**: GPU 2 + GPU 3 (40GB × 2) で開始、4 GPU 拡張は動作確認後
- **仕様**:
  - `torchrun --nproc_per_node=2 finetune_gemma4.py --ddp_mode True --smoke_mode True`
  - per-GPU B=4、effective B=8 (Phase 2e と同 effective batch)
  - 100 step smoke
  - **Acceptance criteria**:
    1. LLM/VB leak check が DDP wrap 後に動く (`model.module.llm.parameters()` access、全 100 step で leak 0)
    2. loss finite、Phase 2e と同 order of magnitude
    3. per-step time 実測 (single GPU 0.87 s との比較)、throughput = 2 × 0.87 / per_step_DDP で評価 (期待 1.6-1.8x)
    4. Checkpoint save (rank 0 only) + load で state_dict に `module.` prefix 付与、single GPU load 時に strip して互換
    5. R16 TF + DDP CUDA init 順序: 各 worker process 冒頭で `tf.config.set_visible_devices([], 'GPU')` 実行、GPU memory grab なし
    6. NCCL all-reduce が frozen param (LLM 5104M + Vision 731M) を除外 (`requires_grad=False` の param が PyTorch DDP で自動除外)、実測 all-reduce 対象サイズが 675M 相当
- **所要**: ~1.5-2 時間
- **Deliverable**: `VLA-Adapter/vla-scripts/finetune_gemma4.py` (DDP mode flag 追加)、`scripts/gemma4/test_18_ddp_smoke.py` (launcher wrapper)、`runs/gemma4/ddp_smoke/result.json`
- **Escalation**: 上記 6 acceptance criteria のいずれか fail で即停止、原因切り分けを User 判断

### Tier B: Phase 2e 完走後 (4/22 以降、Phase 2g と合わせて判断)

#### B1. Bidirectional attention pre-flight smoke

- **目的**: Phase 2i (Bidirectional retrain) 着手前の動作確認、`use_bidirectional_attention=True` で model が forward 通るか
- **環境**: GPU 1 or 任意の空き GPU
- **仕様**: Phase 2c の random-init model で `model.config.text_config.use_bidirectional_attention = True` 設定、100 step training smoke で loss finite 確認
- **所要**: ~30 分
- **Deliverable**: `scripts/gemma4/test_19_bidirectional_smoke.py`
- **判断分岐**: 動作確認取れたら Check-in #18 での Bidirectional 投入判断材料、動かなければ自前 patch 要で Stage 2 scope 再判断 (User check-in)

### Tier C: Check-in #18 判断後 (Bidirectional 投入 or ablation 並列)

Phase 2g 完了後 (Check-in #17)、paper gap 分析で Bidirectional 投入判断 (Check-in #18) 次第で以下 2 分岐:

#### C1 path: Bidirectional 投入 → DDP 4-GPU retrain

- **条件**: Check-in #18 で「Bidirectional 投入価値あり」判断 (success rate 70-85% で頭打ち等)
- **環境**: GPU 2-5 (40GB × 4) で DDP、D1 動作確認済前提
- **仕様**: per-GPU B=2、effective B=8 or per-GPU B=4、effective B=16 (D1 の memory 実測次第)、max_steps=200000、baseline checkpoint から fine-tune
- **所要**: **0.6-0.7 日** (D1 実測 throughput 依存、期待 3-3.5x speedup)、vs 単 GPU 2 日
- **Deliverable**: `configs/gemma4_libero_spatial_bidir_ddp.yaml`、本番 retrain checkpoint、Bidirectional eval result

#### C2 path: Bidirectional skip → 独立 job 並列で ablation

- **条件**: Check-in #18 で「Bidirectional 不要」判断 (既に 95%+ 到達等)
- **環境**: GPU 2-5 (40GB × 4) で独立 4 job 並列
- **Ablation 候補**:
  - Shuffle buffer size ablation (1000 / 100000 / 256000 で success rate variance)
  - Multi-seed runs (seed 42 / 123 / 456 / 789 で paper value との gap variance)
  - Action chunk size ablation (chunk 4 / 8 / 16 で eval rate vs 精度 trade-off)
- **所要**: 各 run 2 日 × 並列 = **2 日 wall clock**、Buffer 消費同等
- **Deliverable**: Mission 1 paper gap 分析の data density 向上、Check-in #17 で提示する success rate に root cause analysis 追加可能

### Tier D: Stage 2 Buffer 末尾 (5/15-5/17 頃、Mission 1-4 達成後)

**D2 (新規、Stage 3 risk hedge)**: DDP 本格 smoke (4-GPU × 500 step)

- **目的**: D1 (2-GPU × 100 step) では検出できない 4-GPU 固有の issue (NCCL communication、activation memory) を検証、Stage 3 着手時の未知リスク除去
- **所要**: ~1 時間
- **実施条件**: Mission 1-4 全て達成済、Buffer 残 2-3 日以上

---

## 4. Check-in Point 追加

Stage 2 Plan v2 の #10-24 に以下を追加:

| # | タイミング | 提示物 |
|---|---|---|
| **12.5** | D1 (DDP pre-flight smoke) 完了後 | 6 acceptance criteria 結果、per-GPU memory、throughput、all-reduce 対象 param サイズ |
| **16.5** | A1 + A2 + A3 完了後 (Tier A まとめ) | Qwen 0.5B query time、breakdown、torch.compile gain、Mission 1 paper gap 分析の data point |
| **18.5** | C1 or C2 分岐判断時 (Check-in #18 後) | Bidirectional 投入可否の最終判断、C1/C2 path 選択根拠 |

---

## 5. Escalation Policy 追加

Stage 2 Plan v2 §9 の既存 18 項目に以下を追加:

19. **A1 (Qwen 0.5B) venv 構築失敗**: `torch` と `transformers` の version 互換性、Qwen tokenizer 不備等。~1 時間の投資で解決せねば A1 断念、他 Tier 優先
20. **A3 (torch.compile) graph break 多数**: R2 `get_per_layer_inputs` 経路で不可避な break、no-op gain (<1.05x) なら Stage 2 scope 断念
21. **D1 acceptance criteria fail**: 6 criteria のいずれか fail、特に NCCL all-reduce が frozen param 巻き込む / LLM leak check が DDP 後に動かない / checkpoint 'module.' prefix handling 失敗
22. **D1 per-step time が single GPU 悪化** (例: per-step 1.5x 以上、throughput <1.3x): DDP overhead 過大、原因切り分け要 (NCCL backend、batch_size 設定、data loader 競合等)
23. **Tier A 実施中に Phase 2e per-step time が 1.0 s/step 超** (+15% 閾値、Escalation #9 warning 1.29 s の前段): Tier A 並列実行が Phase 2e に干渉、Tier A 即停止して Phase 2e 優先

---

## 6. Decision Tree: Phase 2i 以降

Check-in #17 (Phase 2g 結果) → Check-in #18 (Bidirectional 投入判断) の 2x2 判断:

```
Phase 2g success rate
├─ 95-99% (paper 近傍)
│   ├─ Bidirectional 投入: C2 path (ablation 並列)、Mission 1 data density 強化優先
│   └─ Bidirectional skip:  C2 path (同上)、Mission 2 自前データに Buffer 転用
├─ 85-95% (良好だが gap あり)
│   ├─ Bidirectional 投入: C1 path (DDP 4-GPU retrain、0.6-0.7 日)、Mission 4 達成
│   └─ Bidirectional skip:  C2 path (ablation で gap 原因切り分け)
├─ 70-85% (gap 大)
│   └─ Bidirectional 投入ほぼ確定: C1 path、success rate 底上げ試行
└─ <70% (paper と大きく乖離)
    └─ Escalation → 根本原因調査 (pipeline bug、hyperparam、等)、Stage 2 scope 再判断
```

---

## 7. Mission 寄与 Mapping

| Action | Mission 1 (paper gap) | Mission 2 (自前データ) | Mission 3 (Demo) | Mission 4 (Bidirectional) |
|---|---|---|---|---|
| A1 Qwen 0.5B 比較 | ◎ (speed gap 確定) | ○ (推論速度知見) | △ | △ |
| A2 Component breakdown | ◎ (bottleneck 特定) | ○ | △ | ◎ (最適化判断) |
| A3 torch.compile smoke | ○ (推論高速化) | ○ | ◎ (Demo で smoother playback) | ◎ (retrain 後 deploy) |
| D1 DDP pre-flight | — | ○ (fine-tune 高速化) | — | **◎ (Phase 2i 3x 高速化前提)** |
| B1 Bidirectional smoke | ○ | — | — | **◎ (Phase 2i 前提)** |
| C1 DDP Bidirectional retrain | ○ (gap 縮小試行) | — | — | **◎ (Mission 4 本体)** |
| C2 Ablation 並列 | **◎ (variance / hyperparam analysis)** | — | — | — |
| D2 DDP 4-GPU 本格 smoke | — | — | — | ○ (Stage 3 hedge) |

---

## 8. Deliverables

Stage 2 前半 Deliverables (Plan v3 §12) に以下を追加:

| # | ファイル | Phase |
|---|---|---|
| 14 | `scripts/gemma4/test_15_qwen_baseline.py` + `.venv-qwen-orig/` | A1 |
| 15 | `scripts/gemma4/test_16_latency_breakdown.py` | A2 |
| 16 | `scripts/gemma4/test_17_torch_compile_smoke.py` | A3 |
| 17 | `scripts/gemma4/test_18_ddp_smoke.py` + `finetune_gemma4.py` DDP mode 追加 | D1 |
| 18 | `scripts/gemma4/test_19_bidirectional_smoke.py` | B1 |
| 19 | `configs/gemma4_libero_spatial_bidir_ddp.yaml` | C1 (条件付) |
| 20 | `docs/gemma4_migration_log.md` に Stage 2 後半 section 追記、scope 変更 1 行記録 | 各 Phase |
| 21 | `docs/gemma4_stage2_latter_half_plan.md` (本ファイル) copy | 開始時 |

---

## 9. Compute Budget (Stage 2 後半、Day 2-28)

| Day | Phase / Tier |
|---|---|
| Day 2 (4/21) | Phase 2e 継続 + Tier A 並列 (A1 GPU 1、A2/A3 GPU 1 or 2、D1 GPU 2+3) |
| Day 3 (4/22 早朝) | Phase 2e 完走 + Phase 2g (~34 分) + Check-in #17 |
| Day 3 後半 | Phase 2h gap 分析 + B1 + Check-in #18 |
| Day 4-5 | C1 (Bidirectional DDP retrain ~0.7 日) or C2 (ablation 並列 ~2 日) |
| Day 6-7 | Phase 2j (teleop 準備) + Phase 2k (single demo sanity) |
| Day 8-9 | Phase 2l (データ収集 30-50 demo) |
| Day 10 | Phase 2m (RLDS 変換 + fine-tune) |
| Day 11-13 | Phase 2n (Demo 動画 + 提出物) |
| Day 14-27 | **Buffer ~14 日** (debug、retraining、Stage 3 準備、D2 Stage 3 hedge 等) |
| Day 28 (5/18) | ハッカソン提出 |

Stage 2 Plan v2 の Buffer 9 日 → 後半 plan で **~14 日に拡大**。Mission 4 投入 + Mission 2 完全達成 + Stage 3 hedge 余裕。

---

## 10. References

- Stage 1 全履歴: `docs/gemma4_migration_log.md`
- Stage 1 plan: `docs/gemma4_stage1_phase1c1d_plan.md`
- Stage 2 plan v2: `docs/gemma4_stage2_plan.md`
- Stage 2 前半 prompt v3: `docs/claude_code_prompt_stage2.md` (single GPU 版)
- 原実装 warmup: `VLA-Adapter/vla-scripts/finetune.py:1060-1065`
- 原実装 scheduler: `VLA-Adapter/vla-scripts/finetune.py:915-921`
- 原実装 eval: `VLA-Adapter/experiments/robot/libero/run_libero_eval.py`
- HF #45242 (Gemma 4 KV 共有 + GC bug): R6 の根拠
- VLA-Adapter 原実装 (Qwen 0.5B、A1 比較元): `VLA-Adapter/vla-scripts/finetune.py` (Qwen backbone 前提) + `VLA-Adapter/experiments/robot/libero/run_libero_eval.py`

---

## 11. 完了認定 (Stage 2 後半)

以下 4 軸を Stage 2 前半 4 軸と合わせて達成で Stage 2 完了:

**Stage 2 前半 4 軸** (Plan v2 §13):
1. 本番 train 完走 (Phase 2e)
2. LIBERO-Spatial eval + paper gap 分析 (Phase 2g + 2h)
3. 自前データ fine-tune 成立 (Phase 2j-2m)
4. Demo 動画 + ハッカソン提出 (Phase 2n)

**Stage 2 後半追加** (本 plan):
5. Tier A 全件 (A1 + A2 + A3) 完了、Mission 1 paper gap 分析の data density 向上
6. D1 (DDP pre-flight smoke) 6 acceptance criteria 全 pass、Stage 3 risk hedge 完了
7. C1 (Bidirectional DDP retrain) or C2 (ablation 並列) のいずれか完了、Check-in #18 判断根拠明記
