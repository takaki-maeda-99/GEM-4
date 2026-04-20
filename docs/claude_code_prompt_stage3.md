# Claude Code 着手プロンプト: Gemma 4 E2B Stage 3 (X-VLA Soft Prompt 統合 Pretrain)

## このプロンプトの扱い方

- Claude Code の新規 or 継続セッションに貼る (Stage 2 後半 prompt と併走)
- 実装計画の詳細は `docs/gemma4_stage3_plan.md` に記載、これを**必ず最初に精読**
- Stage 1-2 全履歴は `docs/gemma4_migration_log.md`
- 作業は Check-in Point 駆動、着手前宣言・完了後報告、User 明示許可なき自動進行禁止
- **重要**: Stage 3 は Stage 2 と並列、Stage 2 Mission 1-4 を犠牲にしない

---

## Stage 3 Mission

Hackathon 提出 (5/18) までに **X-VLA Soft Prompt 機構を `VLAAdapterGemma4` に統合し、cross-embodiment 実ロボットデータで pretrain を実施**。Stage 2 Mission 2 (自前データ fine-tune) の starting checkpoint として pretrained weight を提供する。

**Main role**: Mission 2/3 (自前実ロボット demo) の quality 担保、few-shot adaptation data efficiency 向上
**Secondary role**: Submission の technical depth 追加
**Non-role**: Ablation、比較検証、paper 値再現は scope 外

**Hard deadline**: **5/13** まで pretrain 結果が出ない場合、fallback (Stage 2 LIBERO checkpoint から自前 fine-tune) に切替。

---

## 最初にやること (セッション開始時)

1. 作業 dir 確認 (`/misc/dl00/takaki/vla-gemma-4/`)
2. 以下を順に read:
   - `docs/gemma4_stage3_plan.md` — 本タスク詳細計画 (必読)
   - `docs/gemma4_migration_log.md` — Stage 1-2 全履歴
   - `docs/gemma4_stage2_latter_half_plan.md` — Stage 2 後半 (並列実行中)
   - `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` — `VLAAdapterGemma4`、Soft Prompt 統合対象
   - `VLA-Adapter/vla-scripts/finetune_gemma4.py` — pretrain mode 追加対象
   - X-VLA paper (https://arxiv.org/abs/2510.10274v1) — Section 3 (Soft Prompt)、Section 4.1 (Architecture)、Appendix C/G を参照
3. 環境確認:
   - `nvidia-smi` で GPU 状態 (Stage 3 用に GPU 2-5 の 40GB × 4 を確保)
   - Disk 空き容量確認 (Tier 1 data ~500 GB 想定)
   - Network bandwidth 確認 (TFDS download 用)

---

## Stage 3 実ロボット仕様 (Pretrain data 選定の基準)

User 自前アーム仕様:
- **DOF**: 6 DOF arm + 1 DOF gripper (= 7-dim action)
- **Camera**: 俯瞰 (overhead) + 手元 (wrist) の 2 カメラ
- **Action space**: EEF xyz + rotation + gripper binary

この仕様に近いほど pretrain transfer 効率が高い。

---

## 絶対ルール (Stage 2 継承 + Stage 3 追加)

### R1-R17 (Stage 1-2 から継承、変更なし)

特に:
- **R4** (LLM + Vision 完全凍結): Stage 3 pretrain でも維持、Soft Prompt は Adapter 扱い
- **R6** (`gradient_checkpointing` 永久禁止): HF #45242
- **R10** (Linear warmup 10% → 100%): Stage 3 pretrain でも `lr_warmup_steps=500` 同様
- **R14** (LoRA 永久 Out of Scope)
- **R15/R16** (DDP Active): Stage 3 pretrain は DDP 4-GPU 前提

### R18 (新規). Stage 3 は Stage 2 Mission 1-4 を犠牲にしない

Stage 3 の Phase / process / GPU 使用は、Stage 2 の Phase 2e / 2g / 2m-2n に干渉してはならない。Stage 3 debug で Stage 2 が遅延する場合、**即 Stage 3 を一時停止、Stage 2 優先**。

### R19 (新規). 5/13 Hard deadline 厳守

5/13 0:00 時点で以下のいずれかを確定:
- (a) Pretrain 完了、zero-shot LIBERO sanity で >20% success、Phase 2m の starting checkpoint として使用
- (b) Pretrain 未完 or sanity 失敗、fallback (Stage 2 LIBERO checkpoint) で Phase 2m kick-off

5/13 以降は Stage 3 の continuation を許可しない、Mission 2/3 に集中。

### R20 (新規). Pretrain は 1 run のみ、Ablation 禁止

User 指示により ablation 不要。pretrain 本番は Tier 1 data × Soft Prompt あり × 1 回 の run のみ。比較検証は Hackathon 後対応。

### R21 (新規). X-VLA paper の一部機能のみ採用

採用:
- Soft Prompt Library
- 配置: 案 B (`inputs_embeds` 前段 concat)
- Two-step adaptation (prompt warmup 1k step → joint)
- Custom LR (soft prompt + vision_projector に backbone の 1/4)
- Balanced data sampling
- Action space 正規化 (EEF + Rotate6D + gripper binary)

採用しない (Stage 3 scope 外):
- Per-dataset Input/Output projection library
- Temporal downsampling
- Flow matching action head (Pro action head 維持)

---

## マイクロフェーズ

### Phase 3a: Data pipeline 構築 (Stage 2 Phase 2e 進行中、**今すぐ着手**)

| # | Step | 内容 | 環境 | 所要 |
|---|---|---|---|---|
| 3a-1 | **Data download 着手** | TFDS download: Taco Play + BridgeData v2 + OXE Fractal | CPU + disk + network | ~10-20h |
| 3a-2 | Dataset statistics 計算 | Per-dataset `dataset_statistics.json` (action 99-percentile) | CPU | ~2-3h |
| 3a-3 | Multi-dataset dataloader 実装 | Weighted sampling、cross-trajectory shuffle、dataset_id 付与、action space 正規化 (EEF + Rotate6D + gripper binary) | CPU | ~1 日 |
| 3a-4 | 動作確認 | 1 batch forward、shape 確認、dataset_id 伝播、正規化 action が [-1, 1] 範囲 | GPU 1 | ~半日 |

**Deliverable**:
- `scripts/stage3/download_data.py`
- `scripts/stage3/compute_dataset_statistics.py`
- `scripts/stage3/multi_dataset_loader.py`
- `runs/gemma4/stage3_data/` (downloaded data、statistics json)

### Phase 3b: Soft Prompt 統合実装 (Phase 2e 完走前後、~3-4 日)

| # | Step | 内容 | 環境 | 所要 |
|---|---|---|---|---|
| 3b-1 | `SoftPromptLibrary` module | `nn.Embedding(num_datasets, num_tokens × hidden_dim)` | — | ~3h |
| 3b-2 | `VLAAdapterGemma4` 統合 | 案 B (inputs_embeds 前段 concat)、`per_layer_inputs` / `attention_mask` / `position_ids` 拡張 | — | ~半日 |
| 3b-3 | `finetune_gemma4.py` pretrain mode | Multi-dataset sampling、custom LR param groups、two-step adaptation toggle、dataset_id 経由の soft_prompt 選択 | — | ~半日 |
| 3b-4 | 100 step smoke | Pretrain mode smoke: R4 (LLM leak = 0)、loss finite、soft_prompt grad 非ゼロ、Stage 2 regression 非破壊 (既存 single-dataset fine-tune が動くことを確認) | GPU 1 | ~半日 |

**Deliverable**:
- `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` (SoftPromptLibrary 追加、forward 拡張)
- `VLA-Adapter/vla-scripts/finetune_gemma4.py` (pretrain mode)
- `scripts/stage3/test_20_pretrain_smoke.py`

### Phase 3c: Pretrain 本番 Run (4/25-5/8 想定、~13-14 日)

| # | Step | 内容 | 環境 | 所要 |
|---|---|---|---|---|
| 3c-1 | DDP kick-off | `torchrun --nproc_per_node=4` GPU 2-5、per-GPU B=4 effective B=16、WandB online | 4 GPU 40GB | kick-off 30 分 |
| 3c-2 | Pretrain 本番 | Tier 1 data × Soft Prompt、max_steps=300000、warmup 500、custom LR、5/13 hard stop or loss plateau | 4 GPU 40GB | continuous run |
| 3c-3 | 途中 checkpoint sanity | 50k/100k/150k/200k step 時点で LIBERO-Spatial zero-shot rollout (1 task × 1 episode、~5 分) | GPU 1 | each ~5 分 |

**Monitoring** (finetune_gemma4.py pretrain mode 組込):
- Per-dataset loss breakdown
- Soft prompt norm trajectory (per-dataset)
- LLM grad leak check (1k step 毎 full scan)
- Per-step time rolling (Escalation #23 継承)
- GPU memory stable

**Deliverable**:
- `scripts/stage3/pretrain_kickoff.py` (DDP launcher)
- `scripts/stage3/zero_shot_libero_sanity.py`
- `runs/gemma4/stage3_pretrain/` (checkpoints、WandB log、sanity result)

### Phase 3d: Fine-tune 分岐 (5/13 hard deadline 判定)

| # | Step | 内容 | 環境 | 所要 |
|---|---|---|---|---|
| 3d-1 | **5/13 判定** | Pretrain 最新 checkpoint の zero-shot rollout success rate 確認 | GPU 1 | ~15 分 |
| 3d-2 | Success branch | Phase 2m (自前 fine-tune) の starting checkpoint として pretrained を使用 | Phase 2m へ引渡 | — |
| 3d-3 | Fallback branch | Stage 2 LIBERO 200k checkpoint を Phase 2m に使用 | Phase 2m へ引渡 | — |

**判定 criteria**:
- Zero-shot LIBERO-Spatial success rate > 20% → Success
- Loss converged (最後の 10k step で stable 減少) → Success
- 上記満たさず → Fallback

---

## Check-in Points

Stage 2 Check-in に追加:

| # | タイミング | 提示物 |
|---|---|---|
| **25** | Phase 3a-1 完了時 (Data download) | Download 成否、各 source のサイズ、TFDS load 動作確認 |
| **26** | Phase 3a-4 完了時 (dataloader) | 1 batch の shape、dataset_id 伝播、action 正規化 range、RAM/VRAM 影響 |
| **27** | Phase 3b-4 完了時 (100 step smoke) | Smoke 結果 (loss、grad、leak)、Stage 2 regression 非破壊確認、soft_prompt norm trajectory |
| **28** | Phase 3c-1 完了時 (DDP kick-off) | WandB run url、per-GPU memory、initial step time、4 GPU throughput (D1 比較) |
| **29** | Phase 3c-3 途中 (50k / 100k / 150k / 200k step sanity) | 各 checkpoint の zero-shot LIBERO success rate、loss trajectory、soft_prompt norm |
| **30** | **5/13 分岐判定** | Success / Fallback 判定根拠、Phase 2m への引渡準備 |

---

## Escalation Policy (Stage 2 継承 + Stage 3 追加)

既存 Escalation #1-23 + 以下追加:

24. **3a data download 失敗**: Tier 1 data のいずれか TFDS 取得不能 → 該当 skip or Tier 2 代替判断
25. **3b smoke で Stage 2 regression 破壊**: soft_prompt 追加で既存 forward path 壊れる → revert、soft_prompt 経路分離強化
26. **3c 本番で 10k step 以内に loss NaN/Inf**: Multi-dataset sampling weight or action space 不整合 → 即停止
27. **3c 本番で LLM grad leak 発生** (R4 違反): 即停止、soft_prompt grad 伝播 debug
28. **3c 途中 checkpoint zero-shot rollout が全 checkpoint で < 5%**: pretrain 効果皆無 → 5/10 時点で fallback 確定
29. **3c 累計期間が 5/13 hard deadline 超過**: 即停止、fallback へ
30. **5/13 時点で Phase 2m が pretrain 待ちで停滞**: fallback 切替、Mission 2 優先

---

## やってはいけないこと

Stage 2 禁止事項に加え:

- **Stage 3 failure で Mission 2/3 を犠牲にする** (R18 違反)
- **5/13 hard deadline を越えて pretrain 継続** (R19 違反)
- **Ablation run の追加** (R20 違反、User 明示で不要)
- **X-VLA paper の scope 外 feature 実装** (Temporal downsampling、per-dataset projection 等) (R21 違反)
- **R1-R9 改変** (placeholder ID、feature_norm、use_cache 等、Stage 2 確定事項を壊さない)
- **Stage 3 の GPU 使用で GPU 0 (Stage 2 Phase 2e) 干渉**
- **Check-in Point スキップ** (特に Check-in #27 smoke regression 確認)

---

## Out of Scope

- Ablation (with/without Soft Prompt 比較)
- Multiple pretrain run (data source 組合せ違い、model size 違い等)
- Paper 値再現、comprehensive eval
- X-VLA の全 feature 実装
- 論文執筆、学会発表準備
- 5/13 以降の pretrain 継続

---

## Deliverables (Stage 3)

| # | ファイル |
|---|---|
| 22 | `scripts/stage3/download_data.py` |
| 23 | `scripts/stage3/compute_dataset_statistics.py` |
| 24 | `scripts/stage3/multi_dataset_loader.py` |
| 25 | `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` (SoftPromptLibrary 統合) |
| 26 | `VLA-Adapter/vla-scripts/finetune_gemma4.py` (pretrain mode) |
| 27 | `scripts/stage3/test_20_pretrain_smoke.py` |
| 28 | `scripts/stage3/pretrain_kickoff.py` (DDP launcher) |
| 29 | `scripts/stage3/zero_shot_libero_sanity.py` |
| 30 | `runs/gemma4/stage3_pretrain/` (checkpoints、WandB log、sanity) |
| 31 | `docs/gemma4_migration_log.md` Stage 3 section + 5/13 分岐判定 entry |

---

## 報告フォーマット (Stage 2 と同じ)

```
=== Phase 3X 完了報告 ===

Status: [OK / FAIL / BLOCKED]
変更ファイル: ...
実行コマンド: ...
環境: [GPU 番号、VRAM]
主要数値: ...
Escalation 該当: [YES / NO、該当の場合は番号]
Check-in Point #NN 判断要請: ...
Stage 2 進行への影響: [なし / あり、具体内容]
```

---

## 即時アクション (User 許可待ち)

**今すぐ kick off 可能な item**:

1. **Phase 3a-1: Data download** — GPU 不要、Phase 2e に非干渉、~10-20h かけて裏で実行
2. **X-VLA paper の Section 3/4/Appendix C/G 精読** — 実装 detail の再確認
3. **Phase 3b-1/2 の code 書き** — Phase 2e GPU 0 独立、smoke は Phase 2e 完走後に実施

**User 許可待ち**:
- Phase 3a-1 着手 (データ download start)
- Phase 3b 着手 (Soft Prompt implementation start)

---

## References

- Stage 3 詳細計画: `docs/gemma4_stage3_plan.md`
- Stage 2 後半 plan: `docs/gemma4_stage2_latter_half_plan.md`
- Stage 2 前半 plan: `docs/gemma4_stage2_plan.md`
- Stage 1-2 migration log: `docs/gemma4_migration_log.md`
- X-VLA paper: https://arxiv.org/abs/2510.10274v1 (Section 3, 4, Appendix C/G 重点)
- X-VLA code: https://github.com/2toinf/X-VLA (reference only、直接 fork せず Gemma 4 統合する)
- VLA-Adapter 原実装: `VLA-Adapter/`

---

## Stage 3 完了認定

以下を満たせば Stage 3 完了:

1. Phase 3a-c 全 phase 完了 (or Phase 3c が 5/13 hard deadline で停止)
2. Phase 3d 判定実施、Success or Fallback branch 確定
3. Phase 2m (自前 fine-tune) の starting checkpoint が選択済
4. Stage 2 Mission 1-4 進行に遅延なし
5. Submission narrative (success / fallback どちらでも) 確定

Hackathon 提出 (5/18) 時点で Mission 2/3 と Stage 3 (success or fallback) の統合結果が提出物に反映されていれば完了。
