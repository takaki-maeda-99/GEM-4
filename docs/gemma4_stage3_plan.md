# Gemma 4 E2B VLA-Adapter Stage 3 Plan: X-VLA Soft Prompt 統合 Pretrain

**作成**: 2026-04-20 夜 (Phase 2e ~26k/200k 進行中)
**位置づけ**: Stage 2 と並列、Hackathon 提出 (5/18) までに実ロボットデータでの事前学習を完了し、自前データ (Mission 2) fine-tune 時に pretrained checkpoint を使用できる状態にする
**Main target**: Mission 2 (自前データ fine-tune) + Mission 3 (Demo 動画) の quality 向上、X-VLA 統合自体は submission の technical depth として機能

---

## 1. Mission 再定義

Hackathon submission の strength は **「Gemma 4 backbone で VLA を構築し、自前実ロボット環境で動作する application として demonstrate」**。X-VLA Soft Prompt pretrain は以下の役割:

- **Primary role**: 自前データ (実ロボット) 10-50 demo 級の few-shot adaptation で good behavior を引き出す、Mission 2/3 の demo quality 担保
- **Secondary role**: Submission narrative の technical depth 追加 (cross-embodiment pretrain 実施の事実)
- **Non-role**: Ablation、比較検証、paper 値再現 (後日対応)

つまり pretrain 成功なら Mission 2 の starting checkpoint として使用、失敗なら Stage 2 LIBERO-Spatial checkpoint を使う fallback。

---

## 2. 実ロボット環境仕様 (Mission 2 target、Pretrain data 選定の基準)

- **DOF**: 6 DOF arm + 1 DOF gripper (7-dim action、Stage 2 の LIBERO と action 空間 compatible)
- **Camera**: 俯瞰 (overhead) + 手元 (wrist) の 2 カメラ
- **Action space**: EEF xyz + EEF rotation (Rotate6D or quaternion or euler、Stage 2 で使用中の Rotate6D 継承可能性高) + gripper binary

**Pretrain data 選定への帰結**: 以下条件を満たす data source を優先:

| 条件 | 理由 |
|---|---|
| 6DOF + gripper の single arm | 自前アーム embodiment 整合 |
| 俯瞰 + wrist 2 カメラ | Camera setup 整合、vision backbone transfer 効率 |
| EEF 座標系の action | Joint 座標系より転移性高 |
| RLDS 形式で即利用可能 | Hackathon 期間制約、変換コスト回避 |

---

## 3. Pretrain Data 選定

前提: 変換コストなし (既 RLDS / TFDS 公開) の data source のみ採用。

| Data source | Episode 数 | Embodiment | Camera | 適合度 | 採用 |
|---|---|---|---|---|---|
| **BridgeData v2** | 60K | WidowX (6DOF + gripper) | 俯瞰 1 cam | ★★★★ | **必須** |
| **OXE Fractal** (RT-1 data) | 130K | Google Robot (7DOF + gripper) | 俯瞰 1 cam | ★★★ | **必須** |
| **OXE Bridge subset** | (BridgeData と重複のため skip) | — | — | — | skip |
| **OXE Kuka** | 580K | Kuka (7DOF + gripper) | 俯瞰 1 cam | ★★★ | **採用** |
| **OXE Taco Play** | 3K | Franka (7DOF + gripper) | 俯瞰 + wrist | ★★★★★ | **最優先** (自前仕様と最も整合) |
| **Droid** | 92K | Franka (7DOF + gripper) | 俯瞰 + wrist + side | ★★★★ | **採用** (公式 RLDS) |
| AGIBOT / RoboMind | — | — | — | 変換要 | skip |

### Tier 1 (Hackathon 期間で確実採用)

- **Taco Play** (Franka 俯瞰 + wrist、自前と同構成): 3K
- **BridgeData v2** (WidowX 6DOF): 60K
- **OXE Fractal** (Google Robot): 130K

合計 ~193K episodes、自前の Franka 系に近い data が主体、cross-embodiment diversity 確保。

### Tier 2 (期間余裕あれば追加)

- **Droid** (Franka 大規模): 92K、俯瞰 + wrist camera setup が自前に近い
- **OXE Kuka**: 580K、data 量最大、embodiment は異なるが generalization 寄与期待

### Download & Setup 所要

- Tier 1 合計 download: ~500 GB (推定)、10MB/s で ~14h、100MB/s LAN で ~1.5h
- RLDS format なので変換不要、dlimp 経由で Stage 2 の dataloader 再利用可能
- Dataset statistics 計算: per-dataset、各 30 分-1 時間

**今すぐ kick off**: Data download は GPU 0 の Phase 2e と完全独立、Claude Code に直ちに指示可能。

---

## 4. Soft Prompt 統合設計 (Minimum Viable 版)

### 採用する X-VLA 要素

| 要素 | 採用 | 実装スコープ |
|---|---|---|
| **Soft Prompt Library** (per-dataset learnable embedding) | ○ | `nn.ParameterList` or `nn.Embedding(num_datasets, num_tokens × hidden_dim)` |
| **配置: 案 B** (`inputs_embeds` 前段に concat) | ○ | R1-R9 の placeholder ID 拡張不要、整合性維持 |
| **Two-step adaptation** (prompt warmup → joint) | ○ | Fine-tune 時に prompt only 1k step → joint |
| **Custom LR** (soft prompt + VL 系 module に reduced LR) | ○ | AdamW param group で実装 |
| **Balanced data sampling** (cross-domain + cross-trajectory shuffle) | ○ | dlimp / TFDS の weighted sampling |
| **Action space 正規化** (EEF + Rotate6D + gripper binary) | ○ | 自前も同 format に統一、pretrain / fine-tune / 自前 data で整合 |
| **Per-dataset Input/Output projection library** | × | Stage 2 の vision_projector / action_head を flip で共通化、per-dataset projection は Hackathon scope 外 |
| **Temporal downsampling** (30 anchor over 4s) | × | Stage 2 Pro action head chunk=8 と整合させるため skip |
| **Flow matching action head** | × | Pro action head 維持 (VLA-Adapter 設計核心、X-VLA action head 書き換えは scope 外) |

### 配置 (案 B):

```python
# VLAAdapterGemma4.forward 追加部分
def forward(self, pixel_values, input_ids, proprio, dataset_id, actions=None):
    # 既存処理: vision encode + input_ids embed + per_layer_inputs 計算
    vision_embeds = self.vision_projector(vision_features)  # (B, 512, hidden)
    proprio_embed = self.proprio_projector(proprio)         # (B, 1, hidden)
    raw_embeddings = self.llm.get_input_embeddings()(input_ids)
    
    # 既存の Option A 上書き (action_queries)
    embeddings = raw_embeddings.clone()
    for b in range(B):
        embeddings[b, action_positions] = self.action_queries.weight
    
    # 新規: Soft Prompt concat 前段
    soft_prompt_embeds = self.soft_prompt_library(dataset_id)  # (B, num_soft_tokens, hidden)
    
    # 最終 inputs_embeds
    inputs_embeds = torch.cat([soft_prompt_embeds, embeddings], dim=1)
    
    # per_layer_inputs は input_ids ベース (R2)、soft_prompt 位置は zero or dummy で拡張
    soft_prompt_ple = torch.zeros(B, num_soft_tokens, ple_dim, device=device)
    per_layer_inputs_extended = torch.cat([soft_prompt_ple, per_layer_inputs], dim=1)
    
    # attention_mask / position_ids も soft_prompt 分拡張
    attention_mask_extended = cat([ones(B, num_soft_tokens), attention_mask], dim=1)
    position_ids_extended = arange(0, num_soft_tokens + L, device=device).unsqueeze(0).expand(B, -1)
    
    # LLM forward
    out = self.llm(inputs_embeds=inputs_embeds_extended, per_layer_inputs=per_layer_inputs_extended, ...)
    
    # Action head は action_queries 位置の hidden state のみ使用 (既存)
    action_hidden = out.last_hidden_state[:, num_soft_tokens + action_offset : num_soft_tokens + action_offset + 64]
    predicted = self.action_head(action_hidden, proprio, ...)
```

### Soft Prompt のパラメータサイズ

- num_datasets: 3 (Taco Play + BridgeData v2 + OXE Fractal)、Tier 2 まで拡張で 5
- num_soft_tokens: 8 (X-VLA paper 24 blocks よりは少なめ、data 3-5 source 規模での最適点)
- hidden: 1536 (Gemma 4 E2B hidden_size)
- 合計: 5 × 8 × 1536 = **61K params** (全体 675.138M の 0.009%、VLA-Adapter のスケール観より小さい)

Stage 2 trainable 675.138M → Stage 3 pretrain trainable 675.138M + 0.06M ≈ 同 order。Memory 影響ほぼなし。

---

## 5. 実装 Phase 構造 (Stage 2 並列実施)

### Phase 3a: Data pipeline 構築 (Stage 2 Phase 2e と並列、即着手)

| Step | 内容 | 環境 | 所要 |
|---|---|---|---|
| 3a-1 | Tier 1 data download (Taco Play + BridgeData v2 + OXE Fractal)、RLDS 展開 | CPU + disk | ~10-20h (network bandwidth 依存) |
| 3a-2 | Per-dataset `dataset_statistics.json` 計算 (action 99-percentile) | CPU | ~2-3h |
| 3a-3 | Multi-dataset dataloader 実装 (weighted sampling、cross-trajectory shuffle、dataset_id 付与) | CPU | ~1 日 |
| 3a-4 | 1 batch forward 動作確認 (shape、dataset_id 伝播、action space 正規化) | GPU 1 (Phase 2e と独立) | ~半日 |

**Phase 2e と完全独立**、今すぐ kick off 可能。

### Phase 3b: Soft Prompt 統合実装

| Step | 内容 | 環境 | 所要 |
|---|---|---|---|
| 3b-1 | `SoftPromptLibrary` module 実装 (`nn.Embedding(num_datasets, num_tokens × hidden)`) | — | ~3h |
| 3b-2 | `VLAAdapterGemma4` に soft_prompt 経路追加 (案 B)、`per_layer_inputs`、`attention_mask`、`position_ids` 拡張ロジック | — | ~半日 |
| 3b-3 | `finetune_gemma4.py` に pretrain mode 追加 (multi-dataset sampling、custom LR param groups、two-step adaptation toggle) | — | ~半日 |
| 3b-4 | 100 step smoke (pretrain mode、R4 LLM leak、loss finite、soft_prompt grad 確認、Stage 2 regression 非破壊) | GPU 1 | ~半日 |

Phase 2e 完走 (4/22 早朝) 前後に完了目標。

### Phase 3c: Pretrain 本番 Run

| Step | 内容 | 環境 | 所要 |
|---|---|---|---|
| 3c-1 | DDP 4-GPU kick-off (GPU 2-5 or 空き 4 基、D1 で動作確認済) | DDP 4 GPU 40GB | kick-off ~30 分 |
| 3c-2 | Pretrain 本番 run (固定期間、5/13 まで or loss plateau 検出で停止) | DDP 4 GPU | 最大 ~13-17 日 |
| 3c-3 | 途中 checkpoint (10k / 50k / 100k / 150k step) で LIBERO-Spatial zero-shot rollout sanity (1-3 episode、~5 分 each) | GPU 1 (Phase 2e 完走後) | each ~5 分 |

**GPU 割当**:

- GPU 0 (80GB): Phase 2e、完走後 (4/22) Phase 2g → Phase 2m-2n
- GPU 1 (80GB): Phase 3b smoke、3c 途中 checkpoint sanity、Stage 2 Tier A (A1/A2/A3 後半)
- GPU 2-5 (40GB × 4): Phase 3c 本番 pretrain DDP
- GPU 6-7 (40GB × 2): 予備 (Bidirectional retrain 投入時、C2 ablation、等)

Phase 2e 完走 (4/22) 後、GPU 0 を Phase 2g/2h/2m/2n に使い、Phase 3c は GPU 2-5 で完全独立運行。

### Phase 3d: Fine-tune 分岐 (Phase 2m との統合)

**Day ~5/13 時点で分岐判定**:

```python
if pretrain_done_and_zero_shot_rollout_success_rate > 20%:
    # Success branch: Pretrained checkpoint から自前データ fine-tune
    starting_checkpoint = "runs/gemma4/stage3_pretrain/latest.pt"
    narrative = "Cross-embodiment pretrain 効果を自前実ロボットで示す"
else:
    # Fallback: Stage 2 LIBERO-Spatial 200k から自前データ fine-tune
    starting_checkpoint = "runs/gemma4/stage2_libero_spatial/200k.pt"
    narrative = "Gemma 4 VLA-Adapter の LIBERO baseline + 自前データ"
```

この判定は Phase 2j (teleop 環境準備) が完了していれば自前データ収集と並行で実施可。**5/13 を hard deadline** として設定、以降は Mission 2/3 に集中。

---

## 6. Timeline (Stage 3 と Stage 2 の実時系列)

Phase 2e 進行中 (2026-04-20 夜~) から 5/18 submission までの 28 日間、Stage 2 と Stage 3 を並列実行:

| Day | Stage 2 | Stage 3 |
|---|---|---|
| 1 (4/20) | Phase 2e 進行、D1 完了 | **Phase 3a-1 start** (Data download) |
| 2 (4/21) | Phase 2e 継続、Tier A + B1 | Phase 3a-1/2 (download 継続、stats 計算) |
| 3 (4/22) | Phase 2e 完走 + Phase 2g + #17/#18 | Phase 3a-3/4 (dataloader 動作確認) |
| 4-5 (4/23-24) | Phase 2h gap 分析 + C1/C2 分岐 | **Phase 3b** (Soft Prompt 統合実装) |
| 6 (4/25) | C1 Bidirectional retrain or C2 ablation | **Phase 3c-1 kick-off** (DDP pretrain start) |
| 7-12 (4/26-5/1) | Phase 2j + 2k (teleop 準備、single demo sanity) | Phase 3c 継続 + 途中 checkpoint sanity |
| 13-14 (5/2-3) | Phase 2l (自前データ収集 30-50 demo) | Phase 3c 継続 |
| 15 (5/4) | Phase 2l 完了 | Phase 3c 継続 |
| 16 (5/5) | Phase 2m kick-off (自前 fine-tune start) | Phase 3c **途中 checkpoint を Phase 2m 投入候補として評価** |
| 17-18 (5/6-7) | Phase 2m 継続 (pretrained から or fallback から) | Phase 3c 継続 (最終 checkpoint 保存) |
| 19 (5/8) | Phase 2m 完了 | Phase 3c 完了 |
| 20-22 (5/9-11) | Phase 2n (Demo 動画撮影、編集) | — |
| 23 (5/13) | **Pretrain 分岐 hard deadline**、Phase 3d 完了 | — |
| 24-27 (5/14-17) | Phase 2n 継続 (動画編集、README、提出物) | — |
| 28 (5/18) | Hackathon 提出 | — |

Buffer は 5/13-18 の 5 日間、自前データ fine-tune の retry + 動画編集時間として機能。

---

## 7. Pretrain config (Phase 3c 本番)

### 基本 config

```python
@dataclass
class PretrainConfig:
    # Architecture
    num_soft_prompt_tokens: int = 8
    num_datasets: int = 3  # Tier 1: Taco Play + BridgeData v2 + OXE Fractal
    
    # Data
    data_sources: List[str] = [
        "taco_play", "bridge_v2", "fractal20220817_data"  # TFDS names
    ]
    sampling_weights: List[float] = [0.20, 0.30, 0.50]  # Taco=Franka 重視、BridgeData 中、Fractal 量で押す
    
    # Training
    batch_size_per_gpu: int = 4  # D1 実測で 40GB A100 で確認値
    effective_batch_size: int = 16  # 4 GPU × 4
    max_steps: int = 300_000   # Hackathon 期間で回せる上限、時間制約で stop
    learning_rate_backbone: float = 2e-4  # Stage 2 準拠
    learning_rate_soft_prompt: float = 5e-5  # Custom LR、backbone の 1/4
    learning_rate_vision_projector: float = 5e-5  # Custom LR
    weight_decay: float = 0.01
    lr_warmup_steps: int = 500
    clip_max_norm: float = 1.0
    
    # Checkpoint
    save_freq: int = 10_000
    save_latest_checkpoint_only: bool = True
    
    # Hard deadline
    hard_stop_datetime: str = "2026-05-13 00:00:00"  # Phase 3d 分岐判定日
    
    # LIBERO sanity rollout
    sanity_rollout_every: int = 50_000  # 途中 checkpoint 毎の zero-shot 試走
```

### Pretrain 中の monitoring

- Loss trajectory (per-dataset breakdown)
- Soft prompt norm trajectory (per-dataset、収束確認)
- LLM grad leak (毎 1k step full scan、Stage 2 R4 継承)
- CPU/GPU memory stable
- Per-step time (Escalation #23 rolling window 活用)

---

## 8. Escalation Policy (Stage 3 追加)

Stage 2 Escalation #1-23 を継承 + 以下追加:

24. **Phase 3a data download 失敗**: Tier 1 3 source のいずれかが TFDS から取得不能 → 該当 source を Tier 2 代替 or skip 判断
25. **Phase 3b smoke で Stage 2 regression 破壊**: `VLAAdapterGemma4` の soft_prompt 追加が既存 forward path を壊す → revert、soft_prompt 経路の分離強化
26. **Phase 3c 本番で 10k step 以内に loss NaN/Inf**: Multi-dataset sampling の weight 偏り or action space 不整合疑い、即停止
27. **Phase 3c 本番で LLM grad leak 発生** (R4 違反): 即停止、soft_prompt 経路の grad 伝播 debug
28. **Phase 3c 途中 checkpoint zero-shot rollout が全 checkpoint で < 5%**: pretrain 効果皆無 → 5/10 時点で fallback 確定
29. **Phase 3a-3c の累計期間が Compute budget 超過 (5/13 hard deadline 違反)**: 即停止、fallback branch へ
30. **5/13 時点で Phase 2m が pretrained checkpoint を待って停滞**: fallback へ切替、Mission 2 優先

---

## 9. やってはいけないこと (Stage 2 継承 + Stage 3 追加)

Stage 2 Plan v2/v3 の禁止事項全てに加え:

- **Stage 3 失敗で Mission 2/3 を犠牲にする**: 5/13 hard deadline 厳守、Mission 2/3 の進行を絶対に遅らせない
- **Pretrain code で R1-R9 改変**: placeholder ID、feature_norm、use_cache 等を勝手に変更しない、soft_prompt は独立経路で統合
- **X-VLA paper の全 feature 実装**: Temporal downsampling、per-dataset projection library 等の scope 外要素は入れない
- **Ablation run の追加**: User 指示で ablation 不要、pretrain 本番 1 run のみ
- **LIBERO eval の本格化**: Sanity rollout のみ、論文比較等の formal eval は scope 外 (Stage 2 Phase 2g で完了)

---

## 10. Deliverables (Stage 3)

Stage 2 Deliverables に追加:

| # | ファイル | Phase |
|---|---|---|
| 22 | `scripts/stage3/download_data.py` (Tier 1 data の TFDS download script) | 3a-1 |
| 23 | `scripts/stage3/compute_dataset_statistics.py` | 3a-2 |
| 24 | `scripts/stage3/multi_dataset_loader.py` (weighted sampling、dataset_id 付与) | 3a-3 |
| 25 | `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` (`SoftPromptLibrary` + forward 拡張) | 3b-2 |
| 26 | `VLA-Adapter/vla-scripts/finetune_gemma4.py` (pretrain mode 追加) | 3b-3 |
| 27 | `scripts/stage3/test_20_pretrain_smoke.py` | 3b-4 |
| 28 | `scripts/stage3/pretrain_kickoff.py` (DDP launcher) | 3c-1 |
| 29 | `scripts/stage3/zero_shot_libero_sanity.py` | 3c-3 |
| 30 | `runs/gemma4/stage3_pretrain/` (checkpoint、WandB log、sanity rollout result) | 3c |
| 31 | `docs/gemma4_migration_log.md` Stage 3 section + 5/13 分岐判定 entry | 各 Phase |

---

## 11. Submission Narrative

5/18 提出時の narrative framework:

### Main pitch (success or fallback 問わず)

- **Gemma 4 E2B backbone で VLA-Adapter を移植** (Stage 1-2 成果)
- **自前実ロボット (6DOF + gripper + 2 カメラ) で動作する application を構築** (Mission 2/3)
- **LIBERO-Spatial で X% success rate** (Stage 2 Phase 2g 結果)
- **Demo 動画で sim + real 両方の動作を示す**

### Success branch 追加

- **Cross-embodiment pretrain (3 data source, ~193K episodes) により自前データ fine-tune の data efficiency 向上**
- 自前データ 10-30 demo で task 成立 (従来 Stage 2 baseline より少ない demo 数で動作)
- Technical depth: X-VLA 流 Soft Prompt 機構を Gemma 4 backbone に initial 統合

### Fallback branch 追加

- **Gemma 4 VLA-Adapter の LIBERO-Spatial baseline を起点に、自前データ fine-tune**
- 自前データ 30-50 demo で task 成立
- Technical depth: Stage 1-2 で確立した Gemma 4 移植 pipeline の reproducibility + 自前環境への transfer

どちらの branch でも Hackathon の主張である **「VLA を構築して現実に使えるアプリケーションを作った」** は達成可能、Stage 3 は success 時の upside として機能。

---

## 12. References

- X-VLA paper: https://arxiv.org/abs/2510.10274v1 (2025-10-14)
- X-VLA code: https://github.com/2toinf/X-VLA
- Stage 2 plan: `docs/gemma4_stage2_plan.md`
- Stage 2 後半 plan: `docs/gemma4_stage2_latter_half_plan.md`
- Stage 1-2 migration log: `docs/gemma4_migration_log.md`
- VLA-Adapter 原実装: `VLA-Adapter/`

---

## 13. Stage 3 完了認定

- Phase 3a-3c 完了 + 5/13 分岐判定実施
- Pretrain checkpoint 保存 (success 時) or fallback 選択 (failure 時) の明示
- Mission 2 (Phase 2m) の starting checkpoint 選択済
- Submission narrative の success/fallback branch 確定

Hackathon 提出 (5/18) 時点で Mission 2/3 と Stage 3 の統合結果が提出物に反映されていれば完了。
