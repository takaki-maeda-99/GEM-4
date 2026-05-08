# VLA-Gemma4 次の実験プラン

**Created**: 2026-04-25 03:30 JST
**Hackathon deadline**: 2026-05-18 (残 23 日)

---

## 現状 (2026-04-25 03:30)

### Winning arch: **Option B** (use_wrist_bridge=True、feature_norm=Identity、wrist_bridge_layer_mode=per_layer)

| ckpt | step | LIBERO-Spatial eval |
|---|---|---|
| Option B 10k | 10k | 73% |
| **Mode B wrist_bridge 20k** | **20k** | **77%** ⭐ |
| Mode A + LoRA64 (eval 実行中) | 10k | TBD @ 03:57 |
| Mode B 35k (32.5k 相当) | 32.5k | TBD @ 04:15 |

### 棄却済 (負けた arch)
- proper_ffn (FFN 拡張): 50% ❌
- LayerNorm feature_norm: 47% ❌ (double norm)
- Option A final_broadcast: 49% ❌ (per-layer 情報損失)
- pure XVLA (cross-attn 撤去): 0.33 loss plateau ❌

### Phase 2f 80% まで残 3pt、Mode B 35k で射程

---

## 既にある resource (download 不要)

| dataset | size | path |
|---|---|---|
| LIBERO Spatial | 1.8G | `data/modified_libero_rlds/libero_spatial_no_noops` |
| LIBERO Goal | 1.8G | `data/modified_libero_rlds/libero_goal_no_noops` |
| LIBERO Object | 2.7G | `data/modified_libero_rlds/libero_object_no_noops` |
| LIBERO 10 | 3.5G | `data/modified_libero_rlds/libero_10_no_noops` |
| Taco Play (OXE) | — | `data/stage3_openx/taco_play` |
| Fractal20220817 (RT-1、OXE) | — | `data/stage3_openx/fractal20220817_data` |

**→ Bridge V2 download 不要**、Taco + Fractal で 2-dataset pretrain 可能

### 不足 resource
- **dataset_statistics.json は libero_spatial_no_noops のみ**
- Goal/Object/10 用 stats を生成 or eval 時に on-the-fly 計算必要

---

## Step 1: LIBERO 全 4 suite 評価 (当日〜翌日)

### 1a. zero-shot eval (現 Option B 10k ckpt 流用)

```bash
# LIBERO-Spatial の stats で他 suite eval する (action_dim 同じなので技術的には可、精度落ち前提)
for suite in libero_goal libero_object libero_10; do
  tmux new-session -d -s eval_zeroshot_$suite \
    "CUDA_VISIBLE_DEVICES=X .venv-gemma4/bin/python scripts/gemma4/eval_libero_gemma4.py \
      --training_mode speed --vision_backbone_type siglip --siglip_use_tensor_transform True \
      --use_xvla_style False --use_wrist_bridge True \
      --checkpoint_path runs/gemma4/...libero_b_siglip_10k_wristb_b16_v2/latest_checkpoint.pt \
      --task_suite_name $suite --num_trials_per_task 10 --num_tasks_limit 0 \
      --run_id_note optionB_10k_zeroshot_$suite"
done
```

**目的**: arch transfer 度合測定、floor baseline。

### 1b. 各 suite で Option B 個別訓練 (paper に合わせる)

```bash
for suite in libero_goal libero_object libero_10; do
  # dataset_statistics.json 生成 (各 suite の action q01/q99/mask 計算)
  # ... compute stats from RLDS ...

  tmux new-session -d -s train_$suite \
    "CUDA_VISIBLE_DEVICES=X,Y .venv-gemma4/bin/torchrun ... \
      --dataset_name ${suite}_no_noops --max_steps 10000 --batch_size 16 \
      --use_wrist_bridge True \
      --run_id_note option_B_10k_$suite"
done
```

**Cost**: 3 suite × 10k × 1.9h = **5.7h** (DDP 2-way 1 GPU pair) or parallel 3 pairs で 1.9h

---

## Step 2: Multi-dataset pretrain with X-VLA SoftPrompt (Day 29-30)

### 目的

**hardware / 視点 / Hz の違う複数 dataset を共同 pretrain**、domain 差は SoftPromptLibrary で吸収。

### 構成

```python
# finetune_gemma4.py の pretrain_mode=True path
num_pretrain_datasets = 2  # Taco + Fractal
num_soft_prompt_tokens = 32
# mixture_spec = [("taco_play", 1.0), ("fractal20220817_data", 1.0)]

# arch: Option B + SoftPrompt active
use_wrist_bridge = True
use_xvla_style = False
# SoftPromptLibrary → action_head self-attn pool に concat (h_sp)
```

### 実装 checkpoint 済

- `SoftPromptLibrary` class (`modeling_prismatic_gemma4.py:42-64`)
- `predict_action(h_sp=...)` (`action_heads.py:50`)
- `MLPResNet.forward` で h_sp → self-attn pool concat
- `build_pretrain_dataloader` で multi-dataset mixture_spec 処理
- **未使用: num_pretrain_datasets=0 の run が殆ど、2 で activate 必要**

### Cost

- Taco + Fractal mixture pretrain 40k step (Option B + SoftPrompt)
- 2-GPU DDP、B=16、s/step 0.7 → 40k × 0.7 = **7.8h**
- ckpt: `runs/gemma4/gemma-4-e2b+pretrain-nd2-sp32+b16+...+taco_fractal_optB_pretrain`

---

## Step 3: pretrained → LIBERO all-suite finetune (Day 31)

pretrain ckpt を init_weights_from で 4 suite 個別 finetune:

```bash
for suite in libero_spatial libero_goal libero_object libero_10; do
  tmux new-session -d -s ft_pretrained_$suite \
    "... --init_weights_from runs/gemma4/.../taco_fractal_optB_pretrain/latest_checkpoint.pt \
      --dataset_name ${suite}_no_noops --max_steps 10000 \
      --run_id_note optB_pretrained_$suite"
done
```

**pretrain 効果判定**: (Step 3 avg across 4 suites) vs (Step 1 avg without pretrain)

---

## Step 4: Mission 2 (自前データ fine-tune、Day 32+)

- 実機 / sim でユーザー独自 demo 収集 (teleop or scripted)
- LeRobot 形式で保存 (`lerobot_libero_loader.py` 互換)
- pretrained Option B + new SoftPrompt (dataset_id=最新) で finetune
- Demo 動画作成 (Mission 3)

---

## Mission ステータス (VLA-Gemma4 hackathon)

| Mission | 内容 | 進捗 |
|---|---|---|
| 1 | LIBERO-Spatial で paper gap 分析 | 77% 達成、80% 射程 |
| 2 | 自前データで 1 タスク以上 fine-tune | 未着手 (環境準備要) |
| 3 | Demo 動画 + ハッカソン提出 | 未着手 |
| 4 (optional) | ablation / accuracy 向上 | 73→77% で 1 round 完了 |

---

## 判断点 (今夜 results で決定)

- **Mode A + LoRA64 eval > 77%**: LoRA 効果 confirmed、wrist を LLM に戻す実験候補 (原論文復元)
- **Mode A + LoRA64 eval ≤ 77%**: Mode B Option B が 本命、LoRA は capacity 余計だけで改善しない
- **Mode B 35k eval ≥ 80%**: Phase 2f 追抜き達成、Mission 1 clear、Mission 2 へシフト
- **Mode B 35k eval < 80%**: step 数以外の改善 (Step 2 SoftPrompt pretrain) が次手

---

## Remember (user reminder 2026-04-25)

- **今後の計画は本 doc が primary reference**
- Mode A + LoRA64 結果 → 03:57 頃に eval 完了
- Mode B 35k → 04:15 頃に完了、eval 追加で ~04:45
- 朝起きた時点で 4 点比較確定
