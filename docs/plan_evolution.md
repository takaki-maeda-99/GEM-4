# VLA-Gemma4 計画書推移サマリ

**Last updated**: 2026-04-24 23:00 JST
**Hackathon deadline**: 2026-05-18 (残 24 日)

元 8 本の plan/spec 文書 + 夜中の iteration を時系列・目的・outcome で 1 枚化。

---

## 0. 全体ロードマップ

```
Phase 1a-1b ──→ Phase 1c-1d ──→ Stage 2 (Phase 2a-2f) ──→ Phase 2f 80% 到達
  (Apr 15-19)    (Apr 19)        (Apr 20-23)                └─ Stage 3 (Taco pretrain) 並走
                                                            
                                ↓ 3 つの構造的損失発見
                                
Dual-track Redesign (Apr 22) ──→ Phase 2f ckpt 廃棄、fresh start
                                ↓
                                ├── Mode A (quality: LoRA + GC)
                                └── Mode B (speed: no_grad)
                                
                                ↓ arch iteration (Apr 23-24)
                                
Option (a) scene concat (68%) → Option B wrist_bridge (73% ⭐) → Mode A + LoRA64 (実行中)
```

---

## 1. Plan 文書別 推移

### Stage 1 Phase 1b (Gemma 4 model 組立)

| 文書 | 日付 | 焦点 | outcome |
|---|---|---|---|
| `gemma4_stage1_phase1b_plan.md` (v5.1) | Apr 18 | VLAAdapterGemma4 + PLE + action_head 組立 | v1 → v5.1、4 回レビュー |
| `gemma4_stage1_phase1b_plan_v5.2.md` | Apr 19 | v5.1 + phase 間 pattern 伝播 | v5.2 確定、LIBERO 10-step smoke 成功 |

**目的**: Gemma 4 E2B + Bridge action_head で forward が finite、backward で param 更新できる状態に。精度問わず。

**主要発見**:
- PLE (Per-Layer Embeddings) は input_ids から事前計算して明示渡し必須 (OOM 回避)
- sliding_window=512 超過対応で attention_mask / position_ids 明示構築
- `get_intermediate_layers` monkey-patch に注意

---

### Stage 1 Phase 1c-1d (Backward scaling + LIBERO smoke)

| 文書 | 日付 | 焦点 | outcome |
|---|---|---|---|
| `gemma4_stage1_phase1c1d_plan.md` | Apr 19 | batch scaling 測定 + 10-step smoke | 10-step LIBERO smoke 成功 |

**目的**: Phase 1b (B=1) → B=2/4/8 で memory / throughput 測定、1d で使う batch 決定。

**成果**: B=8 で動作確認、Stage 2 baseline 準備完了。

---

### Stage 2 前半 (Apr 20-22)

| 文書 | 日付 | 焦点 | outcome |
|---|---|---|---|
| `gemma4_stage2_plan.md` (v2) | Apr 20 | 本番学習 + LIBERO eval + 自前データ | Mission 1-4 定義 |

**Mission**:
- Mission 1: LIBERO-Spatial 本番学習、paper 99.6% との gap 分析
- Mission 2: 自前データで fine-tune (hackathon 提出用)
- Mission 3: Demo 動画
- Mission 4: Bidirectional attention 等 ablation

**outcome**: 
- Phase 2a-2c: eval pipeline 構築、dry run 成功
- Phase 2d: DDP smoke 再開
- Phase 2e: 200k step 本番 run 進行 (DinoSigLIP + 画像 2 枚 LLM arch)
- **Phase 2f eval**: 30k=80%, 100k=80%, 140k=73% ⭐ (LIBERO Spatial)

---

### Stage 2 後半 (Apr 20 夜〜)

| 文書 | 日付 | 焦点 | outcome |
|---|---|---|---|
| `gemma4_stage2_latter_half_plan.md` | Apr 20 | Multi-GPU 拡張 + DDP 復活 | 4 実験並列枠確保 |

**scope 変更**:
- Phase 2a-2c が Day 1 で完了 + Phase 2e 2.0 日で進行 → 13-14 日 buffer
- Multi-GPU DDP で並列 ablation に投入

**成果**: DDP 運用確立、Phase 2e 完走 (140k)、Phase 2f eval 80%。

---

### Stage 3 (並列、Apr 20〜)

| 文書 | 日付 | 焦点 | outcome |
|---|---|---|---|
| `gemma4_stage3_plan.md` | Apr 20 | Taco Play pretrain + X-VLA Soft Prompt | 110k pretrain ckpt 生成 |

**目的**: 自前データ fine-tune 時に使う pretrained weight を作る (few-shot adaptation 前提)。

**実装**: 
- Taco Play (CALVIN) データで 10k-140k step pretrain
- SoftPromptLibrary (case 案 B: LLM 入力前段 concat) 実装
- `small_a_b24_siglip`, `small_b_b24_siglip_tensor_v2` 等の pretrain ckpt 残存

**結果**: pretrain 成立 → Stage 2 LIBERO ckpt の init に使用可。ただし後の Dual-Track Redesign で X-VLA 準拠に是正される。

---

### VLA Redesign (Dual-Track) — Apr 22 の大転換

| 文書 | 日付 | 焦点 | outcome |
|---|---|---|---|
| `docs/superpowers/specs/2026-04-22-vla-redesign-scene-wrist-split-design.md` | Apr 22 | 3 構造損失の識別 + Dual-Track 戦略 | Phase 2f ckpt 廃棄決定 |
| `docs/superpowers/plans/2026-04-22-vla-redesign-dual-track.md` | Apr 22 | 実装計画 (17 tasks) | Task 1-17 完了 |

**3 つの構造的損失を識別**:
1. Wrist を frozen LLM に通してた (OOD、適応不能)
2. DinoSigLIP + vision_projector が redundant (Gemma 4 の native vision 活用せず)
3. SoftPromptLibrary 案 B 配置が X-VLA 原実装と乖離

**G1-G6 共通 goals**:
- G1: Wrist → action_head 直接 (LLM バイパス)
- G2: Scene → Gemma 4 native vision (または SigLIP)
- G3: SoftPromptLibrary → action_head 入力 (X-VLA 準拠)
- G4: Bridge Attention 完全温存
- G5: Mode A/B 切替 flag 化
- G6: 5/18 までに両 mode eval 完了

**Dual-Track 戦略**:
| | Mode A (Quality) | Mode B (Speed) |
|---|---|---|
| action_queries | trainable | frozen (zero init) |
| LLM backward | on (LoRA 経由) | off (torch.no_grad) |
| GC | required | なし |
| batch 上限 | 16 | 32+ |
| 論理 | 一発勝負、quality 重視 | iteration 勝負、scale 重視 |

**Phase 2f 80% ckpt を廃棄**、fresh start で Dual-Track 実装。

**実装**: 17 tasks (WristResNet18 + Bridge h_w concat + X-VLA SoftPrompt 位置修正 + Mode A/B flag + DDP launch script 等)。

---

## 2. Plan 書になってない夜中 iteration (Apr 23-24)

正式 plan 文書外、`troubleshooting.md` (#012-#017) と `results_summary.md` で記録:

| 日付 | 改修 | 目的 | outcome |
|---|---|---|---|
| Apr 23 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (#012) | Mode A 2.2s→5.7s degrade fix | 安定化 |
| Apr 23 | Mode A 1st eval | 0/100 success | LoRA save bug 疑い |
| Apr 23 | #013 LoRA save bug fix | `lora_` key を `llm.*` filter 例外に追加 | trainable_state_dict に LoRA 保存される |
| Apr 24 | **#014 Option (a): scene concat** | `h_v` を self-attn pool に追加、wrist 攻撃性改善 | **68% @ 10k** (Bridge 34%→) |
| Apr 24 | #014 XVLAActionBlock (pure self-attn) 試作 | FFN 貧弱 + 階層性喪失で失敗 | plateau 0.33、kill |
| Apr 24 | #015 **Option B: wrist_bridge** | SigLIP 25 層 per-layer cross-attn、dedicated stream (starvation 構造的回避) | **73% @ 10k** ⭐ |
| Apr 24 | #016 proper_ffn (4× FFN + pre-LN + dual residual) | capacity 増 | **50% @ 10k** (悪化、棄却) |
| Apr 24 | #017 wrist pool 除去 | use_wrist_bridge=True の時 self-attn pool から h_w concat 取り除き、2 経路 redundancy 解消 | 実行中 run で適用 |
| Apr 24 夜 | Mode A + LoRA r=64 attn+MLP + Option B | 論文 LoRA spec + wrist_bridge、LLM adaptation で OOD 吸収狙い | 実行中 (ETA 02:30) |
| Apr 24 夜 | Option B 35k Mode B | step 数で 73% → 80%+ 狙い | 実行中 (ETA 03:10) |

---

## 3. eval 結果推移 (LIBERO Spatial、100 episodes)

| 日付 | config | step | eval | plan doc |
|---|---|---|---|---|
| ~Apr 22 | Phase 2f (DinoSigLIP + 画像2枚LLM) | 30k | **80%** ⭐ | Stage 2 |
| ~Apr 22 | Phase 2f | 60k | 80% | Stage 2 |
| ~Apr 22 | Phase 2f | 100k | 80% | Stage 2 |
| ~Apr 22 | Phase 2f | 140k | 73% | Stage 2 |
| Apr 23-24 | Bridge baseline (dual-track) | 10k | 34% | Dual-Track |
| Apr 24 | Bridge baseline | 35k | 61% | Dual-Track |
| Apr 24 | Mode A E4B (LoRA bug 版) | 10k | 0-1% | #013 発見契機 |
| Apr 24 | **Option (a) Bridge+scene concat** | 10k | **68%** | #014 |
| Apr 24 | **Option B wrist_bridge** | 10k | **73%** ⭐ | #015 |
| Apr 24 | Option B + proper_ffn | 10k | 50% | #016 (棄却) |

**Phase 2f 80% は未だ未追抜** (stage 2 arch は step 数 100k かけてた、dual-track は 10k しか試してない)。

---

## 4. 決定的な判断点

1. **Apr 22 Dual-Track Redesign**: 既存 Phase 2f 80% ckpt を廃棄する決断。理由 = アーキ構造損失が cap しているとの判断、一回 reset で再設計する方が Mission 2 (自前データ) 適応性高くなる。
2. **Apr 24 Option (a) → B**: scene concat が 68%、wrist_bridge が 73%。**attention starvation が構造的問題で、dedicated cross-attn stream で解消**が確認。
3. **proper_ffn 棄却**: capacity 増では補えず、既存の weak FFN でも cross-attn stream 多数あれば機能。
4. **wrist を LLM に戻さない方針維持**: OOD リスクより starvation 回避を重視。LoRA が LLM 側 OOD 適応を担うなら再検討候補 (Mode A + LoRA64 で検証中)。

---

## 5. 未達 / 次アクション

### ハッカソン提出期限までの残課題 (残 24 日)

- [ ] Mode A + LoRA64 10k 完走後 eval → Option B 73% 超えるか (朝 02:30)
- [ ] Mode B Option B 35k 完走後 eval → 80% 狙い (朝 03:10)
- [ ] Mission 2: 自前データ (teleop) 収集 + fine-tune
- [ ] Mission 3: Demo 動画作成
- [ ] LIBERO Goal/Object/10 suite での eval (spatial 以外の汎化能力)

### 未実装アイデア (保留)

- FAST action tokenizer (Option D-lite、3-5 日) — 棄却判定、pass
- Ego-centric wrist encoder (R3M, EgoVLP) — Mode A eval 結果次第
- 原論文復元 (wrist を LLM に戻す + LoRA) — Mode A 結果次第

---

## 6. 参照

- **最新結果**: `runs/results_summary.md`
- **troubleshooting log**: `docs/troubleshooting.md` (bug #001-#017)
- **arch 可視化**: `docs/architecture_option_a.mmd` / `.svg`
- **W&B**: `wandb.ai/takaki-maeda-1999-toyota-technological-institute/vla-gemma4`
- **plan 原本**: `docs/gemma4_stage*.md`, `docs/superpowers/{specs,plans}/*.md`
