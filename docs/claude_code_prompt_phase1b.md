# Claude Code 着手プロンプト: Gemma 4 E2B 移植 Phase 1b

## このプロンプトの扱い方

- Claude Code の**新規セッション冒頭**にこのファイル全文を貼り付ける
- 実装計画の詳細は `gemma4_stage1_phase1b_plan_v5.2.md` に記載、これを**必ず最初に精読**
- v1 troubleshooting.md も読んで教訓を把握
- 実装はファイル 1 個ずつ、Phase 1 ステップずつ。着手前に**必ず宣言**、完了後に**必ず報告**

---

## Mission

VLA-Adapter の LLM backbone を **Qwen2.5-0.5B → Gemma 4 E2B** に置き換え、**LLM 完全凍結状態で LIBERO に対する 10-step smoke train が走る**ところまで到達する。

本タスクは Phase 1b (Model 組み立て) の実装。Phase 0 / 1a は完了済み、Phase 1c / 1d は Phase 1b 完了後に別プロンプトで着手する。

**成功基準**: 10-step smoke train が走り、loss が finite かつ減少傾向 or 安定。精度は問わない。論文値 (99.6%) の再現は Stage 2 以降。

---

## 最初にやること (セッション開始時)

1. 作業ディレクトリに移動 (VLA-Adapter/ を include する親ディレクトリ)
2. 以下のファイルを順に read:
   - `gemma4_stage1_phase1b_plan_v5.2.md` — 本タスクの詳細計画
   - `docs/troubleshooting.md` — v1 の教訓 (PLE OOM, LayerNorm 等)
   - `docs/gemma4_migration_log.md` — Phase 0 / 1a の結果
   - `VLA-Adapter/prismatic/vla/constants_gemma4.py` — Phase 1a で確定した定数
3. `.venv-gemma4/` が存在し `transformers>=5.5.0` が入っていることを確認
4. GPU 1 枚 (A100-80GB) で Gemma 4 E2B のロードが可能な状態かを確認 (`nvidia-smi`)
5. Phase 1b.1 着手前に User に準備完了を報告

---

## 絶対ルール (Stage 1 全期間)

### R1. Module 階層 (transformers 5.5)

```python
# 全コードで統一
llm = model.model.language_model              # NOT model.language_model
embed = llm.embed_tokens
hidden_size = model.config.text_config.hidden_size  # 1536
```

### R2. PLE 逆引き OOM 回避

`inputs_embeds` を渡すときは**必ず** `per_layer_inputs` も一緒に渡す:

```python
per_layer_inputs = llm.get_per_layer_inputs(input_ids, None)
out = llm(inputs_embeds=..., per_layer_inputs=per_layer_inputs, ...)
```

これを怠ると 80GB+ の OOM が出る。**例外なし**。

### R3. Action queries の上書きは Option A (clone + advanced indexing) のみ

```python
embeddings = raw_embeddings.clone()
for b in range(B):
    positions = mask.nonzero(as_tuple=True)[0]
    embeddings[b, positions] = action_queries.weight  # (64, D)
```

**`torch.where` + broadcast は禁止** — 64 個の位置が全て同じベクトルになり mode collapse。しかも assertion をパスする表面的には動く。

### R4. LLM 完全凍結

```python
for p in model.parameters():
    p.requires_grad = False
```

Stage 1 で学習するのは action_head / action_queries / vision_projector / proprio_projector / (optional) feature_norm のみ。

### R5. attention_mask / position_ids を明示構築 (1b.4 以降)

seq > 512 (sliding window) のため、auto generate に任せない:

```python
attention_mask = torch.ones_like(input_ids, dtype=torch.long)
position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, -1).cuda()
```

### R6. `use_cache=True` 明示、gradient_checkpointing は使わない

```python
model.config.use_cache = True   # KV 共有バグ (HF #45242) 回避
assert not model.is_gradient_checkpointing
```

gradient_checkpointing が必要なメモリ状況になったら、**停止して User に報告**。勝手に有効化しない。

---

## Phase 1b マイクロステップ (6 段階)

各段階で独立に走る test script を `scripts/gemma4/test_NN_*.py` として作成。詳細は v5.2 計画書参照。

| # | 内容 | 所要目安 | 主要 check |
|---|---|---|---|
| 1b.1 | Load + text forward + baseline | ~30 分 | peak_memory < 15GB、std 測定 |
| 1b.2 | `inputs_embeds` PLE 罠の切り分け | ~30 分 | OOM 再現 & 対策有効性 |
| 1b.3 | Action queries 注入 + 勾配検証 | ~1 時間 | 勾配 assertion、semantic 検証 |
| 1b.4 | Vision integration (2 カメラ) | ~1 時間 | vision+action 両方上書き、attention_mask 明示 |
| 1b.5 | Action head 接続 + LayerNorm 判断 | ~1 時間 | **Step 0 で named_parameters print 必須** |
| 1b.6 | 統合クラス + 凍結 verify | ~1 時間 | trainable 200-300M、全 grad > 0 |

---

## Check-in Points (自動進行禁止)

以下のタイミングで**必ず停止し、User に結果を見せてから次に進む**:

| # | タイミング | 見せるもの |
|---|---|---|
| 1 | 1b.1 完了後 | 36 entries の std/mean 分布、peak memory |
| 2 | 1b.2 完了後 | 10 試行のメモリスケーリング表 |
| 3 | 1b.3 完了後 | 勾配 / semantic assertion の全 pass 確認 |
| 4 | 1b.5 Step 0 完了後 | `action_head.named_parameters()` の出力 |
| 5 | 1b.6 完了後 | 統合 test の全 assertion 結果 |

**境界ケース** (例: std=2.05 で LayerNorm 要否が曖昧、action_head の層名が想定と違う等) も自動判定せず、停止して User に判断を仰ぐ。

---

## Escalation Policy (即座停止して User 報告)

1. Phase 1b.1 で peak memory が 15GB を超える
2. Phase 1b.2 で Case 1 も Case 2 も OOM
3. Phase 1b.3 の**勾配または semantic assertion が fail**
4. Phase 1b.5 Step 0 で action_head の層構造が**fallback でも特定不能**
5. Phase 1b.6 の LLM 凍結 assertion が fail
6. 80GB GPU で batch=1 すら OOM
7. `Gemma4ForConditionalGeneration` の API が想定と大幅に異なる
8. 単一の問題で 2 時間以上ハマる
9. Stage 1 スコープ外に踏み込まないと解けない問題
10. Check-in Point で境界判定が必要

---

## やってはいけないこと (違反即時停止)

- Monkey patch で問題を隠蔽 (特に KV 共有バグ、PLE 逆引き)
- Workaround で「動いているように見せる」(assertion を潰す、warning を黙らせる)
- 設計判断を単独で行う (LayerNorm 要否、DINO 外し等)
- **`torch.where` + broadcast で action queries 代入**
- **Check-in Point をスキップ**して次 Phase に自動進行
- `model.language_model.*` を使う → `model.model.language_model.*`
- `inputs_embeds` 単独渡し → `per_layer_inputs` を必ず一緒に
- `gradient_checkpointing` を有効化 → KV 共有バグ直撃
- Stage 1 スコープ外 (LoRA / 双方向 attn / DINO 差し替え) に踏み込む

---

## Out of Scope (Stage 2 以降)

- LoRA の導入
- 双方向 attention patch (`use_bidirectional_attention=True` の挙動確認)
- Vision backbone の差し替え (DINO+SigLIP を維持)
- 推論 (LIBERO eval) / rollout
- Multi-GPU DDP / FSDP
- 論文精度 (99.6%) の再現検証
- transformers 4.40 への後方互換性維持

---

## Deliverables (Stage 1 完了時)

| # | ファイル | 生成タイミング |
|---|---|---|
| 1 | `requirements-gemma4.txt` | Phase 1b.1 前 |
| 2 | `VLA-Adapter/prismatic/models/backbones/llm/gemma4.py` | Phase 1b.6 |
| 3 | `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` | Phase 1b.6 |
| 4 | `VLA-Adapter/vla-scripts/finetune_gemma4.py` | Phase 1d (別プロンプト) |
| 5 | `scripts/gemma4/test_01_load.py` 〜 `test_06_full_forward.py` | Phase 1b 各ステップ |
| 6 | `docs/gemma4_migration_log.md` | 全 Phase で随時追記 |

各 test script 完了時に log へ結果 (memory、speed、発見、疑問) を追記。

---

## 報告フォーマット

各 test script 完了後、以下の形式で User に報告:

```
=== Phase 1b.X 完了報告 ===

Status: [OK / FAIL / BLOCKED]
Peak memory: X.X GB
Exit criteria: [全 pass / 以下が fail: ...]

主要な数値:
- ...
- ...

発見:
- ...

次ステップ: [Phase 1b.Y 着手許可をお願いします / Check-in Point #N で判断要請]

(必要なら) User への質問:
- ...
```

Check-in Point では数値表 / print 出力の**生データ**を添付する。要約だけで済ませない。

---

## 開始合図

準備完了したら User に報告し、**Phase 1b.1 着手許可**を得てから作業開始。
勝手に 1b.1 から始めない。準備報告と許可を経てから。

References:
- 実装計画詳細: `gemma4_stage1_phase1b_plan_v5.2.md`
- Phase 0 / 1a 結果: `docs/gemma4_migration_log.md`
- v1 教訓: `docs/troubleshooting.md`
- Phase 1a 定数: `VLA-Adapter/prismatic/vla/constants_gemma4.py`
