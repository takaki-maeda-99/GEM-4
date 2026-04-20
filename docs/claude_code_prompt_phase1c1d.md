# Claude Code 着手プロンプト: Gemma 4 E2B 移植 Phase 1c + 1d

## このプロンプトの扱い方

- Claude Code の**新規セッション冒頭**にこのファイル全文を貼る
- 実装計画の詳細は `gemma4_stage1_phase1c1d_plan.md` に記載、これを**必ず最初に精読**
- `docs/gemma4_migration_log.md` で Phase 1b の結果・発見を把握
- 実装はファイル 1 個ずつ、Phase 1 ステップずつ。着手前に**宣言**、完了後に**報告**

---

## Mission

Phase 1b 完了 (`VLAAdapterGemma4` クラス + 6 assertion 全 pass) を起点に、**LIBERO データで 10-step smoke train** を実行するところまで到達する。

- **Phase 1c**: Batch size scaling を測定 (B=1/2/4/8)、1d で使う batch config を数値決定
- **Phase 1d**: LIBERO-Spatial で 10-step 学習、loss が finite かつ減少傾向 or 安定

**成功基準**: 10 step 完走 + loss finite + LLM leak なし。論文値 (99.6%) の再現は範囲外。

---

## 最初にやること (セッション開始時)

1. 作業ディレクトリ確認 (`/misc/dl00/takaki/vla-gemma-4/` を想定)
2. 以下のファイルを順に read:
   - `gemma4_stage1_phase1c1d_plan.md` — 本タスクの詳細計画
   - `docs/gemma4_migration_log.md` — Phase 0/1a/1b の全結果、4 罠、重要発見
   - `VLA-Adapter/prismatic/vla/constants_gemma4.py` — placeholder ID 定義
   - `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py` — `VLAAdapterGemma4` クラス
   - `scripts/gemma4/test_06_full_forward.py` — 1b.6 の forward 実装 (流用元)
   - `VLA-Adapter/vla-scripts/finetune.py` — `finetune_gemma4.py` の差分元
3. 環境確認:
   - `.venv-gemma4/bin/python -c "import accelerate, peft; print('OK')"` → Blocker B1/B2 解消確認
   - `ls -lh modified_libero_rlds/` → Blocker B3 解消確認 (data download 済みか)
   - `ls VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json` → 流用元の存在確認
4. GPU 状態確認: `nvidia-smi` で GPU 0 が空いていることを確認
5. 準備完了を User に報告、**Phase 1c 着手許可を待つ**

---

## 絶対ルール (Phase 1c/1d 全期間)

Phase 1b から継承した R1-R6 + 1c/1d 用の R7-R9:

### R1. Module 階層 (transformers 5.5)

```python
llm = model.model.language_model              # NOT model.language_model
```

### R2. PLE 逆引き OOM 回避

```python
per_layer_inputs = llm.get_per_layer_inputs(input_ids, None)
out = llm(inputs_embeds=..., per_layer_inputs=per_layer_inputs, ...)
```

### R3. Overwrite pattern は Option A のみ

```python
embeddings = raw_embeddings.clone()
for b in range(B):
    embeddings[b, positions] = action_queries.weight
```

`torch.where` + broadcast は禁止。

### R4. LLM 完全凍結

毎 step で grad leak 検査:

```python
llm_grad_leak = sum(
    1 for p in model_vla.llm.parameters()
    if p.grad is not None and p.grad.abs().sum() > 0
)
assert llm_grad_leak == 0
```

### R5. `attention_mask` / `position_ids` 明示構築

seq > 512 (sliding window) のため auto generate に任せない。

### R6. `use_cache=True` 必須、`gradient_checkpointing` 禁止

```python
model.config.use_cache = True
assert not model.is_gradient_checkpointing
# finetune.py の gradient_checkpointing_enable() 呼び出しは必ずコメントアウト
```

### R7. LoRA は完全に無効化 (Stage 1 スコープ外)

`finetune_gemma4.py` 作成時、以下を全てコメントアウト:
- `get_peft_model` の呼び出し
- `LoraConfig` の構築
- `--use_lora` 関連の CLI option 処理
- ただし import 文は残す (削除すると後で復活させにくい)

### R8. `dataset_statistics.json` は既存再利用

**新規計算しない**。以下を流用:

```
VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json
```

`finetune_gemma4.py` ではこのパスを明示的に読み込む or symlink を貼る。

### R9. 10 step 以上回さない (Stage 1 スコープ外)

`max_steps=10` を厳守。最終 loss が減らなくても 10 step で停止して User 報告。本番学習は Stage 2 以降。

---

## Phase 1c + 1d マイクロステップ

| # | 内容 | 所要目安 | 主要 check |
|---|---|---|---|
| 1c | Batch scaling (B=1/2/4/8) + AdamW state 測定 | ~1 時間 | 1d の batch size 決定 |
| 1d.a | RLDS dataloader → VLAAdapterGemma4 forward 接続 | ~2 時間 | batch dict 構造確認、shape OK |
| 1d.b | 単 step 学習 (2 step 連続、LLM leak check) | ~1 時間 | loss finite、leak 0 |
| 1d.c | 10-step smoke train (`finetune_gemma4.py` 経由) | ~2 時間 | loss トレンド、Stage 1 合否判定 |

---

## Check-in Points (自動進行禁止)

| # | タイミング | 見せるもの |
|---|---|---|
| 6 | 1c 完了後 | 4 cases × fwd/bwd/opt メモリ表、1d で採用する B と grad_accum |
| 7 | 1d.a 完了後 | batch dict の全キー/shape、forward 成立確認、peak memory |
| 8 | 1d.b 完了後 | 2 step loss、LLM leak check、AdamW state 適用後 peak memory |
| 9 | 1d.c 完了後 | **10 step loss ログ全件**、grad norm 推移、Stage 1 合否判定 |

境界ケース (loss 停滞、grad norm 異常、1d.a で batch key が想定と違う等) は **自動判定せず停止、User 判断を仰ぐ**。

---

## Escalation Policy (即停止 + 報告)

1. Phase 1c で batch=1 でも OOM
2. Phase 1c で batch=2 が OOM (1d 実行不能)
3. Phase 1d.a でデータパイプラインが動かない (依存 package 不足、data path 不正)
4. Phase 1d.a で batch dict のキーが想定と大幅に異なる
5. Phase 1d.b で 2 step 以内に loss が NaN/Inf
6. Phase 1d.b で LLM grad leak 発生
7. Phase 1d.c で loss 発散 (初期値の 2 倍以上)
8. Phase 1d.c で grad norm > 100 or 0 に張り付く
9. 単一問題で 2 時間以上ハマる
10. Stage 1 スコープ外に踏み込まないと解けない問題

---

## やってはいけないこと (違反即時停止)

- `gradient_checkpointing` 有効化 (KV 共有バグ直撃)
- LoRA 有効化 (Stage 2 スコープ)
- `dataset_statistics.json` 新規計算 (時間の無駄、既存再利用)
- DDP/FSDP 導入 (単 GPU で smoke 優先)
- Loss NaN の workaround を独断で入れる (scaling tricks、clamp 等)
- Check-in Point スキップ
- `finetune.py` 原本の大幅書き換え (差分で進める、原本は保持)
- `torch.where` + broadcast で placeholder 上書き
- Placeholder ID の勝手な変更 (constants_gemma4.py 厳守)
- Monkey patch で問題隠蔽

---

## Out of Scope (Stage 2 以降)

- LoRA 導入
- Bidirectional attention patch
- 論文精度再現
- Multi-GPU DDP/FSDP
- 10 step 超の学習
- LIBERO eval / rollout

---

## Deliverables

| # | ファイル | 生成タイミング |
|---|---|---|
| 1 | `scripts/gemma4/test_07_batch_scaling.py` | Phase 1c |
| 2 | `scripts/gemma4/test_08_data_pipeline.py` | Phase 1d.a |
| 3 | `scripts/gemma4/test_09_single_step.py` | Phase 1d.b |
| 4 | `VLA-Adapter/vla-scripts/finetune_gemma4.py` | Phase 1d.c |
| 5 | `scripts/gemma4/smoke_train_gemma4.py` (任意、finetune_gemma4.py 内包可) | Phase 1d.c |
| 6 | 更新版 `requirements-gemma4.txt` (accelerate/peft 反映) | Phase 1c 前 |
| 7 | `docs/gemma4_migration_log.md` に Phase 1c/1d セクション追記 | 各 Phase 完了時 |

---

## 報告フォーマット

各 test script 完了後、以下の形式で User に報告:

```
=== Phase 1X.Y 完了報告 ===

Status: [OK / FAIL / BLOCKED]
Peak memory: fwd X.X GB, bwd X.X GB, opt_steady X.X GB
Exit criteria: [全 pass / 以下が fail: ...]

主要な数値:
- ...

発見:
- ...

次ステップ: [Phase 1X.Z 着手許可をお願いします / Check-in Point #N で判断要請]

(必要なら) User への質問:
- ...
```

Check-in Point では数値表 / loss ログ / print 出力の**生データ**を添付する。要約だけで済ませない。

---

## 開始合図

準備完了したら User に報告し、**Phase 1c 着手許可**を得てから作業開始。
勝手に 1c から始めない。

References:
- 実装計画詳細: `gemma4_stage1_phase1c1d_plan.md`
- Phase 0-1b 結果と 4 罠: `docs/gemma4_migration_log.md`
- Phase 1b.6 の VLAAdapterGemma4: `VLA-Adapter/prismatic/extern/hf/modeling_prismatic_gemma4.py`
- Placeholder 定数: `VLA-Adapter/prismatic/vla/constants_gemma4.py`
- `finetune.py` 差分元: `VLA-Adapter/vla-scripts/finetune.py`

---

## Stage 1 完了認定 (Phase 1d.c 後)

10 step 完走 + loss finite + LLM leak 0 + grad norm 安定 = **Stage 1 完了**。

このプロンプトの範囲はここまで。Stage 2 (本番学習、LIBERO eval、実データ収集) は別プロンプトで着手する。Stage 1 完了時点でハッカソン残り期間 (5/18 締切まで) を User と再計画する。
