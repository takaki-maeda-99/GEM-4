# トラブルシューティングログ

開発中に遭遇した問題、原因、対策の記録。

---

## 2026-04-14: Gemma 4 PLE で inputs_embeds が使えない

**問題:** VLAPolicy の encode() で画像・言語・proprio のトークンを結合して `inputs_embeds` として LLM に渡そうとしたら、105GB のテンソルを確保しようとして OOM。

**原因:** Gemma 4 の Per-Layer Embeddings (PLE) アーキテクチャは `input_ids` を前提としており、`inputs_embeds` のみで呼ぶと PLE の計算で巨大なブロードキャストが発生。

**対策:** 「PAD プレースホルダー + 埋め込み上書き」パターンを採用。`input_ids` に PAD を追加して PLE を事前計算し、カスタムトークン位置の埋め込みを上書きしてから `per_layer_inputs` 付きで `language_model` に渡す。Gemma 4 自身がビジョントークンを処理するのと同じパターン。

---

## 2026-04-14: LoRA が vision_tower の Gemma4ClippableLinear にヒットしてエラー

**問題:** `target_modules=["q_proj", "v_proj"]` でLoRAを適用すると、vision_tower 内の `Gemma4ClippableLinear` が PEFT 未対応でエラー。

**原因:** `q_proj` / `v_proj` は LLM と ViT の両方に存在するが、ViT 側は `Gemma4ClippableLinear` でラップされており PEFT が対応していない。

**対策:** `target_modules` を `model.language_model.layers.{i}.self_attn.{proj}` のようにフルパスで指定し、ViT を除外。後に ViT 内の `Linear4bit` レイヤー（ClippableLinear の内部）を個別にターゲット指定して ViT にも LoRA を適用可能にした。

---

## 2026-04-14: torchcodec / torchvision.io.VideoReader が動かない

**問題:** LeRobot のビデオデコードが失敗。torchcodec は `libnppicc.so.12` が見つからない、torchvision は `VideoReader` が廃止済み。

**原因:** torchcodec は CUDA toolkit の NPP ライブラリに依存するが、システムにインストールされていない。torchvision 0.22+ では VideoReader が削除済み。

**対策:** `dataset.py` に PyAV フォールバックパッチを追加。`av` ライブラリでビデオをデコードする `_decode_pyav` 関数で `lerobot.datasets.video_utils.decode_video_frames` をモンキーパッチ。

---

## 2026-04-15: ロスがマイナスに発散

**問題:** ALOHA データで学習するとロスが -2000 まで発散。

**原因:** `MLPHead` がグリッパー次元に BCE loss を使っていたが、ALOHA のグリッパー値は [0,1] 範囲外の関節角度。BCE に負の値を入れると loss が負の無限大に発散する。

**対策:** `gripper_as_binary` フラグを追加。`false`（デフォルト）で全次元 MSE、`true` で最終次元のみ BCE。ALOHA や LIBERO（グリッパーが {-1,1}）では `false` を使用。

---

## 2026-04-15: LIBERO データ形式の互換性問題

**問題:** `lerobot/aloha_sim_transfer_cube_human` が v2.1 形式で、lerobot 0.4.4 は v3.0 を要求。

**原因:** Hub 上のデータセットが古い形式のまま。

**対策:** `tolerance_s=1e6` で厳密なタイムスタンプ検証を緩和。データセット変換コマンドで v3.0 への変換も可能（`python -m lerobot.datasets.v30.convert_dataset_v21_to_v30`）。

---

## 2026-04-15: Gemma4Config.hidden_size が存在しない

**問題:** `gemma.config.hidden_size` でアクセスしようとすると AttributeError。

**原因:** Gemma 4 は `config.text_config.hidden_size` にネストされている。MagicMock では `text_config` が自動生成されて非 int 値を返す。

**対策:** `isinstance(config.text_config.hidden_size, int)` でチェックし、フォールバックで `config.hidden_size` を使用。

---

## 2026-04-15: PEFT ラップ後のモデル階層変化

**問題:** `get_peft_model()` 後に `self.gemma.model.vision_tower` が見つからない。

**原因:** PEFT がトップレベルをラップして階層が1段深くなる。

**対策:** `__init__` で `self._gemma_model`, `self._lang_model`, `self._embed_tokens` として直接参照を保持。PEFT ラップ後もサブモジュールの Python オブジェクトは同一なので参照が有効。

---

## 2026-04-15: uv 環境で PyTorch が GPU 非対応

**問題:** `uv sync` で PyTorch 2.11.0+cu128 がインストールされるが、ドライバー 570 と非互換で `cuda=False`。

**原因:** PyTorch 2.11 のランタイムがドライバー 570 と非互換。

**対策:** `pyproject.toml` で `torch>=2.7.0,<2.8.0` にピン留め。cu128 インデックスから 2.7.x をインストール。

---

## 2026-04-15: CUDA toolkit インストールでドライバー破損

**問題:** `sudo apt install cuda-toolkit-12-8` で nvidia-smi が消え、GPU が使えなくなった。

**原因:** CUDA repo のパッケージが Ubuntu 標準の nvidia-utils-570 と競合し、ドライバーパッケージが壊れた。

**対策:** CUDA repo パッケージを全削除 (`sudo apt remove --purge cuda-*-12-8`) し、Ubuntu 標準のドライバーに戻す。**教訓: GPU ドライバーや CUDA toolkit のインストールは絶対に提案しない。ソフトウェアワークアラウンドで対応する。**

---

## 2026-04-15: チェックポイントでディスク 100% 使用

**問題:** `save_every_n_steps: 50` で 6.5GB のチェックポイントが大量に保存され、ディスク 100%。

**原因:** 保存間隔が短すぎ + 古いチェックポイントを削除していなかった。

**対策:** `save_every_n_steps: 500` に変更。チェックポイント保存先を `output_dir` に統一。`.gitignore` に `outputs/` と `checkpoint_step_*.pt` を追加。

---

## 2026-04-16: ACTHead のロスが下がらない (1回目)

**問題:** ACTHead で学習すると MSE loss が 0.4〜2.0 の間で暴れて収束しない。

**原因:** MSE loss が外れ値を二乗で増幅。20ステップチャンクの後半は予測が難しく、大きな誤差が出やすい。

**対策:** L1 loss に変更（元論文 ACT に準拠）。L1 は外れ値に対してロバスト。

---

## 2026-04-16: ACTHead のロスが下がらない (2回目)

**問題:** L1 に変えたが依然としてロスの下がりが遅い（0.83 → 0.63 で停滞）。

**原因:** chunk_queries に positional encoding がなく、「ステップ1」と「ステップ20」の区別がつかない。また nhead=4, num_layers=2 は元論文（nhead=8, num_layers=6）より大幅に小さい。

**対策:** 元論文に合わせて修正:
- 学習可能な positional embedding を追加（`nn.Embedding(chunk_size, d_model)`）
- query を zero-init + positional embedding に変更
- nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1

---

## 2026-04-16: eval_libero でロボットが天を仰ぐ異常動作

**問題:** 学習済みモデルで評価すると、ロボットが一方向に飛んでいく異常な動き。

**原因:** **eval_libero.py でアクションのデノーマライズが欠落していた。** モデルは正規化空間（mean=0, std=1）でアクションを出力するが、`env.step()` は raw アクション空間を期待。正規化空間の値（例: 0.5）をそのまま delta アクションとして送ると異常に大きな動き。

**対策:** `evaluate_libero()` に `normalizer` を渡し、`policy.predict()` の出力を `normalizer.denormalize()` してから `env.step()` に渡す。

---

## 2026-04-16: 異なるタスク/エピソードで同じ動き (mode collapse)

**問題:** 学習済みモデルが異なるタスク・異なるエピソードで全く同じ軌跡を出力する。

**原因:** LLM の隠れ状態（features）は std≈6 と大きなスケール。ActionHead の学習初期にこの大きなスケールの入力を受けると、入力の微妙な違いを無視して全体の平均軌跡を出力するように収束してしまう（mode collapse）。features 自体はタスクごとに異なる値を持っている（cosine similarity=0.29 で確認済み）が、ActionHead がその違いを活かせていなかった。

**対策:** `encode()` と ActionHead の間に `LayerNorm` を追加。features を mean=0, std=1 に正規化してから ActionHead に渡す。

---

## 2026-04-16: build_policy() で feature_norm が未定義

**問題:** `build_policy()` で構築した policy で学習すると `'VLAPolicy' object has no attribute 'feature_norm'`。

**原因:** `build_policy()` は `VLAPolicy.__init__()` をバイパスして手動でモジュールを構築しているが、新しく追加した `feature_norm` が含まれていなかった。

**対策:** `build_policy()` と `quick_test.py` の手動構築部分に `policy.feature_norm = torch.nn.LayerNorm(hidden_dim).to(device)` を追加。**教訓: `__init__` をバイパスする手動構築は、新しいモジュール追加時に漏れやすい。**

---

## 2026-04-17: `<start_of_image>` タグが特殊トークンとして認識されない

**問題:** LIBERO rollout で全エピソード失敗。loss も 0.28 で下げ止まり。

**原因:** `encode()` で画像プレースホルダーとして `<start_of_image>` というタグを使っていたが、Gemma 4 のトークナイザはこれを特殊トークンとして認識しない。結果、`<`, `start`, `_`, `of`, `_`, `image`, `>` の 7 個の普通トークンに分割され、**画像位置マーカーが完全に消失**。`mm_token_type_ids` が全て 0 になり、画像が一切埋め込まれていない状態で学習していた。

**対策:** `apply_chat_template` 経由でプロンプトを構築するように変更。これにより正しい画像プレースホルダートークン (`<|image|>`, ID=258880) が 256 個並んで展開される。

```python
messages = [{"role": "user", "content": [
    {"type": "image"}, {"type": "image"}, {"type": "text", "text": instr}
]}]
prompt = processor.apply_chat_template(messages, tokenize=False)
# processor() に渡すと input_ids の画像プレースホルダーが展開される
```

**教訓:** 特殊トークンを直接プロンプトに書き込むのではなく、必ず processor の chat template 経由でビルドする。

---

## 2026-04-17: Vision features の次元が LLM hidden dim と不一致

**問題:** `masked_scatter` で CUDA assertion エラー (`totalElements <= srcSize`)。

**原因:** `get_image_features()` は ViT の出力そのまま（768次元）を返す。LLM の hidden dim は 1536。Gemma4 は内部で `embed_vision` モジュールを通して 768→1536 に投影しているが、手動で `get_image_features` を呼ぶだけではこの投影がスキップされる。

**対策:** `get_image_features` の後に `embed_vision` を手動で呼ぶ。最終的には Gemma4 の native `forward()` に任せて全部の内部処理を自動化する方針に転換。

---

## 2026-04-17: LIBERO sim の proprio 形式が学習データと不一致

**問題:** rollout で学習データに似た動きにならない。

**原因:** 学習データ (LeRobot) の proprio は 8D で `[x, y, z, rx, ry, rz, gripper_left, gripper_right]` (xyz + euler/axis-angle + 2-finger gripper)。メタデータには `[x, y, z, rx, ry, rz, rw, gripper]` (quaternion + 1D gripper) と書いてあったが**嘘**。一方 `eval_libero.py` はメタに従って `[eef_pos(3) + eef_quat(4) + gripper_mean(1)]` の 8D を作っていた。

**対策:** `robot0_eef_quat` を euler angle に変換、`robot0_gripper_qpos` (2D) をそのまま使用。

```python
from scipy.spatial.transform import Rotation as R
eef_euler = R.from_quat(obs["robot0_eef_quat"]).as_euler("xyz")
proprio = np.concatenate([obs["robot0_eef_pos"], eef_euler, obs["robot0_gripper_qpos"]])
```

**教訓:** データセットのメタデータ (`names`) は信用せず、実際のデータを見て形式を確認する。

---

## 2026-04-17: LIBERO sim の agentview が 180 度回転

**問題:** rollout 動画と学習データ動画を比較すると向きが違う。

**原因:** LIBERO の `OffScreenRenderEnv` は agentview を上下反転・左右反転された状態で返す。`make_composite_frame` では `[::-1]` で垂直反転だけしていたが、実際は 180 度回転（縦横両方反転）が必要。

**対策:** MSE 比較で向きを特定。

| 変換 | MSE |
|---|---|
| そのまま | 6194 |
| 縦反転 | 4835 |
| 横反転 | 3902 |
| **180度回転** | **538** ✓ |

`convert_obs_to_batch` と `make_composite_frame` の両方で `agent_image[::-1, ::-1]` に修正。

**教訓:** 視覚系のデバッグは数値比較 (MSE) で客観的に判断する。

---

## 2026-04-17: Bridge Attention の post-norm で ActionQuery の勾配消失

**問題:** 新アーキテクチャ (frozen VLM + Bridge + Flow matching) で 1 サンプルすら overfit できない (MSE 0.21 で頭打ち)。

**原因:** Bridge Attention が post-norm 構造 (`query = norm(query + attn(query))`)、かつ ActionQuery の初期化が `* 0.02` と小さすぎた。初期値が小さい query が cross-attention で KV の混合結果に飲まれて全 query 位置で同一化。その後の self-attention と LayerNorm で完全に同じ値になり、**ActionQuery の勾配が完全にゼロ** になっていた。

**確認方法:**
- `action_query.grad.norm() == 0`
- `latents[0].std(0).mean() == 0` (20 query 位置で全て同じ出力)

**対策:**
1. **Pre-norm** 構造に変更 (`h = norm(query); query = query + attn(h)`)
2. ActionQuery の init scale を 0.02 → 0.5 に上げる

これで 1 サンプル overfit MSE が 0.21 → **0.0005** に改善。

**教訓:**
- 新しい attention 構造を組むときは post-norm より pre-norm を優先 (gradient flow が安定)
- 勾配消失は loss カーブだけでは判断できない。`.grad.norm()` と出力の多様性で直接確認する
- Synthetic data での overfit と real data での overfit でアーキテクチャの健全性が違って見えることがある
