# Troubleshooting Log

Gemma 4 E2B VLA 移植で遭遇した問題と対策の記録。
CLAUDE.md ルールに基づき、バグ修正 / 環境構築問題 / デバッグに時間がかかった事象を記録する。

---

## 2026-04-20

### #001  libero_requirements.txt install が evdev build で失敗 (Phase 2a 並行 install)

**現象**:
`.venv-gemma4` (uv 管理) で `uv pip install -r VLA-Adapter/experiments/robot/libero/libero_requirements.txt` 実行時、`robosuite==1.4.1` → `pynput==1.8.1` → `evdev==1.9.3` の build 段階で以下エラーで失敗:

```
× Failed to download and build `evdev==1.9.3`
No solution found when resolving: `setuptools>=77.0`
Because only setuptools<=70.2.0 is available and you require setuptools>=77.0,
we can conclude that your requirements are unsatisfiable.
```

**原因**:
`.venv-gemma4` の index 優先順位で pytorch index (`download.pytorch.org/whl/cu128`) が first-match となり、そこには `setuptools<=70.2.0` しか存在しない。`evdev` の build-system は `setuptools>=77.0` を要求するため、uv default の first-match 方針では解決不能。dependency confusion 防衛のための仕様挙動。

**対策**:
`uv pip install --index-strategy unsafe-best-match` で複数 index を横断検索、pypi から setuptools 新版を引けるようにして解決:

```bash
uv pip install --index-strategy unsafe-best-match -r VLA-Adapter/experiments/robot/libero/libero_requirements.txt
```

Exit 0、33 packages installed (robosuite 1.4.1, bddl 3.6.0, mujoco 3.7.0, gym 0.26.2, evdev 1.9.3, etc.)。Stage 1 critical pkg (torch/transformers/accelerate/timm/tokenizers/peft/dlimp/tensorflow 系) は byte-identical 維持、downgrade なし。

**教訓**:
- uv × pytorch index 混合環境では、native build を含む C 拡張 (evdev, mujoco native, robosuite extensions 等) で同類の build-time setuptools version mismatch が起きうる
- fallback 手順は (1) `--index-strategy unsafe-best-match` 試行、失敗時は (2) `setuptools>=77.0` を先行注入する局所修復
- 事前に Stage 1 `requirements-gemma4.txt` を snapshot し、install 後 diff で critical pkg downgrade の有無を sanity check する手順をルーチン化すべき

---

### #002  draccus × `from __future__ import annotations` で dataclass 検出が壊れる (Phase 2b finetune_gemma4.py 初回実行)

**現象**:
`VLA-Adapter/vla-scripts/finetune_gemma4.py` 初回実行で以下 trace:

```
File ".../draccus/argparsing.py", line 98, in _assert_no_conflicts
    if utils.CONFIG_ARG in [field.name for field in dataclasses.fields(self.config_class)]:
File ".../dataclasses.py", line 1246, in fields
    raise TypeError('must be called with a dataclass type or instance') from None
TypeError: must be called with a dataclass type or instance
```

`@dataclass` decorator は正しく付けているのに draccus は `FinetuneConfig` を dataclass として認識しない。

**原因**:
`from __future__ import annotations` (PEP 563) を file 冒頭に置いたため、関数アノテーションが全て string に evaluate される (`def finetune(cfg: FinetuneConfig)` が `cfg: "FinetuneConfig"` 相当になる)。draccus の `@draccus.wrap()` はデコレータ対象関数のシグネチャから config class を抽出するが、string literal "FinetuneConfig" を class として dataclasses.fields に渡すため TypeError。

**対策**:
`from __future__ import annotations` を削除。Python 3.11 では `tuple[X, Y]` や `list[X]` 等の PEP 585 組み込み generic は future import 不要で使えるため、削除による型アノテーション破綻はない。

```diff
- from __future__ import annotations
- import json
+ # NOTE: draccus の dataclass 検出は annotations を実 class として要求するため
+ #       `from __future__ import annotations` は本 file で使用禁止
+ import json
```

修正後、再実行で問題なく動作 (100-step smoke + resume 検証 5/5 bit-exact PASS)。

**教訓**:
- `@draccus.wrap()` を使う file では `from __future__ import annotations` を使わない
- 同系の dataclass ベースの ser/deser ライブラリ (pydantic の一部 legacy mode、marshmallow_dataclass など) でも類似の string-annotation 非互換が発生しうるため、draccus 以外でも要注意
- file 冒頭 `NOTE:` コメントで将来の編集者への明示アラートを入れる (recurrence 防止)

---

### #003  `experiments.robot.libero.libero_utils` 経由で `openvla_utils` の dead import (Phase 2c dry-run 初回実行)

**現象**:
`scripts/gemma4/eval_libero_gemma4.py` が LIBERO 用 helper (`get_libero_env`, `quat2axisangle` etc.) を使うため、`from experiments.robot.libero.libero_utils import ...` を書いたが、以下の chain で失敗:

```
experiments.robot.libero.libero_utils
  → from experiments.robot.robot_utils import DATE, DATE_TIME
    → from experiments.robot.openvla_utils import ...
      → ModuleNotFoundError: No module named 'json_numpy'
      → (その後 `ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'`)
```

`robot_utils.py` が top-level で `openvla_utils.py` を import しており、そこで (a) json_numpy が未 install、(b) transformers 5.5.4 で `AutoModelForVision2Seq` が削除済 (unified Auto クラスに統合)、の 2 段 break。

**原因**:
- `robot_utils.py` はモデル家族判定 (`get_vla`, `get_vla_action`) を openvla_utils から再輸出する構造で、top-level import にしている
- Gemma 4 VLA eval 経路からは `openvla_utils.*` を一切使わないが、`from experiments.robot.robot_utils import (normalize_gripper_action, ...)` だけで連鎖を引いてしまう
- transformers 5.5.4 は `AutoModelForVision2Seq` を削除、`AutoModelForImageTextToText` 等に整理 (破壊的変更)

**対策**:
Gemma 4 eval script では上流 import を一切せず、必要 helper 6 個 (`get_libero_env`, `get_libero_dummy_action`, `get_libero_image`, `get_libero_wrist_image`, `quat2axisangle`, `normalize_gripper_action`, `invert_gripper_action`, `set_seed_everywhere`) を verbatim で inline コピー。いずれも OpenVLA に依存しない pure function (約 60 行)。これにより `openvla_utils.py` の dead import chain を完全回避。

`libero.libero.benchmark` / `libero.libero.get_libero_path` / `libero.libero.envs.OffScreenRenderEnv` は pip install 済の LIBERO パッケージから直接 import (上流 chain なし) OK。

**教訓**:
- 他プロジェクトの `experiments/` 以下の utility を利用する際は、top-level import chain 全体を確認してから依存する
- Gemma 4 scope では OpenVLA 系の Qwen tokenizer / action tokenizer / FiLM processor 等は全く使わないのに、import chain で引き摺り出されて破綻するパターン頻出 (troubleshooting.md #002, #003 が同系統)
- inline copy のコスト < dead import chain 修復のコスト、と判定したら inline を選ぶ (保守性は署名で原典にリンク)

---

### #004  `torch.load` が weights_only=True 既定で LIBERO 内部 init_states の numpy 読み込みに失敗

**現象**:
Phase 2c dry-run で `task_suite.get_task_init_states(task_id)` (LIBERO 内部) 呼び出し時:

```
File ".../LIBERO/libero/libero/benchmark/__init__.py", line 164, in get_task_init_states
    init_states = torch.load(init_states_path)
_pickle.UnpicklingError: Weights only load failed. ...
WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct was not an allowed global by default.
```

**原因**:
PyTorch 2.6+ で `torch.load` の default `weights_only=True` になり、任意 Python object の pickle load を reject する security 強化。LIBERO の init_states ファイルは numpy array 化された初期状態データ (trusted local file) で、numpy の private symbol `_reconstruct` を含むため reject される。LIBERO source を編集せずに対応する必要あり。

**対策**:
eval script で `torch.load` を monkey-patch、default で `weights_only=False` を付与:

```python
_ORIG_TORCH_LOAD = torch.load
def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)
torch.load = _torch_load_weights_only_false
```

ローカル pip install 済 LIBERO は信頼 source、我々の finetune_gemma4 checkpoint も自前生成のため weights_only=False で安全。eval script 冒頭に置き、子 process (test_12_resume_child など) では不要。

**教訓**:
- PyTorch 2.6 移行時は既存 library の `torch.load` 呼び出し箇所を全点検 (外部依存の source を直接 patch できない場合は monkey-patch で回避)
- 上流 library が新しい PyTorch に未追従なケースは続出予想、project-wide の defensive monkey-patch 層を 1 箇所設けるのが合理的 (eval / train script の共通 prelude)

**Scope 注記** (2026-04-20 User 指示で追加):
- monkey-patch は **process-global**、`eval_libero_gemma4.py` の module 読込時に install される
- `test_13_rollout_dryrun.py` / `test_14_rollout_sanity.py` (Phase 2f) / Phase 2g full eval は eval_libero_gemma4 を import するため patch が効く
- `finetune_gemma4.py` / `test_12_resume_child.py` は eval_libero_gemma4 を import しないため patch 非適用 (これらは自前 `torch.load(..., weights_only=False)` を明示的に呼ぶ)
- **副作用リスク**: WandB / HuggingFace / accelerate の内部で pickled tensor を load する経路が仮にあれば同じ patch が効く (通常は pickle 使わない実装だが、将来 library 更新で使うようになった場合は要再点検)
- 将来的には `contextlib.contextmanager` でスコープ限定する方向を検討 (eval script 内部の LIBERO 呼び出し箇所のみ `with weights_only_false_context(): ...` で覆う)

---

### #005  `prismatic.overwatch` の accelerate.PartialState() が torchrun 下で default process group を pre-init (D1 DDP smoke 初回実行)

**現象**:
`torchrun --nproc_per_node=2 finetune_gemma4.py --ddp_mode True ...` で以下 error、rank 0/1 両方で発火:

```
File ".../finetune_gemma4.py", line 427, in finetune
    dist.init_process_group(backend=cfg.ddp_backend, init_method="env://")
...
ValueError: trying to initialize the default process group twice!
```

**原因**:
`finetune_gemma4.py` が `from test_08_data_pipeline import ...` を経由して `prismatic.*` を import、その中で `prismatic.overwatch.overwatch` が:

```python
# prismatic/overwatch/overwatch.py
from accelerate import PartialState
...
self.distributed_state = PartialState()
```

を **module 読込時に実行**。`PartialState()` は torchrun 環境変数 (LOCAL_RANK, WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT) を検知して内部で `dist.init_process_group` を実行する。後続の `finetune_gemma4.py:finetune()` での `dist.init_process_group` 呼び出しが 2 回目の init と見なされ reject。

**対策**:
`finetune_gemma4.py` の DDP 初期化を `dist.is_initialized()` でガード:

```python
if not dist.is_initialized():
    dist.init_process_group(backend=cfg.ddp_backend, init_method="env://")
else:
    print(f"[rank {global_rank}] dist already initialized by accelerate.PartialState, backend={dist.get_backend()}")
```

accelerate が nccl backend で init していれば互換性あり、後続の `DDP(model)` wrap + NCCL all-reduce は正常動作。D1 smoke で確認済。

**教訓**:
- 他プロジェクトの overwatch / logging helper が import 時に副作用を持つケースに注意 (特に `accelerate.PartialState` / `PartialState()` コンストラクタは torchrun 検知で process group を init)
- DDP 系は「自前 init」前提の実装が多いが、上流に accelerate 系のヘルパーが混入していると pre-init で conflict する
- 対処は is_initialized() ガードで defensive に; 再初期化は禁止

---

### #006  L1RegressionActionHead Pro version で DDP `find_unused_parameters=False` が "partial grad" error (D1 DDP smoke step 2)

**現象**:
D1 smoke 実行時、step 1 は成功、step 2 の forward で:

```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
This error indicates that your module has parameters that were not used in producing loss.
...
Parameter indices which did not receive grad for rank 0: 36 37 59 60 82 83 105 106 ... (48 個)
```

rank 0/1 両方で同じ indices が grad 未受領。

**原因**:
`VLAAdapterGemma4.action_head` の `L1RegressionActionHead(use_pro_version=True)` は 24 block の cross-attention + FiLM stack を持つ。`predict_action` 内で phase="Training" / "Inference" で実行 path が分岐、特定 block の一部 weight (e.g., `blocks.<k>.film_layer.gamma` / `beta`) が Training phase で forward 非経由、backward で grad を受け取らない。

DDP の default (`find_unused_parameters=False`) は全 trainable param が backward で grad を受ける前提で bucket を管理。grad 未受領 param があると step 2 の `_pre_forward` 内で `_rebuild_buckets` が前 step の reduction 未完了を検知して abort。

**対策**:
`DDP(model, find_unused_parameters=True)` に設定。毎 iteration で forward 経由 param を動的検出し、unused は reduction bucket から除外。overhead 5-10% (小 bucket の追加走査)。

```python
model_vla = DDP(model_vla, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)
```

**教訓**:
- アーキテクチャに conditional branch (phase / mode / mask) がある場合は `find_unused_parameters=True` が保守的に安全
- Stage 1 / Phase 2b の single-GPU では「全 trainable が grad 受領」で成立するが、DDP は step-to-step の bucket 状態管理が厳密で partial grad を許さない
- overhead はプロジェクト依存だが 5-10% 許容可な場合は True にして安定性優先
- 原因は `predict_action` 内の block 間分岐、アクティブ path が phase 依存だが C1 本番 DDP retrain でも同設定 (`find_unused_parameters=True`) を維持

---

## 2026-04-22 (Dual-Track Redesign Phase)

### #007  Gemma 4 native vision API が spec 想定と乖離 (Task 6 pre-implementation inspection)

**現象**:
Dual-track redesign の初期 spec / plan では Gemma 4 内蔵 vision を `m.model.vision_tower` + `m.model.multi_modal_projector` で使える前提だった (PaliGemma 的想定)。Task 6 のpre-implementation inspection で `Gemma4ForConditionalGeneration` の実構造を調査したところ複数の乖離が判明:

1. `multi_modal_projector` attribute **存在しない**。代わりに `embed_vision` (`Gemma4MultimodalEmbedder` = RMSNorm + Linear) が projector 役
2. `vision_tower` は **patchified input** を要求: `pixel_values: (B, max_patches, 768)` + `pixel_position_ids: (B, max_patches, 2)` (padding は `(-1, -1)`)、生画像 `(B, 3, H, W)` 不可
3. `vision_tower` 出力は **padding-stripped flat** `(N_valid_total, 768)`、batch 次元が畳まれる
4. `_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)` のみ有効、spec 想定の `{128, 256}` 不可
5. `Gemma4ImageProcessor.preprocess()` は `do_normalize=False` (ImageNet 正規化なし、rescale ÷255 のみ)
6. 画像サイズに関わらず processor が内部で aspect-ratio-preserving resize → 常に `max_patches = max_soft_tokens * 9` のサイズに揃う

**原因**:
Gemma 4 の multimodal 構造は PaliGemma 系 (SigLIP + Linear projector) とは別設計の native variant (Gemma4MultimodalEmbedder)。HF docs が新しく浸透していなかったため spec 執筆時に従来 PaliGemma 前提で書いてしまった。

**対策**:
Spec rev 3 (`6c8d3e5`) + Plan rev 3 (`0a04ebf`) で全面修正:
- `Gemma4ImageProcessor` を直接使用 (自作 patchify 禁止)
- `VLAAdapterGemma4.encode_scene` helper で preprocess → `get_image_features` → reshape back を一気通貫
- `max_soft_tokens` を YAML 可変化 (default 280 → 256 soft tokens/image)
- `NUM_VISION_TOKENS` constant は model 属性 `self.num_vision_tokens` で dynamic 化

**教訓**:
- 新モデルの内部構造は **実環境で `from_pretrained` + `named_children()` + `inspect.signature(forward)` で実測してから spec 書き起こす**。HF docs / paper 類推は誤る
- `AutoProcessor.from_pretrained` は Gemma4VideoProcessor を巻き込んで torchvision extras 不足で import fail することがある。image だけなら `from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor` で直 import が安全
- Padding-stripped flat 出力は batch 復元に `num_soft_tokens_per_image` metadata が必須。固定サイズ入力なら batch 内で均一、reshape `(N, D) → (B, N/B, D)` で戻せる

---

### #008  torch 2.11 用 flash-attn 2.x wheel が存在しない (Task 5)

**現象**:
Task 5 で `attn_implementation = "flash_attention_2"` を default にしようとしたが、`.venv-gemma4` (torch 2.11.0+cu128, Python 3.11) 向けの flash-attn 2.x wheel が HF の releases に一切存在せず (v2.8.3 までで最新は torch2.10 対応)。ソース build は過去セッションで試行して失敗 (>1h コンパイル後タイムアウト/依存解決不能)。

また session 途中で `.venv` (別 venv、torch 2.7.1) に flash-attn 2.8.3 wheel をインストールし Task 5 smoke test を走らせてしまい、"FA-2 動作確認済" と誤報告した事件あり (.venv と `.venv-gemma4` の取り違え)。

**原因**:
1. flash-attn release cycle が torch の release と少し遅れる、torch 2.11 は比較的新しいため wheel 未追いつき
2. 複数 venv (`.venv` と `.venv-gemma4`) が同 repo に共存、どちらを使うかの意識が薄いと即ミス
3. system `python` は 2.7 (ubuntu default)、つい実行すると全然違う env で走る

**対策**:
- `attn_implementation = "sdpa"` を維持決定 (Task 5 commit `b9b19d3` で amend)。torch 2.11 の native SDPA に Flash Attention backend が組み込み済なので Gemma 4 での実効速度は FA-2 と同等レンジ
- Spec rev 3 / Plan rev 3 で全 `"flash_attention_2"` リテラルを `"sdpa"` に置換
- **全ドキュメント / コマンド例に `.venv-gemma4/bin/python` を絶対パスで記載、`python` / `.venv/bin/python` 禁止**

**教訓**:
- 新しい torch に乗り換える時は flash-attn wheel 存在を先に確認 (`curl -s https://api.github.com/repos/Dao-AILab/flash-attention/releases` で grep)
- 複数 venv の共存は事故の元。project root に `.envrc` + direnv、または `UV_PROJECT_ENVIRONMENT` で一本化するのが理想
- subagent への prompt に **Python env を CRITICAL として明示**する (`.venv-gemma4/bin/python`)、複数回繰り返すくらいで丁度いい

---

### #009  Gemma 4 vision 入力は画像サイズに依らず常に同じ patch 数に resize される (Task 6, Task 14)

**現象**:
`Gemma4ImageProcessor.preprocess(img_224x224)` でも `preprocess(img_768x768)` でも出力 `pixel_values.shape` が同じ `(1, max_patches, 768)` になる。valid patch count も `num_soft_tokens_per_image` も同値。

**原因**:
Gemma4ImageProcessor の `aspect_ratio_preserving_resize` が `max_patches = max_soft_tokens * pooling_kernel_size^2` を target に画像を resize するため、入力画像サイズに関わらず常に同じ内部表現に揃える仕様。224x224 を渡しても内部で upsample して 768x768 相当 (max_soft_tokens=280 時) で処理される。

**影響**:
- Vision attention compute は **max_soft_tokens が決定**、入力画像サイズは計算量に影響しない (padding 量だけ変わる)
- max_soft_tokens=70 → 64 soft tokens/image、vision attn O(630²)
- max_soft_tokens=140 → 121 soft tokens/image、vision attn O(1260²)
- max_soft_tokens=280 (default) → 256 soft tokens/image、vision attn O(2520²)
- data loader で 224x224 に事前 resize する必要なし (processor が内部で整える)、ただし I/O / 初期 resize コストは残る

**対策**:
- `VLAAdapterGemma4.__init__` で dummy probe を一度走らせて `self.num_vision_tokens` を実測決定 (初期化コスト数十 ms、副作用なし)
- YAML `max_soft_tokens` で LLM 入力トークン数 vs vision compute を trade-off 可能に (default 280)

**教訓**:
- HF の image processor は黒箱ではなく、source 読んで resize 方針を把握しないと memory / compute 見積もりを誤る
- `do_rescale=True, do_normalize=False, image_mean=[0,0,0], image_std=[1,1,1]` という Gemma 4 default は ImageNet 系と異なる。data loader 側で ImageNet 正規化しない選択が正解

---

### #010  VLAAdapterGemma4 forward interface 変更が downstream `finetune()` main に cascade (Task 11, 13, 14)

**現象**:
Task 6 で `VLAAdapterGemma4.__init__` の `vision_backbone` 引数を削除、Task 7 で `pixel_values` dict schema を `{"dino", "siglip"}` → `{"scene", "wrist"}` に変更、Task 8 で `self.wrist_encoder` 属性追加、Task 11 で `training_mode` 引数追加、といった interface 変更が重なり、`finetune_gemma4.py` の downstream (`finetune()` main 関数内) に 6-8 箇所の stale 参照が残った:

- `build_model` 返り値 3-tuple → 2-tuple 変更 (Task 13)
- `build_pretrain_dataloader(cfg, tok, vision_backbone)` 引数削除
- `inner_model.vision_backbone.parameters()` grad-leak check (存在しない attribute アクセス)
- subprocess spawn の `--vision-backbone-id` 引数
- `FinetuneConfig.vision_backbone_id` 停止必要

Task 11/12/13 実装時の smoke test は **`build_model` 単体呼び出しで検証**していたため、`finetune()` main flow の壊れは検知できず。Task 14 で Implementer が grep 調査して全部洗い出し、一括 cleanup。

**原因**:
Task レベルでの scope 境界が厳しく、Task 6 等では「VLAAdapterGemma4 の内部だけ」を守ったため、downstream の参照元までは触らなかった (plan が "Task 14 (data loader) で吸収" と scope 分離していた)。しかし各 Task の smoke test は通るので "破壊検知" はされない。

**対策**:
Task 14 で:
- `build_model` から `vision_backbone` 引数・関連 `DinoSigLIPViTBackbone` 構築を削除
- `build_pretrain_dataloader` を新 `taco_solo_loader.TacoSoloDataset` 経由に置換
- `build_dataloader` (LIBERO path) は `NotImplementedError` + FIXME 残し (LIBERO migration 別 task)
- grad-leak check は削除 (`llm.parameters()` の freeze 一括で保証)
- subprocess spawn から `--vision-backbone-id` 除去

**教訓**:
- interface 変更を伴う refactor では **caller grep (`grep -rn "<removed_attr>\|<removed_kwarg>"`)** を implementer prompt に組み込む。smoke pass だけでは不十分
- plan で scope を「A の内部だけ」に絞る時、**downstream の壊れは明示的に acknowledge して別 Task に集約**する書き方にする (Task 6 plan で "Task 14 で吸収" と明記していたのが機能した)
- Task 15 (E2E smoke) は interface 整合性の最終検知ポイント、ここで `draccus.parse → build_model → build_param_groups → forward → loss.backward()` までチェーンで実行することで残存 interface mismatch を確実に露出させる

---

### #012  Mode A (LoRA+GC+native vision) の step time が長時間 run で急激に degrade (2026-04-23)

**現象:**
Mode A (quality: LoRA r=16 + gradient_checkpointing + Gemma 4 native vision、B=24 DDP 2-way) で pretrain/fine-tune を走らせると、step ~1000-2000 付近で **step time が 2.2 → 5.7 s/step に急激にジャンプ**。gradual ではなく 100 step 以内で 2.6x 程度増加、Escalation #9 で halt。

観測:
- Medium A (5000 step targeted): step 2968 で halt (~6.95 s/step)
- diag solo run (3500 step targeted、同 config): step 1943 で halt (~5.69 s/step)
- 4 runs 同時実行時は step 989 で halt (早い) → contention で degrade trigger が早まる
- Mode B は 5000 step 安定 (1.28 s/step 不変)
- **Mode A SigLIP 経路 (vision_backbone_type=siglip) は 2000 step 完走** → native vision tower が degrade trigger の可能性

**原因 (確定してない、候補仮説):**
可能性排除済:
- ❌ Memory fragmentation: `torch.cuda.memory_allocated` / `reserved` / `max_allocated` いずれも完全安定 (diag log で 200 step 毎測定、alloc=15.9GB 不変)
- ❌ Pure thermal throttling: GPU temp 80-82°C で solo / concurrent 同値、temp 自体は degrade 前後で変化なし
- ❌ Data loader degradation: Mode B は同 loader で degrade しない

可能性残存:
- **CUDA kernel autotune** (LoRA 計算 path で自動選択が何か閾値を越えてから slow な kernel に切替え)
- **PyTorch autograd graph の cumulative state** (GC recompute が graph 肥大化、某 step で traverse cost がjump)
- **DDP `find_unused_parameters=True` の累積 overhead** (毎 iter 全 autograd graph scan、LoRA 分岐で cost 上昇)
- **System-level bottleneck** (PCIe bus、CPU thread pool、memory bandwidth) で concurrent 時に早く hit

**対策 (暫定):**
- Mode A は **2000 step 未満の run でまず様子見** (Mode A SigLIP は 2000 step 完走実績)
- 長期 run 必要な場合は Mode B を主体に
- 将来: `torch.backends.cudnn.benchmark = False` を試す、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 試す、`find_unused_parameters=False` に切替 (LoRA がすべての path を通るなら安全)

**原因確定 (2026-04-23 追記):** **CUDA caching allocator の internal fragmentation が根本原因**。

Ablation 結果:
- `find_unused_parameters=False`: step 1285 で halt (むしろ悪化) ❌
- `cudnn.benchmark`: default False、元から無効、無関係 ❌
- input_require_grads hook: `gradient_checkpointing_enable()` が自動追加 + PEFT が追加で hook 計 2 重、正常 ❌
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: **3000 step 完走、degrade 完全消失** ✅

メカニズム:
- Default segmented-BiN allocator は fixed-size blocks 方式、free block が特定サイズしか再利用可
- LoRA + GC + 動的活性化 で多様 alloc pattern → block 粒度 mismatch 累積
- `reserved` 総量は 37.6GB 安定だが **実効 usable 空間が fragment で減り続ける**
- 某 step で new alloc の free block 探索 cost が急増 (linear search + split/merge overhead) → step time 2.2s → 5.7s に突然 jump
- Memory 外から見ると "CUDA memory stable なのに遅くなる" という謎症状になる

Fix (採用):
- 学習 launch 時に `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` env var を必ず設定
- 副次効果: `reserved` が 37.6 → 30.7GB (-18%) に縮小、memory 効率も向上
- launch scripts (`scripts/gemma4/launch_pretrain_{quality,speed}.sh`) に永続化

**教訓:**
- `torch.cuda.memory_allocated/reserved/max_allocated` だけ見てても fragmentation は検知できない (外形値は stable)
- 徐々に遅くなる症状 + CUDA memory stable + temp normal は **fragmentation 第一候補**
- PyTorch 2.x の expandable_segments は default OFF (back-compat)、LoRA + GC 系 training では ON が事実上必須


**教訓:**
- step_time の経時 log は必須 (diagnostic 無しでは thermal vs memory vs cuda の切り分け不可)
- Escalation 閾値は batch_size / mode / hardware 依存で経験則的、事前定義は困難
- `torch.cuda.memory_stats()` を定期 dump する軽量 instrumentation を keep、長時間 run の trace として価値あり

---


**現象**:
Task 13 実装時に draccus で config load → `build_model` → param count assertion で `expected = 675.138 ± 0.5 M` が fail。675.138 は Stage 2 (DinoSigLIP + FiLM あり + PEFT なし) の trainable count で、dual-track redesign 後は Mode A ~546M / Mode B ~541M と全く異なる。

**原因**:
Task 2 で `film_gen` 削除 (113M 減)、Task 6 で DinoSigLIP 削除 (~400M 減)、Task 8 で WristResNet18 追加 (~12M 増)、Task 12 で LoRA 追加 (~5M、Mode A のみ) と trainable 総数が mode 次第で変動するようになったが、legacy の tight assertion が旧 baseline のままだった。

**対策**:
- `build_model` 内の tight assertion を **log-only** に変更 (print で mode-aware な値を出力)
- `VLAAdapterGemma4.__init__` 側の loose bound (500-620 M) は残す (明らかな spec 違反を検知する trap として機能)

**教訓**:
- mode / feature flag で param 数が変動する設計では **tight equality assertion を避ける**。loose range + log 出力の組合せが保守性と detection を両立
- Refactor 中の assertion は「過去の正しさ」ではなく「今の契約」を表現すべき。古い baseline を置きっぱなしにすると新 feature 追加のたびに誤陽性になる

---

### #013  Mode A の LoRA weights が checkpoint に保存されず、eval が全 0% (2026-04-24)

**現象**:
Mode A (quality: LoRA r=16 + GC + SigLIP、E4B backbone、LIBERO Spatial 10k finetune) の train loss は健全に収束 (0.62 → 0.13)、Mode B 10k と同等。ところが LIBERO eval で成績が全 task 0-1% に張り付く。Mode B (同 step 数) が 34% 出ているのと極端な差。同じ症状が step 3k preview eval・10k full eval・E2B legacy Mode A B=24 (前 session) の全てで再現。

**原因**:
`save_checkpoint` の trainable_state フィルタ `if not k.startswith("llm.") and not k.startswith("vision_backbone.")` が、PEFT LoRA wrap 後の key (`llm.model.language_model.base_model.model.layers.N.self_attn.q_proj.lora_A.default.weight` 等) を **frozen LLM と誤認して除外**。結果、ckpt の trainable_state_dict に LoRA weight が 0 個保存される。

eval 側の load check `relevant_missing = [k for k in missing if not k.startswith("llm.") and not k.startswith("vision_backbone.")]` も同じロジックで LoRA missing を silent に許容していたため、assertion が発火せず誰も気付かなかった。

action_head は LoRA-modified LLM 出力を前提に訓練されているが、eval で LoRA delta が zero-init のままだと vanilla LLM 出力を受け取り、出力が完全に train distribution 外 (0-1%)。train 側は LoRA active なので loss は下がる。典型的な silent save bug。

**対策**:
- `save_checkpoint`: `k.startswith("llm.")` 判定に **`"lora_" in k` の例外**を追加 (save 対象に)
- `load_model_state` (eval): 同じ修正で missing LoRA を検知 (assert 発火)
- 既存 Mode A ckpt は全て無効、再訓練必須

**教訓**:
- `strict=False` + key-prefix filter の組合せは **silent weight 欠落** の温床。要素数・key 名 parity check を一発入れるべきだった
- 「trainable な params は保存される」という暗黙前提が、PEFT 等の wrap で暗黙前提じゃなくなる。**param が trainable かどうかは `requires_grad` で判定するのが safer**
- train loss が下がっている ≠ 保存も正しい。save round-trip を smoke に入れるべき (B=2 を 1 step → save → reload → forward 同値 check)

---
