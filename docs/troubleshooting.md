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
