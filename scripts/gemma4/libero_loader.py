"""
Task 14 LIBERO fine-tune loader: VLA-Adapter + Gemma 4 新 vision schema 対応.

`scripts/stage3/taco_solo_loader.py` の pattern を踏襲しつつ、LIBERO 系 RLDS 用に調整:
  - LIBERO 系 (libero_{spatial,object,goal,10}_no_noops) 単独用
  - Data path は `data/modified_libero_rlds/<dataset_name>`、RLDS 形式 (TFRecord 16 shards)
  - 上流データ生成 (obs_transforms/traj_transforms/BOUNDS_Q99 正規化 + 8-step chunking 等) は
    既存の `prismatic.vla.datasets.rlds.make_interleaved_dataset` に委譲する
    (test_08_data_pipeline.py と同じ経路)。
  - 新 vision schema ({"scene", "wrist"}) を produce。旧 {"dino", "siglip"} pixel_values とは非互換:
      * scene: (B, 3, 224, 224) float32 [0, 255] (Gemma4ImageProcessor が内部で rescale/patchify)
      * wrist: (B, 3, 224, 224) float32 ImageNet 正規化済 (WristResNet18 用、taco_solo_loader と同じ)
  - proprio: (B, 8) float32、action: (B, 8, 7) float32 normalized (BOUNDS_Q99、RLDS 内で正規化済)
  - dataset_id は 0 固定 (単一 task suite fine-tune)

input_ids layout (taco_solo と一致、caller-injection num_vision_tokens 対応):
  [BOS] + prompt(20) + VISION_PLACEHOLDERS(num_vision_tokens) + [PROPRIO] + ACTION_TOKENS(64) + [EOS]

VLAAdapterGemma4.forward 互換 batch dict:
  {
    "pixel_values": {"scene": (B, 3, 224, 224) float, "wrist": (B, 3, 224, 224) float},
    "input_ids":    (B, L) long,
    "proprio":      (B, 8) float32,
    "actions":      (B, 8, 7) float32 normalized,
    "dataset_id":   (B,) long, 全 0,
    "languages":    [str] × B,
  }

Run:
  # 3 raw samples、shape/dtype/言語 instruction 確認 (GPU 不要、TF CPU only)
  .venv-gemma4/bin/python scripts/gemma4/libero_loader.py --verify

  # B=2 の full batch 構築 (DataLoader 経由、tokenizer 使用)
  .venv-gemma4/bin/python scripts/gemma4/libero_loader.py --smoke-batch
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import torch
from torch.utils.data import IterableDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
sys.path.insert(0, str(VLA_ROOT))

from prismatic.vla.constants_gemma4 import (  # noqa: E402
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTION_TOKENS,
    NUM_VISION_TOKENS,
    PROPRIO_PLACEHOLDER_IDX,
    VISION_PLACEHOLDER_BEGIN_IDX,
)
from prismatic.vla.constants import NormalizationType  # noqa: E402
from prismatic.vla.datasets.rlds import make_interleaved_dataset  # noqa: E402
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights  # noqa: E402


# ===========================================================
# Constants
# ===========================================================
DEFAULT_DATASET_NAME = "libero_spatial_no_noops"
DEFAULT_DATA_ROOT = "data/modified_libero_rlds"
DATASET_ID = 0

PROMPT_MAX_LEN = 20
PROPRIO_DIM = 8
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8

IMAGE_SIZE = 224

# ImageNet 正規化 (WristResNet18 backbone 用、taco_solo_loader と同値)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ===========================================================
# Image preprocessing (taco_solo_loader と同一仕様)
# ===========================================================
def _scene_from_rlds(img_uint8: np.ndarray) -> torch.Tensor:
    """RLDS primary image (H, W, 3) uint8 → (3, 224, 224) float32 [0, 255].

    RLDS make_interleaved_dataset が frame_transform_kwargs.resize_size で 224x224 に
    resize 済なので、ここでは H, W = 224 を想定するが、念のため ensure_224 で再 resize する。
    """
    # NumPy writable 警告回避のため .copy() 経由 (RLDS の uint8 buffer は non-writable)
    t = torch.from_numpy(np.asarray(img_uint8).copy()).float()   # (H, W, 3) float32 [0, 255]
    t = t.permute(2, 0, 1).contiguous()                   # (3, H, W)
    if t.shape[-1] != IMAGE_SIZE or t.shape[-2] != IMAGE_SIZE:
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear", antialias=True
        ).squeeze(0)
    return t


def _wrist_from_rlds(img_uint8: np.ndarray) -> torch.Tensor:
    """RLDS wrist image (H, W, 3) uint8 → (3, 224, 224) ImageNet-normalized float32."""
    arr = np.asarray(img_uint8)
    if arr.shape[0] != IMAGE_SIZE or arr.shape[1] != IMAGE_SIZE:
        # wrist は RLDS 側で resize 済のはず、念のため tf.image.resize で安全網
        arr = tf.image.resize(
            tf.convert_to_tensor(arr), (IMAGE_SIZE, IMAGE_SIZE),
            method=tf.image.ResizeMethod.BILINEAR, antialias=True
        )
        arr = tf.cast(tf.clip_by_value(arr, 0.0, 255.0), tf.uint8).numpy()
    normalized = arr.astype(np.float32) / 255.0                          # (H, W, 3) float [0, 1]
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD              # ImageNet
    t = torch.from_numpy(normalized).float().permute(2, 0, 1).contiguous()  # (3, 224, 224)
    return t


# ===========================================================
# LIBERO dataset (iterable、per-timestep samples via RLDS)
# ===========================================================
class LiberoDataset(IterableDataset):
    """LIBERO 単独 iterable dataset. RLDS make_interleaved_dataset で LIBERO task suite を展開し、
    Gemma 4 新 vision schema ({"scene", "wrist"}) 用に transform した per-timestep dict を yield.

    Per-timestep raw sample (pre-transform、collator に渡る形):
      {
        "scene_img":    (H, W, 3) uint8,   # RLDS primary image (224x224 resize 済)
        "wrist_img":    (H, W, 3) uint8,   # RLDS wrist image (224x224 resize 済)
        "action_chunk": (NUM_ACTIONS_CHUNK, ACTION_DIM) float32 normalized (BOUNDS_Q99、RLDS 内)
        "proprio":      (PROPRIO_DIM,) float32 raw,
        "language":     str,
        "dataset_id":   0,
      }

    NOTE: `num_workers > 0` 対応に関して、RLDS make_interleaved_dataset は内部で TF dataset を
    生成するため、fork 後に TF subsystem を初期化する必要がある。これは `_iter_chunks` 内で
    lazy に dataset を構築することで実現 (worker fork → worker 内で TF init → 安全)。
    `persistent_workers=True` 運用も問題なし。
    """

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_ROOT,
        dataset_name: str = DEFAULT_DATASET_NAME,
        num_actions_chunk: int = NUM_ACTIONS_CHUNK,
        shuffle_buffer_size: int = 1000,
        train: bool = True,
        seed: int = 42,
    ):
        self.data_dir = str(data_dir)
        self.dataset_name = dataset_name
        self.num_actions_chunk = num_actions_chunk
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train = train
        self.seed = seed

        # Sanity check (fail fast if data path missing)
        dataset_root = Path(self.data_dir) / self.dataset_name
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"LIBERO dataset directory not found: {dataset_root}. "
                f"Expected <data_dir>/<dataset_name>/1.0.0/*.tfrecord*"
            )

    def _build_rlds(self):
        """Build RLDS dataset (lazy、per worker process)."""
        mixture_spec = [(self.dataset_name, 1.0)]
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_dir,
            mixture_spec,
            load_camera_views=("primary", "wrist"),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,
                future_action_window_size=self.num_actions_chunk - 1,
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=dict(
                resize_size=(IMAGE_SIZE, IMAGE_SIZE),
                num_parallel_calls=16,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=self.shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=self.train,
        )
        dataset, dataset_length, _stats = make_interleaved_dataset(**rlds_config)
        return dataset, dataset_length

    def _iter_chunks(self) -> Iterator[dict]:
        """Yield per-timestep raw sample (RLDS batch → raw dict、image/action/proprio/language 抽出)."""
        ds_tf, _dataset_length = self._build_rlds()
        for rlds_batch in ds_tf.as_numpy_iterator():
            # RLDS batch layout (test_08_result.json 参考):
            #   observation.image_primary: (1, 224, 224, 3) uint8
            #   observation.image_wrist:   (1, 224, 224, 3) uint8
            #   observation.proprio:       (1, 8) float32
            #   task.language_instruction: bytes
            #   action:                    (NUM_ACTIONS_CHUNK, 7) float32 normalized
            scene = rlds_batch["observation"]["image_primary"][0]    # (224, 224, 3) uint8
            wrist = rlds_batch["observation"]["image_wrist"][0]       # (224, 224, 3) uint8
            proprio = np.asarray(rlds_batch["observation"]["proprio"][0], dtype=np.float32)  # (8,)
            action_chunk = np.asarray(rlds_batch["action"], dtype=np.float32)                # (8, 7)

            lang_raw = rlds_batch["task"]["language_instruction"]
            if isinstance(lang_raw, bytes):
                language = lang_raw.decode("utf-8").strip()
            else:
                language = str(lang_raw).strip()

            yield {
                "scene_img": scene,
                "wrist_img": wrist,
                "action_chunk": action_chunk,
                "proprio": proprio,
                "language": language,
                "dataset_id": DATASET_ID,
            }

    def __iter__(self) -> Iterator[dict]:
        # pretrain (single-task fine-tune) 前提: 無限ループ (train=True)。
        # make_interleaved_dataset(train=True) は RLDS 内で shuffle + repeat するため
        # while True で wrap しないでも OK だが、RLDS iterator が exhausted になった場合の
        # 安全網として while True を残す (taco_solo_loader と同一 pattern)。
        while True:
            for sample in self._iter_chunks():
                yield sample


# ===========================================================
# Batch transform + collate
# ===========================================================
def _build_input_ids(tokenizer, language: str, num_vision_tokens: int = NUM_VISION_TOKENS) -> torch.Tensor:
    """Layout (env var で複数 axis 切替可):
      default:                                [BOS] + V(num_vision_tokens) + prompt(20) + [PROPRIO] + A(64) + [EOS]
      VLA_OLD_PROMPT_FIRST=1:                  [BOS] + prompt(20) + V(num_vision_tokens) + [PROPRIO] + A(64) + [EOS]
      VLA_VISION_PLACEHOLDER_MODE=image_token: V を unique <unused> ID 列の代わりに IMAGE_TOKEN_ID(258880) × num_vision_tokens で構築
                                               (PLE が pretrain 由来の値になる仮説検証用、2026-04-26 ablation)
    """
    from prismatic.vla.constants_gemma4 import IMAGE_TOKEN_ID  # 局所 import (循環回避)

    text = f"What action should the robot take to {language.lower().strip()}?"
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) > PROMPT_MAX_LEN:
        ids = ids[:PROMPT_MAX_LEN]
    else:
        pad = [tokenizer.pad_token_id] * (PROMPT_MAX_LEN - len(ids))
        ids = ids + pad
    if os.environ.get("VLA_VISION_PLACEHOLDER_MODE", "unused") == "image_token":
        vision_block = [IMAGE_TOKEN_ID] * num_vision_tokens
    else:
        vision_block = list(range(VISION_PLACEHOLDER_BEGIN_IDX, VISION_PLACEHOLDER_BEGIN_IDX + num_vision_tokens))
    if os.environ.get("VLA_OLD_PROMPT_FIRST", "0") == "1":
        head = [tokenizer.bos_token_id] + ids + vision_block
    else:
        head = [tokenizer.bos_token_id] + vision_block + ids
    full = (
        head
        + [PROPRIO_PLACEHOLDER_IDX]
        + list(range(ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS))
        + [tokenizer.eos_token_id]
    )
    return torch.tensor(full, dtype=torch.long)


class LiberoCollator:
    """Callable collate: raw sample list → VLAAdapterGemma4-compatible batch dict.

    `DataLoader(collate_fn=LiberoCollator(tokenizer, num_vision_tokens=256))` の形で使う。

    Args:
        tokenizer: Gemma 4 tokenizer
        num_vision_tokens: VISION placeholder 数 (model.num_vision_tokens と揃える、
            soft_tokens=70→64 / 140→121 / 280→256)
    """

    def __init__(self, tokenizer, num_vision_tokens: int = NUM_VISION_TOKENS):
        self.tokenizer = tokenizer
        self.num_vision_tokens = num_vision_tokens

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        scene_tensors = []
        wrist_tensors = []
        input_ids_list = []
        proprio_list = []
        actions_list = []
        dataset_id_list = []
        languages = []

        for s in samples:
            scene_tensors.append(_scene_from_rlds(s["scene_img"]))
            wrist_tensors.append(_wrist_from_rlds(s["wrist_img"]))
            input_ids_list.append(_build_input_ids(self.tokenizer, s["language"], self.num_vision_tokens))
            proprio_list.append(torch.tensor(s["proprio"], dtype=torch.float32))
            actions_list.append(torch.tensor(s["action_chunk"], dtype=torch.float32))
            dataset_id_list.append(int(s["dataset_id"]))
            languages.append(s["language"])

        return {
            "pixel_values": {
                "scene": torch.stack(scene_tensors, dim=0),   # (B, 3, 224, 224) float32 [0, 255]
                "wrist": torch.stack(wrist_tensors, dim=0),   # (B, 3, 224, 224) float32 ImageNet-norm
            },
            "input_ids":  torch.stack(input_ids_list, dim=0),
            "proprio":    torch.stack(proprio_list, dim=0),
            "actions":    torch.stack(actions_list, dim=0),
            "dataset_id": torch.tensor(dataset_id_list, dtype=torch.long),
            "languages":  languages,
        }


def collate_libero(tokenizer, num_vision_tokens: int = NUM_VISION_TOKENS):
    """Factory: return a LiberoCollator bound to the given tokenizer.

    Args:
        tokenizer: Gemma 4 tokenizer
        num_vision_tokens: pass model.num_vision_tokens for soft_tokens alignment
    """
    return LiberoCollator(tokenizer, num_vision_tokens=num_vision_tokens)


# ===========================================================
# Standalone verify
# ===========================================================
def _run_verify(num_samples: int = 3, dataset_name: str = DEFAULT_DATASET_NAME):
    data_dir_abs = str(REPO_ROOT / DEFAULT_DATA_ROOT)
    print(f"=== Task 14 LIBERO --verify: pull {num_samples} raw samples from {dataset_name} ===")
    print(f"Data dir: {data_dir_abs}")
    ds = LiberoDataset(data_dir=data_dir_abs, dataset_name=dataset_name, shuffle_buffer_size=100)
    it = iter(ds._iter_chunks())
    for i in range(num_samples):
        s = next(it)
        print(f"\nSample {i}:")
        print(f"  scene_img:    shape={s['scene_img'].shape}, dtype={s['scene_img'].dtype}, "
              f"range=[{s['scene_img'].min()}, {s['scene_img'].max()}]")
        print(f"  wrist_img:    shape={s['wrist_img'].shape}, dtype={s['wrist_img'].dtype}, "
              f"range=[{s['wrist_img'].min()}, {s['wrist_img'].max()}]")
        print(f"  action_chunk: shape={s['action_chunk'].shape}, dtype={s['action_chunk'].dtype}, "
              f"range=[{s['action_chunk'].min():+.3f}, {s['action_chunk'].max():+.3f}]")
        print(f"  proprio:      shape={s['proprio'].shape}, dtype={s['proprio'].dtype}")
        print(f"  language:     '{s['language']}'")
        print(f"  dataset_id:   {s['dataset_id']}")
    print("\nOK: --verify passed")


def _run_smoke_batch(batch_size: int = 2, dataset_name: str = DEFAULT_DATASET_NAME):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    print(f"=== Task 14 LIBERO --smoke-batch: build 1 batch (B={batch_size}) via DataLoader ===")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_dir_abs = str(REPO_ROOT / DEFAULT_DATA_ROOT)
    ds = LiberoDataset(data_dir=data_dir_abs, dataset_name=dataset_name, shuffle_buffer_size=100)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_libero(tokenizer, num_vision_tokens=NUM_VISION_TOKENS),
        num_workers=0,   # make_interleaved_dataset が内部 parallelism, workers=0 安全
    )
    batch = next(iter(loader))

    print("\nBatch fields:")
    for k, v in batch.items():
        if k == "pixel_values":
            print(f"  pixel_values.scene: {tuple(v['scene'].shape)} {v['scene'].dtype}  "
                  f"range=[{v['scene'].min():.1f}, {v['scene'].max():.1f}]")
            print(f"  pixel_values.wrist: {tuple(v['wrist'].shape)} {v['wrist'].dtype}  "
                  f"range=[{v['wrist'].min():+.3f}, {v['wrist'].max():+.3f}]")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} (len={len(v)}) e.g. {v[:min(2, len(v))]}")

    # Signature compatibility check
    expected_keys = {"pixel_values", "input_ids", "proprio", "actions", "dataset_id"}
    assert expected_keys.issubset(batch.keys()), \
        f"missing keys for VLAAdapterGemma4.forward: {expected_keys - set(batch.keys())}"
    assert isinstance(batch["pixel_values"], dict) and set(batch["pixel_values"].keys()) == {"scene", "wrist"}, \
        f"pixel_values must be dict with keys {{'scene','wrist'}}, got {list(batch['pixel_values'].keys())}"
    B = batch_size
    assert batch["pixel_values"]["scene"].shape == (B, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["pixel_values"]["wrist"].shape == (B, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["proprio"].shape == (B, PROPRIO_DIM)
    assert batch["actions"].shape == (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
    assert batch["dataset_id"].shape == (B,)
    assert batch["dataset_id"].dtype == torch.long
    expected_L = 1 + PROMPT_MAX_LEN + NUM_VISION_TOKENS + 1 + NUM_ACTION_TOKENS + 1
    assert batch["input_ids"].shape == (B, expected_L), \
        f"input_ids length {batch['input_ids'].shape[1]} != expected {expected_L}"
    print(f"\nOK: --smoke-batch passed (input_ids length={expected_L}, matches Task 8 layout)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Pull 3 raw samples, print shapes/dtype")
    parser.add_argument("--smoke-batch", action="store_true", help="Build 1 full batch via DataLoader")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    if args.verify:
        _run_verify(dataset_name=args.dataset_name)
    elif args.smoke_batch:
        _run_smoke_batch(batch_size=args.batch_size, dataset_name=args.dataset_name)
    else:
        print("Use --verify or --smoke-batch")


if __name__ == "__main__":
    main()
