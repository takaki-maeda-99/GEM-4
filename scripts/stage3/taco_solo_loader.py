"""
Task 14: Taco Play 単独 pretrain loader (dual-track redesign plan rev 3, §6.1).

`multi_dataset_loader.py` から派生 (Fractal + image duplication hack を削除):
  - Taco Play のみ (2-cam: rgb_static → scene, rgb_gripper → wrist)
  - Proprio 有効化 (robot_obs [0:7] + [14] で 8-dim: EEF pos(3) + EEF rot(3) + gripper_width(1) + gripper_action(1))
  - pixel_values は dict: {"scene": (B, 3, 224, 224), "wrist": (B, 3, 224, 224)}
    * scene は raw float [0, 255] で返す (VLAAdapterGemma4.encode_scene 内部で Gemma4ImageProcessor が resize+patchify+rescale を行う)
    * wrist は ImageNet 正規化済 float tensor (WristResNet18 用)
  - Canonical 7-dim action (Phase 3a-2 と整合)、BOUNDS_Q99 normalize
  - action chunking 8-step sliding window (残り不足は複製、X-VLA 原実装 behavior 準拠)
  - input_ids layout (Task 6/8 以降):
      [BOS] + prompt(20) + VISION_PLACEHOLDERS(NUM_VISION_TOKENS=256) + [PROPRIO] + ACTION_TOKENS(64) + [EOS]
    * wrist placeholder は input_ids に含めない (WristResNet18 は action_head の concat-to-x で入る、LLM 経由しない)
  - dataset_id は 0 固定 (taco_play 1 種)

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
  # 3 raw samples、shape/dtype 確認
  CUDA_VISIBLE_DEVICES=4 .venv-gemma4/bin/python scripts/stage3/taco_solo_loader.py --verify
  # B=2 の full batch 構築 (DataLoader 経由)、VLAAdapterGemma4 signature と照合
  CUDA_VISIBLE_DEVICES=4 .venv-gemma4/bin/python scripts/stage3/taco_solo_loader.py --smoke-batch
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "stage3_openx"
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
SCRIPTS_GEMMA4 = REPO_ROOT / "scripts" / "gemma4"
sys.path.insert(0, str(VLA_ROOT))
sys.path.insert(0, str(SCRIPTS_GEMMA4))

from prismatic.vla.constants_gemma4 import (  # noqa: E402
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTION_TOKENS,
    NUM_VISION_TOKENS,
    PROPRIO_PLACEHOLDER_IDX,
    VISION_PLACEHOLDER_BEGIN_IDX,
)

# Rev 3 Task 6/14 coupling: NUM_VISION_TOKENS は max_soft_tokens=280 前提の暫定値 (256)。
# YAML で max_soft_tokens を変更する場合は _build_input_ids の layout も追随が必要。
# 現状 constants_gemma4.py の値がズレていれば早期 fail させる:
assert NUM_VISION_TOKENS == 256, (
    f"taco_solo_loader は NUM_VISION_TOKENS=256 (max_soft_tokens=280) 前提。"
    f"現在 {NUM_VISION_TOKENS}。max_soft_tokens を変更するなら、VLAAdapterGemma4 の "
    f"model.num_vision_tokens を loader に注入する caller-injection 設計に移行が必要。"
)

# ===========================================================
# Constants
# ===========================================================
DATASET_NAME = "taco_play"
DATASET_ID = 0

PROMPT_MAX_LEN = 20
PROPRIO_DIM = 8
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8

IMAGE_SIZE = 224

# ImageNet 正規化 (WristResNet18 は torchvision ResNet18 ImageNet pretrained が backbone)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ===========================================================
# Canonical action extractor (Phase 3a-2 と完全一致)
# ===========================================================
def extract_canonical_action_taco_play(step: Dict[str, Any]) -> np.ndarray:
    """Taco Play: action.rel_actions_world (7 dim 直接、eager tensor)."""
    return step["action"]["rel_actions_world"].numpy().astype(np.float32)


# ===========================================================
# Image / proprio / language extractors
# ===========================================================
def extract_images_taco_play(step: Dict[str, Any]):
    """Taco Play: rgb_static (primary → scene、150x200) + rgb_gripper (wrist、84x84)."""
    scene = step["observation"]["rgb_static"].numpy()    # (150, 200, 3) uint8
    wrist = step["observation"]["rgb_gripper"].numpy()   # (84, 84, 3) uint8
    return scene, wrist


def extract_proprio_taco_play(step: Dict[str, Any]) -> np.ndarray:
    """Taco Play robot_obs 15-dim → 8-dim proprio.

    robot_obs layout (Taco Play の仕様):
      [0:3]  tcp_pos (x, y, z)
      [3:6]  tcp_rot (euler or rotvec)
      [6]    gripper_width
      [7:14] arm_joint_states
      [14]   gripper_action (0/1)

    8-dim proprio = [tcp_pos(3), tcp_rot(3), gripper_width(1), gripper_action(1)]
    joint_states は除外 (EEF-space 制御前提、PROPRIO_DIM=8 に合わせる)。
    """
    ro = step["observation"]["robot_obs"].numpy().astype(np.float32)
    return np.concatenate([ro[0:7], ro[14:15]], axis=0)   # (8,)


def extract_language(step: Dict[str, Any]) -> str:
    """natural_language_instruction string."""
    s = step["observation"].get("natural_language_instruction")
    if s is None:
        return ""
    b = s.numpy()
    return b.decode("utf-8") if isinstance(b, bytes) else str(b)


# ===========================================================
# Normalize (BOUNDS_Q99、Stage 2 と対称形)
# ===========================================================
def normalize_action_bounds_q99(action: np.ndarray, stats: dict) -> np.ndarray:
    """mask=True dim のみ [-1, 1] に clip、mask=False (gripper) は raw pass-through."""
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats["mask"], dtype=bool)
    denom = (q99 - q01) + 1e-8
    norm = np.clip(2.0 * (action - q01) / denom - 1.0, -1.0, 1.0)
    out = action.astype(np.float32).copy()
    out[mask] = norm[mask]
    return out


# ===========================================================
# Image preprocessing helpers
# ===========================================================
def _resize_nearest(img_uint8: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Simple bilinear resize via tf (keep on CPU)、(H, W, 3) uint8 → (size, size, 3) uint8."""
    t = tf.convert_to_tensor(img_uint8)
    t = tf.image.resize(t, (size, size), method=tf.image.ResizeMethod.BILINEAR, antialias=True)
    t = tf.cast(tf.clip_by_value(t, 0.0, 255.0), tf.uint8)
    return t.numpy()


def _scene_to_tensor(scene_uint8: np.ndarray) -> torch.Tensor:
    """Scene image raw (H,W,3) uint8 → (3, 224, 224) float32 in [0, 255] range.

    VLAAdapterGemma4.encode_scene は Gemma4ImageProcessor で内部 resize + rescale (÷255) + patchify を行う。
    よって loader 側は 224x224 resize のみ行い、normalize は不要 (raw [0, 255] float で渡す)。
    """
    resized = _resize_nearest(scene_uint8, IMAGE_SIZE)             # (224, 224, 3) uint8
    t = torch.from_numpy(resized).float().permute(2, 0, 1).contiguous()  # (3, 224, 224), [0, 255]
    return t


def _wrist_to_tensor(wrist_uint8: np.ndarray) -> torch.Tensor:
    """Wrist image (H,W,3) uint8 → (3, 224, 224) float32、ImageNet 正規化済.

    WristResNet18 は torchvision ResNet18 (ImageNet pretrained) が backbone のため、
    ImageNet mean/std で正規化する。
    """
    resized = _resize_nearest(wrist_uint8, IMAGE_SIZE).astype(np.float32) / 255.0  # (224, 224, 3) float [0,1]
    normalized = (resized - IMAGENET_MEAN) / IMAGENET_STD                          # broadcasts over last axis
    t = torch.from_numpy(normalized).float().permute(2, 0, 1).contiguous()         # (3, 224, 224)
    return t


# ===========================================================
# Taco Solo dataset (iterable、per-timestep samples)
# ===========================================================
class TacoSoloDataset(IterableDataset):
    """Taco Play 単独 iterable dataset、per-timestep sample を yield.

    Per-timestep sample dict (pre-transform):
      {
        "scene_img":   (224, 224, 3) uint8,   # resize 前 raw uint8 (collate 用には transform 内で変換)
        "wrist_img":   (224, 224, 3) uint8,
        "action_chunk": (8, 7) float32 normalized,
        "proprio":     (8,) float32 raw (正規化なし、VLAAdapterGemma4.proprio_projector が処理),
        "language":    str,
        "dataset_id":  0 int,
      }

    `num_workers > 0` 対応 (Phase 3b-5 と同設計):
      tfds.load は __iter__ 内 lazy load、worker process 内で初期化。
    """

    def __init__(
        self,
        data_dir: str | Path = DATA_ROOT,
        num_actions_chunk: int = NUM_ACTIONS_CHUNK,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.num_actions_chunk = num_actions_chunk
        self.seed = seed

        # Load Taco Play statistics (Phase 3a-2 output)
        stats_path = self.data_dir / DATASET_NAME / "dataset_statistics.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"dataset_statistics.json not found at {stats_path}. "
                f"Run scripts/stage3/compute_dataset_statistics.py first."
            )
        self.stats = json.loads(stats_path.read_text())[DATASET_NAME]["action"]

    def _iter_chunks(self) -> Iterator[dict]:
        """Yield per-timestep raw sample dict (image uint8, action normalized, proprio raw, language)."""
        # Lazy load: worker process 内で初回 call 時に TF dataset を作る (fork 後に安全)
        ds_tf = tfds.load(DATASET_NAME, data_dir=str(self.data_dir), split="train")
        for ep in ds_tf:
            steps_buf = list(ep["steps"])
            for i in range(len(steps_buf)):
                # chunk 8 分の action、残り不足は複製 (X-VLA 原実装 behavior)
                chunk = []
                for k in range(self.num_actions_chunk):
                    idx = min(i + k, len(steps_buf) - 1)
                    a = extract_canonical_action_taco_play(steps_buf[idx])
                    a_norm = normalize_action_bounds_q99(a, self.stats)
                    chunk.append(a_norm)
                action_chunk = np.stack(chunk, axis=0)   # (8, 7)

                scene, wrist = extract_images_taco_play(steps_buf[i])
                proprio = extract_proprio_taco_play(steps_buf[i])
                language = extract_language(steps_buf[i])

                yield {
                    "scene_img": scene,
                    "wrist_img": wrist,
                    "action_chunk": action_chunk,
                    "proprio": proprio,
                    "language": language,
                    "dataset_id": DATASET_ID,
                }

    def __iter__(self) -> Iterator[dict]:
        from torch.utils.data import get_worker_info
        worker_info = get_worker_info()
        if worker_info is not None:
            effective_seed = self.seed + worker_info.id * 1000 + 1
        else:
            effective_seed = self.seed
        rng = random.Random(effective_seed)   # noqa: F841 - reserved for future shuffle

        # Single-dataset なので weighted sampling なし、iter 全周で無限ループ (pretrain 前提)
        while True:
            for sample in self._iter_chunks():
                yield sample


# ===========================================================
# Batch transform + collate (tokenize + placeholder、image transform を collate 内で実施)
# ===========================================================
def _build_input_ids(tokenizer, language: str) -> torch.Tensor:
    """Gemma4BatchTransform と同じ pattern で Gemma 4 tokenize + placeholder 埋め込み.

    Layout:
      [BOS] + prompt(PROMPT_MAX_LEN=20) + VISION_PLACEHOLDERS(256) + [PROPRIO] + ACTION_TOKENS(64) + [EOS]

    wrist placeholder は含めない (Task 8 以降、WristResNet18 は action_head の concat-to-x で
    入り、LLM 経由しないため input_ids にも含めない)。
    """
    text = f"What action should the robot take to {language.lower().strip()}?"
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) > PROMPT_MAX_LEN:
        ids = ids[:PROMPT_MAX_LEN]
    else:
        pad = [tokenizer.pad_token_id] * (PROMPT_MAX_LEN - len(ids))
        ids = ids + pad
    full = (
        [tokenizer.bos_token_id]
        + ids
        + list(range(VISION_PLACEHOLDER_BEGIN_IDX, VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS))
        + [PROPRIO_PLACEHOLDER_IDX]
        + list(range(ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS))
        + [tokenizer.eos_token_id]
    )
    return torch.tensor(full, dtype=torch.long)


class TacoSoloCollator:
    """Callable collate: raw sample list → VLAAdapterGemma4-compatible batch dict.

    `DataLoader(collate_fn=TacoSoloCollator(tokenizer))` の形で使う。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        scene_tensors = []
        wrist_tensors = []
        input_ids_list = []
        proprio_list = []
        actions_list = []
        dataset_id_list = []
        languages = []

        for s in samples:
            scene_tensors.append(_scene_to_tensor(s["scene_img"]))
            wrist_tensors.append(_wrist_to_tensor(s["wrist_img"]))
            input_ids_list.append(_build_input_ids(self.tokenizer, s["language"]))
            proprio_list.append(torch.tensor(s["proprio"], dtype=torch.float32))
            actions_list.append(torch.tensor(s["action_chunk"], dtype=torch.float32))
            dataset_id_list.append(int(s["dataset_id"]))
            languages.append(s["language"])

        return {
            "pixel_values": {
                "scene": torch.stack(scene_tensors, dim=0),   # (B, 3, 224, 224) float32
                "wrist": torch.stack(wrist_tensors, dim=0),   # (B, 3, 224, 224) float32
            },
            "input_ids":  torch.stack(input_ids_list, dim=0),                    # (B, L)
            "proprio":    torch.stack(proprio_list, dim=0),                      # (B, 8)
            "actions":    torch.stack(actions_list, dim=0),                      # (B, 8, 7)
            "dataset_id": torch.tensor(dataset_id_list, dtype=torch.long),       # (B,)
            "languages":  languages,
        }


def collate_taco_solo(tokenizer):
    """Factory: return a collator callable bound to the given tokenizer."""
    return TacoSoloCollator(tokenizer)


# ===========================================================
# Standalone verify
# ===========================================================
def _run_verify(num_samples: int = 3):
    print(f"=== Task 14 --verify: pull {num_samples} raw samples ===")
    ds = TacoSoloDataset()
    it = iter(ds._iter_chunks())
    for i in range(num_samples):
        s = next(it)
        print(f"\nSample {i}:")
        print(f"  scene_img: shape={s['scene_img'].shape}, dtype={s['scene_img'].dtype}")
        print(f"  wrist_img: shape={s['wrist_img'].shape}, dtype={s['wrist_img'].dtype}")
        print(f"  action_chunk: shape={s['action_chunk'].shape}, dtype={s['action_chunk'].dtype}, "
              f"range=[{s['action_chunk'].min():+.3f}, {s['action_chunk'].max():+.3f}]")
        print(f"  proprio: shape={s['proprio'].shape}, dtype={s['proprio'].dtype}, values={s['proprio']}")
        print(f"  language: '{s['language']}'")
        print(f"  dataset_id: {s['dataset_id']}")
    print("\nOK: --verify passed")


def _run_smoke_batch(batch_size: int = 2):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    print(f"=== Task 14 --smoke-batch: build 1 batch (B={batch_size}) via DataLoader ===")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = TacoSoloDataset()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_taco_solo(tokenizer),
        num_workers=0,
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

    # Confirm VLAAdapterGemma4.forward signature compatibility
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
    # input_ids length: 1 + PROMPT_MAX_LEN + NUM_VISION_TOKENS + 1 + NUM_ACTION_TOKENS + 1
    expected_L = 1 + PROMPT_MAX_LEN + NUM_VISION_TOKENS + 1 + NUM_ACTION_TOKENS + 1
    assert batch["input_ids"].shape == (B, expected_L), \
        f"input_ids length {batch['input_ids'].shape[1]} != expected {expected_L}"
    print(f"\nOK: --smoke-batch passed (input_ids length={expected_L}, matches Task 8 layout)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Pull 3 raw samples, print shapes")
    parser.add_argument("--smoke-batch", action="store_true", help="Build 1 full batch via DataLoader")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    if args.verify:
        _run_verify()
    elif args.smoke_batch:
        _run_smoke_batch(batch_size=args.batch_size)
    else:
        print("Use --verify or --smoke-batch")


if __name__ == "__main__":
    main()
