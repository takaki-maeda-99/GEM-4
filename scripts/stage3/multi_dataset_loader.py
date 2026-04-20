"""
Phase 3a-3: Multi-dataset pretrain loader (Taco Play + Fractal).

設計方針 (Plan §3a-3):
  - Weighted cross-dataset sampling: Taco=0.30, Fractal=0.70 (User plan §7、Bridge skip で再分配)
  - dataset_id 付与 (0=taco_play, 1=fractal20220817_data): VLAAdapterGemma4 forward で SoftPromptLibrary 入力
  - Canonical 7-dim action (Phase 3a-2 と整合): xyz delta(3) + rot delta(3) + gripper(1)
  - BOUNDS_Q99 normalize (mask=True dim のみ、gripper は raw pass-through)
  - 2-camera 入力: Taco Play (rgb_static + rgb_gripper)、Fractal (image を 2 camera slot に複製)
  - Proprio 不使用 (X-VLA 原実装準拠、zero tensor 8-dim)
  - Gemma 4 tokenize pattern (Gemma4BatchTransform と同じ placeholder 構成)

VLAAdapterGemma4 forward 呼び出し互換 batch dict:
  {
    "pixel_values": {"dino": (B, 2, 3, 224, 224), "siglip": (B, 2, 3, 224, 224)},
    "input_ids":    (B, L) long,
    "proprio":      (B, 8) bf16 — zeros (pretrain)
    "actions":      (B, 8, 7) bf16 — chunk 8 (X-VLA 原実装準拠)、normalized
    "dataset_id":   (B,) long — 0 or 1
    "languages":    [str] × B
  }

Run (standalone 1 batch 動作確認、Phase 3a-4):
  CUDA_VISIBLE_DEVICES=6 .venv-gemma4/bin/python scripts/stage3/multi_dataset_loader.py --verify
"""
import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import tensorflow_datasets as tfds
import torch
from PIL import Image
from torch.utils.data import IterableDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "stage3_openx"
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
SCRIPTS_GEMMA4 = REPO_ROOT / "scripts" / "gemma4"
sys.path.insert(0, str(VLA_ROOT))
sys.path.insert(0, str(SCRIPTS_GEMMA4))

# Gemma 4 placeholder IDs and helpers (Stage 1 確定済)
from prismatic.vla.constants_gemma4 import (  # noqa: E402
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTION_TOKENS,
    NUM_VISION_TOKENS,
    PROPRIO_PLACEHOLDER_IDX,
    VISION_PLACEHOLDER_BEGIN_IDX,
)

# ===========================================================
# Constants
# ===========================================================
# dataset_id mapping (SoftPromptLibrary dataset 順)
DATASET_ID_MAP = {
    "taco_play": 0,
    "fractal20220817_data": 1,
}
NUM_DATASETS = len(DATASET_ID_MAP)

# 採用 sampling weight (User plan §7、Bridge skip で再正規化)
# 元 plan: taco=0.20, bridge=0.20, fractal=0.60 → 正規化 taco=0.25, fractal=0.75
# User 指示文: "Taco=0.30 (or Plan の 0.20)、Fractal=0.70" → Taco 重視で 0.30 採用
DEFAULT_WEIGHTS = {"taco_play": 0.30, "fractal20220817_data": 0.70}

# Gemma 4 E2B 固定
PROMPT_MAX_LEN = 20
PROPRIO_DIM = 8
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8   # X-VLA 原実装と Stage 2 LIBERO 共通

IMAGE_SIZE = 224


# ===========================================================
# Canonical action extractors (Phase 3a-2 と完全一致)
# ===========================================================
def extract_canonical_action_taco_play(step: Dict[str, Any]) -> np.ndarray:
    """Taco Play: action.rel_actions_world (7 dim 直接、eager tensor)."""
    return step["action"]["rel_actions_world"].numpy().astype(np.float32)


def extract_canonical_action_fractal(step: Dict[str, Any]) -> np.ndarray:
    """Fractal: world_vector(3) + rotation_delta(3) + gripper_closedness_action(1)."""
    wv = step["action"]["world_vector"].numpy().astype(np.float32)
    rd = step["action"]["rotation_delta"].numpy().astype(np.float32)
    gc = step["action"]["gripper_closedness_action"].numpy().astype(np.float32)
    return np.concatenate([wv, rd, gc], axis=0)


# ===========================================================
# Image extractors (2-camera slot 対応)
# ===========================================================
def extract_images_taco_play(step: Dict[str, Any]) -> tuple:
    """Taco Play: rgb_static (primary、150x200) + rgb_gripper (wrist、84x84)."""
    primary = step["observation"]["rgb_static"].numpy()    # (150, 200, 3) uint8
    wrist = step["observation"]["rgb_gripper"].numpy()      # (84, 84, 3) uint8
    return primary, wrist


def extract_images_fractal(step: Dict[str, Any]) -> tuple:
    """Fractal: image (256x320) のみ、wrist 相当なし → 同一 image を 2 slot に duplicate."""
    img = step["observation"]["image"].numpy()              # (256, 320, 3) uint8
    return img, img


# ===========================================================
# Language extractor
# ===========================================================
def extract_language(step: Dict[str, Any]) -> str:
    """natural_language_instruction string (Taco Play / Fractal 共通 key)."""
    s = step["observation"].get("natural_language_instruction")
    if s is None:
        return ""
    b = s.numpy()
    return b.decode("utf-8") if isinstance(b, bytes) else str(b)


# ===========================================================
# Normalize (BOUNDS_Q99、Stage 2 と対称形)
# ===========================================================
def normalize_action_bounds_q99(action: np.ndarray, stats: dict) -> np.ndarray:
    """Stage 2 denormalize の inverse。mask=True dim のみ [-1,1] に clip。"""
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats["mask"], dtype=bool)
    denom = (q99 - q01) + 1e-8
    norm = np.clip(2.0 * (action - q01) / denom - 1.0, -1.0, 1.0)
    out = action.astype(np.float32).copy()
    out[mask] = norm[mask]   # mask=False (gripper) は raw
    return out


# ===========================================================
# Dataset wrappers (per-dataset iter + canonical 変換)
# ===========================================================
class PerDatasetIterable:
    """1 dataset の episode/step を iter、canonical batch dict を yield."""
    def __init__(self, dataset_name: str, stats: dict, action_extractor, image_extractor,
                 num_actions_chunk: int = NUM_ACTIONS_CHUNK):
        self.dataset_name = dataset_name
        self.dataset_id = DATASET_ID_MAP[dataset_name]
        self.stats = stats
        self.action_extractor = action_extractor
        self.image_extractor = image_extractor
        self.num_actions_chunk = num_actions_chunk
        self._tfds = tfds.load(dataset_name, data_dir=str(DATA_ROOT), split="train")

    def iter_chunks(self) -> Iterator[dict]:
        """Yield per-timestep chunk dict: {primary_img, wrist_img, action_chunk 8x7 normalized, language, dataset_id}."""
        for ep in self._tfds:
            # buffer の episode 内全 step、NUM_ACTIONS_CHUNK 先まで取れる step のみ yield
            steps_buf = list(ep["steps"])
            for i in range(len(steps_buf)):
                # chunk 8 分の action を取る、残り不足は複製 (X-VLA 原実装 behavior 準拠)
                chunk = []
                for k in range(self.num_actions_chunk):
                    idx = min(i + k, len(steps_buf) - 1)
                    a = self.action_extractor(steps_buf[idx])
                    a_norm = normalize_action_bounds_q99(a, self.stats)
                    chunk.append(a_norm)
                action_chunk = np.stack(chunk, axis=0)   # (8, 7)

                primary, wrist = self.image_extractor(steps_buf[i])
                language = extract_language(steps_buf[i])

                yield {
                    "primary_img": primary,    # (H, W, 3) uint8
                    "wrist_img": wrist,
                    "action_chunk": action_chunk,
                    "language": language,
                    "dataset_id": self.dataset_id,
                }


# ===========================================================
# Multi-dataset pretrain loader
# ===========================================================
class MultiDatasetPretrainDataset(IterableDataset):
    """Weighted sampling across multiple OXE datasets + Gemma 4 tokenize + image_transform.

    Gemma4BatchTransform に似た構造だが、pretrain specific:
      - canonical 7-dim action with per-dataset stats normalize
      - dataset_id 付与
      - proprio zero-fill (X-VLA pretrain 準拠)
      - 2-camera 対応 (dataset 別 image 抽出、VisionBackbone image_transform 適用)
    """

    def __init__(
        self,
        tokenizer: Any,
        image_transform: Any,
        weights: Dict[str, float] = None,
        num_actions_chunk: int = NUM_ACTIONS_CHUNK,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.num_actions_chunk = num_actions_chunk
        self.seed = seed

        # Load per-dataset statistics (Phase 3a-2 output)
        self.stats = {}
        for ds_name in DATASET_ID_MAP:
            p = DATA_ROOT / ds_name / "dataset_statistics.json"
            if not p.exists():
                raise FileNotFoundError(
                    f"dataset_statistics.json not found for {ds_name} at {p}. "
                    f"Run scripts/stage3/compute_dataset_statistics.py first (Phase 3a-2)."
                )
            data = json.loads(p.read_text())
            self.stats[ds_name] = data[ds_name]["action"]

        # Per-dataset iterable
        self.datasets = {
            "taco_play": PerDatasetIterable(
                "taco_play", self.stats["taco_play"],
                extract_canonical_action_taco_play, extract_images_taco_play,
                num_actions_chunk=num_actions_chunk,
            ),
            "fractal20220817_data": PerDatasetIterable(
                "fractal20220817_data", self.stats["fractal20220817_data"],
                extract_canonical_action_fractal, extract_images_fractal,
                num_actions_chunk=num_actions_chunk,
            ),
        }

        # Sampling weights (dataset_name → weight)
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS.copy()
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, f"weights sum must = 1.0、got {self.weights}"
        self.ds_names = list(self.weights.keys())
        self.ds_weights = [self.weights[n] for n in self.ds_names]
        self.rng = random.Random(seed)

    def _build_input_ids(self, language: str) -> torch.Tensor:
        """Gemma4BatchTransform と同じ pattern で Gemma 4 tokenize + placeholder 埋め込み."""
        text = f"What action should the robot take to {language.lower().strip()}?"
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) > PROMPT_MAX_LEN:
            ids = ids[:PROMPT_MAX_LEN]
        else:
            pad = [self.tokenizer.pad_token_id] * (PROMPT_MAX_LEN - len(ids))
            ids = ids + pad
        full = (
            [self.tokenizer.bos_token_id]
            + ids
            + list(range(VISION_PLACEHOLDER_BEGIN_IDX, VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS))
            + [PROPRIO_PLACEHOLDER_IDX]
            + list(range(ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS))
            + [self.tokenizer.eos_token_id]
        )
        return torch.tensor(full, dtype=torch.long)

    def _transform_images(self, primary: np.ndarray, wrist: np.ndarray) -> Dict[str, torch.Tensor]:
        """image_transform に通して dino/siglip 2 tensor を stacked で返す."""
        pv_p = self.image_transform(Image.fromarray(primary))
        pv_w = self.image_transform(Image.fromarray(wrist))
        return {
            "dino":   torch.stack([pv_p["dino"], pv_w["dino"]], dim=0),     # (2, 3, 224, 224)
            "siglip": torch.stack([pv_p["siglip"], pv_w["siglip"]], dim=0),
        }

    def __iter__(self) -> Iterator[dict]:
        # 各 dataset の iter を keep、weighted sampling で次 dataset を選択、該 iter から 1 chunk 取る
        ds_iters = {n: iter(self.datasets[n].iter_chunks()) for n in self.ds_names}
        while True:
            chosen = self.rng.choices(self.ds_names, weights=self.ds_weights, k=1)[0]
            try:
                ck = next(ds_iters[chosen])
            except StopIteration:
                # Refill iter when exhausted (1 epoch 達成、pretrain は無限 iter 前提)
                ds_iters[chosen] = iter(self.datasets[chosen].iter_chunks())
                ck = next(ds_iters[chosen])

            # Gemma 4 tokenize
            input_ids = self._build_input_ids(ck["language"])

            # 2-camera image transform
            pv = self._transform_images(ck["primary_img"], ck["wrist_img"])

            yield {
                "pixel_values": pv,
                "input_ids": input_ids,
                "proprio": torch.zeros(PROPRIO_DIM, dtype=torch.float32),  # pretrain 不使用
                "actions": torch.tensor(ck["action_chunk"], dtype=torch.float32),   # (8, 7)
                "dataset_id": torch.tensor(ck["dataset_id"], dtype=torch.long),     # scalar
                "language": ck["language"],
            }


def collate_pretrain(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DataLoader collate: single samples → batched tensors."""
    return {
        "pixel_values": {
            "dino":   torch.stack([s["pixel_values"]["dino"] for s in samples], dim=0),
            "siglip": torch.stack([s["pixel_values"]["siglip"] for s in samples], dim=0),
        },
        "input_ids":  torch.stack([s["input_ids"] for s in samples], dim=0),
        "proprio":    torch.stack([s["proprio"] for s in samples], dim=0),
        "actions":    torch.stack([s["actions"] for s in samples], dim=0),
        "dataset_id": torch.stack([s["dataset_id"] for s in samples], dim=0),
        "languages":  [s["language"] for s in samples],
    }


# ===========================================================
# Standalone verify (Phase 3a-4): 1 batch の shape / dataset_id / range 確認
# ===========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="1 batch 動作確認 (GPU 必要)")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if not args.verify:
        print("Use --verify to run 1-batch sanity check on GPU.")
        return

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")   # CUDA_VISIBLE_DEVICES で制御想定

    # --- Build vision backbone + tokenizer (Gemma4BatchTransform と同じ pattern) ---
    from transformers import AutoTokenizer
    from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone

    print("Loading DINO+SigLIP + Gemma 4 tokenizer...")
    vision_backbone = DinoSigLIPViTBackbone(
        vision_backbone_id="dinosiglip-vit-so-224px",
        image_resize_strategy="resize-naive",
        default_image_size=224,
        image_sequence_len=2,
    ).to(device, dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Instantiate loader ---
    dataset = MultiDatasetPretrainDataset(
        tokenizer=tokenizer,
        image_transform=vision_backbone.image_transform,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_pretrain, num_workers=0)

    # --- Pull 1 batch ---
    print(f"\n=== Phase 3a-4 1-batch verify (B={args.batch_size}) ===")
    batch = next(iter(loader))
    print("Batch shapes + dtypes:")
    for k, v in batch.items():
        if k == "pixel_values":
            print(f"  pixel_values.dino:   {tuple(v['dino'].shape)} {v['dino'].dtype}")
            print(f"  pixel_values.siglip: {tuple(v['siglip'].shape)} {v['siglip'].dtype}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} (len={len(v)}) e.g. {v[:2]}")
    print("\ndataset_id distribution:", batch["dataset_id"].tolist())
    print("action range per dim (1st sample, after normalize):")
    dim_names = ["Δx", "Δy", "Δz", "Δrx", "Δry", "Δrz", "grip"]
    first_action = batch["actions"][0].numpy()   # (8, 7)
    for d in range(7):
        print(f"  dim {d} ({dim_names[d]}): [{first_action[:, d].min():+.3f}, {first_action[:, d].max():+.3f}]")


if __name__ == "__main__":
    main()
