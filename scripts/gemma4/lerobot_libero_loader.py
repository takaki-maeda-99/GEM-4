"""
LeRobot 形式 LIBERO loader. 既存 `libero_loader.py` (RLDS) と同じ per-timestep dict schema を
yield するため、`LiberoCollator` と DataLoader は共通で使える (drop-in 代替).

仕様:
  - HF Hub の LeRobot 形式 LIBERO datasets を対象:
      lerobot/libero_spatial_image / libero_goal_image / libero_object_image / libero_10_image
  - LeRobotDataset の delta_timestamps で action を 8-step chunk 化 (fps=10 想定)。
  - 画像 (256x256) は bilinear で 224x224 へ resize、uint8 HWC numpy で yield
    (LiberoCollator 側 _scene_from_rlds / _wrist_from_rlds で ImageNet 正規化される)。
  - action/proprio は LeRobot raw (unnormalized)、本 loader 内で BOUNDS_Q99 正規化を適用
    (既存 dataset_statistics.json を reuse)。

Per-timestep raw sample (collator に渡る形、RLDS loader と identical):
  {
    "scene_img":    (H=224, W=224, 3) uint8,
    "wrist_img":    (H=224, W=224, 3) uint8,
    "action_chunk": (NUM_ACTIONS_CHUNK=8, ACTION_DIM=7) float32 normalized (BOUNDS_Q99),
    "proprio":      (PROPRIO_DIM=8,) float32 raw (collator は raw のまま渡す想定),
    "language":     str,
    "dataset_id":   0,
  }

Run:
  # HF から episode 0 だけ pull して smoke 確認:
  .venv-gemma4/bin/python scripts/gemma4/lerobot_libero_loader.py --verify --repo_id lerobot/libero_spatial_image
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
sys.path.insert(0, str(VLA_ROOT))

DATASET_ID = 0
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8
IMAGE_SIZE = 224
DEFAULT_FPS = 10


# ------------------------------------------------------------------
# BOUNDS_Q99 action normalization (train と整合、eval denormalize の inverse)
# ------------------------------------------------------------------
def _load_action_stats(dataset_statistics_path: str, unnorm_key: str) -> dict:
    all_stats = json.loads(Path(dataset_statistics_path).read_text())
    assert unnorm_key in all_stats, \
        f"unnorm_key '{unnorm_key}' not in {dataset_statistics_path} (keys: {list(all_stats.keys())})"
    action = all_stats[unnorm_key]["action"]
    return {
        "q01": np.array(action["q01"], dtype=np.float32),
        "q99": np.array(action["q99"], dtype=np.float32),
        "mask": np.array(action.get("mask", [True] * len(action["q01"])), dtype=bool),
    }


def _normalize_action_bounds_q99(action_raw: np.ndarray, stats: dict) -> np.ndarray:
    """BOUNDS_Q99 forward (eval の denormalize の inverse)、mask=True dim のみ normalize。

    Args:
        action_raw: (..., 7) float32
        stats: {"q01", "q99", "mask"} all (7,)
    Returns:
        (..., 7) float32 with mask=True dims ∈ [-1, 1] (clip)、mask=False dims unchanged
    """
    q01 = stats["q01"]
    q99 = stats["q99"]
    mask = stats["mask"]
    denom = (q99 - q01) + 1e-8
    norm = np.clip(2.0 * (action_raw - q01) / denom - 1.0, -1.0, 1.0)
    out = action_raw.astype(np.float32).copy()
    # 最終次元 (action dim) 単位で mask 適用、leading batch/chunk dim を保ったまま
    for i in range(len(mask)):
        if mask[i]:
            out[..., i] = norm[..., i]
    return out


# ------------------------------------------------------------------
# Image conversion helpers
# ------------------------------------------------------------------
def _lerobot_img_to_uint8_hwc(img: torch.Tensor, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """LeRobotDataset 返す画像 (torch.Tensor (C, H, W) float [0,1]) → (target_size, target_size, 3) uint8 HWC numpy."""
    assert img.ndim == 3 and img.shape[0] == 3, f"expected (3, H, W), got {tuple(img.shape)}"
    # 256x256 → 224x224 (antialias 付き bilinear、RLDS 側と同じ設定)
    if img.shape[1] != target_size or img.shape[2] != target_size:
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=target_size, mode="bilinear", antialias=True
        ).squeeze(0)
    # [0,1] → [0,255] uint8 HWC
    img_uint8 = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return img_uint8.permute(1, 2, 0).cpu().numpy()


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class LeRobotLiberoDataset(IterableDataset):
    """LeRobot 形式 LIBERO の IterableDataset.

    既存 `LiberoDataset` (RLDS) と同じ per-timestep dict を yield するので、`LiberoCollator`
    で drop-in 互換。

    Args:
        repo_id: HF dataset repo (例: "lerobot/libero_spatial_image")
        dataset_statistics_path: BOUNDS_Q99 stats JSON (既存 libero 訓練済 stats を reuse)
        unnorm_key: stats 内の dataset key (例: "libero_spatial_no_noops")
        episodes: pull する episode subset (None で全件)
        action_chunk_len: action chunk 長 (既存 RLDS と揃えて 8)
        fps: dataset の fps (meta/info.json から取る方が堅牢だが explicit でも可)
        download_videos: False なら video decode skip (parquet-only 形式で安全)
    """

    def __init__(
        self,
        repo_id: str,
        dataset_statistics_path: str | Path,
        unnorm_key: str,
        episodes: Optional[List[int]] = None,
        action_chunk_len: int = NUM_ACTIONS_CHUNK,
        fps: Optional[int] = None,
        download_videos: bool = True,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # fps 自動判定 (meta/info.json を見る)
        if fps is None:
            from huggingface_hub import hf_hub_download
            info_path = hf_hub_download(repo_id=repo_id, filename="meta/info.json", repo_type="dataset")
            with open(info_path) as f:
                fps = json.load(f).get("fps", DEFAULT_FPS)
        self.fps = fps
        self.action_chunk_len = action_chunk_len

        # delta_timestamps: action を [t, t+1/fps, ..., t+(chunk-1)/fps] で取得 → (chunk_len, 7)
        delta = {"action": [i / fps for i in range(action_chunk_len)]}

        self.ds = LeRobotDataset(
            repo_id,
            delta_timestamps=delta,
            episodes=episodes,
            download_videos=download_videos,
        )

        # Task 文字列取得用: ds.meta.tasks は pandas DataFrame、index=task string, column="task_index"
        # → reverse map: task_index (int) → task (str) を eager に build
        tasks_df = self.ds.meta.tasks
        self._task_idx_to_str: Dict[int, str] = {}
        if hasattr(tasks_df, "iterrows"):
            for task_str, row in tasks_df.iterrows():
                self._task_idx_to_str[int(row["task_index"])] = str(task_str).strip()
        elif isinstance(tasks_df, dict):
            for k, v in tasks_df.items():
                self._task_idx_to_str[int(k)] = str(v).strip()
        elif isinstance(tasks_df, (list, tuple)):
            for i, v in enumerate(tasks_df):
                self._task_idx_to_str[i] = str(v).strip()

        # Action 正規化 stats
        stats_path = Path(dataset_statistics_path)
        if not stats_path.is_absolute():
            stats_path = REPO_ROOT / stats_path
        self.action_stats = _load_action_stats(str(stats_path), unnorm_key)
        self.unnorm_key = unnorm_key

    def _resolve_task(self, task_idx: int) -> str:
        return self._task_idx_to_str.get(int(task_idx), "")

    def _sample_to_dict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # LeRobotDataset sample schema (lerobot/libero_spatial_image):
        #   observation.images.image: (3, 256, 256) torch float [0,1]
        #   observation.images.wrist_image: (3, 256, 256) torch float [0,1]
        #   observation.state: (8,) torch float
        #   action: (action_chunk_len, 7) torch float (raw)
        #   task_index: 0-d torch long
        scene_img = _lerobot_img_to_uint8_hwc(sample["observation.images.image"], IMAGE_SIZE)
        wrist_img = _lerobot_img_to_uint8_hwc(sample["observation.images.wrist_image"], IMAGE_SIZE)

        proprio = sample["observation.state"].detach().cpu().numpy().astype(np.float32)  # (8,)
        action_chunk_raw = sample["action"].detach().cpu().numpy().astype(np.float32)     # (chunk, 7)
        action_chunk = _normalize_action_bounds_q99(action_chunk_raw, self.action_stats)

        task_idx_t = sample.get("task_index")
        task_idx = int(task_idx_t.item()) if torch.is_tensor(task_idx_t) else int(task_idx_t)
        language = self._resolve_task(task_idx)

        return {
            "scene_img":    scene_img,
            "wrist_img":    wrist_img,
            "action_chunk": action_chunk,
            "proprio":      proprio,
            "language":     language,
            "dataset_id":   DATASET_ID,
        }

    def __iter__(self) -> Iterator[dict]:
        # LeRobotDataset は __len__ あり (frame 単位)、__getitem__ で iterate 可能。
        # 訓練は無限 loop 前提 (RLDS loader 流)。
        while True:
            for i in range(len(self.ds)):
                sample = self.ds[i]
                # delta_timestamps の最後が episode 終端を超える場合 LeRobotDataset が
                # tolerance 制御するが、念のため shape check
                if sample["action"].shape[0] != self.action_chunk_len:
                    continue
                yield self._sample_to_dict(sample)


# ------------------------------------------------------------------
# CLI verify
# ------------------------------------------------------------------
def _verify(repo_id: str, dataset_statistics_path: str, unnorm_key: str, num_samples: int = 3):
    print(f"=== verify LeRobot loader: {repo_id} ===", flush=True)
    ds = LeRobotLiberoDataset(
        repo_id=repo_id,
        dataset_statistics_path=dataset_statistics_path,
        unnorm_key=unnorm_key,
        episodes=[0],  # 第 1 episode のみ
    )
    it = iter(ds)
    for i in range(num_samples):
        s = next(it)
        print(f"\n--- sample {i} ---")
        print(f"  scene_img:    shape={s['scene_img'].shape}, dtype={s['scene_img'].dtype}, "
              f"range=[{s['scene_img'].min()}, {s['scene_img'].max()}]")
        print(f"  wrist_img:    shape={s['wrist_img'].shape}, dtype={s['wrist_img'].dtype}, "
              f"range=[{s['wrist_img'].min()}, {s['wrist_img'].max()}]")
        print(f"  action_chunk: shape={s['action_chunk'].shape}, dtype={s['action_chunk'].dtype}, "
              f"range=[{s['action_chunk'].min():.3f}, {s['action_chunk'].max():.3f}]")
        print(f"  proprio:      shape={s['proprio'].shape}, dtype={s['proprio'].dtype}, "
              f"range=[{s['proprio'].min():.3f}, {s['proprio'].max():.3f}]")
        print(f"  language:     '{s['language']}'")
        print(f"  dataset_id:   {s['dataset_id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--repo_id", default="lerobot/libero_spatial_image")
    parser.add_argument("--dataset_statistics_path",
                        default="VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json")
    parser.add_argument("--unnorm_key", default="libero_spatial_no_noops")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    if args.verify:
        _verify(args.repo_id, args.dataset_statistics_path, args.unnorm_key, args.num_samples)
    else:
        parser.print_help()
