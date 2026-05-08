"""Multi-dataset pretrain loader (new schema、scene+wrist raw uint8).

2026-04-26 #024: taco_solo_loader.py と同 schema で複数データセット対応版。
legacy multi_dataset_loader.py は image_transform を loader 内で実施する旧 DinoSigLIP schema 用。
本 loader は taco_solo_loader と同じ {scene, wrist} raw uint8 schema、collate で resize/normalize。

dataset_id mapping (SoftPromptLibrary 用、X-VLA 流):
    0 = taco_play
    1 = fractal20220817_data

Fractal は wrist 専用カメラ無しなので、scene image を duplicate (legacy の duplicate hack 踏襲)。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset

# Reuse helpers from taco_solo_loader (same schema)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from taco_solo_loader import (  # noqa: E402
    DATA_ROOT,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    extract_canonical_action_taco_play,
    extract_images_taco_play,
    extract_language,
    extract_proprio_taco_play,
    normalize_action_bounds_q99,
)
from multi_dataset_loader import (  # noqa: E402
    extract_canonical_action_fractal,
    extract_images_fractal,
)

import json

# ===========================================================
DATASET_ID_MAP: Dict[str, int] = {
    "taco_play": 0,
    "fractal20220817_data": 1,
}
DEFAULT_WEIGHTS: Dict[str, float] = {"taco_play": 0.30, "fractal20220817_data": 0.70}


def extract_proprio_zero_8d(_step: Dict[str, Any]) -> np.ndarray:
    """Fractal は proprio nominal の正解値が無いので zero 埋める (X-VLA pretrain と同等)。"""
    return np.zeros(PROPRIO_DIM, dtype=np.float32)


# Per-dataset adapter
class _PerDatasetIter:
    def __init__(
        self,
        dataset_name: str,
        stats: dict,
        action_extractor,
        image_extractor,
        proprio_extractor,
        data_dir: Path,
        num_actions_chunk: int,
    ):
        self.dataset_name = dataset_name
        self.dataset_id = DATASET_ID_MAP[dataset_name]
        self.stats = stats
        self.action_extractor = action_extractor
        self.image_extractor = image_extractor
        self.proprio_extractor = proprio_extractor
        self.data_dir = data_dir
        self.num_actions_chunk = num_actions_chunk

    def iter_chunks(self) -> Iterator[dict]:
        ds_tf = tfds.load(self.dataset_name, data_dir=str(self.data_dir), split="train")
        for ep in ds_tf:
            steps_buf = list(ep["steps"])
            for i in range(len(steps_buf)):
                chunk = []
                for k in range(self.num_actions_chunk):
                    idx = min(i + k, len(steps_buf) - 1)
                    a = self.action_extractor(steps_buf[idx])
                    a_norm = normalize_action_bounds_q99(a, self.stats)
                    chunk.append(a_norm)
                action_chunk = np.stack(chunk, axis=0)

                scene, wrist = self.image_extractor(steps_buf[i])
                proprio = self.proprio_extractor(steps_buf[i])
                language = extract_language(steps_buf[i])

                yield {
                    "scene_img": scene,
                    "wrist_img": wrist,
                    "action_chunk": action_chunk,
                    "proprio": proprio,
                    "language": language,
                    "dataset_id": self.dataset_id,
                }


class MultiSoloDataset(IterableDataset):
    """Weighted-sampling multi-dataset loader、taco_solo_loader と同 schema (scene/wrist raw uint8).

    pretrain 用、collate は taco_solo_loader.collate_taco_solo を流用 (同 schema)。
    """

    def __init__(
        self,
        data_dir: str | Path = DATA_ROOT,
        weights: Dict[str, float] | None = None,
        num_actions_chunk: int = NUM_ACTIONS_CHUNK,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.num_actions_chunk = num_actions_chunk
        self.seed = seed

        weights = weights or DEFAULT_WEIGHTS
        # Load per-dataset stats
        self.stats: Dict[str, dict] = {}
        for ds_name in DATASET_ID_MAP:
            p = self.data_dir / ds_name / "dataset_statistics.json"
            if not p.exists():
                raise FileNotFoundError(
                    f"dataset_statistics.json not found for {ds_name} at {p}."
                )
            data = json.loads(p.read_text())
            self.stats[ds_name] = data[ds_name]["action"]

        self.datasets: Dict[str, _PerDatasetIter] = {
            "taco_play": _PerDatasetIter(
                "taco_play", self.stats["taco_play"],
                extract_canonical_action_taco_play, extract_images_taco_play,
                extract_proprio_taco_play,
                self.data_dir, num_actions_chunk,
            ),
            "fractal20220817_data": _PerDatasetIter(
                "fractal20220817_data", self.stats["fractal20220817_data"],
                extract_canonical_action_fractal, extract_images_fractal,
                extract_proprio_zero_8d,
                self.data_dir, num_actions_chunk,
            ),
        }
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.ds_names: List[str] = list(self.weights.keys())
        self.ds_weights: List[float] = [self.weights[n] for n in self.ds_names]

    def __iter__(self) -> Iterator[dict]:
        from torch.utils.data import get_worker_info
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_seed = self.seed + worker_id * 1000 + 1
        worker_rng = random.Random(worker_seed)

        # Worker-local TFDS iters (lazy init for fork safety)
        ds_iters = {n: iter(self.datasets[n].iter_chunks()) for n in self.ds_names}

        while True:
            chosen = worker_rng.choices(self.ds_names, weights=self.ds_weights, k=1)[0]
            try:
                ck = next(ds_iters[chosen])
            except StopIteration:
                ds_iters[chosen] = iter(self.datasets[chosen].iter_chunks())
                ck = next(ds_iters[chosen])
            yield ck
