import torch
from torch import Tensor
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .normalizer import Normalizer


class VLADataset(Dataset):
    """Wraps a LeRobot dataset into a unified format for VLA training."""

    def __init__(
        self,
        dataset_name: str,
        cameras: list[str],
        proprio_key: str = "observation.state",
        action_key: str = "action",
        language_instruction_key: str = "language_instruction",
        default_instruction: str = "manipulation task",
        chunk_size: int = 1,
        normalizer: Normalizer | None = None,
        **kwargs,
    ):
        self.cameras = cameras
        self.proprio_key = proprio_key
        self.action_key = action_key
        self.language_instruction_key = language_instruction_key
        self.default_instruction = default_instruction
        self.chunk_size = chunk_size
        self.normalizer = normalizer

        # Build delta_timestamps for action chunking
        delta_timestamps = None
        if chunk_size > 1:
            fps = kwargs.pop("fps", 10)
            dt = 1.0 / fps
            delta_timestamps = {
                action_key: [i * dt for i in range(chunk_size)],
            }

        self.lerobot_dataset = LeRobotDataset(
            dataset_name,
            delta_timestamps=delta_timestamps,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.lerobot_dataset[idx]

        images = [sample[cam] for cam in self.cameras]
        proprio = sample[self.proprio_key]
        actions = sample[self.action_key]

        # Ensure actions shape is [T, action_dim]
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        if self.normalizer is not None:
            actions = self.normalizer.normalize(actions)

        instruction = sample.get(
            self.language_instruction_key, self.default_instruction
        )

        return {
            "images": images,
            "instruction": instruction,
            "proprio": proprio,
            "actions": actions,
        }
