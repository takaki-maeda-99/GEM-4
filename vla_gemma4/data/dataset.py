import torch
from torch import Tensor
from torch.utils.data import Dataset

from .normalizer import Normalizer


def _patch_lerobot_video_backend():
    """Patch lerobot's video decoding to use pyav when torchvision/torchcodec fail."""
    try:
        import lerobot.datasets.video_utils as vu

        def _decode_pyav(video_path, timestamps, tolerance_s, backend=None):
            import av
            import numpy as np

            container = av.open(str(video_path))
            stream = container.streams.video[0]
            fps = float(stream.average_rate)

            frames = []
            for ts in timestamps:
                frame_idx = int(round(ts * fps))
                container.seek(frame_idx, stream=stream)
                for frame in container.decode(video=0):
                    img = frame.to_ndarray(format="rgb24")
                    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    frames.append(tensor)
                    break

            container.close()
            return torch.stack(frames) if frames else torch.empty(0)

        vu.decode_video_frames = _decode_pyav
    except ImportError:
        pass


_patch_lerobot_video_backend()

from lerobot.datasets.lerobot_dataset import LeRobotDataset


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
