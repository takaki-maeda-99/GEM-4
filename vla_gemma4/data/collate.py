import torch


def vla_collate_fn(samples: list[dict], num_cameras: int) -> dict:
    """Custom collate for VLADataset output.

    Handles: list-of-tensors (images), strings (instruction), tensors (proprio, actions).
    """
    # Stack images per camera: list of [B, C, H, W]
    images = [
        torch.stack([s["images"][cam_idx] for s in samples])
        for cam_idx in range(num_cameras)
    ]

    return {
        "images": images,
        "instruction": [s["instruction"] for s in samples],
        "proprio": torch.stack([s["proprio"] for s in samples]),
        "actions": torch.stack([s["actions"] for s in samples]),
    }
