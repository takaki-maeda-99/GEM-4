"""Evaluation entrypoint for VLA-Gemma4."""

import argparse
import json
import logging

import torch
from torch.utils.data import DataLoader

import yaml

from vla_gemma4.data.collate import vla_collate_fn
from vla_gemma4.data.dataset import VLADataset
from vla_gemma4.data.normalizer import Normalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(policy, dataloader, action_dim: int, normalizer=None) -> dict:
    """Run offline evaluation on a dataset.

    Returns:
        Dict with evaluation metrics.
    """
    policy.eval()

    all_mse = []
    all_l1 = []
    gripper_correct = 0
    gripper_total = 0
    pose_dim = action_dim - 1  # Last dim is gripper

    with torch.no_grad():
        for batch in dataloader:
            pred = policy.predict(batch)  # [B, T, action_dim]
            actions = batch["actions"].to(pred.device)

            # Denormalize if needed
            if normalizer is not None:
                pred = normalizer.denormalize(pred)
                actions = normalizer.denormalize(actions)

            # Position/rotation metrics (all dims except gripper)
            pose_pred = pred[:, :, :pose_dim]
            pose_target = actions[:, :, :pose_dim]
            all_mse.append(((pose_pred - pose_target) ** 2).mean().item())
            all_l1.append((pose_pred - pose_target).abs().mean().item())

            # Gripper metrics (last dim)
            gripper_pred = (pred[:, :, -1] > 0.5).float()
            gripper_target = (actions[:, :, -1] > 0.5).float()
            gripper_correct += (gripper_pred == gripper_target).sum().item()
            gripper_total += gripper_target.numel()

    metrics = {
        "mse": sum(all_mse) / len(all_mse),
        "l1": sum(all_l1) / len(all_l1),
        "gripper_accuracy": gripper_correct / gripper_total if gripper_total > 0 else 0.0,
        "num_batches": len(all_mse),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA-Gemma4")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    config = load_config(args.config)

    normalizer = None
    if config["data"].get("normalizer_path"):
        normalizer = Normalizer.load(config["data"]["normalizer_path"])

    dataset = VLADataset(
        dataset_name=config["data"]["dataset_name"],
        cameras=config["cameras"],
        proprio_key=config.get("proprio_key", "observation.state"),
        language_instruction_key=config["data"]["language_instruction_key"],
        default_instruction=config["data"]["default_instruction"],
        chunk_size=config["chunk_size"],
        normalizer=normalizer,
        video_backend="pyav",
        tolerance_s=1e6,  # Relaxed for older dataset formats
    )
    logger.info(f"Eval dataset size: {len(dataset)}")

    num_cameras = len(config["cameras"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=lambda batch: vla_collate_fn(batch, num_cameras=num_cameras),
    )

    # Build model with quantization support
    from vla_gemma4.scripts.train import build_policy, apply_lora

    logger.info("Loading model...")
    policy = build_policy(config)

    # Load checkpoint
    if config["training"]["strategy"] == "lora":
        policy = apply_lora(policy, config)
        # Load LoRA adapter weights from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        # Filter to only load matching keys (LoRA + custom modules)
        model_state = checkpoint["model_state_dict"]
        missing, unexpected = policy.load_state_dict(model_state, strict=False)
        logger.info(f"Loaded checkpoint (missing={len(missing)}, unexpected={len(unexpected)} keys)")
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        policy.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded full checkpoint")

    metrics = evaluate(policy, dataloader, action_dim=config["action_dim"], normalizer=normalizer)

    logger.info(f"Evaluation results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
