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
from vla_gemma4.model.vla_policy import VLAPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(policy, dataloader, normalizer=None) -> dict:
    """Run offline evaluation on a dataset.

    Returns:
        Dict with evaluation metrics.
    """
    policy.eval()

    all_mse = []
    all_l1 = []
    gripper_correct = 0
    gripper_total = 0

    with torch.no_grad():
        for batch in dataloader:
            pred = policy.predict(batch)  # [B, T, 7]
            actions = batch["actions"]

            # Denormalize if needed
            if normalizer is not None:
                pred = normalizer.denormalize(pred)
                actions = normalizer.denormalize(actions)

            # Position/rotation metrics (dims 0-5)
            pose_pred = pred[:, :, :6]
            pose_target = actions[:, :, :6]
            all_mse.append(((pose_pred - pose_target) ** 2).mean().item())
            all_l1.append((pose_pred - pose_target).abs().mean().item())

            # Gripper metrics (dim 6)
            gripper_pred = (pred[:, :, 6] > 0.5).float()
            gripper_target = (actions[:, :, 6] > 0.5).float()
            gripper_correct += (gripper_pred == gripper_target).sum().item()
            gripper_total += gripper_target.numel()

    metrics = {
        "mse": sum(all_mse) / len(all_mse),
        "l1": sum(all_l1) / len(all_l1),
        "gripper_accuracy": gripper_correct / gripper_total if gripper_total > 0 else 0.0,
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
    )

    num_cameras = len(config["cameras"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: vla_collate_fn(batch, num_cameras=num_cameras),
    )

    # Load model (handles both full and LoRA checkpoints)
    policy = VLAPolicy(config)
    if config["training"]["strategy"] == "lora":
        from peft import PeftModel
        policy.gemma = PeftModel.from_pretrained(
            policy.gemma, args.checkpoint
        )
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        policy.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)

    metrics = evaluate(policy, dataloader, normalizer)

    logger.info(f"Evaluation results: {metrics}")
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
