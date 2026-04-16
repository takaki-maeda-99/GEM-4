"""Evaluation entrypoint for VLA-Gemma4."""

import argparse
import json
import logging

import torch
from torch.utils.data import DataLoader, Subset

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
    """Run offline evaluation on a dataset."""
    policy.eval()

    all_mse = []
    all_l1 = []
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            pred = policy.predict(batch)
            actions = batch["actions"].to(pred.device)

            if normalizer is not None:
                pred = normalizer.denormalize(pred)
                actions = normalizer.denormalize(actions)

            mse = ((pred - actions) ** 2).mean().item()
            l1 = (pred - actions).abs().mean().item()
            all_mse.append(mse)
            all_l1.append(l1)
            total_samples += pred.shape[0]

            if (i + 1) % 10 == 0:
                logger.info(f"  eval batch {i+1}: mse={mse:.6f}, l1={l1:.6f}")

    metrics = {
        "mse": sum(all_mse) / len(all_mse),
        "l1": sum(all_l1) / len(all_l1),
        "num_batches": len(all_mse),
        "num_samples": total_samples,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA-Gemma4")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--max_samples", type=int, default=200, help="Max samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=4, help="Eval batch size (smaller than training)")
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
        tolerance_s=1e6,
    )

    # Use subset for faster evaluation
    eval_size = min(args.max_samples, len(dataset))
    subset = Subset(dataset, range(eval_size))
    logger.info(f"Evaluating on {eval_size} / {len(dataset)} samples")

    num_cameras = len(config["cameras"])
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: vla_collate_fn(batch, num_cameras=num_cameras),
    )

    # Build model
    from vla_gemma4.scripts.train import build_policy, apply_lora

    logger.info("Loading model...")
    policy = build_policy(config)

    # Load checkpoint — only load trainable weights (LoRA + custom modules)
    if config["training"]["strategy"] == "lora":
        policy = apply_lora(policy, config)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Load only matching keys
    saved_state = checkpoint["model_state_dict"]
    current_state = policy.state_dict()
    filtered_state = {k: v for k, v in saved_state.items() if k in current_state}
    policy.load_state_dict(filtered_state, strict=False)
    logger.info(f"Loaded {len(filtered_state)} / {len(saved_state)} keys from checkpoint")

    # Free checkpoint memory
    del checkpoint, saved_state
    torch.cuda.empty_cache()

    metrics = evaluate(policy, dataloader, action_dim=config["action_dim"], normalizer=normalizer)

    logger.info(f"Evaluation results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
