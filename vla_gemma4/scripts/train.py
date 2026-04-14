"""Training entrypoint for VLA-Gemma4."""

import argparse
import logging
import yaml

import torch
from torch.utils.data import DataLoader

from vla_gemma4.data.collate import vla_collate_fn
from vla_gemma4.data.dataset import VLADataset
from vla_gemma4.data.normalizer import Normalizer
from vla_gemma4.model.vla_policy import VLAPolicy
from vla_gemma4.training.trainer import VLATrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_lora(policy: VLAPolicy, config: dict) -> VLAPolicy:
    """Apply LoRA to the LLM backbone and freeze ViT."""
    from peft import LoraConfig, get_peft_model

    lora_cfg = config["training"]["lora"]

    # Freeze vision tower
    for param in policy.gemma.model.vision_tower.parameters():
        param.requires_grad = False

    # Apply LoRA to LLM
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    policy.gemma = get_peft_model(policy.gemma, lora_config)
    return policy


def main():
    parser = argparse.ArgumentParser(description="Train VLA-Gemma4")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config loaded: {args.config}")

    # Build dataset
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
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: vla_collate_fn(batch, num_cameras=num_cameras),
    )

    # Build model
    policy = VLAPolicy(config)

    # Apply LoRA if configured
    if config["training"]["strategy"] == "lora":
        policy = setup_lora(policy, config)
        logger.info("LoRA applied to LLM backbone, ViT frozen")

    # Train (Accelerator in VLATrainer handles device placement and multi-GPU)
    num_training_steps = len(dataloader) * config["training"]["num_epochs"]
    trainer = VLATrainer(policy, config, num_training_steps=num_training_steps)
    trainer.train(dataloader)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
