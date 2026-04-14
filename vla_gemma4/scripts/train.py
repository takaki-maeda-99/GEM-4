"""Training entrypoint for VLA-Gemma4."""

import argparse
import logging
import os
import yaml

import torch
from torch.utils.data import DataLoader

from vla_gemma4.data.collate import vla_collate_fn
from vla_gemma4.data.dataset import VLADataset
from vla_gemma4.data.normalizer import Normalizer
from vla_gemma4.model.vla_policy import VLAPolicy, _build_action_head
from vla_gemma4.model.proprio_encoder import ProprioEncoder
from vla_gemma4.training.trainer import VLATrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_policy(config: dict) -> VLAPolicy:
    """Build VLAPolicy with optional quantization."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    load_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}

    # Apply quantization if configured
    quant = config.get("inference", {}).get("quantization")
    if quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        if quant == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    gemma = AutoModelForImageTextToText.from_pretrained(config["model_name"], **load_kwargs)
    processor = AutoProcessor.from_pretrained(config["model_name"])

    hidden_dim = gemma.config.text_config.hidden_size
    device = next(gemma.model.language_model.parameters()).device

    # Build VLAPolicy manually with pre-loaded model
    policy = VLAPolicy.__new__(VLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    policy.gemma = gemma
    policy.processor = processor
    policy._gemma_model = gemma.model
    policy._lang_model = gemma.model.language_model
    policy._embed_tokens = gemma.model.language_model.embed_tokens

    policy.proprio_encoder = ProprioEncoder(
        proprio_dim=config["proprio_dim"],
        hidden_dim=hidden_dim,
    ).to(device)

    num_act = config["num_action_tokens"]
    policy.act_tokens = torch.nn.Parameter(
        torch.randn(1, num_act, hidden_dim, device=device) * 0.02
    )
    policy.action_head = _build_action_head(config, input_dim=hidden_dim).to(device)

    return policy


def apply_lora(policy: VLAPolicy, config: dict) -> VLAPolicy:
    """Apply LoRA to the LLM backbone and freeze ViT."""
    from peft import LoraConfig, get_peft_model

    lora_cfg = config["training"]["lora"]

    # Freeze vision tower
    for param in policy.gemma.model.vision_tower.parameters():
        param.requires_grad = False

    # Target only language_model layers (avoid vision_tower's Gemma4ClippableLinear)
    num_layers = policy.gemma.config.text_config.num_hidden_layers
    target_modules = [
        f"model.language_model.layers.{i}.self_attn.{proj}"
        for i in range(num_layers)
        for proj in lora_cfg["target_modules"]
    ]

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=target_modules,
    )
    policy.gemma = get_peft_model(policy.gemma, lora_config)
    policy.gemma.print_trainable_parameters()
    return policy


def main():
    parser = argparse.ArgumentParser(description="Train VLA-Gemma4")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Config loaded: {args.config}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build dataset
    normalizer = None
    if config["data"].get("normalizer_path"):
        normalizer = Normalizer.load(config["data"]["normalizer_path"])

    # Use lerobot_cameras if available (LeRobot dataset keys differ from sim keys)
    train_cameras = config.get("lerobot_cameras", config["cameras"])

    dataset = VLADataset(
        dataset_name=config["data"]["dataset_name"],
        cameras=train_cameras,
        proprio_key=config.get("proprio_key", "observation.state"),
        language_instruction_key=config["data"]["language_instruction_key"],
        default_instruction=config["data"]["default_instruction"],
        chunk_size=config["chunk_size"],
        normalizer=normalizer,
        video_backend="pyav",
        tolerance_s=1e6,  # Relaxed for older dataset formats
    )
    logger.info(f"Dataset size: {len(dataset)}")

    num_cameras = len(train_cameras)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda batch: vla_collate_fn(batch, num_cameras=num_cameras),
    )

    # Build model
    logger.info("Loading model...")
    policy = build_policy(config)

    # Apply LoRA if configured
    if config["training"]["strategy"] == "lora":
        policy = apply_lora(policy, config)
        logger.info("LoRA applied to LLM backbone, ViT frozen")

    # Train
    num_training_steps = len(dataloader) * config["training"]["num_epochs"]
    trainer = VLATrainer(
        policy, config,
        num_training_steps=num_training_steps,
        output_dir=args.output_dir,
    )
    trainer.train(dataloader)

    logger.info(f"Training complete. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
