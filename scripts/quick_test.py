"""Quick test: load model, train a few steps, run inference, print results."""

import logging
import sys
import yaml
import torch
from torch.utils.data import DataLoader, Subset

from vla_gemma4.data.collate import vla_collate_fn
from vla_gemma4.data.dataset import VLADataset
from vla_gemma4.model.vla_policy import VLAPolicy, _build_action_head
from vla_gemma4.model.proprio_encoder import ProprioEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "vla_gemma4/configs/aloha_sim_test.yaml"
    config = load_config(config_path)

    # ---- 1. Load dataset (small subset) ----
    logger.info("Loading dataset...")
    dataset = VLADataset(
        dataset_name=config["data"]["dataset_name"],
        cameras=config["cameras"],
        proprio_key=config.get("proprio_key", "observation.state"),
        language_instruction_key=config["data"]["language_instruction_key"],
        default_instruction=config["data"]["default_instruction"],
        chunk_size=config["chunk_size"],
        video_backend="pyav",
        tolerance_s=1e6,  # Relaxed for older dataset formats
    )
    logger.info(f"Dataset size: {len(dataset)}")

    subset = Subset(dataset, range(min(100, len(dataset))))
    num_cameras = len(config["cameras"])
    dataloader = DataLoader(
        subset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: vla_collate_fn(batch, num_cameras=num_cameras),
    )

    # ---- 2. Load model with 4-bit quantization ----
    logger.info("Loading Gemma 4 E2B with 4-bit quantization...")
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    gemma = AutoModelForImageTextToText.from_pretrained(
        config["model_name"],
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(config["model_name"])

    hidden_dim = gemma.config.text_config.hidden_size
    logger.info(f"Model loaded. Hidden size: {hidden_dim}")

    # Build VLAPolicy manually with pre-loaded model
    policy = VLAPolicy.__new__(VLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    policy.gemma = gemma
    policy.processor = processor
    policy._gemma_model = gemma.model
    policy._lang_model = gemma.model.language_model
    policy._embed_tokens = gemma.model.language_model.embed_tokens

    device = next(gemma.model.language_model.parameters()).device
    policy.proprio_encoder = ProprioEncoder(
        proprio_dim=config["proprio_dim"],
        hidden_dim=hidden_dim,
    ).to(device)

    num_act = config["num_action_tokens"]
    policy.act_tokens = torch.nn.Parameter(
        torch.randn(1, num_act, hidden_dim, device=device) * 0.02
    )
    policy.feature_norm = torch.nn.LayerNorm(hidden_dim).to(device)
    policy.action_head = _build_action_head(config, input_dim=hidden_dim).to(device)

    # ---- 3. Apply LoRA ----
    logger.info("Applying LoRA...")
    from peft import LoraConfig, get_peft_model

    lora_cfg = config["training"]["lora"]
    for param in gemma.model.vision_tower.parameters():
        param.requires_grad = False

    # Target only language_model layers (avoid vision_tower's Gemma4ClippableLinear)
    target_modules = [
        f"model.language_model.layers.{i}.self_attn.{proj}"
        for i in range(35)
        for proj in ["q_proj", "v_proj"]
    ]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=target_modules,
    )
    policy.gemma = get_peft_model(policy.gemma, lora_config)
    policy.gemma.print_trainable_parameters()

    # ---- 4. Train a few steps ----
    num_steps = 20
    logger.info(f"Training {num_steps} steps...")

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["training"]["lr"])

    policy.train()
    step = 0
    for batch in dataloader:
        if step >= num_steps:
            break

        try:
            features = policy.encode(batch).float()
            dev = next(policy.action_head.parameters()).device
            loss_dict = policy.action_head.compute_loss(
                features.to(dev), batch["actions"].to(dev).float()
            )

            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config["training"]["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()

            if step % 5 == 0:
                logger.info(
                    f"  Step {step}: loss={loss_dict['loss'].item():.4f} "
                    f"mse={loss_dict.get('mse_loss', torch.tensor(0)):.4f} "
                    f"bce={loss_dict.get('bce_loss', torch.tensor(0)):.4f}"
                )
            step += 1

        except Exception as e:
            logger.error(f"Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break

    # ---- 5. Inference ----
    logger.info("\nRunning inference on 3 samples...")
    policy.eval()

    eval_loader = DataLoader(
        Subset(dataset, range(3)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: vla_collate_fn(batch, num_cameras=num_cameras),
    )

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            pred = policy.predict(batch)
            gt = batch["actions"]

            logger.info(f"\n--- Sample {i} ---")
            logger.info(f"  Instruction: {batch['instruction'][0]}")
            logger.info(f"  Predicted action: {pred[0, 0, :7].cpu().tolist()}")
            logger.info(f"  Ground truth:     {gt[0, 0, :7].cpu().tolist()}")
            logger.info(f"  Pred shape: {pred.shape}")
            logger.info(f"  MSE: {((pred.cpu() - gt.cpu()) ** 2).mean().item():.6f}")

    logger.info("\nDone!")

    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Peak GPU memory: {allocated:.2f} GB")


if __name__ == "__main__":
    main()
