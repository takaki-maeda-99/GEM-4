import logging
import os

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate import Accelerator

logger = logging.getLogger(__name__)


class VLATrainer:
    """Training loop for VLAPolicy with Accelerate for multi-GPU and mixed precision."""

    def __init__(
        self,
        policy: nn.Module,
        config: dict,
        num_training_steps: int = 10000,
        output_dir: str = "outputs",
    ):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        train_cfg = config["training"]

        # Accelerator handles device placement, mixed precision, and multi-GPU
        self.accelerator = Accelerator(
            mixed_precision=train_cfg.get("mixed_precision", "no"),
        )

        # Separate learning rates: higher for action head + custom modules
        head_lr = train_cfg.get("head_lr", train_cfg["lr"] * 10)
        backbone_params = []
        head_params = []
        for name, param in policy.named_parameters():
            if not param.requires_grad:
                continue
            if "action_head" in name or "proprio_encoder" in name or "act_tokens" in name or "feature_norm" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = AdamW([
            {"params": backbone_params, "lr": train_cfg["lr"]},
            {"params": head_params, "lr": head_lr},
        ], weight_decay=train_cfg["weight_decay"])
        logger.info(f"Optimizer: backbone lr={train_cfg['lr']}, head lr={head_lr} "
                     f"(backbone={len(backbone_params)}, head={len(head_params)} param groups)")

        # LR scheduler: linear warmup then cosine decay
        warmup_steps = train_cfg["warmup_steps"]
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(num_training_steps - warmup_steps, 1)
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # Wrap with Accelerator for multi-GPU + AMP
        self.policy, self.optimizer, self.scheduler = self.accelerator.prepare(
            policy, self.optimizer, self.scheduler
        )

        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.num_epochs = train_cfg["num_epochs"]
        self.save_every_n_steps = train_cfg["save_every_n_steps"]
        self.eval_every_n_steps = train_cfg["eval_every_n_steps"]
        self.global_step = 0

    def _batch_to_device(self, batch: dict) -> dict:
        """Move batch tensors to the accelerator device."""
        device = self.accelerator.device
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device)
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                result[k] = [t.to(device) for t in v]
            else:
                result[k] = v
        return result

    def train_step(self, batch: dict) -> dict:
        self.policy.train()
        self.optimizer.zero_grad()

        batch = self._batch_to_device(batch)

        with self.accelerator.autocast():
            loss_dict = self.policy.compute_loss(batch)

        self.accelerator.backward(loss_dict["loss"])

        self.accelerator.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        return {k: v.item() if hasattr(v, "item") else v for k, v in loss_dict.items()}

    def train_epoch(self, dataloader) -> list[dict]:
        metrics = []
        for batch in dataloader:
            step_metrics = self.train_step(batch)
            metrics.append(step_metrics)

            if self.global_step % 50 == 0:
                logger.info(
                    f"  step {self.global_step}: loss={step_metrics['loss']:.4f}"
                )

            if self.global_step % self.save_every_n_steps == 0:
                self._save_checkpoint()

        return metrics

    def train(self, train_dataloader, eval_dataloader=None) -> None:
        train_dataloader = self.accelerator.prepare(train_dataloader)

        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            metrics = self.train_epoch(train_dataloader)

            avg_loss = sum(m["loss"] for m in metrics) / len(metrics)
            logger.info(f"  avg_loss: {avg_loss:.4f}")

        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        path = os.path.join(self.output_dir, f"checkpoint_step_{self.global_step}.pt")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrapped = self.accelerator.unwrap_model(self.policy)
            torch.save(
                {
                    "step": self.global_step,
                    "model_state_dict": unwrapped.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                path,
            )
            logger.info(f"Checkpoint saved to {path}")
