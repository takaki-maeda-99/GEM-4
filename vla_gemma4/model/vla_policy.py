import torch
from torch import Tensor, nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from .action_heads.base import ActionHead
from .action_heads.mlp_head import MLPHead
from .proprio_encoder import ProprioEncoder


def _build_action_head(config: dict, input_dim: int) -> ActionHead:
    """Factory to build action head from config."""
    head_config = config["action_head"]
    head_type = head_config["type"]

    if head_type == "mlp":
        return MLPHead(
            input_dim=input_dim,
            action_dim=config["action_dim"],
            chunk_size=config["chunk_size"],
            hidden_dims=head_config.get("hidden_dims"),
        )
    else:
        raise ValueError(f"Unknown action head type: {head_type}")


class VLAPolicy(nn.Module):
    """VLA policy that integrates Gemma 4 backbone with swappable action heads."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Load Gemma 4 model and processor
        self.gemma = AutoModelForImageTextToText.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(config["model_name"])

        hidden_dim = self.gemma.config.hidden_size

        # Proprio encoder
        self.proprio_encoder = ProprioEncoder(
            proprio_dim=config["proprio_dim"],
            hidden_dim=hidden_dim,
        )

        # Learnable [ACT] tokens
        num_act = config["num_action_tokens"]
        self.act_tokens = nn.Parameter(torch.randn(1, num_act, hidden_dim) * 0.02)

        # Action head
        self.action_head = _build_action_head(config, input_dim=hidden_dim)

    def encode(self, batch: dict) -> Tensor:
        """Encode observations into features for the action head.

        Returns:
            [B, N_act, D] features from [ACT] token positions.
        """
        images = batch["images"]  # list of [B, C, H, W]
        instructions = batch["instruction"]  # list of str
        proprio = batch["proprio"]  # [B, proprio_dim]

        B = proprio.shape[0]
        device = proprio.device

        # 1. Process images through Gemma 4 vision encoder
        image_embeds_list = []
        for cam_images in images:
            # Get image features via the vision tower (SiglipVisionModel)
            vision_outputs = self.gemma.model.vision_tower(
                pixel_values=cam_images.to(self.gemma.dtype),
            )
            img_features = vision_outputs.last_hidden_state
            # Project to LLM space via embed_vision
            img_embeds = self.gemma.model.embed_vision(inputs_embeds=img_features)
            image_embeds_list.append(img_embeds)

        # 2. Encode language instruction
        text_inputs = self.processor.tokenizer(
            instructions, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        text_embeds = self.gemma.model.embed_tokens(text_inputs.input_ids)
        # Track text attention mask for padding
        text_attention_mask = text_inputs.attention_mask  # [B, text_len]

        # 3. Encode proprioception
        proprio_embeds = self.proprio_encoder(proprio)  # [B, 1, D]

        # 4. Expand [ACT] tokens for batch
        act_embeds = self.act_tokens.expand(B, -1, -1)  # [B, N_act, D]

        # 5. Concatenate all embeddings: [images...] [text] [proprio] [ACT]
        all_embeds = torch.cat(
            image_embeds_list + [text_embeds, proprio_embeds, act_embeds],
            dim=1,
        )

        # 6. Build attention mask (1 for real tokens, 0 for text padding)
        num_image_tokens = sum(e.shape[1] for e in image_embeds_list)
        num_proprio_tokens = 1
        num_act_tokens = self.config["num_action_tokens"]
        # Image, proprio, and ACT tokens are always attended to (ones)
        non_text_mask = torch.ones(
            B, num_image_tokens + num_proprio_tokens + num_act_tokens,
            device=device, dtype=text_attention_mask.dtype,
        )
        # Insert text mask between image tokens and proprio/ACT tokens
        attention_mask = torch.cat(
            [
                non_text_mask[:, :num_image_tokens],
                text_attention_mask,
                non_text_mask[:, num_image_tokens:],
            ],
            dim=1,
        )

        # 7. Forward through LLM backbone
        outputs = self.gemma.model(
            inputs_embeds=all_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        # 8. Extract [ACT] token features from the end
        num_act = self.config["num_action_tokens"]
        features = hidden_states[:, -num_act:, :]  # [B, N_act, D]

        return features

    def compute_loss(self, batch: dict) -> dict:
        features = self.encode(batch)
        return self.action_head.compute_loss(features, batch["actions"])

    def predict(self, batch: dict) -> Tensor:
        with torch.no_grad():
            features = self.encode(batch)
            return self.action_head.predict(features)
