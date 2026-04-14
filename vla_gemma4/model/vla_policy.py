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
    """VLA policy that integrates Gemma 4 backbone with swappable action heads.

    Uses the "PLE pre-computation + custom token injection" pattern to work
    with Gemma 4's Per-Layer Embeddings architecture. Custom tokens (proprio,
    ACT) use PAD token IDs as PLE placeholders, then their embeddings are
    overwritten — the same pattern Gemma 4 uses for vision/audio soft tokens.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Load Gemma 4 model and processor
        self.gemma = AutoModelForImageTextToText.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(config["model_name"])

        # Gemma 4 stores hidden_size in text_config; fall back for mocks
        gemma_config = self.gemma.config
        if (
            hasattr(gemma_config, "text_config")
            and hasattr(gemma_config.text_config, "hidden_size")
            and isinstance(gemma_config.text_config.hidden_size, int)
        ):
            hidden_dim = gemma_config.text_config.hidden_size
        else:
            hidden_dim = gemma_config.hidden_size

        # Store direct references to sub-modules (survive PEFT wrapping)
        self._gemma_model = self.gemma.model  # Gemma4Model
        self._lang_model = self.gemma.model.language_model  # Gemma4TextModel
        self._embed_tokens = self.gemma.model.language_model.embed_tokens

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

    def _get_pad_token_id(self) -> int:
        """Get PAD token ID, handling both real and mock configs."""
        try:
            pad_id = self.processor.tokenizer.pad_token_id
            if isinstance(pad_id, int):
                return pad_id
        except (AttributeError, TypeError):
            pass
        try:
            pad_id = self.gemma.config.text_config.pad_token_id
            if isinstance(pad_id, int):
                return pad_id
        except (AttributeError, TypeError):
            pass
        return 0

    def encode(self, batch: dict) -> Tensor:
        """Encode observations into features for the action head.

        Uses PLE pre-computation + custom token injection pattern:
        1. Build input_ids with image placeholders + PAD for custom tokens
        2. Compute PLE from full input_ids (PAD gets PAD's PLE — acceptable)
        3. Merge vision features via get_image_features + masked_scatter
        4. Overwrite PAD embeddings at custom positions with proprio/ACT
        5. Forward through language_model with pre-computed per_layer_inputs

        Returns:
            [B, N_act, D] features from [ACT] token positions.
        """
        images = batch["images"]  # list of [B, C, H, W] per camera
        instructions = batch["instruction"]  # list[str]
        proprio = batch["proprio"]  # [B, proprio_dim]

        B = proprio.shape[0]
        num_act = self.config["num_action_tokens"]
        num_custom = 1 + num_act  # 1 proprio + N ACT tokens
        pad_id = self._get_pad_token_id()

        # --- Step 1: Process text + images via processor ---
        # Build prompts with <start_of_image> for each camera
        img_tag = "<start_of_image>"
        prompts = []
        all_pil_images = []
        for i in range(B):
            # Each camera gets an image tag
            tags = img_tag * len(images)
            prompt = f"<start_of_turn>user\n{tags}{instructions[i]}<end_of_turn>"
            prompts.append(prompt)
            # Collect images for this sample
            sample_imgs = []
            for cam_images in images:
                # cam_images: [B, C, H, W] → take sample i, convert to PIL
                img_tensor = cam_images[i]  # [C, H, W]
                if img_tensor.dtype == torch.float32 or img_tensor.dtype == torch.float64:
                    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
                else:
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                from PIL import Image
                sample_imgs.append(Image.fromarray(img_np))
            all_pil_images.append(sample_imgs)

        # Processor expects images as list[list[PIL.Image]] matching text batch
        proc_inputs = self.processor(
            text=prompts,
            images=all_pil_images,
            return_tensors="pt",
            padding=True,
        )
        # Move to model device
        device = next(self._lang_model.parameters()).device
        input_ids = proc_inputs["input_ids"].to(device)
        attention_mask = proc_inputs["attention_mask"].to(device)
        pixel_values = proc_inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        image_position_ids = proc_inputs.get("image_position_ids")
        if image_position_ids is not None:
            image_position_ids = image_position_ids.to(device)
        mm_token_type_ids = proc_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(device)

        # --- Step 2: Append PAD token IDs for proprio + ACT positions ---
        custom_ids = torch.full(
            (B, num_custom), pad_id, dtype=input_ids.dtype, device=device
        )
        input_ids_ext = torch.cat([input_ids, custom_ids], dim=1)

        # Extend attention mask (custom tokens are real → mask=1)
        custom_mask = torch.ones(B, num_custom, dtype=attention_mask.dtype, device=device)
        attention_mask_ext = torch.cat([attention_mask, custom_mask], dim=1)

        # Extend mm_token_type_ids (custom tokens are text-type = 0)
        if mm_token_type_ids is not None:
            custom_mm = torch.zeros(B, num_custom, dtype=mm_token_type_ids.dtype, device=device)
            mm_token_type_ids_ext = torch.cat([mm_token_type_ids, custom_mm], dim=1)
        else:
            mm_token_type_ids_ext = None

        # --- Step 3: Compute embeddings + PLE from extended input_ids ---
        lang_model = self._lang_model
        inputs_embeds = self._embed_tokens(input_ids_ext)  # [B, S_ext, D]

        # Compute PLE
        if lang_model.hidden_size_per_layer_input:
            # Get image placeholder mask from the extended input_ids
            image_mask, _, _ = self._gemma_model.get_placeholder_mask(
                input_ids_ext, mm_token_type_ids_ext
            )

            # For PLE: replace image positions with PAD (same as Gemma4Model does)
            llm_input_ids = input_ids_ext.clone()
            llm_input_ids[image_mask] = pad_id

            pad_emb = self._embed_tokens.weight[pad_id]
            llm_inputs_embeds = torch.where(
                image_mask[..., None], pad_emb.view(1, 1, -1), inputs_embeds
            )
            per_layer_inputs = lang_model.get_per_layer_inputs(
                llm_input_ids, llm_inputs_embeds
            )
        else:
            per_layer_inputs = None

        # --- Step 4: Merge vision features into embeddings ---
        if pixel_values is not None and image_mask.any():
            vision_out = self._gemma_model.get_image_features(
                pixel_values, image_position_ids
            )
            # get_image_features returns BaseModelOutputWithPast
            image_features = vision_out.last_hidden_state.to(
                device=device, dtype=inputs_embeds.dtype
            )

            # Scatter vision features into image placeholder positions
            image_mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds = inputs_embeds.masked_scatter(image_mask_3d, image_features)

        # --- Step 5: Replace PAD embeddings at custom positions with proprio/ACT ---
        proprio_embeds = self.proprio_encoder(proprio.to(device))  # [B, 1, D]
        act_embeds = self.act_tokens.expand(B, -1, -1)  # [B, N_act, D]
        custom_embeds = torch.cat([proprio_embeds, act_embeds], dim=1)  # [B, num_custom, D]

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[:, -num_custom:, :] = custom_embeds.to(inputs_embeds.dtype)

        # --- Step 6: Project PLE ---
        if per_layer_inputs is not None:
            per_layer_inputs = lang_model.project_per_layer_inputs(
                inputs_embeds, per_layer_inputs
            )

        # --- Step 7: Forward through language model ---
        outputs = lang_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask_ext,
        )
        hidden_states = outputs.last_hidden_state

        # --- Step 8: Extract [ACT] token features from the end ---
        features = hidden_states[:, -num_act:, :]  # [B, N_act, D]

        return features

    def compute_loss(self, batch: dict) -> dict:
        features = self.encode(batch).float()  # Cast bf16 → float32 for action head
        dev = self.action_head_device
        actions = batch["actions"].to(dev).float()
        return self.action_head.compute_loss(features.to(dev), actions)

    @property
    def action_head_device(self) -> torch.device:
        return next(self.action_head.parameters()).device

    def predict(self, batch: dict) -> Tensor:
        with torch.no_grad():
            features = self.encode(batch).float()
            return self.action_head.predict(features.to(self.action_head_device))
