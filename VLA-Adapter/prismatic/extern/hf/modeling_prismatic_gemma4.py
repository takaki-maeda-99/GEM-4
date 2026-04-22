"""
modeling_prismatic_gemma4.py

VLA-Adapter backbone を Gemma 4 E2B に置き換えた統合 nn.Module。
Stage 1 (Phase 1b.6) で作成。

構成:
  - LLM: Gemma4ForConditionalGeneration (frozen) — vision_tower + embed_vision 同梱
  - ProprioProjector: proprio_dim → llm_dim → llm_dim (trainable)
  - action_queries: Embedding(64, llm_dim), zero init (trainable)
  - feature_norm: Identity (1b.1 判定)、LayerNorm 差し替え可
  - action_head: L1RegressionActionHead(use_pro_version=True, ~640M, trainable)

Task 6 (rev 3) で DinoSigLIP + VisionProjector を廃し、Gemma 4 純正の vision_tower +
embed_vision (Gemma4MultimodalEmbedder) に置換。画像前処理は Gemma4ImageProcessor に委譲。

forward pattern (1b.3-1b.5 で検証済み):
  - input_ids: placeholder ID 含む系列 (vision N + action 64 + proprio 1 + bos/eos/prompt)
    N = self.num_vision_tokens (max_soft_tokens=280 → 256)
  - PLE は input_ids から事前計算 (OOM 回避)
  - clone + advanced indexing で vision / action placeholder を上書き
  - Gemma4TextModel を直接呼ぶ (wrapper スキップ)
  - attention_mask / position_ids を明示構築 (sliding_window=512 超過対応)
  - entries 0-24 slice を action_head に渡す
"""
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor

from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants_gemma4 import (
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTION_TOKENS,
    PROPRIO_PLACEHOLDER_IDX,
    VISION_PLACEHOLDER_BEGIN_IDX,
)


class SoftPromptLibrary(nn.Module):
    """Stage 3 (X-VLA 準拠): dataset_id indexed な learnable prompt library.

    X-VLA 原実装 (X-VLA/models/transformer.py:336) では `nn.Embedding(num_domains, len × hidden)`
    で action head 内の transformer 入力末尾に concat。本 Gemma 4 実装は User plan §R21 配置 "案 B"
    に基づき、**LLM の `inputs_embeds` 前段に concat** (案 B deviation from X-VLA、migration_log 記録)。

    Attrs:
        embedding: `nn.Embedding(num_datasets, num_tokens * hidden_dim)`、std=0.02 init
        num_datasets, num_tokens, hidden_dim
    """
    def __init__(self, num_datasets: int, num_tokens: int = 32, hidden_dim: int = 1536):
        super().__init__()
        self.embedding = nn.Embedding(num_datasets, num_tokens * hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.num_datasets = num_datasets
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

    def forward(self, dataset_id: torch.LongTensor) -> torch.Tensor:
        """dataset_id: (B,) → soft_prompts: (B, num_tokens, hidden_dim)."""
        B = dataset_id.shape[0]
        return self.embedding(dataset_id).view(B, self.num_tokens, self.hidden_dim)


class ProprioProjector(nn.Module):
    """2 層 MLP: proprio_dim → llm_dim → llm_dim."""

    def __init__(self, proprio_dim: int, llm_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim)
        self.fc2 = nn.Linear(llm_dim, llm_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionProjector(nn.Module):
    """Ablation only (DinoSigLIP path): 3 層 MLP vision_dim → initial_projection_dim → llm_dim → llm_dim.

    Task 6 で VLAAdapterGemma4 から取り除いた pre-Task-6 実装を復活させたもの。
    vision_backbone_type="dinosiglip" のとき、DinoSigLIPViTBackbone (2176-dim concat) の
    patch 出力を Gemma4 LLM hidden size へ射影するために使う。
    """

    def __init__(self, vision_dim: int, llm_dim: int, initial_projection_dim: int = 8192):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, initial_projection_dim, bias=True)
        self.fc2 = nn.Linear(initial_projection_dim, llm_dim, bias=True)
        self.fc3 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


class VLAAdapterGemma4(nn.Module):
    """Gemma 4 E2B backbone + VLA-Adapter 設計 の統合モデル (Stage 1)."""

    def __init__(
        self,
        gemma_model: nn.Module,                 # Gemma4ForConditionalGeneration instance
        feature_norm: Optional[nn.Module] = None,
        proprio_dim: int = 8,
        action_dim: int = 7,
        num_action_chunks: int = 8,
        max_soft_tokens: int = 280,              # Task 6 rev 3 default (→ 256 vision tokens)
        # --- Stage 3 Soft Prompt (X-VLA 準拠、配置 案 B = LLM inputs_embeds 前段) ---
        # num_pretrain_datasets=0 (default) で soft_prompt 無効化、Stage 1-2 と backward compatible
        num_pretrain_datasets: int = 0,
        num_soft_prompt_tokens: int = 32,
        # --- Dual-Track (Task 11): "quality" | "speed" ---
        #   quality: action_queries trainable + LLM grad 流れる (GC/LoRA 併用前提、finetune side)
        #   speed:   action_queries frozen (zero init)、forward 内で LLM を torch.no_grad() で wrap
        training_mode: str = "quality",
        # --- Scene vision backbone ablation (2026-04-22): "gemma4_native" | "dinosiglip" | "siglip" ---
        # gemma4_native: Gemma 4 純正 vision_tower + embed_vision (Task 6 以降の default)
        # dinosiglip:    Pre-Task-6 の DinoV2-L + SigLIP-So400m concat + VisionProjector 経路
        # siglip:        SigLIP-So400m 単独 + VisionProjector (DinoV2 無し、半分の params / compute)
        vision_backbone_type: str = "gemma4_native",
        dinosiglip_backbone_id: str = "dinosiglip-vit-so-224px",
        dinosiglip_resize_strategy: str = "resize-naive",
        dinosiglip_image_size: int = 224,
        siglip_backbone_id: str = "siglip-vit-so400m",
        siglip_resize_strategy: str = "resize-naive",
        siglip_image_size: int = 224,
        siglip_use_tensor_transform: bool = False,  # True: PIL 経由なし、GPU tensor 直接変換
    ):
        super().__init__()
        assert training_mode in ("quality", "speed"), \
            f"training_mode must be 'quality' or 'speed', got {training_mode!r}"
        assert vision_backbone_type in ("gemma4_native", "dinosiglip", "siglip"), \
            f"vision_backbone_type must be 'gemma4_native' / 'dinosiglip' / 'siglip', got {vision_backbone_type!r}"
        self.training_mode = training_mode
        self.vision_backbone_type = vision_backbone_type
        self.llm = gemma_model
        self.llm.config.use_cache = True                    # G1 対策
        assert not getattr(self.llm, "is_gradient_checkpointing", False), \
            "gradient_checkpointing must be off (Stage 1 前提、KV 共有バグ回避)"
        # Freeze ALL of Gemma 4 (vision_tower + embed_vision + language_model + audio_tower + embed_audio)
        for p in self.llm.parameters():
            p.requires_grad = False

        llm_dim = self.llm.config.text_config.hidden_size   # Gemma 4 E2B = 1536

        # --- Scene vision branch (2 modes) ---
        if vision_backbone_type == "gemma4_native":
            # Gemma4ImageProcessor は aspect-ratio-preserving resize + patchify + position_ids 生成を担う。
            self.image_processor = Gemma4ImageProcessor(max_soft_tokens=max_soft_tokens)
            self.max_soft_tokens = max_soft_tokens   # retained for introspection / logging (__repr__)

            # num_vision_tokens を実測で決める (224x224 dummy で processor に聞く)
            # 実測: max_soft_tokens=70→64, 140→121, 280→256
            _dummy = torch.zeros(3, 224, 224)   # content irrelevant, only shape matters; avoids RNG contamination
            _out = self.image_processor.preprocess(_dummy, return_tensors="pt")
            self.num_vision_tokens: int = int(_out["num_soft_tokens_per_image"][0])
            # 以下の属性は dinosiglip path では使われない
            self.vision_backbone = None
            self.vision_projector = None
        elif vision_backbone_type == "dinosiglip":
            # Ablation path: Pre-Task-6 wrapper を復活 (VisionProjector + DinoSigLIPViTBackbone)。
            # ※ Gemma 4 native image_processor は構築しない (scene は DinoSigLIP 経路)。
            #   ただし loader は同一 (scene は (B, 3, 224, 224) float [0,255])、encode_scene 内部で
            #   PIL 経由で DinoSigLIP image_transform を適用する。
            from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone
            self.image_processor = None
            self.max_soft_tokens = None
            self.vision_backbone = DinoSigLIPViTBackbone(
                vision_backbone_id=dinosiglip_backbone_id,
                image_resize_strategy=dinosiglip_resize_strategy,
                default_image_size=dinosiglip_image_size,
                image_sequence_len=1,   # scene のみ (wrist は ResNet18 別経路)
            )
            # freeze + eval (BN など無いが dropout 念のため)
            for p in self.vision_backbone.parameters():
                p.requires_grad = False
            self.vision_backbone.eval()
            vision_embed_dim = self.vision_backbone.embed_dim   # DINO (1024) + SigLIP (1152) = 2176
            self.vision_projector = VisionProjector(
                vision_dim=vision_embed_dim,
                llm_dim=llm_dim,
                initial_projection_dim=8192,
            )
            # DinoSigLIP at 224 / patch14 → 16×16 = 256 patches (image_sequence_len=1)
            self.num_vision_tokens: int = int(self.vision_backbone.num_patches)
            assert self.num_vision_tokens == 256, (
                f"DinoSigLIP-vit-so-224px expected 256 patches, got {self.num_vision_tokens}. "
                "loader (NUM_VISION_TOKENS=256) と不整合。"
            )
        else:  # "siglip"
            # SigLIP-only ablation (2026-04-23): DinoV2 廃して SigLIP-So400m 単独、VisionProjector に通す。
            # Gemma 4 vision tower 自体が SigLIP 系由来のため、"独立 SigLIP vs LLM 共同学習 SigLIP" の
            # 最もクリーンな同族比較。params / compute は DinoSigLIP の半分。
            from prismatic.models.backbones.vision.siglip_vit import SigLIPViTBackbone
            self.image_processor = None
            self.max_soft_tokens = None
            self.vision_backbone = SigLIPViTBackbone(
                vision_backbone_id=siglip_backbone_id,
                image_resize_strategy=siglip_resize_strategy,
                default_image_size=siglip_image_size,
            )
            for p in self.vision_backbone.parameters():
                p.requires_grad = False
            self.vision_backbone.eval()
            vision_embed_dim = self.vision_backbone.embed_dim   # SigLIP-So400m = 1152
            self.vision_projector = VisionProjector(
                vision_dim=vision_embed_dim,
                llm_dim=llm_dim,
                initial_projection_dim=8192,
            )
            # SigLIP at 224 / patch14 → 16×16 = 256 patches
            self.num_vision_tokens: int = int(self.vision_backbone.num_patches)
            assert self.num_vision_tokens == 256, (
                f"siglip-vit-so400m expected 256 patches, got {self.num_vision_tokens}. "
                "loader (NUM_VISION_TOKENS=256) と不整合。"
            )
            self.siglip_use_tensor_transform = siglip_use_tensor_transform

        self.proprio_projector = ProprioProjector(proprio_dim=proprio_dim, llm_dim=llm_dim)

        # --- Wrist encoder (Task 8): trainable ResNet18, bypasses frozen LLM ---
        from prismatic.models.backbones.vision.wrist_resnet18 import WristResNet18
        self.wrist_encoder = WristResNet18(out_dim=llm_dim)

        self.action_queries = nn.Embedding(NUM_ACTION_TOKENS, llm_dim)
        self.action_queries.weight.data.zero_()

        self.feature_norm = feature_norm if feature_norm is not None else nn.Identity()

        self.action_head = L1RegressionActionHead(
            input_dim=llm_dim,
            hidden_dim=llm_dim,
            action_dim=action_dim,
            num_task_tokens=self.num_vision_tokens,
            use_pro_version=True,
        )

        self._num_action_chunks = num_action_chunks

        # --- Stage 3: Soft Prompt Library (optional、Stage 1-2 は num_pretrain_datasets=0 で skip) ---
        self.num_pretrain_datasets = num_pretrain_datasets
        self.num_soft_prompt_tokens = num_soft_prompt_tokens
        if num_pretrain_datasets > 0:
            self.soft_prompt_library = SoftPromptLibrary(
                num_datasets=num_pretrain_datasets,
                num_tokens=num_soft_prompt_tokens,
                hidden_dim=llm_dim,
            )
        else:
            self.soft_prompt_library = None

        # --- Trainable param count (Task 8: wrist_encoder ~12M added) ---
        # 実測: ~541 M (ActionHead ~528M + ProprioProjector + action_queries + WristResNet18 + SoftPromptLibrary)
        # DinoSigLIP ablation path では VisionProjector (~32M) が加わり ~573M になる。
        # 幅広の sanity range (500-660 M) に留め、exact な 577M 仮説には tie しない。
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        print(f"[VLAAdapterGemma4] trainable params: {total_trainable:.3f} M "
              f"(vision_backbone_type={self.vision_backbone_type})")
        assert 500.0 < total_trainable < 660.0, \
            f"trainable params out of expected range: got {total_trainable:.3f} M (expected ~541M baseline)"

    # -----------------------------------------------------------------
    # 便宜 property
    # -----------------------------------------------------------------
    @property
    def llm_dim(self) -> int:
        return self.llm.config.text_config.hidden_size

    @property
    def text_model(self):
        return self.llm.model.language_model

    # -----------------------------------------------------------------
    # Vision encoding (Task 6 rev 3 / 2026-04-22 DinoSigLIP ablation)
    # -----------------------------------------------------------------
    def encode_scene(self, scene_images) -> torch.Tensor:
        """scene image → (B, num_vision_tokens, llm_dim).

        vision_backbone_type に応じて 3 経路を分岐:
          - "gemma4_native": Gemma4ImageProcessor → vision_tower → embed_vision (Task 6 default)
          - "dinosiglip":    DinoSigLIPViTBackbone (DINO+SigLIP concat) → VisionProjector
          - "siglip":        SigLIPViTBackbone only → VisionProjector
        """
        if self.vision_backbone_type == "gemma4_native":
            return self._encode_scene_gemma4_native(scene_images)
        elif self.vision_backbone_type == "dinosiglip":
            return self._encode_scene_dinosiglip(scene_images)
        elif self.vision_backbone_type == "siglip":
            return self._encode_scene_siglip(scene_images)
        else:  # pragma: no cover (guarded in __init__)
            raise ValueError(f"unknown vision_backbone_type={self.vision_backbone_type!r}")

    def _encode_scene_gemma4_native(self, scene_images) -> torch.Tensor:
        """Gemma 4 native vision で scene image → (B, num_vision_tokens, llm_dim).

        Args:
            scene_images: PIL images / numpy / torch tensor。
                tensor の場合は (B, 3, H, W) で [0, 255] float か uint8。
                processor が内部で aspect-ratio-preserving resize + patchify + rescale(÷255) を行う。

        Returns:
            h_v: (B, num_vision_tokens, llm_dim) reshaped pooled tokens.

        NOTE: batch 内すべての画像が同じ num_soft_tokens_per_image を生むことを前提としている
        (fixed-size square input ではこれが保証される)。異なる aspect ratio / size を混ぜた
        batch では pooler_output が不均等長になり assertion 失敗する。その場合は
        out["num_soft_tokens_per_image"] を使って per-row slice が必要 (Task 14 で data
        loader が固定サイズを保証する前提なので、現時点では assertion で十分)。
        """
        device = self.llm.device
        dtype = self.llm.dtype  # bfloat16

        out = self.image_processor.preprocess(scene_images, return_tensors="pt")
        pv = out["pixel_values"].to(device, dtype=dtype)             # (B, max_patches, 768)
        pi = out["image_position_ids"].to(device)                    # (B, max_patches, 2)
        B = pv.shape[0]

        feats = self.llm.model.get_image_features(pv, pi)
        # feats.pooler_output: (N_valid_total, 1536) padding-stripped flat

        n = self.num_vision_tokens
        assert feats.pooler_output.shape[0] == B * n, \
            f"pooler_output flat length {feats.pooler_output.shape[0]} != B*n = {B}*{n}"
        h_v = feats.pooler_output.view(B, n, -1)
        return h_v

    def _dinosiglip_transform_batch(self, scene_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(B, 3, H, W) float [0,255] / uint8 tensor → {"dino": (B,3,224,224), "siglip": (B,3,224,224)}.

        DinoSigLIP の image_transform は PIL Image を要求するため、一度 CPU uint8 PIL に落として
        dino/siglip の 2 transform を適用 → stack で GPU (bf16) に戻す。
        CPU bound なので batch 数が大きいと bottleneck になり得る。User 承知済 (ablation 比較用の
        scene-encoder isolate 目的、速度比較は「PIL 変換 overhead 含む」と明記)。
        """
        from PIL import Image as PILImage

        # tensor を CPU uint8 にする (元が float なら clamp + cast)
        if scene_images.is_floating_point():
            scene_cpu = scene_images.detach().float().clamp(0, 255).to("cpu", dtype=torch.uint8)
        else:
            scene_cpu = scene_images.detach().to("cpu", dtype=torch.uint8)

        dino_list, siglip_list = [], []
        for i in range(scene_cpu.shape[0]):
            arr = scene_cpu[i].permute(1, 2, 0).numpy()   # (H, W, 3) uint8
            pil = PILImage.fromarray(arr)
            d = self.vision_backbone.image_transform(pil)  # {"dino": tensor, "siglip": tensor}
            dino_list.append(d["dino"])
            siglip_list.append(d["siglip"])

        device = self.llm.device
        dtype = self.llm.dtype   # bfloat16
        pv = {
            "dino":   torch.stack(dino_list, dim=0).to(device=device, dtype=dtype),
            "siglip": torch.stack(siglip_list, dim=0).to(device=device, dtype=dtype),
        }
        return pv

    def _encode_scene_dinosiglip(self, scene_images: torch.Tensor) -> torch.Tensor:
        """DinoSigLIP ablation path.

        Args:
            scene_images: (B, 3, H, W) float [0,255] or uint8 tensor (loader 出力そのまま)。

        Returns:
            h_v: (B, 256, llm_dim) projected tokens. DinoSigLIP backbone は frozen、VisionProjector のみ train.
        """
        pv = self._dinosiglip_transform_batch(scene_images)

        # DinoSigLIP backbone は frozen + eval。grad も autograd graph も不要 (projector のみ trainable)。
        with torch.no_grad():
            vision_features = self.vision_backbone(pv)    # (B, 256, 2176) bf16

        # VisionProjector は trainable、grad 流す。
        h_v = self.vision_projector(vision_features)       # (B, 256, llm_dim)
        return h_v

    def _siglip_transform_batch(self, scene_images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float [0,255] / uint8 tensor → (B, 3, 224, 224) SigLIP-normalized bf16 on GPU.

        SigLIPViTBackbone の image_transform (timm 由来 Compose) は PIL Image 入力を期待するため、
        DinoSigLIP 版と同様に一度 CPU uint8 PIL に落として変換。CPU bound、batch 大で bottleneck。
        """
        from PIL import Image as PILImage

        if scene_images.is_floating_point():
            scene_cpu = scene_images.detach().float().clamp(0, 255).to("cpu", dtype=torch.uint8)
        else:
            scene_cpu = scene_images.detach().to("cpu", dtype=torch.uint8)

        out_list = []
        for i in range(scene_cpu.shape[0]):
            arr = scene_cpu[i].permute(1, 2, 0).numpy()   # (H, W, 3) uint8
            pil = PILImage.fromarray(arr)
            t = self.vision_backbone.image_transform(pil)  # (3, 224, 224) tensor
            out_list.append(t)

        device = self.llm.device
        dtype = self.llm.dtype   # bfloat16
        return torch.stack(out_list, dim=0).to(device=device, dtype=dtype)

    def _siglip_transform_batch_gpu(self, scene_images: torch.Tensor) -> torch.Tensor:
        """No-PIL, GPU-native batch transform (SigLIP-So400m @ 224).

        timm の SigLIP transform pipeline:
          Resize(248, bicubic, antialias) → CenterCrop(224) → MaybeToTensor → Normalize(0.5, 0.5)

        を torch.nn.functional で GPU 上バッチ一括実行。PIL round-trip なし、
        per-sample Python loop なし、num_workers=4 parallel な data loader と独立に高速。

        Args:
            scene_images: (B, 3, H, W) float [0,255] or uint8 (loader 出力)。

        Returns:
            (B, 3, 224, 224) bf16 on self.llm.device、timm transform と数値的にほぼ一致。
        """
        import torch.nn.functional as F

        device = self.llm.device
        x = scene_images.to(device=device, dtype=torch.float32)
        # rescale to [0, 1]
        if x.max() > 2.0:   # uint8 or [0,255] float
            x = x / 255.0
        # Resize to 248 (bicubic + antialias、timm default と同仕様)
        x = F.interpolate(x, size=248, mode="bicubic", antialias=True)
        # CenterCrop to 224 (248→224、margin (248-224)/2 = 12)
        x = x[:, :, 12:236, 12:236]
        # Normalize: (x - 0.5) / 0.5 = 2x - 1
        x = 2.0 * x - 1.0
        return x.to(dtype=self.llm.dtype)   # bfloat16

    def _encode_scene_siglip(self, scene_images: torch.Tensor) -> torch.Tensor:
        """SigLIP-only ablation path.

        Args:
            scene_images: (B, 3, H, W) float [0,255] or uint8 tensor。

        Returns:
            h_v: (B, 256, llm_dim) projected tokens. SigLIP backbone frozen、VisionProjector のみ train.
        """
        if getattr(self, "siglip_use_tensor_transform", False):
            pv = self._siglip_transform_batch_gpu(scene_images)  # GPU tensor 直接、no PIL
        else:
            pv = self._siglip_transform_batch(scene_images)      # PIL 経由 (legacy、fair 比較用)

        with torch.no_grad():
            vision_features = self.vision_backbone(pv)     # (B, 256, 1152) bf16

        h_v = self.vision_projector(vision_features)        # (B, 256, llm_dim)
        return h_v

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        pixel_values: Dict[str, torch.Tensor],   # {"scene": (B, 3, H, W) raw or processed, "wrist": (B, 3, 224, 224) bf16}
        # scene: Gemma4ImageProcessor が内部で resize + patchify (encode_scene 経由)
        # wrist: WristResNet18 が (B, 49, llm_dim) に変換 (Task 8 で active)
        input_ids: torch.LongTensor,            # (B, L) placeholder 込み
        proprio: torch.Tensor,                  # (B, proprio_dim) raw
        actions: Optional[torch.Tensor] = None, # (B, 8, 7) or None
        dataset_id: Optional[torch.LongTensor] = None,   # Stage 3 (B,)、soft_prompt_library indexing 用
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L = input_ids.shape
        device = input_ids.device
        llm = self.text_model

        # Stage 3 Soft Prompt 配置 (rev 3 Task 7): X-VLA 原実装に合わせ action head 入力に渡す。
        # library 構築済 AND dataset_id 渡された場合のみ active。
        use_soft_prompt = (self.soft_prompt_library is not None) and (dataset_id is not None)

        # ---- Vision features (Task 6: Gemma 4 native; scene path) ----
        scene_imgs = pixel_values["scene"] if isinstance(pixel_values, dict) else pixel_values
        h_v = self.encode_scene(scene_imgs)                # (B, num_vision_tokens, llm_dim)

        # ---- Wrist features (Task 8: trainable ResNet18, bypasses frozen LLM) ----
        wrist_pixel_values = pixel_values["wrist"]         # (B, 3, 224, 224) bf16 on GPU
        h_w = self.wrist_encoder(wrist_pixel_values)       # (B, 49, llm_dim)

        # ---- PLE 事前計算 (OOM 回避) ----
        with torch.no_grad():
            per_layer_inputs = llm.get_per_layer_inputs(input_ids, None)   # (B, L, num_layers=35, ple_dim=256)
            raw_embeddings = llm.embed_tokens(input_ids)                   # (B, L, llm_dim=1536)
        embeddings = raw_embeddings.clone()

        # ---- Option A: 2 種類の placeholder を両方上書き (original input_ids 空間で) ----
        amask = (input_ids >= ACTION_TOKEN_BEGIN_IDX) & (
            input_ids < ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS
        )
        vmask = (input_ids >= VISION_PLACEHOLDER_BEGIN_IDX) & (
            input_ids < VISION_PLACEHOLDER_BEGIN_IDX + self.num_vision_tokens
        )
        for b in range(B):
            apos = amask[b].nonzero(as_tuple=True)[0]
            vpos = vmask[b].nonzero(as_tuple=True)[0]
            embeddings[b, apos] = self.action_queries.weight
            embeddings[b, vpos] = h_v[b]

        # ---- Soft Prompt は action head 入力に回す (X-VLA 原実装準拠, rev 3 Task 7) ----
        # LLM inputs_embeds への prepend (案 B deviation) は廃止。action head 側 predict_action に h_sp で渡す。
        if use_soft_prompt:
            h_sp = self.soft_prompt_library(dataset_id)   # (B, num_soft_prompt_tokens, llm_dim)
        else:
            h_sp = None

        # ---- attention_mask / position_ids (R6) ----
        L_total = L   # Soft Prompt 相当の offset はもう無い
        attention_mask = torch.ones(B, L_total, dtype=torch.long, device=device)
        position_ids = torch.arange(L_total, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

        # ---- LLM forward (Gemma4TextModel 直接) ----
        # Dual-Track Task 11:
        #   speed mode は LLM call を torch.no_grad() で wrap (autograd graph 作らず、
        #   activation 保存せず、backward 不可)。out.hidden_states は detached で返る。
        #   quality mode は従来通り grad 流す (finetune 側で GC 有効化想定)。
        if self.training_mode == "speed":
            with torch.no_grad():
                out = llm(
                    inputs_embeds=embeddings,
                    per_layer_inputs=per_layer_inputs,
                    use_cache=True,
                    output_hidden_states=True,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
        else:  # quality
            out = llm(
                inputs_embeds=embeddings,
                per_layer_inputs=per_layer_inputs,
                use_cache=True,
                output_hidden_states=True,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        # ---- Action head input 組み立て (entries 0-24) ----
        all_hidden = torch.stack(out.hidden_states, dim=1)              # (B, 36, L_total, llm_dim)
        hidden_subset = all_hidden[:, :25, :, :]                        # (B, 25, L_total, llm_dim)

        # batch 内で placeholder 位置は共通と仮定 (LIBERO は固定 layout)。
        # Soft Prompt を LLM 入力に concat しなくなったため offset 加算は不要 (rev 3 Task 7)。
        apos0 = amask[0].nonzero(as_tuple=True)[0]
        vpos0 = vmask[0].nonzero(as_tuple=True)[0]

        vision_hidden = self.feature_norm(hidden_subset[:, :, vpos0, :])  # (B, 25, num_vision_tokens, llm_dim)
        action_hidden = self.feature_norm(hidden_subset[:, :, apos0, :])  # (B, 25, 64, llm_dim)
        combined = torch.cat([vision_hidden, action_hidden], dim=2)       # (B, 25, N_v+64, llm_dim)

        predicted = self.action_head.predict_action(
            actions_hidden_states=combined,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            phase="Training" if self.training else "Inference",
            h_w=h_w,
            h_sp=h_sp,
        )

        if actions is None:
            return predicted
        loss = F.l1_loss(predicted, actions)
        return predicted, loss
