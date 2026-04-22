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
    ):
        super().__init__()
        self.llm = gemma_model
        self.llm.config.use_cache = True                    # G1 対策
        assert not getattr(self.llm, "is_gradient_checkpointing", False), \
            "gradient_checkpointing must be off (Stage 1 前提、KV 共有バグ回避)"
        # Freeze ALL of Gemma 4 (vision_tower + embed_vision + language_model + audio_tower + embed_audio)
        for p in self.llm.parameters():
            p.requires_grad = False

        # --- Gemma 4 native vision preprocessor ---
        # Gemma4ImageProcessor は aspect-ratio-preserving resize + patchify + position_ids 生成を担う。
        self.image_processor = Gemma4ImageProcessor(max_soft_tokens=max_soft_tokens)
        self.max_soft_tokens = max_soft_tokens   # retained for introspection / logging (__repr__)

        # num_vision_tokens を実測で決める (224x224 dummy で processor に聞く)
        # 実測: max_soft_tokens=70→64, 140→121, 280→256
        _dummy = torch.zeros(3, 224, 224)   # content irrelevant, only shape matters; avoids RNG contamination
        _out = self.image_processor.preprocess(_dummy, return_tensors="pt")
        self.num_vision_tokens: int = int(_out["num_soft_tokens_per_image"][0])

        llm_dim = self.llm.config.text_config.hidden_size   # Gemma 4 E2B = 1536

        self.proprio_projector = ProprioProjector(proprio_dim=proprio_dim, llm_dim=llm_dim)

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
    # Vision encoding (Task 6 rev 3)
    # -----------------------------------------------------------------
    def encode_scene(self, scene_images) -> torch.Tensor:
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

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        pixel_values: Dict[str, torch.Tensor],
        # pixel_values formats:
        #   If dict: {"scene": (B, 3, H, W), "wrist": (B, 3, H, W)} (Task 14 format; wrist path added in Task 8)
        #   If tensor: (B, 3, H, W) scene-only (legacy smoke path)
        # Processed by self.image_processor (Gemma4ImageProcessor) inside encode_scene().
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

        # ---- Vision features (Task 6: Gemma 4 native; scene only. wrist は Task 8 で扱う) ----
        scene_imgs = pixel_values["scene"] if isinstance(pixel_values, dict) else pixel_values
        h_v = self.encode_scene(scene_imgs)                # (B, num_vision_tokens, llm_dim)

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
            h_w=None,     # Task 8 で wrist feature を計算して差し替える
            h_sp=h_sp,
        )

        if actions is None:
            return predicted
        loss = F.l1_loss(predicted, actions)
        return predicted, loss
