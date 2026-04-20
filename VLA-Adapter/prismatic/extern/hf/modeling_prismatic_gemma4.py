"""
modeling_prismatic_gemma4.py

VLA-Adapter backbone を Gemma 4 E2B に置き換えた統合 nn.Module。
Stage 1 (Phase 1b.6) で作成。

構成:
  - LLM: Gemma4ForConditionalGeneration (frozen)
  - Vision backbone: DinoSigLIPViTBackbone (frozen)
  - VisionProjector: vision_embed_dim → 8192 → llm_dim → llm_dim (trainable)
  - ProprioProjector: proprio_dim → llm_dim → llm_dim (trainable)
  - action_queries: Embedding(64, llm_dim), zero init (trainable)
  - feature_norm: Identity (1b.1 判定)、LayerNorm 差し替え可
  - action_head: L1RegressionActionHead(use_pro_version=True, ~640M, trainable)

forward pattern (1b.3-1b.5 で検証済み):
  - input_ids: placeholder ID 含む系列 (vision 512 + action 64 + proprio 1 + bos/eos/prompt)
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

from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.vision.dinosiglip_vit import DinoSigLIPViTBackbone
from prismatic.vla.constants_gemma4 import (
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTION_TOKENS,
    NUM_VISION_TOKENS,
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


class VisionProjector(nn.Module):
    """3 層 MLP: vision_embed_dim → initial_projection_dim → llm_dim → llm_dim."""

    def __init__(self, vision_dim: int, llm_dim: int, initial_projection_dim: int = 8192):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, initial_projection_dim, bias=True)
        self.fc2 = nn.Linear(initial_projection_dim, llm_dim, bias=True)
        self.fc3 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


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
        vision_backbone: DinoSigLIPViTBackbone, # 実 DINO+SigLIP instance
        feature_norm: Optional[nn.Module] = None,
        proprio_dim: int = 8,
        action_dim: int = 7,
        num_action_chunks: int = 8,
        initial_projection_dim: int = 8192,
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
        for p in self.llm.parameters():
            p.requires_grad = False

        self.vision_backbone = vision_backbone
        for p in self.vision_backbone.parameters():
            p.requires_grad = False
        self.vision_backbone.eval()

        llm_dim = self.llm.config.text_config.hidden_size
        vision_dim = self.vision_backbone.embed_dim        # dinosiglip: DINO + SigLIP concat

        self.vision_projector = VisionProjector(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            initial_projection_dim=initial_projection_dim,
        )
        self.proprio_projector = ProprioProjector(proprio_dim=proprio_dim, llm_dim=llm_dim)

        self.action_queries = nn.Embedding(NUM_ACTION_TOKENS, llm_dim)
        self.action_queries.weight.data.zero_()

        self.feature_norm = feature_norm if feature_norm is not None else nn.Identity()

        self.action_head = L1RegressionActionHead(
            input_dim=llm_dim,
            hidden_dim=llm_dim,
            action_dim=action_dim,
            num_task_tokens=NUM_VISION_TOKENS,
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
    def vision_dim(self) -> int:
        return self.vision_backbone.embed_dim

    @property
    def text_model(self):
        return self.llm.model.language_model

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        pixel_values: Dict[str, torch.Tensor],  # {"dino": (B, T, 3, H, W), "siglip": (B, T, 3, H, W)}
        input_ids: torch.LongTensor,            # (B, L) placeholder 込み
        proprio: torch.Tensor,                  # (B, proprio_dim) raw
        actions: Optional[torch.Tensor] = None, # (B, 8, 7) or None
        dataset_id: Optional[torch.LongTensor] = None,   # Stage 3 (B,)、soft_prompt_library indexing 用
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L = input_ids.shape
        device = input_ids.device
        llm = self.text_model

        # Stage 3 Soft Prompt 配置 (案 B) 判定: library 構築済 AND dataset_id 渡された場合のみ active
        use_soft_prompt = (self.soft_prompt_library is not None) and (dataset_id is not None)
        num_sp = self.num_soft_prompt_tokens if use_soft_prompt else 0

        # ---- Vision features ----
        # vision_backbone is frozen + eval, 活性化の勾配は不要 (projector の input として使用のみ)
        vision_features = self.vision_backbone(pixel_values)           # (B, 512, vision_dim)
        vision_projected = self.vision_projector(vision_features)       # (B, 512, llm_dim)

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
            input_ids < VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS
        )
        for b in range(B):
            apos = amask[b].nonzero(as_tuple=True)[0]
            vpos = vmask[b].nonzero(as_tuple=True)[0]
            embeddings[b, apos] = self.action_queries.weight
            embeddings[b, vpos] = vision_projected[b]

        # ---- Stage 3 (optional): Soft Prompt を inputs_embeds 前段に concat (案 B) ----
        # X-VLA 原実装は action head 内 transformer で末尾 concat、本 plan §R21 は LLM 前段 (案 B) で deviation。
        # migration_log 記録済。
        if use_soft_prompt:
            soft_prompts = self.soft_prompt_library(dataset_id)             # (B, num_sp, llm_dim)
            embeddings = torch.cat([soft_prompts, embeddings], dim=1)        # (B, num_sp + L, llm_dim)
            # per_layer_inputs は soft_prompt 位置には contribution なし → zero-pad (B, num_sp, num_layers, ple_dim)
            zero_ple = torch.zeros(
                B, num_sp, per_layer_inputs.size(2), per_layer_inputs.size(3),
                device=per_layer_inputs.device, dtype=per_layer_inputs.dtype,
            )
            per_layer_inputs = torch.cat([zero_ple, per_layer_inputs], dim=1)  # (B, num_sp + L, 35, 256)

        # ---- attention_mask / position_ids (R6) — extended length L' = num_sp + L ----
        L_total = num_sp + L
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

        # batch 内で placeholder 位置は共通と仮定 (LIBERO は固定 layout)、Soft Prompt あれば offset +num_sp
        apos0 = amask[0].nonzero(as_tuple=True)[0] + num_sp   # original position を extended space に shift
        vpos0 = vmask[0].nonzero(as_tuple=True)[0] + num_sp

        vision_hidden = self.feature_norm(hidden_subset[:, :, vpos0, :])  # (B, 25, 512, llm_dim)
        action_hidden = self.feature_norm(hidden_subset[:, :, apos0, :])  # (B, 25, 64, llm_dim)
        combined = torch.cat([vision_hidden, action_hidden], dim=2)       # (B, 25, 576, llm_dim)

        predicted = self.action_head.predict_action(
            actions_hidden_states=combined,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            phase="Training" if self.training else "Inference",
        )

        if actions is None:
            return predicted
        loss = F.l1_loss(predicted, actions)
        return predicted, loss
