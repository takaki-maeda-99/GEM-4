import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import ActionHead


class ACTHead(ActionHead):
    """ACT (Action Chunking with Transformers) action head.

    Uses a Transformer decoder with learnable positional queries to predict
    chunk_size future actions. Follows the original ACT paper:
    - Zero-initialized query content + learned positional embeddings
    - Positional embeddings added at every decoder layer
    - L1 loss
    - No causal mask (bidirectional self-attention)

    CVAE is not included (future extension).
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 7,
        chunk_size: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        temporal_ensemble_m: float = 0.01,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.temporal_ensemble_m = temporal_ensemble_m

        # Project backbone features to decoder dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learned positional embeddings for chunk queries (added every layer)
        self.query_pos_embed = nn.Embedding(chunk_size, d_model)

        # Transformer decoder with dropout
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, action_dim)

        # Temporal ensemble state (inference only)
        self._ensemble_buffer: list[Tensor] = []

    def _decode(self, features: Tensor) -> Tensor:
        """Run Transformer decoder with positional query embeddings.

        Following ACT paper: query content is zeros, positional embeddings
        are added at every decoder layer via the tgt input.
        """
        B = features.shape[0]
        device = features.device

        # Project features to decoder dim
        memory = self.input_proj(features)  # [B, N_act, d_model]

        # Query = zeros + positional embedding (ACT paper pattern)
        pos_ids = torch.arange(self.chunk_size, device=device)
        query_pos = self.query_pos_embed(pos_ids)  # [chunk_size, d_model]
        queries = query_pos.unsqueeze(0).expand(B, -1, -1)  # [B, chunk_size, d_model]

        # Decode: queries cross-attend to backbone features
        decoded = self.decoder(tgt=queries, memory=memory)  # [B, chunk_size, d_model]

        # Project to action space
        return self.output_proj(decoded)  # [B, chunk_size, action_dim]

    def compute_loss(self, features: Tensor, actions: Tensor, **kwargs) -> dict:
        pred = self._decode(features)  # [B, chunk_size, action_dim]
        l1_loss = F.l1_loss(pred, actions)
        return {"loss": l1_loss, "l1_loss": l1_loss}

    def predict(self, features: Tensor, **kwargs) -> Tensor:
        pred_chunk = self._decode(features)  # [B, chunk_size, action_dim]

        # Add to ensemble buffer
        self._ensemble_buffer.append(pred_chunk.detach())

        # Keep only last chunk_size predictions
        if len(self._ensemble_buffer) > self.chunk_size:
            self._ensemble_buffer = self._ensemble_buffer[-self.chunk_size:]

        # Temporal ensemble: weighted average of overlapping predictions
        m = self.temporal_ensemble_m
        weighted_sum = torch.zeros_like(pred_chunk[:, 0:1, :])  # [B, 1, action_dim]
        weight_sum = 0.0

        for k, past_pred in enumerate(reversed(self._ensemble_buffer)):
            if k < past_pred.shape[1]:
                w = math.exp(-m * k)
                weighted_sum += w * past_pred[:, k:k + 1, :]
                weight_sum += w

        action = weighted_sum / weight_sum  # [B, 1, action_dim]
        return action

    def reset_ensemble(self):
        """Clear temporal ensemble buffer. Call at episode start."""
        self._ensemble_buffer = []
