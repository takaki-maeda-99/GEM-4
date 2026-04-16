import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import ActionHead


class ACTHead(ActionHead):
    """ACT (Action Chunking with Transformers) action head.

    Uses a Transformer decoder with learnable chunk queries to predict
    chunk_size future actions. At inference, applies temporal ensemble
    (exponential decay weighted average) over overlapping predictions.

    Always uses MSE loss on all action dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 7,
        chunk_size: int = 20,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        temporal_ensemble_m: float = 0.01,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.temporal_ensemble_m = temporal_ensemble_m

        self.input_proj = nn.Linear(input_dim, d_model)

        self.chunk_queries = nn.Parameter(
            torch.randn(chunk_size, d_model) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, action_dim)

        self._ensemble_buffer: list[Tensor] = []

    def _decode(self, features: Tensor) -> Tensor:
        B = features.shape[0]
        memory = self.input_proj(features)
        queries = self.chunk_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.output_proj(decoded)

    def compute_loss(self, features: Tensor, actions: Tensor, **kwargs) -> dict:
        pred = self._decode(features)
        l1_loss = F.l1_loss(pred, actions)
        return {"loss": l1_loss, "l1_loss": l1_loss}

    def predict(self, features: Tensor, **kwargs) -> Tensor:
        pred_chunk = self._decode(features)

        self._ensemble_buffer.append(pred_chunk.detach())

        if len(self._ensemble_buffer) > self.chunk_size:
            self._ensemble_buffer = self._ensemble_buffer[-self.chunk_size:]

        m = self.temporal_ensemble_m
        weighted_sum = torch.zeros_like(pred_chunk[:, 0:1, :])
        weight_sum = 0.0

        for k, past_pred in enumerate(reversed(self._ensemble_buffer)):
            if k < past_pred.shape[1]:
                w = math.exp(-m * k)
                weighted_sum += w * past_pred[:, k:k+1, :]
                weight_sum += w

        action = weighted_sum / weight_sum
        return action

    def reset_ensemble(self):
        self._ensemble_buffer = []
