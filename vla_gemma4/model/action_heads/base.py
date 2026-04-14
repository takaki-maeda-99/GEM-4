from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class ActionHead(nn.Module, ABC):
    """Base class for all action heads.

    All action heads receive features from the LLM backbone and
    produce action predictions. They encapsulate their own loss
    computation so head-specific logic (KL, VQ, denoising) stays
    internal.
    """

    @abstractmethod
    def compute_loss(self, features: Tensor, actions: Tensor, **kwargs) -> dict:
        """Compute training loss.

        Args:
            features: [B, N_act, D] features from backbone [ACT] tokens.
            actions: [B, T, action_dim] ground-truth actions.

        Returns:
            Dict with at least "loss" key (scalar Tensor), plus any metrics.
        """

    @abstractmethod
    def predict(self, features: Tensor, **kwargs) -> Tensor:
        """Predict actions at inference time.

        Args:
            features: [B, N_act, D] features from backbone [ACT] tokens.

        Returns:
            [B, T, action_dim] predicted actions.
        """
