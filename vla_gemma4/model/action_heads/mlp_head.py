import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import ActionHead


class MLPHead(ActionHead):
    """MLP action head with configurable loss.

    When gripper_dim is set (default: None), the last dimension uses BCE loss
    with sigmoid output. Otherwise, all dimensions use MSE loss.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 7,
        chunk_size: int = 1,
        hidden_dims: list[int] | None = None,
        gripper_as_binary: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.gripper_as_binary = gripper_as_binary

        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_loss(self, features: Tensor, actions: Tensor, **kwargs) -> dict:
        pred = self._forward(features)

        if self.gripper_as_binary:
            # Split: all dims except last = MSE, last dim = BCE
            pose_dim = self.action_dim - 1
            mse_loss = F.mse_loss(pred[:, :, :pose_dim], actions[:, :, :pose_dim])
            bce_loss = F.binary_cross_entropy_with_logits(
                pred[:, :, pose_dim:], actions[:, :, pose_dim:]
            )
            loss = mse_loss + bce_loss
            return {"loss": loss, "mse_loss": mse_loss, "bce_loss": bce_loss}
        else:
            # All dimensions: MSE
            mse_loss = F.mse_loss(pred, actions)
            return {"loss": mse_loss, "mse_loss": mse_loss}

    def predict(self, features: Tensor, **kwargs) -> Tensor:
        pred = self._forward(features)

        if self.gripper_as_binary:
            pose_dim = self.action_dim - 1
            pred_pose = pred[:, :, :pose_dim]
            pred_gripper = torch.sigmoid(pred[:, :, pose_dim:])
            return torch.cat([pred_pose, pred_gripper], dim=-1)
        else:
            return pred

    def _forward(self, features: Tensor) -> Tensor:
        x = features.squeeze(1)
        x = self.mlp(x)
        return x.reshape(x.shape[0], self.chunk_size, self.action_dim)
