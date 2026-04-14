import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import ActionHead


class MLPHead(ActionHead):
    """MLP action head with separate MSE (position/rotation) and BCE (gripper) losses."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 7,
        chunk_size: int = 1,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.pose_dim = action_dim - 1
        self.gripper_dim = 1

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
        pred_pose = pred[:, :, :self.pose_dim]
        pred_gripper = pred[:, :, self.pose_dim:]
        target_pose = actions[:, :, :self.pose_dim]
        target_gripper = actions[:, :, self.pose_dim:]

        mse_loss = F.mse_loss(pred_pose, target_pose)
        bce_loss = F.binary_cross_entropy_with_logits(pred_gripper, target_gripper)
        loss = mse_loss + bce_loss

        return {"loss": loss, "mse_loss": mse_loss, "bce_loss": bce_loss}

    def predict(self, features: Tensor, **kwargs) -> Tensor:
        pred = self._forward(features)
        pred_pose = pred[:, :, :self.pose_dim]
        pred_gripper = torch.sigmoid(pred[:, :, self.pose_dim:])
        return torch.cat([pred_pose, pred_gripper], dim=-1)

    def _forward(self, features: Tensor) -> Tensor:
        x = features.squeeze(1)
        x = self.mlp(x)
        return x.reshape(x.shape[0], self.chunk_size, self.action_dim)
