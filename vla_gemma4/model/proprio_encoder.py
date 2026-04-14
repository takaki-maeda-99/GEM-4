from torch import Tensor, nn


class ProprioEncoder(nn.Module):
    """Encodes proprioception (EEF pose + gripper) into a single token embedding."""

    def __init__(self, proprio_dim: int = 7, hidden_dim: int = 1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, proprio: Tensor) -> Tensor:
        """
        Args:
            proprio: [B, proprio_dim]
        Returns:
            [B, 1, hidden_dim] — single token embedding
        """
        return self.mlp(proprio).unsqueeze(1)
