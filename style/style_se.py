import torch
import torch.nn as nn


class StyleSE(nn.Module):
    def __init__(self, embedding_dim: int = 512, reduction: int = 16):
        super().__init__()
        hidden = max(1, embedding_dim // reduction)
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embedding_dim),
            nn.Sigmoid(),
        )

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x) + x


__all__ = ["StyleSE"]
