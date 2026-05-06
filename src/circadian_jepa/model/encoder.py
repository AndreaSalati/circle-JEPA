from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """MLP encoder mapping R^n_genes → R^embedding_dim (default R^2)."""

    def __init__(
        self,
        n_genes: int,
        hidden_dims: list[int] = [64, 32],
        embedding_dim: int = 2,
        dropout: float = 0.1,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.normalize_output = normalize_output

        dims = [n_genes] + hidden_dims
        layers: list[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.LayerNorm(out_d))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.normalize_output:
            z = nn.functional.normalize(z, dim=-1)
        return z

    def phase(self, z: torch.Tensor) -> torch.Tensor:
        return torch.atan2(z[..., 1], z[..., 0])

    def amplitude(self, z: torch.Tensor) -> torch.Tensor:
        return torch.norm(z, dim=-1)
