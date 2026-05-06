from __future__ import annotations

import torch
import torch.nn as nn

from .ema import EMATeacher
from .encoder import Encoder
from .predictor import RotationPredictor


class CircadianJEPA(nn.Module):
    """Full JEPA model: student encoder + EMA teacher + rotation predictor."""

    def __init__(
        self,
        n_genes: int,
        hidden_dims: list[int] = [64, 32],
        embedding_dim: int = 2,
        ema_momentum: float = 0.996,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        self.student_encoder = Encoder(
            n_genes=n_genes,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            normalize_output=normalize_output,
        )
        self.ema = EMATeacher(self.student_encoder, momentum=ema_momentum)
        self.predictor = RotationPredictor(learn_delta=False)

    def forward(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_a = self.student_encoder(view_a)
        z_b_target = self.ema.forward(view_b)
        z_a_pred = self.predictor(z_a, delta)
        return {"z_a": z_a, "z_b_target": z_b_target, "z_a_pred": z_a_pred}

    def step_ema(self) -> None:
        self.ema.update(self.student_encoder)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.student_encoder(x)
