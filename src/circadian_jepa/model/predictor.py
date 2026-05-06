from __future__ import annotations

import torch
import torch.nn as nn


class RotationPredictor(nn.Module):
    """Predict teacher embedding by rotating student embedding by angle delta.

    With learn_delta=False and no delta provided, this is identity — correct for
    downsampling and same-batch pairs where delta=0.
    """

    def __init__(self, learn_delta: bool = False) -> None:
        super().__init__()
        self.learn_delta = learn_delta

    @staticmethod
    def rotate(z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply 2-D rotation by per-sample angle delta.

        Parameters
        ----------
        z:     (batch, 2)
        delta: (batch,) angles in radians
        """
        cos_d = torch.cos(delta)  # (batch,)
        sin_d = torch.sin(delta)
        z0 = cos_d * z[:, 0] - sin_d * z[:, 1]
        z1 = sin_d * z[:, 0] + cos_d * z[:, 1]
        return torch.stack([z0, z1], dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.learn_delta or delta is None:
            return z
        return self.rotate(z, delta)
