from __future__ import annotations

import torch
import torch.nn.functional as F


def predictive_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity, mean over batch."""
    sim = F.cosine_similarity(z_pred, z_target, dim=-1, eps=1e-8)
    return (1.0 - sim).mean()


def collapse_regularizer(z: torch.Tensor) -> torch.Tensor:
    """Penalise circular concentration: R_bar^2, where R_bar is the mean resultant length.

    R_bar → 0 means phases are spread uniformly; R_bar → 1 means all collapsed.
    """
    theta = torch.atan2(z[..., 1], z[..., 0])
    r_bar = torch.sqrt(theta.cos().mean() ** 2 + theta.sin().mean() ** 2)
    return r_bar ** 2


def amplitude_regularizer(z: torch.Tensor, target_radius: float = 1.0) -> torch.Tensor:
    """Weak L2 penalty keeping ||z|| near target_radius."""
    return ((torch.norm(z, dim=-1) - target_radius) ** 2).mean()


def total_loss(
    out: dict,
    lambda_collapse: float = 1.0,
    lambda_amplitude: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Combine all loss components.

    Parameters
    ----------
    out:
        Output dict from CircadianJEPA.forward with keys z_a, z_b_target, z_a_pred.

    Returns
    -------
    (loss, components): scalar tensor and dict with per-component float values.
    """
    l_pred = predictive_loss(out["z_a_pred"], out["z_b_target"])
    l_collapse = collapse_regularizer(out["z_a"])
    l_amplitude = amplitude_regularizer(out["z_a"])
    loss = l_pred + lambda_collapse * l_collapse + lambda_amplitude * l_amplitude
    components = {
        "predict": l_pred.item(),
        "collapse": l_collapse.item(),
        "amplitude": l_amplitude.item(),
        "total": loss.item(),
    }
    return loss, components
