from __future__ import annotations

import torch
import torch.nn.functional as F


def predictive_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity, mean over batch."""
    sim = F.cosine_similarity(z_pred, z_target, dim=-1, eps=1e-8)
    return (1.0 - sim).mean()


def harmonic_collapse_regularizer(z: torch.Tensor, n_harmonics: int = 2) -> torch.Tensor:
    """Penalise circular concentration across K harmonics.

    For each harmonic k = 1 … n_harmonics, computes R̄_k² where
    R̄_k = ||mean(exp(i·k·θ))|| is the k-th mean resultant length.
    Returns the mean over harmonics so the scale is independent of K.

    - k=1 catches unimodal collapse (all cells at one point).
    - k=2 additionally catches antipodal/bimodal collapse (two opposite poles).
    - k=K catches any K-fold symmetric collapse.

    All R̄_k → 0 for a uniform distribution; → 1 for perfect k-fold collapse.
    """
    theta = torch.atan2(z[..., 1], z[..., 0])
    total = z.new_zeros(())
    for k in range(1, n_harmonics + 1):
        r_bar_k = torch.sqrt(torch.cos(k * theta).mean() ** 2 + torch.sin(k * theta).mean() ** 2)
        total = total + r_bar_k ** 2
    return total / n_harmonics


# backward-compatible alias (K=1 matches the old single-harmonic behaviour)
def collapse_regularizer(z: torch.Tensor) -> torch.Tensor:
    """Legacy alias for harmonic_collapse_regularizer(z, n_harmonics=1)."""
    return harmonic_collapse_regularizer(z, n_harmonics=1)


def amplitude_regularizer(z: torch.Tensor, target_radius: float = 1.0) -> torch.Tensor:
    """Weak L2 penalty keeping ||z|| near target_radius."""
    return ((torch.norm(z, dim=-1) - target_radius) ** 2).mean()


def total_loss(
    out: dict,
    lambda_collapse: float = 1.0,
    lambda_amplitude: float = 0.1,
    n_harmonics: int = 2,
) -> tuple[torch.Tensor, dict]:
    """Combine all loss components.

    Parameters
    ----------
    out:
        Output dict from CircadianJEPA.forward with keys z_a, z_b_target, z_a_pred.
    n_harmonics:
        Number of circular harmonics to penalise in the collapse term (see
        harmonic_collapse_regularizer). Default 2 catches both unimodal and antipodal collapse.

    Returns
    -------
    (loss, components): scalar tensor and dict with per-component float values.
    """
    l_pred = predictive_loss(out["z_a_pred"], out["z_b_target"])
    l_collapse = harmonic_collapse_regularizer(out["z_a"], n_harmonics=n_harmonics)
    l_amplitude = amplitude_regularizer(out["z_a"])
    loss = l_pred + lambda_collapse * l_collapse + lambda_amplitude * l_amplitude
    components = {
        "predict": l_pred.item(),
        "collapse": l_collapse.item(),
        "amplitude": l_amplitude.item(),
        "total": loss.item(),
    }
    return loss, components
