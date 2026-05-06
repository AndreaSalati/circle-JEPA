from __future__ import annotations

import numpy as np


def circular_correlation(theta_pred: np.ndarray, theta_true: np.ndarray) -> float:
    """Jammalamadaka-Sarma circular correlation coefficient.

    Invariant to independent constant offsets in each variable. Range [-1, 1];
    sign indicates whether pred and true co-vary in the same or opposite direction.
    """
    theta_pred = np.asarray(theta_pred, dtype=float)
    theta_true = np.asarray(theta_true, dtype=float)

    mu_pred = np.arctan2(np.sin(theta_pred).mean(), np.cos(theta_pred).mean())
    mu_true = np.arctan2(np.sin(theta_true).mean(), np.cos(theta_true).mean())

    dp = np.sin(theta_pred - mu_pred)
    dt = np.sin(theta_true - mu_true)

    num = (dp * dt).sum()
    den = np.sqrt((dp**2).sum() * (dt**2).sum())
    if den == 0.0:
        return 0.0
    return float(num / den)


def circular_distance(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    """Wrapped angular distance: min(|a-b|, 2π - |a-b|)."""
    theta_a = np.asarray(theta_a, dtype=float)
    theta_b = np.asarray(theta_b, dtype=float)
    diff = np.abs(theta_a - theta_b) % (2 * np.pi)
    return np.minimum(diff, 2 * np.pi - diff)


def _mean_angle(angles: np.ndarray) -> float:
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def align_phase(
    theta_pred: np.ndarray,
    theta_true: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Find best rigid alignment (rotation + optional reflection) of theta_pred to theta_true.

    Tries both sign=+1 (pure rotation) and sign=-1 (reflection then rotation),
    picks whichever gives the smaller mean circular distance.

    Returns
    -------
    aligned : ndarray of same shape as theta_pred, in [0, 2π)
    offset  : float, optimal rotation angle added after sign flip
    sign    : +1 or -1
    """
    theta_pred = np.asarray(theta_pred, dtype=float)
    theta_true = np.asarray(theta_true, dtype=float)

    best_aligned, best_offset, best_sign, best_err = None, 0.0, 1, np.inf

    for sign in (1, -1):
        flipped = sign * theta_pred
        offset = _mean_angle(theta_true - flipped)
        aligned = (flipped + offset) % (2 * np.pi)
        err = circular_distance(aligned, theta_true).mean()
        if err < best_err:
            best_err = err
            best_aligned = aligned
            best_offset = offset
            best_sign = sign

    return best_aligned, best_offset, best_sign
