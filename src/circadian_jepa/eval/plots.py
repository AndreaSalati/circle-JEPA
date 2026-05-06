from __future__ import annotations

import math

import anndata
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .cosinor import fit_cosinor


def plot_embedding(
    z: np.ndarray,
    color_by: np.ndarray | None = None,
    ax: Axes | None = None,
    cmap: str = "hsv",
    s: float = 8.0,
    alpha: float = 0.7,
) -> Axes:
    """Scatter plot of the 2D planar embedding with an optional unit-circle reference.

    Parameters
    ----------
    z        : (n_cells, 2) embedding array
    color_by : (n_cells,) values for colour mapping (e.g. true_phase)
    ax       : existing Axes to draw into; new figure/axes created if None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    z = np.asarray(z)
    theta_circle = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta_circle), np.sin(theta_circle), "k-", lw=0.8, alpha=0.4)

    scatter_kw: dict = {"s": s, "alpha": alpha, "linewidths": 0}
    if color_by is not None:
        sc = ax.scatter(z[:, 0], z[:, 1], c=color_by, cmap=cmap, **scatter_kw)
        plt.colorbar(sc, ax=ax, label="phase (rad)")
    else:
        ax.scatter(z[:, 0], z[:, 1], **scatter_kw)

    ax.set_aspect("equal")
    ax.set_xlabel("z₀")
    ax.set_ylabel("z₁")
    ax.set_title("2D embedding")
    return ax


def plot_phase_vs_truth(
    theta_pred: np.ndarray,
    theta_true: np.ndarray,
    ax: Axes | None = None,
    s: float = 8.0,
    alpha: float = 0.5,
) -> Axes:
    """Scatter inferred phase against true phase. Perfect recovery → diagonal.

    Both axes are in [0, 2π].
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    theta_pred = np.asarray(theta_pred) % (2 * np.pi)
    theta_true = np.asarray(theta_true) % (2 * np.pi)

    ax.scatter(theta_true, theta_pred, s=s, alpha=alpha, linewidths=0)
    ax.plot([0, 2 * math.pi], [0, 2 * math.pi], "r--", lw=1.0, label="y=x")
    ax.set_xlim(0, 2 * math.pi)
    ax.set_ylim(0, 2 * math.pi)
    ax.set_xlabel("true phase (rad)")
    ax.set_ylabel("inferred phase (rad)")
    ax.set_title("Inferred vs true phase")
    ax.legend(fontsize=8)
    return ax


def plot_gene_rhythm(
    adata: anndata.AnnData,
    gene: str,
    phase_key: str = "inferred_phase",
    ax: Axes | None = None,
    s: float = 8.0,
    alpha: float = 0.5,
    n_curve: int = 200,
) -> Axes:
    """Scatter one gene's expression against inferred phase with a fitted cosine overlay.

    Parameters
    ----------
    adata     : AnnData with phase in .obs[phase_key]
    gene      : gene name present in adata.var_names
    phase_key : column in .obs to use as x-axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names.")
    if phase_key not in adata.obs:
        raise ValueError(f"Phase key '{phase_key}' not found in adata.obs.")

    phase = np.asarray(adata.obs[phase_key], dtype=float)
    g_idx = adata.var_names.get_loc(gene)
    expr = adata.X[:, g_idx]
    expr = expr.toarray().ravel() if hasattr(expr, "toarray") else np.asarray(expr).ravel()

    ax.scatter(phase, expr, s=s, alpha=alpha, linewidths=0, label="data")

    # Fit cosinor and overlay
    fit = fit_cosinor(expr, phase)[0]
    phi_range = np.linspace(0, 2 * np.pi, n_curve)
    fitted = (
        fit["baseline"]
        + fit["amplitude"] * np.cos(phi_range - fit["phase_offset"])
    )
    ax.plot(phi_range, fitted, "r-", lw=1.5, label=f"cosinor (R²={fit['r_squared']:.2f})")

    ax.set_xlabel(f"{phase_key} (rad)")
    ax.set_ylabel("expression")
    ax.set_title(f"{gene} rhythm")
    ax.legend(fontsize=8)
    return ax
