from __future__ import annotations

import math

import numpy as np
import scipy.stats


def fit_cosinor(expression: np.ndarray, phase: np.ndarray) -> dict:
    """Fit a cosinor model to each gene: expr = a + b*cos(phase) + c*sin(phase).

    Parameters
    ----------
    expression : (n_cells, n_genes) array or (n_cells,) for a single gene
    phase      : (n_cells,) array of phase values in radians

    Returns
    -------
    dict mapping gene index (int) or 0 to per-gene result dicts with keys:
        amplitude    : float — sqrt(b² + c²)
        phase_offset : float — peak phase in [0, 2π), = atan2(c, b)
        baseline     : float — mean expression intercept
        r_squared    : float — coefficient of determination
        p_value      : float — F-test p-value against intercept-only null
    """
    expression = np.asarray(expression, dtype=float)
    phase = np.asarray(phase, dtype=float)

    if expression.ndim == 1:
        expression = expression[:, np.newaxis]

    n, n_genes = expression.shape
    X = np.column_stack([np.ones(n), np.cos(phase), np.sin(phase)])  # (n, 3)

    results: dict = {}
    for g in range(n_genes):
        y = expression[:, g]
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a, b, c = coeffs

        y_pred = X @ coeffs
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # F-test: 2 oscillatory parameters (b, c) vs intercept-only null
        k = 2
        df_model = k
        df_resid = max(n - k - 1, 1)
        if ss_tot > 0 and (1.0 - r2) > 0:
            f_stat = (r2 / df_model) / ((1.0 - r2) / df_resid)
            p_value = float(1.0 - scipy.stats.f.cdf(f_stat, df_model, df_resid))
        else:
            p_value = 1.0

        results[g] = {
            "amplitude": float(math.sqrt(b**2 + c**2)),
            "phase_offset": float(math.atan2(c, b) % (2 * math.pi)),
            "baseline": float(a),
            "r_squared": float(r2),
            "p_value": p_value,
        }

    return results


def fit_cosinor_named(expression: np.ndarray, phase: np.ndarray, gene_names: list[str]) -> dict:
    """Wrapper that returns results keyed by gene name instead of integer index."""
    by_idx = fit_cosinor(expression, phase)
    return {gene_names[i]: v for i, v in by_idx.items()}


# Known circadian phase relationships (human gene names)
KNOWN_PHASE_PAIRS: list[tuple[str, str, float]] = [
    ("BMAL1", "PER2", math.pi),
    ("BMAL1", "NR1D1", math.pi),
    ("ARNTL", "PER2", math.pi),   # ARNTL is the alias for BMAL1
    ("ARNTL", "NR1D1", math.pi),
]

# Mouse equivalents
KNOWN_PHASE_PAIRS_MOUSE: list[tuple[str, str, float]] = [
    ("Bmal1", "Per2", math.pi),
    ("Bmal1", "Nr1d1", math.pi),
    ("Arntl", "Per2", math.pi),
    ("Arntl", "Nr1d1", math.pi),
]


def check_known_phase_relationships(
    cosinor_results: dict,
    gene_pairs: list[tuple[str, str, float]] | None = None,
) -> dict:
    """Compare observed phase offsets between gene pairs against expected biology.

    Parameters
    ----------
    cosinor_results :
        Dict mapping gene name → fit_cosinor result dict (with 'phase_offset').
    gene_pairs :
        List of (gene_a, gene_b, expected_phase_diff_radians). If None, uses
        KNOWN_PHASE_PAIRS then KNOWN_PHASE_PAIRS_MOUSE, filtered to present genes.

    Returns
    -------
    Dict mapping (gene_a, gene_b) → {"observed_diff", "expected_diff", "error"}
    """
    if gene_pairs is None:
        all_pairs = KNOWN_PHASE_PAIRS + KNOWN_PHASE_PAIRS_MOUSE
        gene_pairs = [
            p for p in all_pairs if p[0] in cosinor_results and p[1] in cosinor_results
        ]

    results: dict = {}
    for gene_a, gene_b, expected_diff in gene_pairs:
        if gene_a not in cosinor_results or gene_b not in cosinor_results:
            continue
        phi_a = cosinor_results[gene_a]["phase_offset"]
        phi_b = cosinor_results[gene_b]["phase_offset"]
        observed_diff = (phi_b - phi_a) % (2 * math.pi)
        # Wrap error to [-π, π]
        error = ((observed_diff - expected_diff) + math.pi) % (2 * math.pi) - math.pi
        results[(gene_a, gene_b)] = {
            "observed_diff": observed_diff,
            "expected_diff": expected_diff,
            "error": float(error),
        }
    return results
