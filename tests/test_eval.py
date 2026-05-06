"""Tests for Phase 5: evaluation and validation utilities."""

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from circadian_jepa.data.dataset import CircadianDataset
from circadian_jepa.data.synthetic import make_synthetic_circadian, make_synthetic_sinusoidal
from circadian_jepa.data.views import ViewGenerator
from circadian_jepa.eval.circular import (
    align_phase,
    circular_correlation,
    circular_distance,
)
from circadian_jepa.eval.cosinor import (
    check_known_phase_relationships,
    fit_cosinor,
    fit_cosinor_named,
)
from circadian_jepa.eval.inference import infer_phase
from circadian_jepa.model.jepa import CircadianJEPA
from circadian_jepa.training.trainer import Trainer


# ── circular metrics ─────────────────────────────────────────────────────────


def test_circular_correlation_identical():
    theta = np.linspace(0.01, 2 * math.pi - 0.01, 100)  # avoid perfect uniformity
    corr = circular_correlation(theta, theta)
    assert abs(corr - 1.0) < 1e-10


def test_circular_correlation_constant_offset():
    theta = np.linspace(0.01, 2 * math.pi - 0.01, 100)
    corr = circular_correlation(theta, theta + 0.5)
    assert corr > 0.99, f"Expected near 1 for constant offset, got {corr:.6f}"


def test_circular_correlation_antipodal():
    theta = np.linspace(0.01, 2 * math.pi - 0.01, 100)
    corr = circular_correlation(theta, theta + math.pi)
    assert corr > 0.5, f"Expected positive correlation for antipodal, got {corr:.4f}"


def test_circular_correlation_random_near_zero():
    rng = np.random.default_rng(0)
    theta_a = rng.uniform(0, 2 * math.pi, 500)
    theta_b = rng.uniform(0, 2 * math.pi, 500)
    corr = circular_correlation(theta_a, theta_b)
    assert abs(corr) < 0.15, f"Expected ~0 for independent random, got {corr:.4f}"


def test_circular_correlation_small_input():
    assert abs(circular_correlation(np.array([0.0]), np.array([0.0]))) < 1e-10


def test_circular_distance_zero():
    d = circular_distance(np.array([1.0]), np.array([1.0]))
    assert d[0] < 1e-10


def test_circular_distance_quarter():
    d = circular_distance(np.array([0.0]), np.array([math.pi / 2]))
    assert abs(d[0] - math.pi / 2) < 1e-10


def test_circular_distance_wrapping():
    d = circular_distance(np.array([0.01]), np.array([2 * math.pi - 0.01]))
    assert d[0] < 0.03, "Should wrap around the circle"


# ── align_phase ──────────────────────────────────────────────────────────────


def test_align_phase_pure_rotation():
    rng = np.random.default_rng(42)
    theta_true = rng.uniform(0, 2 * math.pi, 100)
    offset = 1.23
    theta_pred = (theta_true + offset) % (2 * math.pi)
    aligned, found_offset, sign = align_phase(theta_pred, theta_true)
    assert sign == 1, "Should not need reflection"
    assert abs(found_offset) > 0, "Should find a non-zero offset"
    assert circular_correlation(aligned, theta_true) > 0.99


def test_align_phase_reflection():
    rng = np.random.default_rng(42)
    theta_true = rng.uniform(0, 2 * math.pi, 100)
    theta_pred = (-theta_true) % (2 * math.pi)
    aligned, offset, sign = align_phase(theta_pred, theta_true)
    # Either sign=+1 with rotation or sign=-1 with reflection works;
    # either way aligned must match theta_true
    assert circular_correlation(aligned, theta_true) > 0.99, (
        f"aligned phase should match true phase (sign={sign}, offset={offset:.4f})"
    )


# ── cosinor ──────────────────────────────────────────────────────────────────


def test_fit_cosinor_single_gene():
    phase = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    expr = 5.0 + 2.0 * np.cos(phase - 1.0) + np.random.randn(200) * 0.05
    result = fit_cosinor(expr, phase)
    assert 0 in result
    r = result[0]
    assert abs(r["amplitude"] - 2.0) < 0.2
    assert abs(r["phase_offset"] - 1.0) < 0.1
    assert abs(r["baseline"] - 5.0) < 0.2
    assert r["r_squared"] > 0.9


def test_fit_cosinor_multi_gene():
    phase = np.linspace(0, 2 * math.pi, 100, endpoint=False)
    expr = np.column_stack([
        3.0 + 1.0 * np.cos(phase - 0.0),
        3.0 + 1.0 * np.cos(phase - math.pi),
    ])
    results = fit_cosinor(expr, phase)
    assert len(results) == 2
    # Guard against 2π-wrapping from tiny negative atan2 due to numerical noise
    def _wrap(x):
        return x if x < math.pi else x - 2 * math.pi
    assert abs(_wrap(results[0]["phase_offset"]) - 0.0) < 0.1, (
        f"Expected phase_offset ≈ 0, got {results[0]['phase_offset']}"
    )
    assert abs(results[1]["phase_offset"] - math.pi) < 0.1


def test_fit_cosinor_named():
    phase = np.linspace(0, 2 * math.pi, 50, endpoint=False)
    expr = 2.0 * np.cos(phase)
    results = fit_cosinor_named(expr[:, None], phase, ["GeneA"])
    assert "GeneA" in results


def test_check_known_phase_relationships():
    # Create cosinor results matching BMAL1->PER2 antiphase
    cosinor_results = {
        "BMAL1": {"phase_offset": 0.0, "amplitude": 1.0, "baseline": 0.0, "r_squared": 0.9, "p_value": 0.01},
        "PER2": {"phase_offset": math.pi, "amplitude": 1.0, "baseline": 0.0, "r_squared": 0.9, "p_value": 0.01},
    }
    results = check_known_phase_relationships(cosinor_results)
    key = ("BMAL1", "PER2")
    assert key in results
    assert abs(results[key]["error"]) < 0.1


def test_check_known_phase_relationships_default_pairs():
    cosinor_results = {
        "BMAL1": {"phase_offset": 0.0, "amplitude": 1.0, "baseline": 0.0, "r_squared": 0.9, "p_value": 0.01},
        "PER2": {"phase_offset": math.pi, "amplitude": 1.0, "baseline": 0.0, "r_squared": 0.9, "p_value": 0.01},
    }
    results = check_known_phase_relationships(cosinor_results, gene_pairs=None)
    assert ("BMAL1", "PER2") in results


# ── inference shape sanity ────────────────────────────────────────────────────


def test_infer_phase_output_keys():
    """infer_phase returns an AnnData copy with expected .obs columns."""
    torch.manual_seed(42)
    model = CircadianJEPA(n_genes=15)
    adata = make_synthetic_circadian(n_cells=30, seed=42)
    gene_list = list(adata.var_names)
    result = infer_phase(model, adata, gene_list, device="cpu")
    assert "inferred_phase" in result.obs
    assert "inferred_amplitude" in result.obs
    assert "z_0" in result.obs
    assert "z_1" in result.obs
    assert len(result.obs["inferred_phase"]) == 30


# ── integration: train + evaluate on synthetic data ──────────────────────────


@pytest.mark.slow
def test_full_pipeline_circular_correlation():
    """Train on clean synthetic data, infer phase, check circular correlation > 0.7.

    Uses make_synthetic_sinusoidal with seq_depth=30000 (≈ 20k median UMIs/cell)
    which provides a strong circadian signal while retaining realistic count noise.
    """
    torch.manual_seed(42)
    n_cells = 720
    n_timepoints = 6

    adata = make_synthetic_sinusoidal(
        n_cells=n_cells, n_genes=15, n_timepoints=n_timepoints,
        seq_depth=30000, noise=0.2, dropout_rate=0.3, seed=42,
    )
    n_genes = adata.n_vars
    gene_list = list(adata.var_names)

    vg = ViewGenerator(view_mode="asymmetric", seed=42)
    dataset = CircadianDataset(adata, view_generator=vg)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=0
    )

    model = CircadianJEPA(n_genes=n_genes, ema_momentum=0.95)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, optimizer=optimizer, device="cpu", log_every=9999)
    trainer.fit(loader, n_epochs=50, lambda_collapse=1.0, lambda_amplitude=0.1)

    result = infer_phase(model, adata, gene_list, device="cpu")
    inferred = result.obs["inferred_phase"].values
    true = result.obs["true_phase"].values

    # Align inferred phase to true phase before computing correlation
    aligned, _, _ = align_phase(inferred, true)
    corr = circular_correlation(aligned, true)
    assert corr > 0.7, (
        f"Circular correlation {corr:.4f} < 0.7 — model did not learn phase"
    )
