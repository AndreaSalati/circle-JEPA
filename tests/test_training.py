from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader

from circadian_jepa.data.dataset import CircadianDataset
from circadian_jepa.data.synthetic import make_synthetic_circadian
from circadian_jepa.data.views import ViewGenerator
from circadian_jepa.model.jepa import CircadianJEPA
from circadian_jepa.training.config import TrainConfig
from circadian_jepa.training.losses import (
    amplitude_regularizer,
    collapse_regularizer,
    predictive_loss,
    total_loss,
)
from circadian_jepa.training.run import train_from_config
from circadian_jepa.training.trainer import Trainer


def _rand_z(n: int = 32, dim: int = 2) -> torch.Tensor:
    return torch.randn(n, dim)


# ── unit tests: individual loss components ──────────────────────────────────

def test_predictive_loss_scalar_nonneg():
    loss = predictive_loss(_rand_z(), _rand_z())
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_collapse_regularizer_scalar_nonneg():
    loss = collapse_regularizer(_rand_z())
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_amplitude_regularizer_scalar_nonneg():
    loss = amplitude_regularizer(_rand_z())
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_collapse_regularizer_uniform():
    # Perfectly uniform phases → R_bar ≈ 0 → loss ≈ 0
    n = 1000
    theta = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / n), n)
    z = torch.stack([theta.cos(), theta.sin()], dim=-1)
    assert collapse_regularizer(z).item() < 0.01


def test_collapse_regularizer_collapsed():
    # All pointing in the same direction → R_bar = 1 → loss = 1
    z = torch.ones(64, 2)
    loss = collapse_regularizer(z).item()
    assert loss > 0.9


def test_total_loss_keys_and_scalar():
    out = {
        "z_a": _rand_z(),
        "z_b_target": _rand_z(),
        "z_a_pred": _rand_z(),
    }
    loss, components = total_loss(out)
    assert loss.ndim == 0
    assert set(components.keys()) == {"predict", "collapse", "amplitude", "total"}


# ── integration test ─────────────────────────────────────────────────────────

def test_integration_loss_decreases():
    torch.manual_seed(42)
    adata = make_synthetic_circadian(n_cells=180, n_timepoints=6, seed=42)
    n_genes = adata.n_vars

    vg = ViewGenerator(view_mode="asymmetric", seed=42)
    dataset = CircadianDataset(adata, view_generator=vg)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=0)

    model = CircadianJEPA(n_genes=n_genes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model=model, optimizer=optimizer, device="cpu", log_every=9999)

    history = trainer.fit(loader, n_epochs=5, lambda_collapse=1.0, lambda_amplitude=0.1)

    first_loss = history[0]["total"]
    last_loss = history[-1]["total"]
    assert last_loss < first_loss, (
        f"Total loss did not decrease: epoch1={first_loss:.4f} -> epoch5={last_loss:.4f}"
    )
