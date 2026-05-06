import numpy as np
import pytest
import torch

from circadian_jepa.data.dataset import CircadianDataset
from circadian_jepa.data.gene_lists import get_circadian_genes, get_default_beta
from circadian_jepa.data.synthetic import make_synthetic_circadian
from circadian_jepa.data.views import ViewGenerator


# ---------------------------------------------------------------------------
# gene_lists
# ---------------------------------------------------------------------------


def test_default_beta_loads():
    beta = get_default_beta()
    assert len(beta) == 15
    for col in ("a_0", "a_1", "b_1"):
        assert col in beta.columns


def test_gene_list_mouse():
    genes = get_circadian_genes("mouse")
    assert len(genes) == 15
    assert genes[0][0].isupper()  # capitalized, e.g. Bmal1
    assert not genes[0].isupper()  # not all-caps


def test_gene_list_human():
    genes = get_circadian_genes("human")
    assert all(g == g.upper() for g in genes)


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------


def test_make_synthetic_shape():
    adata = make_synthetic_circadian(500, seed=0)
    assert adata.shape == (500, 15)
    assert "true_phase" in adata.obs
    phases = adata.obs["true_phase"].values
    assert np.all(phases >= 0) and np.all(phases < 2 * np.pi)


def test_make_synthetic_timepoints():
    adata = make_synthetic_circadian(600, n_timepoints=6, seed=0)
    assert adata.shape == (600, 15)
    unique_batches = adata.obs["batch"].unique()
    assert len(unique_batches) == 6


def test_make_synthetic_timepoints_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        make_synthetic_circadian(100, n_timepoints=7)


def test_nb_counts_nonneg():
    adata = make_synthetic_circadian(200, seed=1)
    assert np.all(adata.X >= 0)


def test_counts_layer_matches_x():
    adata = make_synthetic_circadian(100, seed=2)
    assert "counts" in adata.layers
    np.testing.assert_array_equal(adata.X, adata.layers["counts"])


# ---------------------------------------------------------------------------
# ViewGenerator
# ---------------------------------------------------------------------------


def test_view_generator_shape():
    counts = torch.randint(0, 50, (32, 15), dtype=torch.float32)
    vg = ViewGenerator(view_mode="symmetric_split", seed=0)
    view_a, view_b = vg.make_pair(counts)
    assert view_a.shape == counts.shape
    assert view_b.shape == counts.shape


def test_view_generator_downsample_differs():
    counts = torch.randint(5, 100, (64, 15), dtype=torch.float32)
    vg = ViewGenerator(view_mode="symmetric_split", seed=0)
    view_a, view_b = vg.make_pair(counts)
    assert not torch.allclose(view_a, view_b), "Downsampled views should differ"
    # Total counts in each view should be roughly half the original
    totals_a = torch.exp(view_a) - 1  # rough inverse of log1p
    totals_b = torch.exp(view_b) - 1
    ratio = totals_a.sum() / totals_b.sum()
    assert 0.7 < ratio.item() < 1.4, f"Unexpected ratio {ratio.item():.2f}"


def test_same_batch_pairing():
    counts = torch.randint(5, 50, (60, 15), dtype=torch.float32)
    # 3 batches of 20 cells each
    batch_labels = torch.tensor([i // 20 for i in range(60)], dtype=torch.long)
    vg = ViewGenerator(view_mode="symmetric_split", same_batch=True, seed=42)
    view_a, view_b = vg.make_batch_pairs(counts, batch_labels)
    assert view_a.shape == (60, 15)
    assert view_b.shape == (60, 15)


# ---------------------------------------------------------------------------
# CircadianDataset
# ---------------------------------------------------------------------------


def test_dataset_getitem():
    adata = make_synthetic_circadian(100, n_timepoints=4, seed=0)
    vg = ViewGenerator(view_mode="symmetric_split", seed=0)
    ds = CircadianDataset(adata, vg)
    assert len(ds) == 100
    item = ds[0]
    assert "view_a" in item
    assert "view_b" in item
    assert item["view_a"].shape == (15,)
    assert "true_phase" in item
    assert "batch_label" in item


def test_dataset_same_batch():
    adata = make_synthetic_circadian(120, n_timepoints=6, seed=0)
    vg = ViewGenerator(view_mode="symmetric_split", same_batch=True, seed=0)
    ds = CircadianDataset(adata, vg)
    item = ds[0]
    assert "view_a" in item and "view_b" in item
    assert item["view_a"].shape == (15,)
