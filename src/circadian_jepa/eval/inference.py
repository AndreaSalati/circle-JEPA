from __future__ import annotations

import warnings

import anndata
import numpy as np
import torch

from ..model.jepa import CircadianJEPA


def infer_phase(
    model: CircadianJEPA,
    adata: anndata.AnnData,
    gene_list: list[str],
    device: str = "cpu",
) -> anndata.AnnData:
    """Run the student encoder on adata and store inferred phase and amplitude.

    Preprocessing mirrors the training pipeline: raw counts from .layers['counts']
    (or .X if that layer is absent) are total-count normalised to 1e4 and log1p
    transformed before being fed to the model.

    Genes in gene_list that are missing from adata are warned about and padded
    with zeros so the gene ordering matches what the model expects.

    Parameters
    ----------
    model : trained CircadianJEPA
    adata : AnnData to annotate (not modified in-place; a copy is returned)
    gene_list : ordered list of gene names the model was trained on
    device : torch device string

    Returns
    -------
    AnnData copy with new columns in .obs:
        - 'inferred_phase'     — inferred angle in [0, 2π)
        - 'inferred_amplitude' — ||z|| for each cell
        - 'z_0', 'z_1'        — raw 2D embedding coordinates
    """
    adata = adata.copy()

    # Build (n_cells, n_genes_model) count matrix aligned to gene_list
    n_cells = adata.n_obs
    n_genes_model = len(gene_list)
    counts_aligned = np.zeros((n_cells, n_genes_model), dtype=np.float32)

    for j, gene in enumerate(gene_list):
        if gene in adata.var_names:
            col_idx = adata.var_names.get_loc(gene)
            if "counts" in adata.layers:
                col = adata.layers["counts"][:, col_idx]
            else:
                col = adata.X[:, col_idx]
            col = col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()
            counts_aligned[:, j] = col
        else:
            warnings.warn(f"Gene '{gene}' not found in adata; padded with zeros.", stacklevel=2)

    # Same normalisation as ViewGenerator._normalize
    counts_t = torch.tensor(counts_aligned, dtype=torch.float32)
    totals = counts_t.sum(dim=-1, keepdim=True).clamp(min=1.0)
    x_norm = torch.log1p(counts_t / totals * 1e4)

    # Forward pass through student encoder
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        z = model.embed(x_norm.to(device)).cpu().numpy()

    phase = np.arctan2(z[:, 1], z[:, 0]) % (2 * np.pi)
    amplitude = np.linalg.norm(z, axis=-1)

    adata.obs["inferred_phase"] = phase.astype(np.float32)
    adata.obs["inferred_amplitude"] = amplitude.astype(np.float32)
    adata.obs["z_0"] = z[:, 0].astype(np.float32)
    adata.obs["z_1"] = z[:, 1].astype(np.float32)
    return adata
