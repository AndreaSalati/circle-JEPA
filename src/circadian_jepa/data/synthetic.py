import anndata
import numpy as np
import torch
from scritmo.ml.simulations.simulations import design_matrix, generate_nb_data

from .gene_lists import get_default_beta


def make_synthetic_circadian(
    n_cells: int,
    beta=None,
    seq_depth: int = 1000,
    noise_model: str = "nb",
    dispersion: float = 0.1,
    n_timepoints: int | None = None,
    seed: int = 0,
) -> anndata.AnnData:
    """Generate synthetic circadian scRNA-seq data using scritmo's NB model.

    Parameters
    ----------
    n_cells:
        Number of cells to generate.
    beta:
        scritmo.Beta object with Fourier gene parameters. Uses the bundled
        ccg_zhang_context.csv (15 mouse clock genes) when None.
    seq_depth:
        Total UMI count per cell.
    noise_model:
        'nb' (negative binomial) or 'poisson'.
    dispersion:
        NB dispersion parameter (ignored for Poisson).
    n_timepoints:
        If set, cells are assigned to this many evenly-spaced phases. Requires
        n_cells % n_timepoints == 0. If None, phases are uniformly distributed.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    AnnData with:
        .X              — raw NB count data (float32), shape (n_cells, n_genes)
        .var_names      — gene names from beta
        .obs['true_phase']  — simulated phase in [0, 2π)
        .obs['batch']       — timepoint index string
        .obs['seq_depth']   — UMI target
        .layers['counts']   — copy of .X
    """
    if beta is None:
        beta = get_default_beta()

    n_genes = len(beta)

    # --- phase generation ---
    if n_timepoints is not None:
        if n_cells % n_timepoints != 0:
            raise ValueError(
                f"n_cells ({n_cells}) must be divisible by n_timepoints ({n_timepoints})."
            )
        cells_per_tp = n_cells // n_timepoints
        base_angles = torch.linspace(0, 2 * np.pi, n_timepoints + 1)[:-1]
        phi_c = base_angles.repeat_interleave(cells_per_tp)
        batch = np.array(
            [str(t) for t in range(n_timepoints) for _ in range(cells_per_tp)]
        )
    else:
        phi_c = torch.linspace(0, 2 * np.pi, n_cells + 1)[:-1]
        batch = np.array(["0"] * n_cells)

    # --- single-context design matrix ---
    context = np.array(["A"] * n_cells)
    dm = design_matrix(context)
    m_yg = np.zeros((1, n_genes))
    lambdaa = np.ones(1)
    counts_per_cell = torch.ones(n_cells) * seq_depth

    data = generate_nb_data(
        beta,
        phi_c=phi_c,
        counts=counts_per_cell,
        dm=dm,
        m_yg=m_yg,
        lambdaa=lambdaa,
        noise_model=noise_model,
        dispersion=dispersion,
        seed=seed,
    ).numpy()  # (n_cells, n_genes)

    adata = anndata.AnnData(data.astype(np.float32))
    adata.var_names = beta.index.tolist()
    adata.obs["true_phase"] = phi_c.numpy().astype(np.float32)
    adata.obs["batch"] = batch
    adata.obs["seq_depth"] = float(seq_depth)
    adata.layers["counts"] = data.copy()
    return adata
