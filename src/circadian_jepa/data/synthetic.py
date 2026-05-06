import anndata
import numpy as np
import torch
from scritmo.ml.simulations.simulations import design_matrix, generate_nb_data

from .gene_lists import get_default_beta


def make_synthetic_circadian(
    n_cells: int,
    beta=None,
    seq_depth: int = 20000,
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


def make_synthetic_sinusoidal(
    n_cells: int,
    n_genes: int = 15,
    n_timepoints: int | None = None,
    baseline: float = 3.0,
    amplitude_range: tuple[float, float] = (0.5, 1.5),
    noise: float = 0.2,
    dropout_rate: float = 0.3,
    seq_depth: int = 2000,
    seed: int = 0,
) -> anndata.AnnData:
    """Generate clean synthetic circadian data with controlled noise.

    Follows the sinusoidal model described in the build plan (Phase 2, Task 2).

    True log-expression for cell i, gene g:
        log_expr = baseline + A_g * cos(theta_i - phi_g) + noise * randn()

    Converted to counts via Poisson(exp(log_expr)) * library_size factor,
    with random dropout applied.

    Parameters
    ----------
    n_cells : int
    n_genes : int
        Number of genes (each gets a random phase offset and amplitude).
    n_timepoints : int or None
        If set, cells are evenly assigned to discrete phases; must divide n_cells.
    baseline : float
        Baseline log-expression.
    amplitude_range : (float, float)
        Uniform range for per-gene amplitudes A_g.
    noise : float
        Std of Gaussian noise added to log-expression.
    dropout_rate : float
        Fraction of nonzero entries randomly zeroed.
    seq_depth : int
        Target UMI count per cell (library size).
    seed : int

    Returns
    -------
    AnnData with .X = raw counts, .layers['counts'] = copy,
    .obs['true_phase'], .obs['batch'].
    """
    rng: np.random.Generator = np.random.default_rng(seed)

    # Per-gene parameters
    phi_g = rng.uniform(0, 2 * np.pi, size=n_genes)  # phase offset
    A_g = rng.uniform(*amplitude_range, size=n_genes)  # amplitude

    # Cell phases
    if n_timepoints is not None:
        if n_cells % n_timepoints != 0:
            raise ValueError(
                f"n_cells ({n_cells}) must be divisible by n_timepoints ({n_timepoints})."
            )
        cells_per_tp = n_cells // n_timepoints
        theta = np.linspace(0, 2 * np.pi, n_timepoints + 1)[:-1]
        theta = np.repeat(theta, cells_per_tp)
        batch = np.array(
            [str(t) for t in range(n_timepoints) for _ in range(cells_per_tp)]
        )
    else:
        theta = rng.uniform(0, 2 * np.pi, size=n_cells)
        batch = np.array(["0"] * n_cells)

    # Log-expression
    log_expr = baseline + A_g[np.newaxis, :] * np.cos(
        theta[:, np.newaxis] - phi_g[np.newaxis, :]
    )
    log_expr += noise * rng.normal(size=(n_cells, n_genes))

    # Library size per cell (total UMI count, lognormal around seq_depth)
    lib_size = rng.lognormal(mean=np.log(seq_depth), sigma=0.3, size=n_cells)

    # Convert log-expression to expected counts via Poisson.
    # exp(log_expr) gives relative expression proportions per gene.
    # Normalise per cell so total expected count per cell ≈ lib_size.
    expr = np.exp(log_expr)  # (n_cells, n_genes)
    rate = (expr / expr.sum(axis=1, keepdims=True)) * lib_size[:, np.newaxis]
    counts = rng.poisson(rate).astype(np.float32)

    # Dropout
    if dropout_rate > 0:
        mask = rng.binomial(1, 1 - dropout_rate, size=counts.shape)
        counts = counts * mask

    adata = anndata.AnnData(counts)
    adata.var_names = [f"gene_{g}" for g in range(n_genes)]
    adata.obs["true_phase"] = theta.astype(np.float32)
    adata.obs["batch"] = batch
    adata.layers["counts"] = counts.copy()
    return adata
