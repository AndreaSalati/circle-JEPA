# Phase 2 Plan: Data Module (scritmo-compatible)

## Context

The original BUILD_PLAN.md Phase 2 spec used a hand-rolled cosine+Poisson synthetic data generator and hardcoded gene lists. This phase makes the repo compatible with **scRITMO**, which uses:
- `scritmo.Beta` — a `pd.DataFrame` subclass storing Fourier coefficients (`a_0`, `a_1`, `b_1`, …) and metadata (`phase`, `amp`, `log2fc`, `disp`) per gene, loadable from a CSV path.
- `generate_nb_data(params_g, phi_c, counts, dm, m_yg, lambdaa, noise_model, dispersion)` — generates NB count data using the Fourier model: `log_expr_cg = a_0_g + λ_c*(a_1_g*cos(φ_c) + b_1_g*sin(φ_c))`.
- `ccg_zhang_context.csv` — 15 mouse circadian clock genes with fitted parameters (from `/Users/salati/Documents/CODE/github/scRITMO/ccg_zhang_context.csv`).

**Goal**: Replace the naive synthetic generator with scritmo-backed simulation, bundle the gene parameters CSV, and keep the rest of the data module interface (loader, views, dataset) unchanged.

---

## Files to create / modify

| Path | Action |
|------|--------|
| `src/circadian_jepa/data/resources/ccg_zhang_context.csv` | Copy from scritmo repo |
| `src/circadian_jepa/data/gene_lists.py` | New — load gene names from Beta |
| `src/circadian_jepa/data/synthetic.py` | New — wrap scritmo's generate_nb_data |
| `src/circadian_jepa/data/loader.py` | New — unchanged from original spec |
| `src/circadian_jepa/data/views.py` | New — unchanged from original spec |
| `src/circadian_jepa/data/dataset.py` | New — unchanged from original spec |
| `tests/test_data.py` | New — scritmo-aware tests |
| `pyproject.toml` | Edit — add `scritmo` and package-data |

---

## Task 1: Bundle gene parameters

1. Create `src/circadian_jepa/data/resources/` and copy `ccg_zhang_context.csv` into it.
2. In `pyproject.toml`, add `[tool.setuptools.package-data]` with `circadian_jepa = ["data/resources/*.csv"]`.
3. Add `scritmo` to package dependencies.

---

## Task 2: Gene lists (`src/circadian_jepa/data/gene_lists.py`)

```python
from importlib.resources import files
import scritmo as sr

def get_default_beta_path() -> Path:
    return files("circadian_jepa.data.resources").joinpath("ccg_zhang_context.csv")

def get_default_beta() -> sr.Beta:
    return sr.Beta(str(get_default_beta_path()))

def get_circadian_genes(species: str = "mouse") -> list[str]:
    genes = get_default_beta().index.tolist()
    if species == "human":
        return [g.upper() for g in genes]
    return genes
```

---

## Task 3: Synthetic data (`src/circadian_jepa/data/synthetic.py`)

Reuse scritmo's `generate_nb_data` and `design_matrix` directly (not `simulate_data`, to avoid pulling in the trainer/context_model dependencies).

**Function signature**:
```python
def make_synthetic_circadian(
    n_cells: int,
    beta: sr.Beta | None = None,       # if None, loads default ccg_zhang_context.csv
    seq_depth: int = 1000,
    noise_model: str = "nb",
    dispersion: float = 0.1,
    n_timepoints: int | None = None,   # if None, phases are uniform over [0, 2π]
    kappa: float = 0.0,                # >0 adds Von Mises noise around timepoints
    seed: int = 0,
) -> AnnData
```

**Implementation**:
1. If `beta` is None, call `get_default_beta()`.
2. Generate `phi_c`:
   - `n_timepoints=None`: `torch.linspace(0, 2π, n_cells)`, batch label = `"0"` for all cells.
   - `n_timepoints` set, `kappa=0`: `generate_phases(n_cells, n_timepoints)` (requires `n_cells % n_timepoints == 0`), batch label = timepoint index.
   - `kappa > 0`: `generate_phases(n_cells, n_timepoints, kappa)` returns `(phase_labels, phi_c)`.
3. Build trivial single-context design: `dm = design_matrix(["A"]*n_cells)`, `m_yg = zeros((1, n_genes))`, `lambdaa = ones(1)`.
4. Call `generate_nb_data(beta, phi_c, torch.ones(n_cells)*seq_depth, dm, m_yg, lambdaa, noise_model, dispersion, seed)`.
5. Package into AnnData:
   - `.X` = counts as dense float32 numpy.
   - `.var_names` = `beta.index.tolist()`.
   - `.obs['true_phase']` = `phi_c.numpy()` in `[0, 2π)`.
   - `.obs['batch']` = timepoint index string.
   - `.obs['seq_depth']` = seq_depth.
   - `.layers['counts']` = copy of `.X`.

---

## Task 4: Data loader (`src/circadian_jepa/data/loader.py`)

```python
def load_and_preprocess(
    adata: AnnData,
    gene_list: list[str],
    min_cells_per_gene: int = 10,
    log_normalize: bool = True,
) -> AnnData
```
- Subset to `gene_list ∩ adata.var_names`, warn on missing genes.
- Store raw counts in `.layers['counts']`.
- Total-count normalize then `log1p` into `.X`.

---

## Task 5: View generation (`src/circadian_jepa/data/views.py`)

```python
class ViewGenerator(downsample=True, same_batch=False, batch_key="batch", mask_prob=0.0, seed=0):
    def make_pair(counts: Tensor) -> tuple[Tensor, Tensor]
    def make_batch_pairs(counts: Tensor, batch_labels: Tensor) -> tuple[Tensor, Tensor]
```
- `make_pair`: Binomial split per count, then total-count normalize + log1p each view. If `mask_prob > 0`, randomly zero genes after normalization.
- `make_batch_pairs`: For each cell, pick a random same-batch partner; each pair member independently downsampled.

---

## Task 6: Dataset (`src/circadian_jepa/data/dataset.py`)

```python
class CircadianDataset(Dataset):
    # __getitem__ returns dict: {view_a, view_b, [batch_label], [true_phase]}
```

---

## Task 7: Tests (`tests/test_data.py`)

1. `test_default_beta_loads` — 15 rows, columns include `a_0, a_1, b_1`.
2. `test_gene_list_mouse` — returns capitalized names (e.g. `Bmal1`).
3. `test_gene_list_human` — returns all-uppercase names (e.g. `BMAL1`).
4. `test_make_synthetic_shape` — `make_synthetic_circadian(500)` → shape `(500, 15)`, `true_phase` in `[0, 2π)`.
5. `test_make_synthetic_timepoints` — with `n_timepoints=6`, `batch` has 6 unique values.
6. `test_nb_counts_nonneg` — all `.X` values are non-negative.
7. `test_view_generator_shape` — two views have same shape as input.
8. `test_view_generator_downsample_differs` — views differ but per-cell totals are similar.
9. `test_same_batch_pairing` — same-batch pairs share batch label.
10. `test_dataset_getitem` — returns dict with keys `view_a`, `view_b`.

---

## Verification checklist

- [ ] `pytest tests/test_data.py` passes (all 10 tests).
- [ ] REPL: `from circadian_jepa.data.synthetic import make_synthetic_circadian; adata = make_synthetic_circadian(1000); print(adata)` shows `AnnData object with n_obs × n_vars = 1000 × 15`.
- [ ] REPL: `from circadian_jepa.data.gene_lists import get_default_beta; b = get_default_beta(); b.plot_genes()` shows 15 sinusoidal profiles.
- [ ] Visual check: `Bmal1` expression vs `true_phase` is sinusoidal, peaking near phase ~5.9 rad.
- [ ] `ruff check src/` passes.

---

## Notes

- Import from scritmo simulations: `from scritmo.ml.simulations.simulations import generate_nb_data, design_matrix, generate_phases`.
- Do NOT import `simulate_data` — it pulls in trainer/context_model dependencies not needed here.
- `generate_phases` requires `n_cells % n_timepoints == 0`; document this constraint.
- `scritmo` is installed in the `ML-gpu` conda environment.
