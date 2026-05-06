# Circadian-JEPA: Build Plan

A JEPA-style architecture for unsupervised circadian (and eventually cell cycle) phase inference from scRNA-seq data, using a small set of known rhythmic genes. The latent space is `R^2` (a 2D plane), where the angular component encodes phase and the radial component encodes rhythm amplitude/confidence.

## Project Philosophy

- **Python only**, type-hinted, modular.
- Each module should be independently testable.
- No premature optimization. Get a working pipeline end-to-end first, then iterate.
- Use `pytorch` for the model, `scanpy`/`anndata` for data, `numpy`/`scipy` for everything else.
- Keep the codebase small enough that the whole thing fits in one Claude Code session for later debugging.

## Build Phases

The plan is split into 6 phases. Run each as a separate Claude Code session. At the end of each phase there is a verification checklist; do not move on until it passes.

Phases:
1. Repo scaffolding and environment setup
2. Data module (loading, preprocessing, view generation)
3. Model module (encoder, predictor, EMA teacher)
4. Loss and training loop
5. Evaluation and validation utilities
6. End-to-end example notebook and synthetic-data sanity check

A final optional phase (7) covers extensions to torus geometry for joint cell-cycle + circadian inference.

---

## Phase 1: Repo scaffolding and environment

### Goal
Set up a clean Python repo with proper packaging, dependencies, testing, and CI-friendly structure. No model code yet.

### Tasks for Claude Code

1. Create the following directory structure:
```
circadian-jepa/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── circadian_jepa/
│       ├── __init__.py
│       ├── data/
│       │   └── __init__.py
│       ├── model/
│       │   └── __init__.py
│       ├── training/
│       │   └── __init__.py
│       └── eval/
│           └── __init__.py
├── tests/
│   └── __init__.py
├── notebooks/
└── configs/
```

2. Write `pyproject.toml` with:
   - Project name `circadian-jepa`, Python >= 3.10.
   - Dependencies: `torch>=2.0`, `numpy`, `scipy`, `scanpy`, `anndata`, `pandas`, `scikit-learn`, `tqdm`, `pyyaml`.
   - Dev dependencies: `pytest`, `pytest-cov`, `ruff`, `mypy`.
   - Use `setuptools` or `hatchling` as backend. Keep it simple.

3. Write a `.gitignore` covering Python, PyTorch checkpoints, Jupyter, common data formats (`*.h5ad`, `*.loom`), and editor files.

4. Write a brief `README.md` with: project description (one paragraph from the project philosophy above), installation instructions (`pip install -e ".[dev]"`), and a placeholder "Usage" section.

5. Set up `ruff` config in `pyproject.toml` with line length 100 and the rule sets `E`, `F`, `I`, `B`, `UP`.

6. Create a `tests/test_smoke.py` that just imports `circadian_jepa` and asserts the version attribute exists.

### Verification checklist
- [ ] `pip install -e ".[dev]"` succeeds in a clean venv.
- [ ] `pytest` runs and the smoke test passes.
- [ ] `ruff check src/` passes with no errors.
- [ ] `python -c "import circadian_jepa; print(circadian_jepa.__version__)"` works.

### Handoff note for next phase
The data module will need access to `scanpy` and `anndata`, which should already be installed. Phase 2 will add gene-list resources and synthetic data generation.

---

## Phase 2: Data module

### Goal
Load scRNA-seq data, restrict to a curated rhythmic gene set, and produce paired views for JEPA training. Include a synthetic data generator for unit testing without needing real data.

### Background context for Claude Code
The two view-generation strategies are:
- **Downsampling** (always on): given a UMI count vector for a cell, split each count via `Binomial(n, 0.5)` to produce two independent half-counts. This is principled because each half is a valid lower-depth observation of the same cell.
- **Same-batch pooling** (optional): given a batch label (e.g. mouse ID, sacrifice time), pair cells from the same batch as views of each other. Only use when the batch label corresponds to a shared phase variable (true for circadian time-of-day experiments, false for asynchronous cell cycle data).

Views can be combined: same-batch pair + independent downsampling per cell.

### Tasks for Claude Code

1. **Gene lists** (`src/circadian_jepa/data/gene_lists.py`):
   - Hardcode a list of ~15 core circadian clock genes for mouse and human:
     - Core: `BMAL1` (a.k.a. `ARNTL`), `CLOCK`, `PER1`, `PER2`, `PER3`, `CRY1`, `CRY2`, `NR1D1` (REV-ERBα), `NR1D2` (REV-ERBβ), `RORA`, `RORB`, `RORC`, `DBP`, `NPAS2`, `TEF`.
   - Provide both human (uppercase) and mouse (capitalized: `Bmal1`, etc.) versions.
   - Expose `get_circadian_genes(species: str = "human") -> list[str]`.
   - Also expose `get_cell_cycle_genes(species: str)` using Tirosh et al. lists. If too many to hardcode, leave this as a stub returning `NotImplementedError` for now.

2. **Synthetic data** (`src/circadian_jepa/data/synthetic.py`):
   - Function `make_synthetic_circadian(n_cells: int, n_genes: int = 15, n_timepoints: int = 6, noise: float = 0.3, dropout_rate: float = 0.5, seed: int = 0) -> AnnData`.
   - Generate cells uniformly distributed across `n_timepoints` discrete circadian phases (true phase stored in `adata.obs['true_phase']` as radians in `[0, 2pi)`).
   - For each gene `g`, assign a phase offset `phi_g` in `[0, 2pi)` and amplitude `A_g ~ Uniform(0.5, 1.5)`.
   - True log-expression: `log_expr[i, g] = baseline + A_g * cos(theta_i - phi_g) + noise * randn()`.
   - Convert to count data: `counts ~ Poisson(exp(log_expr) * library_size)`, with library size sampled from a log-normal.
   - Apply random dropout: zero out fraction `dropout_rate` of nonzero entries.
   - Return as `AnnData` with `.X` as raw counts (sparse or dense), `.var_names` as `["gene_0", "gene_1", ...]`, and `.obs` containing `true_phase` and a `batch` label (the timepoint index).

3. **Data loading** (`src/circadian_jepa/data/loader.py`):
   - Function `load_and_preprocess(adata: AnnData, gene_list: list[str], min_cells_per_gene: int = 10, log_normalize: bool = True) -> AnnData`.
   - Subset to genes in `gene_list` that exist in `adata.var_names` (warn if any are missing).
   - Filter cells with too few counts in the gene subset.
   - Apply standard normalization: total-count normalize per cell, then `log1p`. Store both raw counts (in `.layers['counts']`) and normalized (in `.X`).

4. **View generation** (`src/circadian_jepa/data/views.py`):
   - Class `ViewGenerator(downsample: bool = True, same_batch: bool = False, batch_key: str = "batch", mask_prob: float = 0.0, seed: int = 0)`.
   - Method `make_pair(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`:
     - Input: integer count tensor of shape `(n_cells, n_genes)`.
     - If `downsample`: for each cell, split each gene count via `Binomial(n, 0.5)`. Returns two views.
     - Re-normalize each view (total-count + log1p).
     - If `mask_prob > 0`: randomly zero out genes per view with this probability (apply after normalization).
   - Method `make_batch_pairs(counts: torch.Tensor, batch_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`:
     - For each cell, pick a random partner from the same batch. Each cell is then independently downsampled. Returns the two view tensors aligned as pairs.

5. **PyTorch Dataset wrapper** (`src/circadian_jepa/data/dataset.py`):
   - Class `CircadianDataset(adata: AnnData, view_generator: ViewGenerator, ...)` extending `torch.utils.data.Dataset`.
   - `__getitem__` returns a dict with keys `view_a`, `view_b`, optionally `batch_label`, `true_phase` (if available, for evaluation only — never used in loss).

6. **Tests** (`tests/test_data.py`):
   - Test that `make_synthetic_circadian` produces an AnnData with the right shape and that `true_phase` is in `[0, 2pi)`.
   - Test that `ViewGenerator` produces tensors of the correct shape and dtype.
   - Test that with `downsample=True`, the two views differ but have similar total counts.
   - Test that `same_batch=True` correctly pairs cells with the same batch label.

### Verification checklist
- [ ] `pytest tests/test_data.py` passes.
- [ ] In a Python REPL: `from circadian_jepa.data.synthetic import make_synthetic_circadian; adata = make_synthetic_circadian(1000); print(adata)` works and shows reasonable dimensions.
- [ ] Visual check: plot one gene's expression across `true_phase` from synthetic data; should show a clear sinusoid.

### Handoff note for next phase
The data module produces tensors of shape `(batch_size, n_genes)` with `n_genes ≈ 15`. The model needs to map these to a 2D embedding. Phase 3 will build the encoder, predictor, and EMA teacher.

---

## Phase 3: Model module

### Goal
Build the core JEPA components: a small MLP encoder mapping `R^15 → R^2`, a rotation-equivariant predictor, and an EMA-updated teacher network.

### Background context for Claude Code

The architecture in plain terms:
- **Student encoder** `f_s(x) → z ∈ R^2`. Phase is `θ = atan2(z[1], z[0])`. Amplitude is `r = ||z||`.
- **Teacher encoder** `f_t(x) → z' ∈ R^2`. Same architecture, but parameters are an EMA of the student. No gradients flow through the teacher.
- **Predictor** `g(z, Δ) = R(Δ) · z`, where `R(Δ)` is a 2D rotation matrix by angle `Δ`. The predictor's job is to predict the teacher's embedding of view B from the student's embedding of view A, accounting for any phase shift `Δ` between the two views.
- For the simplest case (two views of the same cell via downsampling), `Δ = 0` and the predictor reduces to identity. For same-batch pairs (potentially different cells, same timepoint), `Δ = 0` is also assumed. `Δ` becomes nontrivial only if you later do pseudotime-neighbor pairs.

### Tasks for Claude Code

1. **Encoder** (`src/circadian_jepa/model/encoder.py`):
   - Class `Encoder(nn.Module)`:
     - `__init__(n_genes: int, hidden_dims: list[int] = [64, 32], embedding_dim: int = 2, dropout: float = 0.1, normalize_output: bool = False)`.
     - MLP with `LayerNorm` and `GELU` between layers.
     - Final linear layer maps to `embedding_dim` (default 2 for the planar embedding).
     - If `normalize_output=True`, normalize output to unit norm (this would force pure-S^1; default `False` to keep the radial degree of freedom for amplitude).
   - Method `phase(z: torch.Tensor) -> torch.Tensor`: returns `atan2(z[..., 1], z[..., 0])`.
   - Method `amplitude(z: torch.Tensor) -> torch.Tensor`: returns `torch.norm(z, dim=-1)`.

2. **Predictor** (`src/circadian_jepa/model/predictor.py`):
   - Class `RotationPredictor(nn.Module)`:
     - `__init__(learn_delta: bool = False)`.
     - If `learn_delta=False`, `forward(z, delta=None)` just returns `z` (identity), since for downsampling/same-batch pairs `Δ=0`.
     - If `learn_delta=True`, accept a per-pair `delta` tensor and apply `R(delta) · z`.
     - Useful helper: a static method `rotate(z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor` that applies 2D rotation. `delta` shape `(batch,)`, `z` shape `(batch, 2)`.
   - Keep this simple. The complexity here is intentionally low; the predictor's main role is to enforce that the embedding space respects rotational structure.

3. **EMA teacher** (`src/circadian_jepa/model/ema.py`):
   - Class `EMATeacher`:
     - `__init__(student: nn.Module, momentum: float = 0.996)`.
     - Internally creates `self.teacher` as a deep copy of `student`, sets all teacher params to `requires_grad=False`.
     - Method `update(student: nn.Module)`: in-place EMA update of teacher params: `teacher.param = momentum * teacher.param + (1 - momentum) * student.param`.
     - Method `forward(x)`: runs `x` through teacher with `torch.no_grad()`. Equivalent to `teacher(x)` but explicit about no-grad.
   - The `momentum` parameter typically follows a schedule (e.g., 0.996 → 1.0 over training); for now keep it constant and add a `set_momentum` setter for later scheduling.

4. **Top-level model** (`src/circadian_jepa/model/jepa.py`):
   - Class `CircadianJEPA(nn.Module)`:
     - `__init__(n_genes: int, hidden_dims, embedding_dim=2, ema_momentum=0.996, normalize_output=False)`.
     - Holds `student_encoder`, `ema` (which wraps a copy as teacher), and `predictor`.
     - Method `forward(view_a: torch.Tensor, view_b: torch.Tensor, delta: torch.Tensor | None = None) -> dict`:
       - Compute `z_a = student(view_a)`.
       - Compute `z_b_target = teacher(view_b)` (no grad).
       - Compute `z_a_pred = predictor(z_a, delta)`.
       - Return `{"z_a": z_a, "z_b_target": z_b_target, "z_a_pred": z_a_pred}`.
     - Method `step_ema()`: calls `self.ema.update(self.student_encoder)`. Should be called by the training loop after each optimizer step.
     - Method `embed(x)`: convenience for inference, returns student embedding.

5. **Tests** (`tests/test_model.py`):
   - Test `Encoder` produces correct output shape.
   - Test `phase()` and `amplitude()` produce expected values for known inputs (e.g., `[1, 0]` should give `phase=0, amplitude=1`).
   - Test `RotationPredictor.rotate` correctly rotates a known vector by a known angle.
   - Test `EMATeacher.update` actually moves teacher params toward student params, and test that teacher params have `requires_grad=False`.
   - Test `CircadianJEPA.forward` produces output dict with correct shapes.
   - Test that gradients flow through `z_a` and `z_a_pred` but NOT through `z_b_target`.

### Verification checklist
- [ ] `pytest tests/test_model.py` passes.
- [ ] All shapes are correct end-to-end on a synthetic batch.
- [ ] Gradient check: `z_b_target.grad_fn is None`.
- [ ] Sanity check in REPL: instantiate `CircadianJEPA`, run a forward pass on random input of shape `(64, 15)`, confirm output shapes.

### Handoff note for next phase
The model is now stateless about loss and training. Phase 4 builds the loss function (with anti-collapse regularizers) and the training loop.

---

## Phase 4: Loss and training loop

### Goal
Implement the JEPA loss with circular-variance anti-collapse regularization, and a clean training loop that supports both view-generation strategies.

### Background context for Claude Code

The total loss is:
```
L = L_predict + λ_collapse * L_collapse + λ_amplitude * L_amplitude
```

- **`L_predict`**: cosine distance between `z_a_pred` and `z_b_target`. Cosine (rather than MSE) makes the loss invariant to the radial component, so the predictive task focuses on phase, while the radial component is shaped only by the collapse regularizer and amplitude regularizer.
- **`L_collapse`**: penalizes circular concentration of phases over the batch. Using the resultant length: `R̄ = ||(mean(cos θ), mean(sin θ))||`. Then `L_collapse = R̄^2`. When `R̄ → 0`, phases are spread uniformly; when `R̄ → 1`, all phases collapsed.
- **`L_amplitude`**: weak L2 keeping `||z||` near 1, so embeddings don't drift to the origin (which would make phase undefined). Specifically `L_amplitude = mean((||z|| - 1)^2)` but with a small coefficient.

### Tasks for Claude Code

1. **Loss functions** (`src/circadian_jepa/training/losses.py`):
   - Function `predictive_loss(z_pred: Tensor, z_target: Tensor) -> Tensor`:
     - Compute `1 - cosine_similarity(z_pred, z_target)`, mean over batch.
     - Add small epsilon for numerical stability.
   - Function `collapse_regularizer(z: Tensor) -> Tensor`:
     - Compute `theta = atan2(z[..., 1], z[..., 0])`.
     - Compute `R_bar = sqrt(mean(cos(theta))^2 + mean(sin(theta))^2)`.
     - Return `R_bar ** 2`.
   - Function `amplitude_regularizer(z: Tensor, target_radius: float = 1.0) -> Tensor`:
     - Return `mean((torch.norm(z, dim=-1) - target_radius) ** 2)`.
   - Function `total_loss(out: dict, lambda_collapse: float = 1.0, lambda_amplitude: float = 0.1) -> tuple[Tensor, dict]`:
     - Combines the three. Returns `(loss, {"predict": ..., "collapse": ..., "amplitude": ..., "total": ...})` for logging.

2. **Trainer** (`src/circadian_jepa/training/trainer.py`):
   - Class `Trainer`:
     - `__init__(model: CircadianJEPA, optimizer, scheduler=None, device='cuda', log_every: int = 50)`.
     - Method `train_epoch(dataloader, view_generator, lambda_collapse=1.0, lambda_amplitude=0.1) -> dict`:
       - Iterate batches.
       - For each batch, generate views via `view_generator`.
       - Forward pass, compute loss, backward, optimizer step, EMA step.
       - Track running averages of all loss components.
       - Return dict of epoch-mean metrics.
     - Method `fit(train_dataloader, n_epochs, view_generator, ...)`: full training with optional logging callback.

3. **Configuration** (`src/circadian_jepa/training/config.py`):
   - Dataclass `TrainConfig`:
     - All hyperparameters: `n_genes`, `hidden_dims`, `embedding_dim`, `lr`, `weight_decay`, `n_epochs`, `batch_size`, `lambda_collapse`, `lambda_amplitude`, `ema_momentum`, `view_strategy` (one of `"downsample"` or `"same_batch"`), `mask_prob`, `seed`.
     - Method `from_yaml(path: str) -> TrainConfig`.
   - Provide a default config YAML in `configs/default.yaml`.

4. **Training script** (`src/circadian_jepa/training/run.py`):
   - Entry point `train_from_config(config: TrainConfig, adata: AnnData) -> CircadianJEPA`.
   - Sets seeds, builds dataset, dataloader, model, optimizer, trainer, runs `fit`, returns trained model.
   - Add a CLI wrapper using `argparse` so it can be called as `python -m circadian_jepa.training.run --config configs/default.yaml --data path/to/adata.h5ad`.

5. **Tests** (`tests/test_training.py`):
   - Test that each loss component returns a scalar tensor and is non-negative.
   - Test `collapse_regularizer` returns ~0 for uniformly distributed phases, ~1 for collapsed phases.
   - Integration test: run 5 epochs of training on synthetic data with 200 cells, confirm total loss decreases.

### Verification checklist
- [ ] `pytest tests/test_training.py` passes, including the integration test.
- [ ] Manual run: `train_from_config` on synthetic data for 20 epochs runs without NaNs.
- [ ] Loss curves: predictive loss decreases, collapse stays low (R̄ < 0.3 by end of training).

### Handoff note for next phase
After training, you have a `CircadianJEPA` model that produces planar embeddings. Phase 5 is about validating that those embeddings are biologically meaningful.

---

## Phase 5: Evaluation and validation utilities

### Goal
Tools to assess whether the learned embedding actually captures circadian phase, both with and without ground truth.

### Background context for Claude Code

Three validation regimes:
- **Synthetic data with known `true_phase`**: compare inferred to true with circular correlation.
- **Real data with sparse timepoint labels**: e.g. mice sacrificed at known ZTs. Check that inferred phase clusters by timepoint and that the ordering on the circle matches the time order.
- **Real data with no labels at all**: check that core clock genes show sinusoidal expression as a function of inferred phase, and that phase relationships between genes match known biology (e.g., `BMAL1` and `PER2` should be antiphase).

### Tasks for Claude Code

1. **Circular metrics** (`src/circadian_jepa/eval/circular.py`):
   - Function `circular_correlation(theta_pred: ndarray, theta_true: ndarray) -> float`:
     - Implements Jammalamadaka-Sarma circular correlation coefficient.
     - Range `[-1, 1]`, sign indicates direction.
   - Function `circular_distance(theta_a: ndarray, theta_b: ndarray) -> ndarray`:
     - Wrapped distance: `min(|a-b|, 2π - |a-b|)`.
   - Function `align_phase(theta_pred: ndarray, theta_true: ndarray) -> tuple[ndarray, float, int]`:
     - Find best rigid alignment (rotation + reflection) of `theta_pred` to `theta_true`.
     - Returns aligned predictions, optimal offset, optimal sign (+1 or -1 for reflection).

2. **Phase inference** (`src/circadian_jepa/eval/inference.py`):
   - Function `infer_phase(model: CircadianJEPA, adata: AnnData, gene_list: list[str], device: str = "cpu") -> AnnData`:
     - Preprocess `adata` with the same pipeline used for training.
     - Run student encoder on all cells.
     - Add `adata.obs['inferred_phase']` (radians) and `adata.obs['inferred_amplitude']`.
     - Return modified `adata`.

3. **Cosinor sanity check** (`src/circadian_jepa/eval/cosinor.py`):
   - Function `fit_cosinor(expression: ndarray, phase: ndarray) -> dict`:
     - For each gene, fit `expr = baseline + A * cos(phase - phi)` via least squares.
     - Return per-gene `{"amplitude", "phase_offset", "r_squared", "p_value"}`.
   - Function `check_known_phase_relationships(cosinor_results: dict, gene_pairs: list[tuple[str, str, float]]) -> dict`:
     - `gene_pairs` is a list of `(gene_a, gene_b, expected_phase_diff_radians)`.
     - For circadian, hardcode some known relationships, e.g.:
       - `(BMAL1, PER2, π)` — antiphase.
       - `(BMAL1, NR1D1, ~π)` — antiphase (REV-ERBα is repressed by BMAL1 directly but transcriptionally peaks shortly after BMAL1).
     - Returns observed vs expected phase differences with errors.

4. **Plotting utilities** (`src/circadian_jepa/eval/plots.py`):
   - Use `matplotlib`. Add `matplotlib` to dependencies if not already.
   - Function `plot_embedding(z: ndarray, color_by: ndarray | None = None, ax=None) -> Axes`:
     - Scatter of the planar embedding, with optional coloring (e.g. by true phase or batch).
     - Draws the unit circle for reference.
   - Function `plot_phase_vs_truth(theta_pred, theta_true, ax=None) -> Axes`:
     - Scatter on a torus-flattened diagram (each axis is `[0, 2π]`, points should fall on a diagonal if perfect).
   - Function `plot_gene_rhythm(adata: AnnData, gene: str, phase_key: str = "inferred_phase", ax=None) -> Axes`:
     - For one gene: scatter expression vs inferred phase, overlay fitted cosine curve.

5. **Tests** (`tests/test_eval.py`):
   - Test `circular_correlation` is 1 for identical inputs, near 0 for random, robust to constant offset.
   - Test `align_phase` recovers a known rotation.
   - Test full pipeline on synthetic data: train for 30 epochs, infer phase, check circular correlation with `true_phase` is > 0.7 (if this fails, the architecture has a fundamental issue and you need to debug before continuing).

### Verification checklist
- [ ] All eval tests pass.
- [ ] On synthetic data with 1000 cells and 6 timepoints: trained model achieves circular correlation > 0.7.
- [ ] Plots render correctly and show the expected structure (cells distributed around the unit circle, colored by true phase forming a smooth color wheel).

### Handoff note for next phase
You now have a working end-to-end pipeline. Phase 6 packages it into a clean demo notebook.

---

## Phase 6: End-to-end example notebook

### Goal
A single notebook that demonstrates the full pipeline on synthetic data, serving as both a tutorial and a sanity check.

### Tasks for Claude Code

1. Create `notebooks/01_synthetic_demo.ipynb` covering:
   - Generate synthetic circadian data with 2000 cells, 15 genes, 8 timepoints.
   - Show one gene's expression vs `true_phase` (should be sinusoidal).
   - Set up training config.
   - Train the model for ~50 epochs (CPU should be fast enough at this scale).
   - Plot training loss curves (predictive, collapse, amplitude, total).
   - Infer phases, compute circular correlation, plot `inferred_phase` vs `true_phase`.
   - Plot the planar embedding colored by true phase.
   - Run cosinor analysis on inferred phase, show that recovered gene phase offsets match the ones used in synthesis.
   - Markdown cells throughout explaining each step in plain language.

2. Create `notebooks/02_real_data_template.ipynb` as a *template* (not run, since it requires real data):
   - Skeleton cells for loading a real h5ad file.
   - Subsetting to circadian genes.
   - Training with both `view_strategy="downsample"` and `view_strategy="same_batch"` and comparing results.
   - Validation via known gene phase relationships (since no `true_phase` is available).

3. Update `README.md` with:
   - Quick-start example (5–10 lines of code from data → trained model → inferred phase).
   - Pointer to the demo notebook.
   - Brief explanation of the architecture and why JEPA + planar embedding (this is for someone discovering the repo, so keep it accessible).

### Verification checklist
- [ ] `01_synthetic_demo.ipynb` runs end-to-end without errors.
- [ ] Final circular correlation on synthetic data is > 0.8 (with proper hyperparameters this should be achievable).
- [ ] All plots render and look reasonable.
- [ ] README quick-start works as written.

### Handoff note
At this point the repo is functional and demonstrably solves the synthetic problem. Real-data validation is up to you with your specific datasets.

