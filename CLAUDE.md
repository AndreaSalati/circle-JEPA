# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use the `torch` conda environment (if present, otherwise `ML-gpu`). Use PyTorch for all ML implementations.

## Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Lint
ruff check src/

# Run all tests
pytest

# Run a single test file
pytest tests/test_model.py

# Run only fast tests (exclude slow)
pytest -m "not slow"

# Train from config + h5ad
python -m circadian_jepa.training.run --config configs/default.yaml --data path/to/data.h5ad
```

## Architecture

**Circadian-JEPA** is a JEPA-style self-supervised model for inferring circadian phase from scRNA-seq data. The 2D latent space encodes phase as the angle and confidence as the radius.

### Data pipeline

- [src/circadian_jepa/data/synthetic.py](src/circadian_jepa/data/synthetic.py) — two synthetic data generators: `make_synthetic_circadian` (uses `scritmo`'s NB model + bundled CCG beta coefficients) and `make_synthetic_sinusoidal` (clean sinusoidal model, no `scritmo` dependency).
- [src/circadian_jepa/data/views.py](src/circadian_jepa/data/views.py) — `ViewGenerator` produces paired views via three strategies: `"asymmetric"` (full counts → teacher, thinned → student; the default), `"symmetric_split"` (UMI-level Binomial split), `"light_independent"` (both views independently thinned). Normalisation is `log1p(counts / total * 1e4)`.
- [src/circadian_jepa/data/dataset.py](src/circadian_jepa/data/dataset.py) — `CircadianDataset` wraps AnnData; reads raw counts from `.layers['counts']` when available, otherwise `.X`. Same-batch pairing pre-computes partner indices at init time.
- [src/circadian_jepa/data/gene_lists.py](src/circadian_jepa/data/gene_lists.py) — loads the bundled 15-gene CCG beta table via `scritmo.Beta`; `scritmo` is not on PyPI and must be installed separately in the conda env.

### Model

- [src/circadian_jepa/model/encoder.py](src/circadian_jepa/model/encoder.py) — `Encoder`: MLP (Linear → LayerNorm → GELU → Dropout) × n_layers, projecting to `embedding_dim=2`. `phase()` and `amplitude()` are thin helpers on the 2D output.
- [src/circadian_jepa/model/ema.py](src/circadian_jepa/model/ema.py) — `EMATeacher`: frozen deep-copy of the student, updated via exponential moving average after each step. No gradients flow through it.
- [src/circadian_jepa/model/predictor.py](src/circadian_jepa/model/predictor.py) — `RotationPredictor`: with `learn_delta=False` (the default) this is identity — the prediction task reduces to matching embeddings directly. A 2D rotation by `delta` is applied when `learn_delta=True`.
- [src/circadian_jepa/model/jepa.py](src/circadian_jepa/model/jepa.py) — `CircadianJEPA` combines the three above. `forward()` returns `{z_a, z_b_target, z_a_pred}`. Call `step_ema()` after each optimizer step to advance the teacher. `embed()` runs only the student in `no_grad`.

### Training

- [src/circadian_jepa/training/losses.py](src/circadian_jepa/training/losses.py) — three loss terms: `predictive_loss` (1 − cosine similarity between predicted and target embedding), `collapse_regularizer` (penalises R̄² where R̄ is the mean resultant length of angles in the batch), `amplitude_regularizer` (L2 penalty toward unit radius).
- [src/circadian_jepa/training/trainer.py](src/circadian_jepa/training/trainer.py) — `Trainer.fit()` runs the standard loop: forward → loss → backward → step → EMA update.
- [src/circadian_jepa/training/config.py](src/circadian_jepa/training/config.py) — `TrainConfig` dataclass, loadable from YAML via `TrainConfig.from_yaml()`.
- [src/circadian_jepa/training/run.py](src/circadian_jepa/training/run.py) — `train_from_config()` ties everything together; also doubles as a CLI entry point (`__main__`).

### Evaluation

- [src/circadian_jepa/eval/inference.py](src/circadian_jepa/eval/inference.py) — `infer_phase()`: aligns gene list, applies the same `log1p` normalisation as training, runs the student encoder, writes `inferred_phase`, `inferred_amplitude`, `z_0`, `z_1` into a copy of `adata.obs`.
- [src/circadian_jepa/eval/circular.py](src/circadian_jepa/eval/circular.py) — `circular_correlation` (Jammalamadaka-Sarma), `circular_distance`, `align_phase` (best rigid alignment via rotation + optional reflection, searching both orientations).
- [src/circadian_jepa/eval/cosinor.py](src/circadian_jepa/eval/cosinor.py) — cosinor baseline.
- [src/circadian_jepa/eval/plots.py](src/circadian_jepa/eval/plots.py) — visualisation helpers.

### Key design invariants

- The latent space is unconstrained R²; phase and amplitude are derived by `atan2` and `norm` respectively. Output normalisation (`normalize_output=True`) is available but off by default.
- `scritmo` is a hard dependency for `make_synthetic_circadian` and `get_circadian_genes` but is not on PyPI. Tests that need it are expected to be skipped or marked `slow` when the env lacks it.
- `configs/default.yaml` mirrors `TrainConfig` defaults exactly; keeping them in sync is the responsibility of whoever changes either.
