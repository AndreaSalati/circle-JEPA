# Circadian-JEPA

A JEPA-style architecture for unsupervised circadian (and eventually cell cycle) phase inference from scRNA-seq data, using a small set of known rhythmic genes. The latent space is R² (a 2D plane), where the angular component encodes phase and the radial component encodes rhythm amplitude/confidence. The design is modular and type-hinted throughout: each component (data loading, encoder, predictor, EMA teacher) is independently testable, and the codebase is intentionally small enough to be held in a single session.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

## Usage

_Placeholder — filled in during Phase 6._

## Project structure

```
src/circadian_jepa/
├── data/       # AnnData loading, preprocessing, view generation
├── model/      # Encoder, predictor, EMA teacher
├── training/   # Loss functions and training loop
└── eval/       # Phase inference metrics and visualisation
```

## Development

```bash
ruff check src/   # lint
pytest            # tests
```
