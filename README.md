# Circadian-JEPA

A JEPA-style architecture for unsupervised circadian and cell cycle phase inference from scRNA-seq data. A circular latent space is constructed with JEPA-style learning.

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
