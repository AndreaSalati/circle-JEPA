from __future__ import annotations

import argparse
import random

import anndata
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import CircadianDataset
from ..data.views import ViewGenerator
from ..model.jepa import CircadianJEPA
from .config import TrainConfig
from .trainer import Trainer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_from_config(config: TrainConfig, adata: anndata.AnnData) -> CircadianJEPA:
    """Build everything from config, run full training, return trained model."""
    _set_seed(config.seed)

    vg = ViewGenerator(
        view_mode=config.view_mode,
        same_batch=(config.view_strategy == "same_batch"),
        mask_prob=config.mask_prob,
        seed=config.seed,
    )
    dataset = CircadianDataset(adata, view_generator=vg)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    model = CircadianJEPA(
        n_genes=config.n_genes,
        hidden_dims=config.hidden_dims,
        embedding_dim=config.embedding_dim,
        ema_momentum=config.ema_momentum,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    trainer = Trainer(model=model, optimizer=optimizer, device=config.device)
    trainer.fit(
        train_dataloader=dataloader,
        n_epochs=config.n_epochs,
        lambda_collapse=config.lambda_collapse,
        lambda_amplitude=config.lambda_amplitude,
    )
    return model


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train CircadianJEPA from config + h5ad")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data", required=True, help="Path to .h5ad AnnData file")
    args = parser.parse_args()

    import scanpy as sc

    adata = sc.read_h5ad(args.data)
    config = TrainConfig.from_yaml(args.config)
    train_from_config(config, adata)


if __name__ == "__main__":
    _main()
