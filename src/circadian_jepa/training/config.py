from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class TrainConfig:
    n_genes: int = 15
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    embedding_dim: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 50
    batch_size: int = 128
    lambda_collapse: float = 1.0
    lambda_amplitude: float = 0.1
    ema_momentum: float = 0.996
    view_strategy: str = "downsample"
    view_mode: str = "asymmetric"
    mask_prob: float = 0.0
    seed: int = 0
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: str) -> TrainConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})
