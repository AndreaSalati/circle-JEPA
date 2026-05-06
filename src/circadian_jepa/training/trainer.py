from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader

from ..model.jepa import CircadianJEPA
from .losses import total_loss


class Trainer:
    def __init__(
        self,
        model: CircadianJEPA,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        log_every: int = 50,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_every = log_every

    def train_epoch(
        self,
        dataloader: DataLoader,
        view_generator=None,
        lambda_collapse: float = 1.0,
        lambda_amplitude: float = 0.1,
        n_harmonics: int = 2,
    ) -> dict:
        """Run one training epoch. Views are taken directly from the dataloader batch."""
        self.model.train()
        running = {"predict": 0.0, "collapse": 0.0, "amplitude": 0.0, "total": 0.0}
        n_batches = 0

        for i, batch in enumerate(dataloader):
            view_a = batch["view_a"].to(self.device)
            view_b = batch["view_b"].to(self.device)

            out = self.model(view_a, view_b)
            loss, components = total_loss(out, lambda_collapse, lambda_amplitude, n_harmonics)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.step_ema()

            for k, v in components.items():
                running[k] += v
            n_batches += 1

            if (i + 1) % self.log_every == 0:
                avg = {k: v / n_batches for k, v in running.items()}
                print(
                    f"  step {i + 1}: "
                    + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
                )

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    def fit(
        self,
        train_dataloader: DataLoader,
        n_epochs: int,
        view_generator=None,
        lambda_collapse: float = 1.0,
        lambda_amplitude: float = 0.1,
        n_harmonics: int = 2,
        callback: Callable[[int, dict], None] | None = None,
    ) -> list[dict]:
        """Train for n_epochs, returning per-epoch metric history."""
        history: list[dict] = []

        for epoch in range(1, n_epochs + 1):
            metrics = self.train_epoch(
                train_dataloader, view_generator, lambda_collapse, lambda_amplitude, n_harmonics
            )
            if self.scheduler is not None:
                self.scheduler.step()
            history.append(metrics)

            if callback is not None:
                callback(epoch, metrics)
            else:
                print(
                    f"Epoch {epoch}/{n_epochs} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                )

        return history
