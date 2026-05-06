from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMATeacher:
    """Exponential moving-average copy of a student encoder.

    The teacher is never directly optimized; its parameters track the student
    via: teacher_param = momentum * teacher_param + (1 - momentum) * student_param.
    """

    def __init__(self, student: nn.Module, momentum: float = 0.996) -> None:
        self.momentum = momentum
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def set_momentum(self, momentum: float) -> None:
        self.momentum = momentum

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        for t_param, s_param in zip(
            self.teacher.parameters(), student.parameters()
        ):
            t_param.data.mul_(self.momentum).add_(
                s_param.data, alpha=1.0 - self.momentum
            )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher(x)
