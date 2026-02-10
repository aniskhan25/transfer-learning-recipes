"""Compatibility wrapper for supervised training.

The canonical implementation lives in ``methods.supervised``.
Keep this module to avoid breaking older imports.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from methods.supervised import SupervisedResult, run_supervised as _run_supervised

TrainResult = SupervisedResult


def run_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    use_progress: bool = False,
) -> TrainResult:
    return _run_supervised(
        model=model,
        labeled_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        use_progress=use_progress,
    )
