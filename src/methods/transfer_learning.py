"""Transfer-learning training recipes and instrumentation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.transfer_resnet import TransferResNet18
from utils.progress import progress


@dataclass
class FitHistory:
    history: List[Dict[str, float]]


def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return (correct / total) if total else 0.0


def evaluate_with_external_head(
    backbone_model: TransferResNet18,
    eval_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    backbone_model.eval()
    eval_head.eval()
    eval_head = eval_head.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            feat = backbone_model.extract_features(images)
            pred = eval_head(feat).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return (correct / total) if total else 0.0


def collect_features(model: TransferResNet18, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    feats = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            feats.append(model.extract_features(images).cpu())
    return torch.cat(feats, dim=0) if feats else torch.zeros((0, model.feature_dim))


def feature_drift(reference: torch.Tensor, current: torch.Tensor) -> float:
    if reference.numel() == 0 or current.numel() == 0:
        return 0.0
    ref = torch.nn.functional.normalize(reference.float(), dim=1)
    cur = torch.nn.functional.normalize(current.float(), dim=1)
    cos = (ref * cur).sum(dim=1).mean().item()
    return float(max(0.0, 1.0 - cos))


def average_grad_norm(model: nn.Module) -> float:
    norms = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            norms.append(float(p.grad.norm().item()))
    return float(np.mean(norms)) if norms else 0.0


def pretrain_source(
    model: TransferResNet18,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    momentum: float,
    use_progress: bool = False,
) -> FitHistory:
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    ce = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []

    for epoch in progress(range(epochs), enabled=use_progress, desc="source pretrain"):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = ce(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        source_acc = evaluate_classifier(model, test_loader, device)
        history.append({"epoch": float(epoch), "source_test_acc": float(source_acc)})

    return FitHistory(history=history)


def run_target_adaptation(
    model: TransferResNet18,
    target_train: DataLoader,
    target_test: DataLoader,
    target_probe: DataLoader,
    source_test: DataLoader | None,
    source_head: nn.Module | None,
    device: torch.device,
    strategy: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    momentum: float,
    gradual_schedule: Dict[int, List[str]] | None = None,
    use_progress: bool = False,
) -> FitHistory:
    model.to(device)
    ce = nn.CrossEntropyLoss()
    gradual_schedule = gradual_schedule or {}

    if strategy == "feature_extraction":
        model.freeze_backbone()
        for p in model.head.parameters():
            p.requires_grad = True
    elif strategy == "naive_finetune":
        model.unfreeze_backbone_all()
    elif strategy == "gradual_unfreeze":
        model.freeze_backbone()
        for p in model.head.parameters():
            p.requires_grad = True
    elif strategy == "scratch":
        model.unfreeze_backbone_all()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    feature_ref = collect_features(model, target_probe, device)
    history: List[Dict[str, float]] = []

    for epoch in progress(range(epochs), enabled=use_progress, desc=f"{strategy} target train"):
        if strategy == "gradual_unfreeze" and epoch in gradual_schedule:
            model.unfreeze_stages(gradual_schedule[epoch])
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        model.train()
        grad_norm_steps = []
        for images, labels in target_train:
            images, labels = images.to(device), labels.to(device)
            loss = ce(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            grad_norm_steps.append(average_grad_norm(model))
            optimizer.step()

        target_acc = evaluate_classifier(model, target_test, device)
        drift = feature_drift(feature_ref, collect_features(model, target_probe, device))
        if source_test is not None and source_head is not None:
            source_retention = evaluate_with_external_head(model, source_head, source_test, device)
        else:
            source_retention = float("nan")

        history.append(
            {
                "epoch": float(epoch),
                "target_test_acc": float(target_acc),
                "feature_drift": float(drift),
                "source_retention_acc": float(source_retention),
                "grad_norm": float(np.mean(grad_norm_steps)) if grad_norm_steps else 0.0,
            }
        )

    return FitHistory(history=history)


def build_transferred_model(
    source_model: TransferResNet18,
    target_num_classes: int,
) -> tuple[TransferResNet18, nn.Module]:
    transferred = deepcopy(source_model)
    source_head = deepcopy(source_model.head).cpu()
    transferred.replace_head(target_num_classes)
    return transferred, source_head
