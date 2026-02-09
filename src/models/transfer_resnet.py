"""Transfer-learning friendly ResNet-18 model."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn
from torchvision.models import resnet18


class TransferResNet18(nn.Module):
    """ResNet-18 with explicit backbone/head split and freeze helpers."""

    def __init__(self, num_classes: int):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def replace_head(self, num_classes: int) -> None:
        self.head = nn.Linear(self.feature_dim, num_classes)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone_all(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_stages(self, stages: Iterable[str]) -> None:
        stage_set = {s.strip() for s in stages}
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in stage_set):
                param.requires_grad = True

    def trainable_parameter_groups(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
