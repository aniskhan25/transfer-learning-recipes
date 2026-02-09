"""CIFAR-10 transfer-learning data utilities.

Builds source and target tasks from class subsets with deterministic sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from data.augment import cifar_weak


class RemappedSubset(Dataset):
    """Subset wrapper that remaps class ids into a compact [0, K-1] space."""

    def __init__(self, dataset: Dataset, indices: np.ndarray, class_map: Dict[int, int]):
        self.dataset = dataset
        self.indices = indices.astype(np.int64)
        self.class_map = class_map

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        image, label = self.dataset[int(self.indices[i])]
        return image, self.class_map[int(label)]


@dataclass
class TransferDataLoaders:
    source_train: DataLoader
    source_test: DataLoader
    target_train: DataLoader
    target_test: DataLoader
    source_probe: DataLoader
    target_probe: DataLoader
    source_num_classes: int
    target_num_classes: int


def _sample_class_subset(
    y: np.ndarray,
    classes: Iterable[int],
    per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices: List[int] = []
    for cls in classes:
        cls_idx = np.where(y == int(cls))[0]
        cls_idx = rng.permutation(cls_idx)
        take = min(int(per_class), int(cls_idx.shape[0]))
        indices.extend(cls_idx[:take].tolist())
    return np.array(indices, dtype=np.int64)


def _build_loader(
    dataset: Dataset,
    indices: np.ndarray,
    class_map: Dict[int, int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = RemappedSubset(dataset, indices, class_map)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_cifar10_transfer(
    data_dir: str,
    source_classes: List[int],
    target_classes: List[int],
    source_train_per_class: int,
    source_test_per_class: int,
    target_train_per_class: int,
    target_test_per_class: int,
    probe_per_class: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> TransferDataLoaders:
    root = Path(data_dir)
    rng = np.random.default_rng(seed)

    train_source_ds = CIFAR10(root, train=True, download=True, transform=cifar_weak())
    train_target_ds = CIFAR10(root, train=True, download=True, transform=cifar_weak())
    test_ds = CIFAR10(root, train=False, download=True, transform=transforms.ToTensor())
    probe_ds = CIFAR10(root, train=True, download=True, transform=transforms.ToTensor())

    y_train = np.array(CIFAR10(root, train=True, download=True, transform=transforms.ToTensor()).targets)
    y_test = np.array(CIFAR10(root, train=False, download=True, transform=transforms.ToTensor()).targets)

    src_train_idx = _sample_class_subset(y_train, source_classes, source_train_per_class, rng)
    src_test_idx = _sample_class_subset(y_test, source_classes, source_test_per_class, rng)
    src_probe_idx = _sample_class_subset(y_train, source_classes, probe_per_class, rng)

    tgt_train_idx = _sample_class_subset(y_train, target_classes, target_train_per_class, rng)
    tgt_test_idx = _sample_class_subset(y_test, target_classes, target_test_per_class, rng)
    tgt_probe_idx = _sample_class_subset(y_train, target_classes, probe_per_class, rng)

    source_map = {int(cls): i for i, cls in enumerate(source_classes)}
    target_map = {int(cls): i for i, cls in enumerate(target_classes)}

    source_train = _build_loader(
        train_source_ds,
        src_train_idx,
        source_map,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    source_test = _build_loader(
        test_ds,
        src_test_idx,
        source_map,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    source_probe = _build_loader(
        probe_ds,
        src_probe_idx,
        source_map,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    target_train = _build_loader(
        train_target_ds,
        tgt_train_idx,
        target_map,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    target_test = _build_loader(
        test_ds,
        tgt_test_idx,
        target_map,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    target_probe = _build_loader(
        probe_ds,
        tgt_probe_idx,
        target_map,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return TransferDataLoaders(
        source_train=source_train,
        source_test=source_test,
        target_train=target_train,
        target_test=target_test,
        source_probe=source_probe,
        target_probe=target_probe,
        source_num_classes=len(source_classes),
        target_num_classes=len(target_classes),
    )
