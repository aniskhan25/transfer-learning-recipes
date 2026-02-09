import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data.cifar10_transfer import get_cifar10_transfer
from methods.transfer_learning import (
    build_transferred_model,
    pretrain_source,
    run_target_adaptation,
)
from models.transfer_resnet import TransferResNet18
from utils.logging import CSVLogger
from utils.plots import plot_series, savefig
from utils.seed import set_seed


def _plot_target_curves(rows_by_method: Dict[str, List[Dict[str, float]]], out_path: Path) -> None:
    if not rows_by_method:
        return
    first = next(iter(rows_by_method.values()))
    xs = list(range(len(first)))
    series = {k: [r["target_test_acc"] for r in v] for k, v in rows_by_method.items()}
    plot_series(xs, series, "Target accuracy by strategy", "target_acc")
    savefig(out_path)


def _plot_stability_curves(rows_by_method: Dict[str, List[Dict[str, float]]], out_path: Path) -> None:
    if not rows_by_method:
        return
    first = next(iter(rows_by_method.values()))
    xs = list(range(len(first)))
    series = {k: [r["feature_drift"] for r in v] for k, v in rows_by_method.items()}
    plot_series(xs, series, "Feature drift by strategy", "drift")
    savefig(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--use-progress", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(int(cfg.get("seed", 0)))

    out_dir = Path(cfg.get("output_dir", "outputs"))
    logs_dir = out_dir / "logs"
    figs_dir = out_dir / "figures"

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    methods: List[str] = cfg.get("methods", ["scratch", "feature_extraction", "gradual_unfreeze", "naive_finetune"])

    loaders = get_cifar10_transfer(
        data_dir=data_cfg["data_dir"],
        source_classes=data_cfg["source_classes"],
        target_classes=data_cfg["target_classes"],
        source_train_per_class=data_cfg["source_train_per_class"],
        source_test_per_class=data_cfg["source_test_per_class"],
        target_train_per_class=data_cfg["target_train_per_class"],
        target_test_per_class=data_cfg["target_test_per_class"],
        probe_per_class=data_cfg["probe_per_class"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        seed=cfg["seed"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_model = TransferResNet18(num_classes=loaders.source_num_classes)
    source_hist = pretrain_source(
        source_model,
        loaders.source_train,
        loaders.source_test,
        device=device,
        epochs=train_cfg["source_epochs"],
        lr=train_cfg["lr_source"],
        weight_decay=train_cfg["weight_decay"],
        momentum=train_cfg["momentum"],
        use_progress=args.use_progress,
    )

    source_logger = CSVLogger(logs_dir, "transfer_source_pretrain")
    for row in source_hist.history:
        source_logger.log(row)
    source_logger.flush()

    rows_by_method: Dict[str, List[Dict[str, float]]] = {}
    schedule = {int(k): v for k, v in train_cfg.get("gradual_schedule", {}).items()}

    for method_name in methods:
        if method_name == "scratch":
            model = TransferResNet18(num_classes=loaders.target_num_classes)
            source_head = None
            source_test = None
        else:
            model, source_head = build_transferred_model(source_model, loaders.target_num_classes)
            source_test = loaders.source_test

        result = run_target_adaptation(
            model=model,
            target_train=loaders.target_train,
            target_test=loaders.target_test,
            target_probe=loaders.target_probe,
            source_test=source_test,
            source_head=source_head,
            device=device,
            strategy=method_name,
            epochs=train_cfg["target_epochs"],
            lr=train_cfg["lr_target"],
            weight_decay=train_cfg["weight_decay"],
            momentum=train_cfg["momentum"],
            gradual_schedule=schedule,
            use_progress=args.use_progress,
        )

        logger = CSVLogger(logs_dir, f"transfer_{method_name}")
        for row in result.history:
            logger.log(row)
        logger.flush()
        rows_by_method[method_name] = result.history

    _plot_target_curves(rows_by_method, figs_dir / "transfer_target_acc.png")
    _plot_stability_curves(rows_by_method, figs_dir / "transfer_feature_drift.png")


if __name__ == "__main__":
    main()
