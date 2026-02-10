from __future__ import annotations

import copy
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml


def find_repo_root(start: Path | None = None) -> Path:
    cursor = (start or Path.cwd()).resolve()
    while True:
        if (cursor / "configs").is_dir() and (cursor / "scripts" / "run_transfer.py").is_file():
            return cursor
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    raise FileNotFoundError("Could not locate repository root.")


def _deep_update(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_fast_dev_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    data = cfg.setdefault("data", {})
    train = cfg.setdefault("train", {})

    for key, cap in {
        "source_train_per_class": 220,
        "source_test_per_class": 80,
        "target_train_per_class": 30,
        "target_test_per_class": 80,
        "probe_per_class": 40,
    }.items():
        if key in data:
            data[key] = min(int(data[key]), cap)

    data["batch_size"] = min(int(data.get("batch_size", 128)), 64)
    data["num_workers"] = 0

    if "source_epochs" in train:
        train["source_epochs"] = min(int(train["source_epochs"]), 2)
    if "target_epochs" in train:
        train["target_epochs"] = min(int(train["target_epochs"]), 3)

    train["gradual_schedule"] = {
        "1": ["backbone.layer4"],
        "2": ["backbone.layer3"],
    }
    return cfg


@dataclass(frozen=True)
class TransferNotebookLab:
    root: Path

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def logs_dir(self) -> Path:
        out = self.root / "outputs" / "logs"
        out.mkdir(parents=True, exist_ok=True)
        return out

    @property
    def figures_dir(self) -> Path:
        out = self.root / "outputs" / "figures"
        out.mkdir(parents=True, exist_ok=True)
        return out

    @classmethod
    def from_root(cls, root: Path) -> "TransferNotebookLab":
        return cls(root=root.resolve())

    @classmethod
    def from_cwd(cls) -> "TransferNotebookLab":
        return cls.from_root(find_repo_root())

    def make_profiled_config(
        self,
        base_name: str,
        notebook_tag: str,
        fast_dev_run: bool,
        overrides: Mapping[str, Any] | None = None,
    ) -> Path:
        cfg = yaml.safe_load((self.configs_dir / base_name).read_text())
        if fast_dev_run:
            cfg = _apply_fast_dev_profile(cfg)
        if overrides:
            cfg = _deep_update(cfg, copy.deepcopy(dict(overrides)))

        suffix = "fast" if fast_dev_run else "full"
        out = self.configs_dir / f"tmp_{notebook_tag}_{suffix}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return out

    def run_config(self, config_path: Path, use_progress: bool = True) -> None:
        cmd = ["python", str(self.root / "scripts" / "run_transfer.py"), "--config", str(config_path)]
        if use_progress:
            cmd.append("--use-progress")
        subprocess.run(cmd, cwd=self.root, check=True)

    def run(
        self,
        base_name: str,
        notebook_tag: str,
        fast_dev_run: bool = False,
        overrides: Mapping[str, Any] | None = None,
        use_progress: bool | None = None,
    ) -> Path:
        config_path = self.make_profiled_config(
            base_name=base_name,
            notebook_tag=notebook_tag,
            fast_dev_run=fast_dev_run,
            overrides=overrides,
        )
        self.run_config(config_path, use_progress=(not fast_dev_run) if use_progress is None else use_progress)
        return config_path

    def read_method(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.logs_dir / f"transfer_{name}.csv")

    def load_methods(self, names: Sequence[str]) -> dict[str, pd.DataFrame]:
        return {name: self.read_method(name) for name in names}

    def savefig(self, fig: Any, filename: str, dpi: int = 150) -> Path:
        out = self.figures_dir / filename
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        return out
