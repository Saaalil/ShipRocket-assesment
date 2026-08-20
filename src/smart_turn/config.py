from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    data_ref = config.get("data")
    if data_ref:
        data_cfg = load_yaml(data_ref)
        config = {**data_cfg, **config}
    return config
