from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DemoConfig:
    raw: dict[str, Any]
    config_path: Path

    @property
    def references_dir(self) -> Path:
        return Path(self.raw["data"]["references_dir"])

    @property
    def production_dir(self) -> Path:
        return Path(self.raw["data"]["production_dir"])

    @property
    def outputs_dir(self) -> Path:
        return Path(self.raw["data"]["outputs_dir"])

    @property
    def target_id(self) -> str | None:
        value = self.raw["data"].get("target_id")
        return str(value) if value else None


def load_config(path: str | Path) -> DemoConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return DemoConfig(raw=raw, config_path=config_path)
