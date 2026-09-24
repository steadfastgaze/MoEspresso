"""Shared source configuration inputs for model-specific package builders."""

import json
from pathlib import Path

INVENTORY_NAME = "source_inventory.json"


def _read_config(model_dir: Path) -> dict:
    config = Path(model_dir) / "config.json"
    return json.loads(config.read_text()) if config.exists() else {}


def _layer_types(config: dict) -> list[str] | None:
    return config.get("text_config", config).get("layer_types")
