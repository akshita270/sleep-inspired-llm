"""Load YAML config files from the configs/ folder."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str) -> dict:
    """load_config("model_config") -> dict read from configs/model_config.yaml"""
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
