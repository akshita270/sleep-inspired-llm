"""Facts about adapter folders on disk. (Which adapter is champion lives in
src/registry/model_registry.py.)

An adapter folder (models/adapters/cycle_001/) holds:
    adapters.safetensors   the learned LoRA matrices (the only thing that changes)
    adapter_config.json    how to rebuild them on the base model
"""
from pathlib import Path

import mlx.core as mx

from src.utils.config_loader import PROJECT_ROOT

ADAPTER_ROOT = PROJECT_ROOT / "models" / "adapters"


def adapter_size_bytes(adapter_dir: Path) -> int:
    return (Path(adapter_dir) / "adapters.safetensors").stat().st_size


def non_lora_keys(adapter_dir: Path) -> list[str]:
    """Names of saved tensors that are NOT LoRA matrices. Should be empty:
    it proves the file contains only adapter weights, not copies of base weights."""
    weights = mx.load(str(Path(adapter_dir) / "adapters.safetensors"))
    return [k for k in weights if "lora_a" not in k and "lora_b" not in k]
