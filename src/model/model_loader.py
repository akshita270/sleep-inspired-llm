"""Load the base Qwen model, optionally with a LoRA adapter on top.

The base weights are never modified. A LoRA adapter is a separate small file
that mlx-lm applies on top of the frozen weights:
    W_effective = W_base + delta_W_LoRA
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

from mlx_lm import load

from src.utils.config_loader import load_config


@dataclass
class ModelBundle:
    model: object
    tokenizer: object
    model_name: str
    adapter_path: str | None
    load_seconds: float
    config: dict = field(default_factory=dict)

    @property
    def version(self) -> str:
        """A readable label recorded with every result."""
        if self.adapter_path:
            return f"{self.model_name}+{Path(self.adapter_path).name}"
        return f"{self.model_name} (base)"


def load_model(adapter_path: str | None = None) -> ModelBundle:
    """Load the base model; pass adapter_path to load a trained LoRA adapter too."""
    cfg = load_config("model_config")
    name = cfg["model"]["name"]
    adapter = adapter_path or cfg["model"]["adapter_path"]

    start = time.time()
    model, tokenizer = load(name, adapter_path=adapter)
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_name=name,
        adapter_path=adapter,
        load_seconds=time.time() - start,
        config=cfg,
    )
