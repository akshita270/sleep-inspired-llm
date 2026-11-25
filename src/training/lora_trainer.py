"""LoRA fine-tuning of Qwen with mlx-lm.

What LoRA does: for a frozen weight matrix W, learn two small matrices A and B
and use    W_effective = W_base + B @ A     (the "delta W").
Only A and B are trained and saved. W_base is never modified.

Training runs in a SEPARATE PROCESS so the 8 GB model used for chat/question
writing is not in memory at the same time (16 GB Mac), and so a crash in
training cannot take the controller down.
"""
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.model import adapter_manager as am
from src.training.training_monitor import parse_log
from src.utils.config_loader import load_config


def base_model_folder(model_name: str, cache_dir: Path | None = None) -> Path:
    """Locate the cached weight files. We read the cache layout directly because
    huggingface_hub calls a snapshot 'incomplete' when only README/LICENSE are missing."""
    from huggingface_hub.constants import HF_HUB_CACHE
    repo = Path(cache_dir or HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}"
    commit = (repo / "refs" / "main").read_text().strip()
    return repo / "snapshots" / commit


def hash_base_model(model_name: str, cache_dir: Path | None = None) -> str:
    """One SHA-256 over every weight file of the cached base model."""
    folder = base_model_folder(model_name, cache_dir)
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors weight files found in {folder}")
    h = hashlib.sha256()
    for f in files:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


class LoRATrainer:
    """Callable with the signature SleepController expects: (dataset_dir, cycle_id) -> dict."""

    def __init__(self, adapter_root: Path = am.ADAPTER_ROOT, resume_from: Path | None = None,
                 overrides: dict | None = None):
        self.adapter_root = Path(adapter_root)
        self.resume_from = Path(resume_from) if resume_from else None   # continue an earlier adapter
        # config file values, optionally overridden for one experiment (saved in train_config.yaml)
        self.lora = {**load_config("training_config")["lora"], **(overrides or {})}
        self.model_name = load_config("model_config")["model"]["name"]

    def build_config(self, dataset_dir: Path, out_dir: Path, n_train: int) -> dict:
        c = self.lora
        steps_per_epoch = math.ceil(n_train / c["batch_size"])
        cfg = {
            "model": self.model_name, "train": True, "test": True,
            "data": str(dataset_dir), "adapter_path": str(out_dir),
            "fine_tune_type": "lora", "num_layers": c["num_layers"],
            "batch_size": c["batch_size"], "iters": steps_per_epoch * c["epochs"],
            "learning_rate": c["learning_rate"], "max_seq_length": c["max_seq_length"],
            "grad_checkpoint": c["grad_checkpoint"], "mask_prompt": c["mask_prompt"],
            "steps_per_report": c["steps_per_report"], "steps_per_eval": steps_per_epoch,
            "val_batches": -1, "save_every": 10_000, "seed": c["seed"],
            "lora_parameters": {"rank": c["rank"], "scale": c["scale"], "dropout": c["dropout"]},
        }
        if self.resume_from:
            cfg["resume_adapter_file"] = str(self.resume_from / "adapters.safetensors")
        return cfg

    def __call__(self, dataset_dir: Path, cycle_id: str) -> dict:
        dataset_dir = Path(dataset_dir)
        out_dir = self.adapter_root / cycle_id
        out_dir.mkdir(parents=True, exist_ok=False)      # never overwrite an old adapter

        n_train = count_lines(dataset_dir / "train.jsonl")
        cfg = self.build_config(dataset_dir, out_dir, n_train)
        cfg_file = out_dir / "train_config.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False))

        verify = self.lora["verify_base_weights"]
        hash_before = hash_base_model(self.model_name) if verify else None

        started = time.time()
        log_path = out_dir / "train.log"
        with open(log_path, "w") as log, subprocess.Popen(
                [sys.executable, "-m", "mlx_lm", "lora", "-c", str(cfg_file)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if line.startswith(("Iter", "Trainable", "Test", "Loading")):
                    print(line.rstrip(), flush=True)
        seconds = time.time() - started
        # NOTE: this is process (CPU-side) memory only. MLX keeps model and activations in
        # GPU/unified memory that this number does NOT include, so it badly UNDERCOUNTS
        # (measured 1.8 GB for a run that holds an 8 GB model). Do not report it as peak memory.
        peak_rss_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e9
        if proc.returncode != 0:
            raise RuntimeError(f"mlx_lm lora failed (exit {proc.returncode}); see {log_path}")

        # Base model must be byte-for-byte identical after training.
        hash_after = hash_base_model(self.model_name) if verify else None
        if verify and hash_before != hash_after:
            raise RuntimeError("BASE MODEL FILES CHANGED DURING TRAINING")

        stray = am.non_lora_keys(out_dir)
        if stray:
            raise RuntimeError(f"adapter file contains non-LoRA tensors: {stray[:3]}")

        for ckpt in out_dir.glob("0*_adapters.safetensors"):    # keep only the final adapter
            ckpt.unlink()

        metrics = parse_log(log_path.read_text())
        metrics.update({
            "training_seconds": round(seconds, 1), "n_train_examples": n_train,
            "cpu_side_rss_gb_UNDERCOUNTS_gpu": round(peak_rss_gb, 2),
            "iters": cfg["iters"], "adapter_size_bytes": am.adapter_size_bytes(out_dir),
            "base_hash_before": hash_before, "base_hash_after": hash_after,
            "base_unchanged": (hash_before == hash_after) if verify else None,
            "resumed_from": str(self.resume_from) if self.resume_from else None,
            "config": cfg,
        })
        (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
        return {"adapter_version": f"adapter_{cycle_id}", "adapter_path": out_dir, "metrics": metrics}
