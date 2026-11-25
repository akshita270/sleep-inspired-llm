"""Run the three fixed benchmarks on any model and summarise the result.

Works for the base model, a model with a LoRA adapter, or either one with
retrieved memory pasted into the prompt (pass `context_fn`).
"""
import statistics
from datetime import datetime
from typing import Callable

import mlx.core as mx
import pandas as pd

from src.evaluation.benchmark_manager import BENCHMARKS, is_correct, load_benchmark
from src.evaluation.knowledge_evaluator import hallucination_stats
from src.model.inference import ask
from src.model.model_loader import ModelBundle


def evaluate_bundle(bundle: ModelBundle,
                    context_fn: Callable[[str], str | None] | None = None,
                    names=None, loader=None) -> pd.DataFrame:
    """One row per question: the answer, whether it is correct, and its timing.
    By default runs the three fixed evaluation sets; pass names/loader for the dev sets."""
    names = names or BENCHMARKS
    loader = loader or load_benchmark
    ask(bundle, "Say hello.")   # warm-up, not timed or scored
    rows = []
    for bench in names:
        for item in loader(bench):
            out = ask(bundle, item["question"], context=context_fn(item["question"]) if context_fn else None)
            rows.append({
                "benchmark": bench, "id": item["id"], "type": item.get("type"),
                "question": item["question"], "expected": " | ".join(item["answers"]),
                "model_answer": out["answer"], "correct": is_correct(out["answer"], item["answers"]),
                "latency_s": out["latency_s"], "tokens": out["tokens"],
                "tokens_per_s": out["tokens_per_s"],
            })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, label: str, load_seconds: float | None = None) -> dict:
    """Headline numbers. GPU peak memory is measured in THIS process since the last
    mx.reset_peak_memory(), so call reset before loading the model."""
    reasoning = df[df["benchmark"] == "reasoning"]
    return {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "accuracy": df.groupby("benchmark")["correct"].mean().round(4).to_dict(),
        "counts": {k: {"correct": int(v["sum"]), "total": int(v["count"])}
                   for k, v in df.groupby("benchmark")["correct"].agg(["sum", "count"]).to_dict("index").items()},
        "reasoning_by_type": reasoning.groupby("type")["correct"].mean().round(4).to_dict(),
        "hallucination": hallucination_stats(df),
        "median_latency_s": round(statistics.median(df["latency_s"]), 3),
        "median_tokens_per_s": round(statistics.median(df["tokens_per_s"]), 2),
        "gpu_peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        "model_load_s": round(load_seconds, 1) if load_seconds is not None else None,
    }
