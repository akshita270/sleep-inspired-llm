"""Experiment A: original Qwen, no memory, no fine-tuning.

Run from the project root:   python -m experiments.baseline
Saves JSON + CSV into experiments/results/.
"""
import json
import statistics
from datetime import datetime

import pandas as pd

from src.evaluation.benchmark_manager import BENCHMARKS, is_correct, load_benchmark
from src.model.inference import ask
from src.model.model_loader import load_model
from src.utils.config_loader import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def run_baseline() -> dict:
    bundle = load_model()
    print(f"Loaded {bundle.version} in {bundle.load_seconds:.1f}s")

    ask(bundle, "Say hello.")  # warm-up: first call compiles GPU kernels, so don't time it

    rows = []
    for bench in BENCHMARKS:
        for item in load_benchmark(bench):
            out = ask(bundle, item["question"])
            rows.append({
                "benchmark": bench,
                "id": item["id"],
                "question": item["question"],
                "expected": " | ".join(item["answers"]),
                "model_answer": out["answer"],
                "correct": is_correct(out["answer"], item["answers"]),
                "latency_s": out["latency_s"],
                "tokens": out["tokens"],
                "tokens_per_s": out["tokens_per_s"],
            })
        done = [r for r in rows if r["benchmark"] == bench]
        print(f"{bench}: {sum(r['correct'] for r in done)}/{len(done)} correct")

    df = pd.DataFrame(rows)
    summary = {
        "experiment": "A_original_qwen",
        "model": bundle.version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "generation_config": bundle.config["generation"],
        "accuracy": df.groupby("benchmark")["correct"].mean().round(4).to_dict(),
        "counts": df.groupby("benchmark")["correct"].agg(["sum", "count"]).to_dict("index"),
        "median_latency_s": round(statistics.median(df["latency_s"]), 3),
        "median_tokens_per_s": round(statistics.median(df["tokens_per_s"]), 2),
        "peak_memory_gb": round(out["peak_memory_gb"], 2),
        "model_load_s": round(bundle.load_seconds, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "baseline_details.csv", index=False)
    with open(RESULTS_DIR / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=int)
    print(json.dumps(summary, indent=2, default=int))
    return summary


if __name__ == "__main__":
    run_baseline()
