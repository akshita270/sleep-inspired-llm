"""Experiment B: original Qwen + episodic memory (retrieval), no fine-tuning.

Run from the project root:   python -m experiments.episodic_memory

Same model, same generation settings, same 90 test questions as Experiment A.
The only difference: relevant approved facts are pasted into the prompt.
"""
import json
import statistics
import time
from datetime import datetime

import pandas as pd

from src.evaluation.benchmark_manager import BENCHMARKS, is_correct, load_benchmark
from src.memory.database import get_connection
from src.memory.episodic_memory import EpisodicMemory
from src.model.inference import ask
from src.model.model_loader import load_model
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.validation.approval_manager import ApprovalManager

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
DB_FILE = RESULTS_DIR / "experiment_b.db"   # separate DB so sleep_learning.db stays clean


def build_memory() -> tuple[EpisodicMemory, dict]:
    """Feed the 30 learning facts through the real validation pipeline."""
    DB_FILE.unlink(missing_ok=True)
    conn = get_connection(DB_FILE)
    mgr = ApprovalManager(conn)

    decisions = {"APPROVED": 0, "PENDING": 0, "REJECTED": 0, "SKIPPED": 0}
    for item in load_benchmark("new_knowledge"):
        res = mgr.submit({
            "experience_id": item["id"], "session_id": "exp_b", "interaction_type": "NEW_FACT",
            "original_input": item["fact"], "corrected_answer": item["fact"],
            "category": "general", "source": "experiment_fixture",
            "previous_answer": None, "previous_question": None,
            "learn_requested": False, "requires_explicit_approval": False,
        })
        decisions[res["status"]] += 1
        if res["status"] != "APPROVED":
            print("  not approved:", item["id"], res["notes"])

    memory = EpisodicMemory(conn)
    memory.refresh()
    return memory, decisions


def run() -> dict:
    memory, decisions = build_memory()
    print("Validation decisions for the 30 facts:", decisions)

    bundle = load_model()
    ask(bundle, "Say hello.")  # warm-up

    rows = []
    for bench in BENCHMARKS:
        for item in load_benchmark(bench):
            t0 = time.perf_counter()
            context, hits = memory.context_for(item["question"])
            retrieval_s = time.perf_counter() - t0
            out = ask(bundle, item["question"], context=context)
            rows.append({
                "benchmark": bench, "id": item["id"], "question": item["question"],
                "expected": " | ".join(item["answers"]), "model_answer": out["answer"],
                "correct": is_correct(out["answer"], item["answers"]),
                "retrieved_ids": ",".join(h["experience_id"] for h in hits),
                # retrieval hit: the fact written for THIS question was in the retrieved set
                "retrieval_hit": (item["id"] in [h["experience_id"] for h in hits])
                                 if bench == "new_knowledge" else None,
                "n_retrieved": len(hits),
                "retrieval_s": retrieval_s, "latency_s": out["latency_s"] + retrieval_s,
                "tokens": out["tokens"],
            })
        done = [r for r in rows if r["benchmark"] == bench]
        print(f"{bench}: {sum(r['correct'] for r in done)}/{len(done)} correct")

    df = pd.DataFrame(rows)
    new = df[df.benchmark == "new_knowledge"]
    others = df[df.benchmark != "new_knowledge"]
    summary = {
        "experiment": "B_qwen_plus_episodic_memory",
        "model": bundle.version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "memory_config": load_config("system_config")["episodic_memory"],
        "validation_decisions": decisions,
        "accuracy": df.groupby("benchmark")["correct"].mean().round(4).to_dict(),
        "counts": df.groupby("benchmark")["correct"].agg(["sum", "count"]).to_dict("index"),
        "retrieval_accuracy_top_k": round(float(new["retrieval_hit"].astype(float).mean()), 4),
        "unrelated_questions_that_got_memory_context": int((others["n_retrieved"] > 0).sum()),
        "unrelated_questions_total": int(len(others)),
        "median_latency_s": round(statistics.median(df["latency_s"]), 3),
        "median_retrieval_ms": round(statistics.median(df["retrieval_s"]) * 1000, 1),
        "extra_storage_bytes": DB_FILE.stat().st_size,
        "peak_memory_gb": round(out["peak_memory_gb"], 2),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "episodic_details.csv", index=False)
    with open(RESULTS_DIR / "episodic_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=int)

    with open(RESULTS_DIR / "baseline_summary.json") as f:
        a = json.load(f)
    print("\n            A: original   B: + memory")
    for b in BENCHMARKS:
        print(f"{b:<14} {a['accuracy'][b]:>8.1%}   {summary['accuracy'][b]:>8.1%}")
    print(f"{'latency (s)':<14} {a['median_latency_s']:>8}   {summary['median_latency_s']:>8}")
    print(f"retrieval accuracy (top-{summary['memory_config']['top_k']}): {summary['retrieval_accuracy_top_k']:.1%}")
    print(f"unrelated questions that still received memory text: "
          f"{summary['unrelated_questions_that_got_memory_context']}/{summary['unrelated_questions_total']}")
    return summary


if __name__ == "__main__":
    run()
