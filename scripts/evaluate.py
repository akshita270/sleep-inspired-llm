"""Evaluate a candidate adapter, run the gate, and promote or reject it.

    python -m scripts.evaluate --db PATH --version adapter_cycle_003 --out DIR

Steps: state -> EVALUATING, load model + adapter, run 220 questions, compare with the
champion and the original model, apply the gate, record the decision, wake up.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import pandas as pd

from src.evaluation.evaluation_runner import evaluate_bundle, summarise
from src.evaluation.forgetting_evaluator import forgetting_report
from src.evaluation.knowledge_evaluator import taught_accuracy
from src.memory.database import get_connection
from src.model.model_loader import load_model
from src.registry import champion_manager as cm
from src.registry import model_registry as reg
from src.sleep.sleep_state import EVALUATING, SystemState
from src.utils.config_loader import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "experiments" / "results"


def champion_accuracy(conn) -> dict:
    """Accuracy of whoever is serving now: the current champion, else the original model."""
    champ = reg.champion_version(conn)
    if champ:
        return reg.get_metrics(conn, champ)["evaluation"]["summary"]["accuracy"]
    return json.loads((RESULTS / "baseline_summary.json").read_text())["accuracy"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    conn = get_connection(args.db)
    version = reg.get_version(conn, args.version)
    if version["evaluation_status"] != "PENDING":
        raise SystemExit(f"{args.version} is already {version['evaluation_status']}")
    cycle = conn.execute("SELECT cycle_id FROM sleep_cycles WHERE adapter_version=?",
                         (args.version,)).fetchone()["cycle_id"]
    taught = [r[0] for r in conn.execute(
        "SELECT experience_id FROM cycle_experiences WHERE cycle_id=?", (cycle,))]

    with SystemState(conn).busy(EVALUATING):
        champ_acc = champion_accuracy(conn)
        mx.reset_peak_memory()
        bundle = load_model(version["adapter_path"])
        details = evaluate_bundle(bundle)
        summary = summarise(details, bundle.version, bundle.load_seconds)

        base_details = pd.read_csv(RESULTS / "baseline_details.csv")
        forgetting = forgetting_report(details, base_details)
        training = reg.get_metrics(conn, args.version).get("training", {})
        gate = cm.run_gate(summary["accuracy"], champ_acc, forgetting["combined"]["forgetting_rate"],
                           taught_accuracy(details, taught), training.get("base_unchanged"))
        evaluation = {"summary": summary, "forgetting": forgetting, "gate": gate,
                      "champion_accuracy_before": champ_acc,
                      "taught_accuracy": taught_accuracy(details, taught)}

        (cm.promote if gate["passed"] else cm.reject)(conn, args.version, cycle, evaluation)

    args.out.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.out / f"{args.version}_details.csv", index=False)
    (args.out / f"{args.version}_evaluation.json").write_text(json.dumps(evaluation, indent=2))

    print(json.dumps({"accuracy": summary["accuracy"], "counts": summary["counts"],
                      "forgetting_combined": forgetting["combined"], "taught_accuracy": evaluation["taught_accuracy"],
                      "gate": gate, "latency_s": summary["median_latency_s"],
                      "gpu_peak_gb": summary["gpu_peak_memory_gb"]}, indent=2))
    print("DECISION:", "PROMOTED" if gate["passed"] else "REJECTED", "| champion now:",
          reg.champion_version(conn) or "original base model")


if __name__ == "__main__":
    main()
