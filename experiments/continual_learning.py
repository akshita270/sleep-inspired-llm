"""Experiment C, multi-cycle: does LoRA keep what it learned, and does replay help?

Run from the project root:
    python -m experiments.continual_learning --arm no_replay
    python -m experiments.continual_learning --arm general_replay
    python -m experiments.continual_learning --arm full_replay
    python -m experiments.continual_learning --arm no_replay --smoke      # 2 quick cycles, dev set only

The 30 fictional facts are taught 6 per cycle over 5 cycles. ONE adapter keeps training
from cycle to cycle (the base model stays frozen). After EVERY cycle the full 220-question
evaluation is run, so we can see learning, retention of earlier facts, and forgetting.

Three arms differ only in what each cycle trains on:
    no_replay       the cycle's new facts only
    general_replay  new facts + general questions answered by the original model (1x)
    full_replay     new facts + general replay (1x) + examples of facts from EARLIER cycles (1x)

STUDY MODE: the promotion gate is computed and recorded for every cycle ("would this adapter
have been promoted?"), but the chain continues regardless. Otherwise a rejection at cycle 1
would make cycles 2-5 impossible to study.
Settings (rank 16, 16 layers, lr 5e-5, 5 epochs) are the ones chosen on the dev set in Phase 8.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import pandas as pd

from experiments.common import add_fixture_facts, fresh_fixture_db
from src.evaluation import dev_evaluator as dev
from src.evaluation.benchmark_manager import load_benchmark
from src.evaluation.evaluation_runner import evaluate_bundle, summarise
from src.evaluation.forgetting_evaluator import forgetting_report
from src.evaluation.knowledge_evaluator import taught_accuracy
from src.model.model_loader import load_model
from src.registry import champion_manager as cm
from src.registry import model_registry as reg
from src.sleep.cycle_manager import CycleManager
from src.sleep.question_generator import LLMQuestionGenerator
from src.sleep.sleep_controller import SleepController
from src.training.lora_trainer import LoRATrainer
from src.utils.config_loader import PROJECT_ROOT, load_config

RESULTS = PROJECT_ROOT / "experiments" / "results"
FACTS_PER_CYCLE = 6
N_CYCLES = 5

ARMS = {
    "no_replay": {"replay_ratio": 0.0, "previous_replay_ratio": 0.0},
    "general_replay": {"replay_ratio": 1.0, "previous_replay_ratio": 0.0},
    "full_replay": {"replay_ratio": 1.0, "previous_replay_ratio": 1.0},
}


def fact_group(experience_id: str) -> int:
    """new_001..new_006 -> cycle 1, new_007..new_012 -> cycle 2, ..."""
    return (int(experience_id.split("_")[1]) - 1) // FACTS_PER_CYCLE + 1


def cycle_row(arm: str, k: int, details: pd.DataFrame, base_details: pd.DataFrame,
              base_accuracy: dict, summary: dict, training: dict, manifest: dict,
              taught_ids: list[str], base_verified: bool) -> dict:
    """All the numbers for one (arm, cycle) cell, plus the study-mode gate verdict."""
    fg = forgetting_report(details, base_details)["combined"]
    new = details[details["benchmark"] == "new_knowledge"].copy()
    new["group"] = new["id"].map(fact_group)
    seen, earlier = new[new["group"] <= k], new[new["group"] < k]
    gate_cfg = {**load_config("evaluation_config")["gate"], "require_base_unchanged": base_verified}
    gate = cm.run_gate(summary["accuracy"], base_accuracy, fg["forgetting_rate"],
                       taught_accuracy(details, taught_ids), True if base_verified else None, gate_cfg)
    return {
        "arm": arm, "cycle": k, "facts_taught_so_far": k * FACTS_PER_CYCLE,
        "n_train_examples": training["n_train_examples"],
        "n_general_replay": manifest["replay"]["added"], "n_previous_replay": manifest["replay"]["previous_added"],
        "training_seconds": training["training_seconds"], "final_val_loss": training["final_val_loss"],
        "adapter_mb": round(training["adapter_size_bytes"] / 1e6, 1),
        "acc_new_all30": summary["accuracy"]["new_knowledge"],
        "acc_taught_so_far": round(float(seen["correct"].mean()), 4),
        "acc_this_cycle_facts": round(float(new[new["group"] == k]["correct"].mean()), 4),
        "acc_earlier_cycles_facts": round(float(earlier["correct"].mean()), 4) if len(earlier) else None,
        "acc_by_fact_group": {int(g): round(float(v), 4) for g, v in new.groupby("group")["correct"].mean().items()},
        "acc_old": summary["accuracy"]["old_knowledge"], "acc_reasoning": summary["accuracy"]["reasoning"],
        "forgetting_rate": fg["forgetting_rate"], "questions_lost": fg["lost"],
        "confidently_wrong_old": summary["hallucination"]["old_knowledge"]["confidently_wrong"],
        "median_latency_s": summary["median_latency_s"], "median_tokens_per_s": summary["median_tokens_per_s"],
        "gpu_peak_gb": summary["gpu_peak_memory_gb"], "base_hash_verified": base_verified,
        "gate_would_pass": gate["passed"],
        "gate_failed_checks": [c["check"] for c in gate["checks"] if not c["passed"]],
    }


def free_memory() -> None:
    gc.collect()
    mx.clear_cache()


def run_arm(arm: str, smoke: bool = False) -> list[dict]:
    n_cycles = 2 if smoke else N_CYCLES
    run = RESULTS / ("continual_smoke" if smoke else "continual") / arm
    conn, _ = fresh_fixture_db(run / "experiment.db", ids=[])           # empty database
    items = load_benchmark("new_knowledge")
    base_details = pd.read_csv(RESULTS / "baseline_details.csv")
    base_accuracy = json.loads((RESULTS / "baseline_summary.json").read_text())["accuracy"]
    base_dev = dev.load_base_dev() if smoke else None

    rows, prev_adapter, prev_datasets = [], None, []
    for k in range(1, n_cycles + 1):
        t0 = time.time()
        facts = items[(k - 1) * FACTS_PER_CYCLE: k * FACTS_PER_CYCLE]
        print(f"\n=== {arm} | cycle {k}/{n_cycles} | teaching {facts[0]['id']}..{facts[-1]['id']} ===", flush=True)
        decisions = add_fixture_facts(conn, facts)
        assert decisions["APPROVED"] == len(facts), decisions

        trainer = LoRATrainer(
            adapter_root=run / "adapters", resume_from=prev_adapter,
            overrides={"verify_base_weights": k in (1, n_cycles),        # hashing 8 GB is slow: first and last only
                       **({"epochs": 1} if smoke else {})})
        result = SleepController(
            conn, LLMQuestionGenerator(), train_fn=trainer, dataset_root=run / "datasets",
            dataset_overrides={**ARMS[arm], "previous_datasets": [str(d) for d in prev_datasets]}).run()
        assert result["status"] == "TRAINED", result
        adapter_dir, cycle_id, dataset_dir = Path(result["adapter_path"]), result["cycle_id"], Path(result["dataset_dir"])
        manifest = json.loads((dataset_dir / "manifest.json").read_text())
        print(f"trained in {result['metrics']['training_seconds']}s on {result['metrics']['n_train_examples']} examples "
              f"(replay: general {manifest['replay']['added']}, previous {manifest['replay']['previous_added']})", flush=True)

        mx.reset_peak_memory()
        bundle = load_model(str(adapter_dir))
        if smoke:
            d = dev.evaluate_dev(bundle)
            para, n = dev.paraphrase_accuracy(bundle, dataset_dir)
            rep = dev.dev_report(d, base_dev, para, n)
            print("SMOKE dev:", rep["dev_accuracy"], "forgetting", rep["forgetting"]["forgetting_rate"],
                  "paraphrase", rep["paraphrase_accuracy"], flush=True)
            del bundle; free_memory()
            prev_adapter, prev_datasets = adapter_dir, prev_datasets + [dataset_dir]
            CycleManager(conn).finish(cycle_id, "SUCCESS", notes="smoke test")
            continue

        details = evaluate_bundle(bundle)
        summary = summarise(details, bundle.version, bundle.load_seconds)
        del bundle; free_memory()

        row = cycle_row(arm, k, details, base_details, base_accuracy, summary, result["metrics"], manifest,
                        [f["id"] for f in facts], base_verified=bool(result["metrics"]["base_unchanged"]))
        rows.append(row)
        details.to_csv(run / f"cycle_{k:03d}_details.csv", index=False)
        (run / "results.json").write_text(json.dumps(rows, indent=2))

        # study mode: record the verdict, but keep the chain going and mark the facts as ingested
        reg.set_evaluation(conn, result["adapter_version"], "APPROVED" if row["gate_would_pass"] else "REJECTED",
                           {"study_mode": True, "row": row})
        CycleManager(conn).finish(cycle_id, "SUCCESS", evaluation_result=f"STUDY MODE; gate would pass: {row['gate_would_pass']}")
        prev_adapter, prev_datasets = adapter_dir, prev_datasets + [dataset_dir]

        print(f"cycle {k}: new(all30) {row['acc_new_all30']:.1%} | this-cycle facts {row['acc_this_cycle_facts']:.1%} | "
              f"earlier facts {row['acc_earlier_cycles_facts']} | old {row['acc_old']:.1%} | reasoning {row['acc_reasoning']:.1%} | "
              f"forgetting {row['forgetting_rate']:.1%} | gate {row['gate_would_pass']} {row['gate_failed_checks']} "
              f"| {time.time() - t0:.0f}s", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run_arm(args.arm, args.smoke)


if __name__ == "__main__":
    main()
