"""Choose training settings using the DEV set only (never the final evaluation sets).

Run from the project root:   python -m experiments.tuning

PRE-REGISTERED PROTOCOL (written before any config was run):
  Candidates: the four configs below, plus 'T0_no_replay' (the adapter that was rejected
  in Phase 8, scored on the dev set for comparison).
  A config PASSES if dev forgetting <= 5% AND paraphrase accuracy >= 80%
  (the same limits as the promotion gate).
  Selection: among passing configs, highest paraphrase accuracy; ties go to the fewer
  training examples. If none passes, report that honestly and choose nothing.
"""
import gc
import json
from pathlib import Path

import mlx.core as mx

from experiments.common import fresh_fixture_db
from src.evaluation import dev_evaluator as dev
from src.model.model_loader import load_model
from src.sleep.question_generator import LLMQuestionGenerator
from src.sleep.sleep_controller import SleepController
from src.training.lora_trainer import LoRATrainer
from src.utils.config_loader import PROJECT_ROOT

OUT = PROJECT_ROOT / "experiments" / "results" / "tuning"
OLD_RUN = PROJECT_ROOT / "experiments" / "results" / "experiment_c_30facts"

# lr stays 5e-5 (the value that already trained stably); we vary replay, epochs and capacity.
CONFIGS = [
    {"name": "T1_replay1_ep5", "replay_ratio": 1.0, "lora": {"epochs": 5}},
    {"name": "T2_replay1_ep3", "replay_ratio": 1.0, "lora": {"epochs": 3}},
    {"name": "T3_replay2_ep3", "replay_ratio": 2.0, "lora": {"epochs": 3}},
    {"name": "T4_replay2_ep3_small", "replay_ratio": 2.0, "lora": {"epochs": 3, "rank": 8, "num_layers": 8}},
]


def free_memory() -> None:
    gc.collect()
    mx.clear_cache()


def score_adapter(adapter_dir, dataset_dir, base_df) -> dict:
    bundle = load_model(str(adapter_dir) if adapter_dir else None)
    dev_df = dev.evaluate_dev(bundle)
    para, n = dev.paraphrase_accuracy(bundle, dataset_dir)
    report = dev.dev_report(dev_df, base_df, para, n)
    del bundle
    free_memory()
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not dev.BASE_DEV_FILE.exists():
        base = load_model()
        dev.evaluate_dev(base).to_csv(dev.BASE_DEV_FILE, index=False)
        del base
        free_memory()
    base_df = dev.load_base_dev()
    print("base model on dev:", base_df.groupby("benchmark")["correct"].mean().round(3).to_dict())

    results = {}
    results["T0_no_replay"] = score_adapter(
        OLD_RUN / "adapters" / "cycle_003", OLD_RUN / "datasets" / "cycle_003", base_df)
    results["T0_no_replay"]["n_train_examples"] = 113
    print("T0_no_replay", json.dumps(results["T0_no_replay"]["forgetting"]), results["T0_no_replay"]["paraphrase_accuracy"])

    for cfg in CONFIGS:
        run = OUT / cfg["name"]
        conn, _ = fresh_fixture_db(run / "experiment.db")
        trainer = LoRATrainer(adapter_root=run / "adapters",
                              overrides={**cfg["lora"], "verify_base_weights": False})  # verified in Phase 7
        generator = LLMQuestionGenerator()

        def train_fn(dataset_dir, cycle_id, _t=trainer, _g=generator):
            _g.release(); free_memory()
            return _t(dataset_dir, cycle_id)

        result = SleepController(conn, generator, train_fn=train_fn, dataset_root=run / "datasets",
                                 dataset_overrides={"replay_ratio": cfg["replay_ratio"]}).run()
        report = score_adapter(result["adapter_path"], result["dataset_dir"], base_df)
        report.update(config=cfg, training_seconds=result["metrics"]["training_seconds"],
                      final_val_loss=result["metrics"]["final_val_loss"],
                      n_train_examples=result["metrics"]["n_train_examples"])
        results[cfg["name"]] = report
        print(cfg["name"], "forgetting", report["forgetting"]["forgetting_rate"],
              "paraphrase", report["paraphrase_accuracy"], "passes", report["passes_dev"], flush=True)
        (OUT / "tuning_summary.json").write_text(json.dumps(results, indent=2))

    passing = {k: v for k, v in results.items() if v["passes_dev"] and k != "T0_no_replay"}
    chosen = (max(passing, key=lambda k: (passing[k]["paraphrase_accuracy"], -passing[k]["n_train_examples"]))
              if passing else None)
    results["_selected"] = chosen
    (OUT / "tuning_summary.json").write_text(json.dumps(results, indent=2))

    print("\nconfig                  forgetting  paraphrase  dev_know  dev_reason  train_ex  passes")
    for k, v in results.items():
        if k.startswith("_"):
            continue
        a = v["dev_accuracy"]
        print(f"{k:<22} {v['forgetting']['forgetting_rate']:>9.1%} {v['paraphrase_accuracy']:>11.1%} "
              f"{a['dev_knowledge']:>9.1%} {a['dev_reasoning']:>11.1%} {v['n_train_examples']:>9} {str(v['passes_dev']):>7}")
    print("SELECTED:", chosen or "none passed")


if __name__ == "__main__":
    main()
