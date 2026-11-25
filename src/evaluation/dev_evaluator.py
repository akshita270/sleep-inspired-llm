"""Development-set measurements, used to CHOOSE training settings.

Three numbers per candidate adapter:
  dev forgetting      of the dev questions the ORIGINAL model got right, the share now wrong
  paraphrase accuracy on reworded questions about the taught facts (the dataset's validation split)
  dev accuracy        knowledge and reasoning accuracy on the dev sets

None of this touches the final evaluation sets (data/evaluation/).
"""
import json
from pathlib import Path

import pandas as pd

from src.evaluation.benchmark_manager import DEV_BENCHMARKS, is_correct, load_benchmark, load_dev
from src.evaluation.evaluation_runner import evaluate_bundle
from src.model.inference import ask
from src.model.model_loader import ModelBundle
from src.utils.config_loader import PROJECT_ROOT, load_config

BASE_DEV_FILE = PROJECT_ROOT / "data" / "dev" / "base_dev_results.csv"


def evaluate_dev(bundle: ModelBundle) -> pd.DataFrame:
    return evaluate_bundle(bundle, names=DEV_BENCHMARKS, loader=load_dev)


def load_base_dev() -> pd.DataFrame:
    return pd.read_csv(BASE_DEV_FILE)


def paraphrase_accuracy(bundle: ModelBundle, dataset_dir: Path) -> tuple[float, int]:
    """Ask the validation-split wordings of the taught facts and score them with the
    facts' accepted answers. Replay rows are ignored."""
    dataset_dir = Path(dataset_dir)
    answers = {i["id"]: i["answers"] for i in load_benchmark("new_knowledge")}
    valid_lines = [json.loads(l) for l in open(dataset_dir / "valid.jsonl") if l.strip()]
    index = [json.loads(l) for l in open(dataset_dir / "index.jsonl") if l.strip()]
    hits = []
    for row in index:
        if row["split"] != "valid" or row["experience_id"] not in answers:
            continue
        question = valid_lines[row["line"]]["messages"][1]["content"]
        hits.append(is_correct(ask(bundle, question)["answer"], answers[row["experience_id"]]))
    return (sum(hits) / len(hits) if hits else 0.0), len(hits)


def dev_forgetting(candidate: pd.DataFrame, base: pd.DataFrame) -> dict:
    m = base.merge(candidate, on=["benchmark", "id"], suffixes=("_base", "_cand"))
    right = m[m["correct_base"]]
    lost = right[~right["correct_cand"]]
    return {"base_correct": int(len(right)), "lost": int(len(lost)),
            "forgetting_rate": round(len(lost) / len(right), 4) if len(right) else 0.0,
            "lost_ids": sorted(lost["id"])}


def dev_report(candidate: pd.DataFrame, base: pd.DataFrame, para_acc: float, n_para: int) -> dict:
    """Numbers plus the pre-registered pass/fail on the dev set."""
    gate = load_config("evaluation_config")["gate"]
    f = dev_forgetting(candidate, base)
    acc = candidate.groupby("benchmark")["correct"].mean().round(4).to_dict()
    return {
        "dev_accuracy": acc,
        "dev_accuracy_base": base.groupby("benchmark")["correct"].mean().round(4).to_dict(),
        "forgetting": f,
        "paraphrase_accuracy": round(para_acc, 4), "paraphrase_n": n_para,
        "passes_dev": f["forgetting_rate"] <= gate["max_forgetting_rate"] + 1e-9
                      and para_acc + 1e-9 >= gate["min_taught_accuracy"],
    }
