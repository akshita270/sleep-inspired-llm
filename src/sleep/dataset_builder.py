"""Turn approved experiences into versioned instruction-response training files.

One approved fact, e.g.
    "The capital of Zorvania is Kelmirth."
becomes several (question -> answer) pairs, each worded differently:
    "Which city is the capital of Zorvania?"  ->  "The capital of Zorvania is Kelmirth."
so the model learns the fact, not one fixed phrasing.

Safety rules in this file:
  - Leak guard: any training question that equals or closely resembles a question
    from the fixed evaluation sets is DROPPED. The evaluation stays unseen.
  - Duplicate guard: the same question never appears twice.
  - Splits are made per fact: each fact keeps some wordings for training and
    some for validation/test, so validation loss measures generalisation to new
    wordings of facts we taught.
  - A dataset folder is written once and never overwritten (reproducibility).
"""
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from src.evaluation.benchmark_manager import BENCHMARKS, load_benchmark
from src.utils.config_loader import PROJECT_ROOT
from src.validation.duplicate_detector import jaccard, normalise, tokens

TRAINING_DIR = PROJECT_ROOT / "data" / "training"


class QuestionGenerator(Protocol):
    name: str

    def generate(self, statement: str, n: int) -> list[str]: ...


@dataclass
class Example:
    experience_id: str
    instruction: str
    response: str
    origin: str          # "user_question" (real question from chat) or "generated"


def eval_questions() -> list[str]:
    """Every question in the three fixed evaluation sets."""
    return [item["question"] for name in BENCHMARKS for item in load_benchmark(name)]


def clean_statement(statement: str) -> str:
    """'Canberra is the capital, not Sydney.' -> keep only the correct part."""
    s = re.sub(r",\s*not\s+[^.]*", "", statement.strip())
    return s if s.endswith((".", "!", "?")) else s + "."


def build_examples(experiences: list[dict], generator: QuestionGenerator, n_questions: int,
                   eval_qs: list[str], leak_similarity: float) -> tuple[list[Example], dict]:
    eval_sets = [set(tokens(q)) for q in eval_qs]
    eval_norm = {normalise(q) for q in eval_qs}

    def leaks(question: str) -> bool:
        t = set(tokens(question))
        return normalise(question) in eval_norm or any(jaccard(t, e) >= leak_similarity for e in eval_sets)

    examples: list[Example] = []
    seen: dict[str, str] = {}
    report = {"dropped_eval_leak": 0, "dropped_duplicate": 0, "dropped_contradicting_answer": 0,
              "experiences_without_examples": []}

    for exp in experiences:
        response = clean_statement(exp["corrected_answer"])
        candidates = []
        if exp.get("previous_question"):
            candidates.append((exp["previous_question"], "user_question"))
        candidates += [(q, "generated") for q in generator.generate(response, n_questions)]

        kept = 0
        for question, origin in candidates:
            key = normalise(question)
            if not key:
                continue
            if leaks(question):
                report["dropped_eval_leak"] += 1
            elif key in seen:
                report["dropped_duplicate" if seen[key] == normalise(response)
                       else "dropped_contradicting_answer"] += 1
            else:
                seen[key] = normalise(response)
                examples.append(Example(exp["experience_id"], question.strip(), response, origin))
                kept += 1
        if kept == 0:
            report["experiences_without_examples"].append(exp["experience_id"])
    return examples, report


def split_examples(examples: list[Example], seed: int) -> dict[str, list[Example]]:
    """Per fact: 1 wording to validation, 1 to test (if enough wordings), the rest to train."""
    by_exp: dict[str, list[Example]] = {}
    for ex in examples:
        by_exp.setdefault(ex.experience_id, []).append(ex)

    splits = {"train": [], "valid": [], "test": []}
    for exp_id, items in by_exp.items():
        items = sorted(items, key=lambda e: e.instruction)
        random.Random(f"{seed}:{exp_id}").shuffle(items)
        n_valid = 1 if len(items) >= 3 else 0
        n_test = 1 if len(items) >= 4 else 0
        splits["valid"] += items[:n_valid]
        splits["test"] += items[n_valid:n_valid + n_test]
        splits["train"] += items[n_valid + n_test:]
    return splits


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(cycle_id: str, splits: dict[str, list[Example]], system_prompt: str,
                  extra_manifest: dict, root: Path = TRAINING_DIR) -> Path:
    """Write train/valid/test.jsonl + index + manifest. Refuses to overwrite."""
    out = root / cycle_id
    out.mkdir(parents=True, exist_ok=False)

    index_rows = []
    for split, items in splits.items():
        with open(out / f"{split}.jsonl", "w") as f:
            for i, ex in enumerate(items):
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ex.instruction},
                    {"role": "assistant", "content": ex.response},
                ]}, ensure_ascii=False) + "\n")
                index_rows.append({"split": split, "line": i, **asdict(ex)})
    with open(out / "index.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in index_rows)

    manifest = {
        "cycle_id": cycle_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {k: len(v) for k, v in splits.items()},
        "sha256": {f"{s}.jsonl": _sha256(out / f"{s}.jsonl") for s in splits},
        "system_prompt": system_prompt,
        **extra_manifest,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return out
