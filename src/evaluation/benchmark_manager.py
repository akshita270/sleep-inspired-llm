"""Load the fixed evaluation sets and score answers.

Scoring is rule-based (not another LLM judging), so results are repeatable:
an answer is CORRECT if any accepted answer string appears in it as a whole word.
"""
import json
import re

from src.utils.config_loader import PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"

# short name -> file. These three files are fixed and never used for training.
BENCHMARKS = {
    "old_knowledge": "old_knowledge.jsonl",
    "new_knowledge": "new_knowledge.jsonl",
    "reasoning": "reasoning_tests.jsonl",
}


# The DEV sets are for choosing training settings. They are separate from the evaluation sets
# above, which are the final exam and must not be used to pick settings.
DEV_DIR = PROJECT_ROOT / "data" / "dev"
DEV_BENCHMARKS = {"dev_knowledge": "dev_knowledge.jsonl", "dev_reasoning": "dev_reasoning.jsonl"}


def load_benchmark(name: str) -> list[dict]:
    with open(EVAL_DIR / BENCHMARKS[name]) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dev(name: str) -> list[dict]:
    with open(DEV_DIR / DEV_BENCHMARKS[name]) as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalise(text: str) -> str:
    text = text.lower().replace("‑", "-")
    text = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))  # H₂O -> h2o
    text = re.sub(r"(?<=\d),(?=\d{3})", ",", text)  # keep 6,412 style numbers
    return re.sub(r"\s+", " ", text).strip()


def is_correct(answer: str, accepted: list[str]) -> bool:
    """True if any accepted string appears in the answer as a whole word/phrase."""
    text = _normalise(answer)
    for acc in accepted:
        pattern = r"(?<![\w])" + re.escape(_normalise(acc)) + r"(?![\w])"
        if re.search(pattern, text):
            return True
    return False
