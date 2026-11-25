"""Replay ("rehearsal") examples: general questions mixed into every training run.

Why: training only on new facts teaches the model that EVERY answer is about those
facts (we measured this: "Largest planet?" -> "The fictional planet Orlune..."). Mixing
in ordinary question/answer pairs reminds it how to behave on everything else.

The answers are the ORIGINAL model's own answers (self-distillation). So replay
asks the adapter to keep doing what the base model already did. It needs no
hand-written ground truth, and a wrong base answer is simply preserved, not "fixed".

The replay questions are kept disjoint from the evaluation and dev sets.
"""
import json
import random
import re
from pathlib import Path

from src.model.inference import ask
from src.model.model_loader import ModelBundle
from src.sleep.dataset_builder import Example
from src.utils.config_loader import PROJECT_ROOT
from src.validation.duplicate_detector import jaccard, normalise, tokens

REPLAY_DIR = PROJECT_ROOT / "data" / "replay"
QUESTIONS_FILE = REPLAY_DIR / "replay_questions.jsonl"
POOL_FILE = REPLAY_DIR / "replay_pool.jsonl"


def load_questions() -> list[str]:
    return [json.loads(l)["question"] for l in open(QUESTIONS_FILE) if l.strip()]


def build_pool(bundle: ModelBundle) -> list[dict]:
    """Ask the ORIGINAL model each replay question and keep its answer."""
    pool = []
    for q in load_questions():
        answer = ask(bundle, q)["answer"]
        if answer.strip():
            pool.append({"question": q, "answer": answer, "source": "base_model_self_answer"})
    POOL_FILE.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pool))
    return pool


def load_pool() -> list[dict]:
    return [json.loads(l) for l in open(POOL_FILE) if l.strip()]


_TEMPLATE = re.compile(r"^What does this Python code print\?\s*")


def _content_tokens(question: str) -> set:
    """Words of a question, ignoring the fixed 'What does this Python code print?' template,
    which would otherwise make every code question look alike."""
    return set(tokens(_TEMPLATE.sub("", question)))


def find_overlaps(questions: list[str], others: list[str], threshold: float) -> list[tuple[str, str]]:
    """Pairs (question, other) that are identical or at least `threshold` word-overlap alike."""
    other_sets = [(o, _content_tokens(o)) for o in others]
    hits = []
    for q in questions:
        t = _content_tokens(q)
        for o, ot in other_sets:
            if normalise(q) == normalise(o) or jaccard(t, ot) >= threshold:
                hits.append((q, o))
    return hits


def sample_replay(pool: list[dict], n: int, seed: int, eval_questions: list[str],
                  leak_similarity: float) -> tuple[list[Example], int]:
    """n replay examples (repeating pool items if n > pool size). Returns (examples, n_leaky_dropped)."""
    leaky = {q for q, _ in find_overlaps([p["question"] for p in pool], eval_questions, leak_similarity)}
    usable = [p for p in pool if p["question"] not in leaky]
    if not usable or n <= 0:
        return [], len(leaky)
    rng = random.Random(seed)
    chosen = []
    while len(chosen) < n:
        batch = usable[:]
        rng.shuffle(batch)
        chosen += batch[: n - len(chosen)]
    return [Example("replay", p["question"], p["answer"], "replay") for p in chosen], len(leaky)


def sample_previous(dataset_dirs: list, n: int, seed: int) -> list[Example]:
    """Replay of facts taught in EARLIER cycles: n examples drawn from those cycles' training
    wordings (no general replay rows, no earlier replay copies)."""
    rows = []
    for d in dataset_dirs:
        for line in open(Path(d) / "index.jsonl"):
            r = json.loads(line)
            if r["split"] == "train" and r["origin"] in ("generated", "user_question"):
                rows.append(Example(r["experience_id"], r["instruction"], r["response"], "previous_replay"))
    random.Random(seed).shuffle(rows)
    return rows[: max(n, 0)]
