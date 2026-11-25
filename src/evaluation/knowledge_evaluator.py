"""Knowledge and hallucination measurements on a results table.

A results table (pandas DataFrame) has one row per question with columns
benchmark, id, model_answer, correct, ... (see evaluation_runner).
"""
import re

import pandas as pd

from src.utils.config_loader import load_config


def is_abstention(answer: str, patterns: list[str] | None = None) -> bool:
    """True if the answer admits not knowing instead of stating something."""
    patterns = patterns or load_config("evaluation_config")["hallucination"]["abstain_patterns"]
    text = answer.lower().replace("’", "'")
    return any(re.search(p, text) for p in patterns)


def hallucination_stats(df: pd.DataFrame) -> dict:
    """Per benchmark: correct / abstained / confidently wrong.

    'Confidently wrong' = not correct and not an abstention. Rule-based, so a
    correct answer worded unusually could be counted as wrong (see limitations)."""
    patterns = load_config("evaluation_config")["hallucination"]["abstain_patterns"]
    out = {}
    for bench, g in df.groupby("benchmark"):
        abstained = g.apply(lambda r: (not r["correct"]) and is_abstention(str(r["model_answer"]), patterns), axis=1)
        wrong = (~g["correct"]) & (~abstained)
        out[bench] = {"n": int(len(g)), "correct": int(g["correct"].sum()),
                      "abstained": int(abstained.sum()), "confidently_wrong": int(wrong.sum()),
                      "confidently_wrong_rate": round(float(wrong.mean()), 4)}
    return out


def taught_accuracy(df: pd.DataFrame, experience_ids: list[str]) -> float | None:
    """Accuracy on the new-knowledge questions whose facts were taught in this cycle.
    Returns None if none of the taught experiences correspond to evaluation questions."""
    g = df[(df["benchmark"] == "new_knowledge") & (df["id"].isin(experience_ids))]
    return float(g["correct"].mean()) if len(g) else None
