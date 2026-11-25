"""Measure forgetting against the ORIGINAL model.

Forgetting rate = of the questions the original (base) model answered
correctly, the share the candidate now answers wrongly.

Only old-knowledge and reasoning questions count: new-knowledge questions the
base model could never answer cannot be 'forgotten'.
"""
import pandas as pd

FORGETTING_BENCHMARKS = ("old_knowledge", "reasoning")


def forgetting_report(candidate: pd.DataFrame, base: pd.DataFrame) -> dict:
    merged = base.merge(candidate, on=["benchmark", "id"], suffixes=("_base", "_cand"))
    merged = merged[merged["benchmark"].isin(FORGETTING_BENCHMARKS)]

    report = {}
    for name, g in [*merged.groupby("benchmark"), ("combined", merged)]:
        base_right = g[g["correct_base"]]
        lost = base_right[~base_right["correct_cand"]]
        gained = g[(~g["correct_base"]) & (g["correct_cand"])]
        report[name] = {
            "base_correct": int(len(base_right)),
            "lost": int(len(lost)),
            "forgetting_rate": round(len(lost) / len(base_right), 4) if len(base_right) else 0.0,
            "lost_ids": sorted(lost["id"]),
            "gained_ids": sorted(gained["id"]),
        }
    return report
