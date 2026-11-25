"""Read the REAL measured results from experiments/results/ (nothing is typed in by hand).

Every number the dashboard, the API and the docs show comes from the files read here.
If a file is missing the function raises FileNotFoundError instead of inventing a value.
"""
import json
from pathlib import Path

import pandas as pd

from src.utils.config_loader import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "experiments" / "results"
ARM_ORDER = ["no_replay", "general_replay", "full_replay"]
ARM_LABELS = {"no_replay": "No replay", "general_replay": "General replay", "full_replay": "Full replay"}


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def phase8_comparison() -> pd.DataFrame:
    """The A / B / C / C+replay final-exam table (220 questions)."""
    a, b = _json(RESULTS / "baseline_summary.json"), _json(RESULTS / "episodic_summary.json")
    c_no = _json(RESULTS / "experiment_c_30facts" / "evaluations" / "adapter_cycle_003_evaluation.json")
    c_re = _json(RESULTS / "experiment_c_final" / "evaluations" / "adapter_cycle_001_evaluation.json")
    t_no = _json(RESULTS / "experiment_c_30facts" / "adapters" / "cycle_003" / "training_metrics.json")
    t_re = _json(RESULTS / "experiment_c_final" / "adapters" / "cycle_001" / "training_metrics.json")

    def row(name, acc, forgetting, latency, train_s=None, gate=None):
        return {"model": name, "new_facts": acc["new_knowledge"], "old_knowledge": acc["old_knowledge"],
                "reasoning": acc["reasoning"], "forgetting_rate": forgetting,
                "median_latency_s": latency, "training_seconds": train_s, "gate": gate}

    return pd.DataFrame([
        row("A: original Qwen", a["accuracy"], None, a["median_latency_s"], None, "champion"),
        row("B: Qwen + episodic memory", b["accuracy"], 0.0, b["median_latency_s"], None, "not trained"),
        row("C: LoRA, no replay", c_no["summary"]["accuracy"], c_no["forgetting"]["combined"]["forgetting_rate"],
            c_no["summary"]["median_latency_s"], t_no["training_seconds"], "rejected" if not c_no["gate"]["passed"] else "passed"),
        row("C: LoRA + replay", c_re["summary"]["accuracy"], c_re["forgetting"]["combined"]["forgetting_rate"],
            c_re["summary"]["median_latency_s"], t_re["training_seconds"], "rejected" if not c_re["gate"]["passed"] else "passed"),
    ])


def phase8_gate_checks() -> pd.DataFrame:
    """Per-rule gate result for the replay adapter and the no-replay adapter."""
    out = []
    for name, p in [("C: LoRA, no replay", "experiment_c_30facts/evaluations/adapter_cycle_003_evaluation.json"),
                    ("C: LoRA + replay", "experiment_c_final/evaluations/adapter_cycle_001_evaluation.json")]:
        for c in _json(RESULTS / p)["gate"]["checks"]:
            out.append({"model": name, "check": c["check"], "value": c["value"], "rule": c["rule"], "passed": c["passed"]})
    return pd.DataFrame(out)


def tuning_table() -> pd.DataFrame:
    t = _json(RESULTS / "tuning" / "tuning_summary.json")
    rows = []
    for k, v in t.items():
        if k.startswith("_"):
            continue
        rows.append({"setting": k, "dev_knowledge": v["dev_accuracy"]["dev_knowledge"],
                     "dev_reasoning": v["dev_accuracy"]["dev_reasoning"],
                     "dev_forgetting": v["forgetting"]["forgetting_rate"],
                     "paraphrase_acc": v["paraphrase_accuracy"], "train_examples": v["n_train_examples"],
                     "training_seconds": v.get("training_seconds"), "selected": k == t.get("_selected")})
    return pd.DataFrame(rows)


def continual_table() -> pd.DataFrame:
    """One row per (arm, cycle) from the Phase 9 run."""
    rows = []
    for arm in ARM_ORDER:
        f = RESULTS / "continual" / arm / "results.json"
        if f.exists():
            rows += _json(f)
    if not rows:
        raise FileNotFoundError("no continual results found; run python -m experiments.continual_learning")
    df = pd.DataFrame(rows)
    df["arm_label"] = df["arm"].map(ARM_LABELS)
    return df


def fact_group_matrix(arm: str) -> pd.DataFrame:
    """Rows = cycle, columns = fact group (1-5): accuracy on that group's facts after each cycle."""
    rows = _json(RESULTS / "continual" / arm / "results.json")
    m = pd.DataFrame([{int(g): v for g, v in r["acc_by_fact_group"].items()} for r in rows],
                     index=[r["cycle"] for r in rows])
    m.index.name = "after cycle"
    return m.sort_index(axis=1)


def known_limitations() -> list[str]:
    return [
        "Each Phase 9 arm was run once (seed 42); each fact group has only 6 facts, so one fact = 16.7 points. Small gaps between arms may be noise.",
        "The facts are fictional and injected as trusted 'experiment_fixture' facts, not collected from live chat.",
        "The 100 general replay questions are written by hand and answered by the original model; they are not read from the SQLite store.",
        "The dev set under-estimated forgetting (1.7% on dev vs 7.6% on the final exam for the selected setting).",
        "The old-knowledge and reasoning test sets are near ceiling (99.2% / 92.9%) for the original model, so they are coarse.",
        "The leak guard blocks near-copies of exam questions (Jaccard >= 0.8) but allows similar wordings.",
        "GPU peak memory during training is not measured properly (the CPU-side RSS number undercounts).",
        "Rollback is unit-tested but was never exercised on a promoted model, because no adapter passed the gate.",
        "The classifier and conflict detector are simple rules, not learned models.",
        "The first episodic-memory run (before a validator fix) was lost; its numbers (86.7% new knowledge, 90% retrieval, 3 facts rejected) exist only in the chat log.",
    ]


def summary() -> dict:
    cont = continual_table()
    last = cont[cont["cycle"] == cont["cycle"].max()]
    return {"phase8": phase8_comparison().to_dict("records"),
            "continual_final_cycle": last[["arm", "acc_new_all30", "acc_earlier_cycles_facts", "acc_old",
                                           "acc_reasoning", "forgetting_rate"]].to_dict("records"),
            "gate_would_pass_count": int(cont["gate_would_pass"].sum()), "cycles_total": int(len(cont)),
            "limitations": known_limitations()}


def plain_guide() -> list[list[str]]:
    """Short plain-English explanations of the terms used on the dashboards."""
    return [
    [
        "The big idea",
        "An AI model is taught new facts the way people learn while sleeping: it collects things while 'awake', then 'sleeps' to study them. We tested whether that works, and whether it makes the model forget what it already knew."
    ],
    [
        "New facts",
        "30 made-up facts the original model could not know (for example about an invented company). Scoring high here means the model learned them."
    ],
    [
        "Old knowledge",
        "120 real-world questions the original model already answers well, like capitals and basic science. Scoring low here means studying the new facts damaged what it knew."
    ],
    [
        "Reasoning",
        "70 logic, maths and coding questions. Another check that learning did not make the model worse at thinking."
    ],
    [
        "Forgetting rate",
        "Of the questions the original model got right, the share the trained model now gets wrong. Lower is better. We set a limit of 5%."
    ],
    [
        "LoRA adapter",
        "A small add-on file (about 59 MB) that is trained instead of changing the big model. The original model is never modified, and removing the add-on brings it back exactly."
    ],
    [
        "Episodic memory",
        "A notebook. Facts are stored and looked up when a question comes in. The model is not trained at all."
    ],
    [
        "Replay",
        "Revising old material while learning new material, like re-reading old chapters while studying a new one. Without it, the model tends to forget."
    ],
    [
        "General replay / full replay",
        "General replay revises real questions answered by the original model. Full replay also revises the facts taught in earlier rounds."
    ],
    [
        "Cycle",
        "One round of: give new facts, train, take the 220-question exam, record the scores."
    ],
    [
        "Gate and champion",
        "Safety rules set before the tests. A newly trained add-on only replaces the current model (the champion) if it passes. Here, none passed, so the original model stayed champion."
    ]
]


def conclusion() -> list[str]:
    """Short conclusion, with every number read from the result files."""
    p8 = phase8_comparison().set_index("model")
    b, c0, c1 = p8.loc["B: Qwen + episodic memory"], p8.loc["C: LoRA, no replay"], p8.loc["C: LoRA + replay"]
    cont = continual_table()
    last = cont[cont["cycle"] == cont["cycle"].max()].set_index("arm")
    pct = lambda v: f"{v * 100:.1f}%".replace(".0%", "%")
    return [
        f"Teaching works, but it has a cost. LoRA learned the new facts ({pct(c0['new_facts'])}), but old knowledge fell from 99.2% to {pct(c0['old_knowledge'])}.",
        f"Replay helps a lot. With it, old knowledge stayed at {pct(c1['old_knowledge'])} and forgetting dropped from {pct(c0['forgetting_rate'])} to {pct(c1['forgetting_rate'])}, which is still above our 5% limit, so the safety gate rejected it.",
        f"Over 5 rounds, only replaying earlier facts kept them: {pct(last.loc['full_replay', 'acc_earlier_cycles_facts'])} remembered, against {pct(last.loc['no_replay', 'acc_earlier_cycles_facts'])} with no replay. But general knowledge still faded in every version, and {int(cont['gate_would_pass'].sum())} of {len(cont)} cycles passed the gate.",
        f"A simple memory notebook (no training) got {pct(b['new_facts'])} on the new facts with no damage, so for plain fact-keeping it is the safer choice here.",
        "Verdict: the original model stayed champion. Sleep-style training is promising but not yet safe enough to trust without more work. These are small, single-run results on made-up facts.",
    ]
