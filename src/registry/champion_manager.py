"""The champion/candidate gate.

run_gate() compares a candidate with the current champion against the
thresholds in configs/evaluation_config.yaml. promote()/reject() then record
the decision. Nothing here ever deletes or overwrites an adapter: a rejected
candidate stays on disk, and the previous champion keeps serving.
"""
import sqlite3

from src.registry import model_registry as reg
from src.sleep.cycle_manager import CycleManager
from src.utils.config_loader import load_config

EPS = 1e-9


def run_gate(candidate_acc: dict, champion_acc: dict, forgetting_rate: float,
             taught_accuracy: float | None, base_unchanged: bool | None,
             cfg: dict | None = None) -> dict:
    """Return {'passed': bool, 'checks': [...]}. Every check must pass."""
    cfg = cfg or load_config("evaluation_config")["gate"]
    checks = []

    def add(name, passed, value, rule):
        checks.append({"check": name, "passed": bool(passed), "value": value, "rule": rule})

    if cfg["require_base_unchanged"]:
        add("base_weights_unchanged", base_unchanged is True, base_unchanged, "must be True")
    if taught_accuracy is None:
        checks.append({"check": "taught_facts_learned", "passed": True, "value": None,
                       "rule": "skipped: taught facts do not map to evaluation questions"})
    else:
        add("taught_facts_learned", taught_accuracy + EPS >= cfg["min_taught_accuracy"],
            round(taught_accuracy, 4), f">= {cfg['min_taught_accuracy']}")
    drop = lambda b: round(champion_acc[b] - candidate_acc[b], 4)   # positive = got worse
    add("new_knowledge_not_lost", drop("new_knowledge") <= cfg["max_new_knowledge_drop"] + EPS,
        drop("new_knowledge"), f"drop <= {cfg['max_new_knowledge_drop']}")
    add("old_knowledge_drop", drop("old_knowledge") <= cfg["max_old_knowledge_drop"] + EPS,
        drop("old_knowledge"), f"drop <= {cfg['max_old_knowledge_drop']}")
    add("reasoning_drop", drop("reasoning") <= cfg["max_reasoning_drop"] + EPS,
        drop("reasoning"), f"drop <= {cfg['max_reasoning_drop']}")
    add("forgetting_rate", forgetting_rate <= cfg["max_forgetting_rate"] + EPS,
        round(forgetting_rate, 4), f"<= {cfg['max_forgetting_rate']}")
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def promote(conn: sqlite3.Connection, version_id: str, cycle_id: str, evaluation: dict) -> None:
    """Candidate passed: it becomes the champion and its experiences count as learned."""
    reg.set_evaluation(conn, version_id, "APPROVED", evaluation)
    CycleManager(conn).finish(cycle_id, "SUCCESS", adapter_version=version_id,
                              evaluation_result="PROMOTED: all gate checks passed")


def reject(conn: sqlite3.Connection, version_id: str, cycle_id: str, evaluation: dict) -> None:
    """Candidate failed: the old champion keeps serving; its experiences stay eligible."""
    failed = [c["check"] for c in evaluation["gate"]["checks"] if not c["passed"]]
    reg.set_evaluation(conn, version_id, "REJECTED", evaluation)
    CycleManager(conn).finish(cycle_id, "REJECTED", adapter_version=version_id,
                              evaluation_result="REJECTED: failed " + ", ".join(failed))
