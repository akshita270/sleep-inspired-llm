"""Shared helpers for experiments."""
from pathlib import Path

from src.evaluation.benchmark_manager import load_benchmark
from src.memory.database import get_connection
from src.validation.approval_manager import ApprovalManager


def add_fixture_facts(conn, items: list[dict]) -> dict:
    """Push fixture facts (benchmark items with a 'fact' field) through the validator."""
    mgr = ApprovalManager(conn)
    decisions = {"APPROVED": 0, "PENDING": 0, "REJECTED": 0, "SKIPPED": 0}
    for item in items:
        res = mgr.submit({
            "experience_id": item["id"], "session_id": "fixture", "interaction_type": "NEW_FACT",
            "original_input": item["fact"], "corrected_answer": item["fact"],
            "category": "general", "source": "experiment_fixture",
            "previous_answer": None, "previous_question": None,
            "learn_requested": False, "requires_explicit_approval": False})
        decisions[res["status"]] += 1
    return decisions


def fresh_fixture_db(db_file: Path, ids: list[str] | None = None):
    """Create an empty experiment database and push the 30 learning facts through
    the real validation pipeline (source 'experiment_fixture' = trusted).

    Returns (connection, {"APPROVED": n, ...}). Refuses to reuse an existing file,
    so one experiment run can never contaminate another.
    """
    db_file.parent.mkdir(parents=True, exist_ok=True)
    if db_file.exists():
        raise FileExistsError(f"{db_file} already exists; choose a new name or delete it yourself")
    conn = get_connection(db_file)
    items = [i for i in load_benchmark("new_knowledge") if ids is None or i["id"] in ids]
    return conn, add_fixture_facts(conn, items)
