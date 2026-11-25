"""Undo a promotion: withdraw the current champion and fall back to the one before it.

Nothing is deleted. The withdrawn adapter folder stays on disk and its row is
kept (marked rolled_back), so the decision can be audited. Its experiences
become eligible for learning again.
"""
import json
import sqlite3
from datetime import datetime

from src.registry import model_registry as reg
from src.sleep.cycle_manager import CycleManager


class RollbackError(Exception):
    pass


def rollback(conn: sqlite3.Connection, reason: str) -> str | None:
    """Withdraw the current champion. Returns the new champion's version id,
    or None meaning the plain base model is now the champion."""
    current = reg.champion_version(conn)
    if current is None:
        raise RollbackError("already on the original base model; nothing to roll back")

    metrics = reg.get_metrics(conn, current)
    metrics["rollback"] = {"reason": reason, "at": datetime.now().isoformat(timespec="seconds")}
    conn.execute("UPDATE model_versions SET rolled_back=1, metrics=? WHERE version_id=?",
                 (json.dumps(metrics), current))
    conn.execute(
        "UPDATE sleep_cycles SET status='REJECTED', "
        "evaluation_result=COALESCE(evaluation_result,'') || ' | ROLLED BACK: ' || ? "
        "WHERE adapter_version=?", (reason, current))
    return reg.champion_version(conn)
