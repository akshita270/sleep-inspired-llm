"""The list of model versions (adapters) and which one is the champion.

Champion  = the newest APPROVED version that has not been rolled back.
            If there is none, the champion is the plain base model.
Candidate = a freshly trained adapter, PENDING until it passes evaluation.

Only paths and numbers are stored here, never weights.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path


def register_candidate(conn: sqlite3.Connection, version_id: str, base_model: str,
                       adapter_path: Path, metrics: dict | None = None) -> None:
    """Record a freshly trained adapter as PENDING, chained to the current champion."""
    conn.execute(
        """INSERT INTO model_versions
           (version_id, base_model, adapter_path, parent_version, created_at, evaluation_status, metrics)
           VALUES (?, ?, ?, ?, ?, 'PENDING', ?)""",
        (version_id, base_model, str(adapter_path), champion_version(conn),
         datetime.now().isoformat(timespec="seconds"), json.dumps({"training": metrics or {}})))


def champion_version(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT version_id FROM model_versions WHERE evaluation_status='APPROVED' AND rolled_back=0 "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
    return row[0] if row else None


def get_version(conn: sqlite3.Connection, version_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM model_versions WHERE version_id=?", (version_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown model version {version_id}")
    return row


def champion_adapter_path(conn: sqlite3.Connection) -> str | None:
    """Adapter folder to load for normal use, or None meaning the plain base model."""
    v = champion_version(conn)
    return get_version(conn, v)["adapter_path"] if v else None


def get_metrics(conn: sqlite3.Connection, version_id: str) -> dict:
    return json.loads(get_version(conn, version_id)["metrics"] or "{}")


def set_evaluation(conn: sqlite3.Connection, version_id: str, status: str, evaluation: dict) -> None:
    """Store the decision and the evidence behind it."""
    metrics = get_metrics(conn, version_id)
    metrics["evaluation"] = evaluation
    conn.execute("UPDATE model_versions SET evaluation_status=?, metrics=? WHERE version_id=?",
                 (status, json.dumps(metrics), version_id))


def list_versions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT version_id, evaluation_status, parent_version, rolled_back, created_at, adapter_path "
        "FROM model_versions ORDER BY created_at, rowid").fetchall()
    champion = champion_version(conn)
    return [{**dict(r), "is_champion": r["version_id"] == champion} for r in rows]
