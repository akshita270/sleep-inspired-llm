import sqlite3

import pytest

from src.memory.database import get_connection, transaction


@pytest.fixture
def conn(tmp_path):
    return get_connection(tmp_path / "test.db")


def _insert(conn, eid, status="PENDING", h="h1"):
    conn.execute(
        "INSERT INTO experiences (experience_id, session_id, interaction_type, original_input, "
        "category, source, status, content_hash, created_at) "
        "VALUES (?, 's', 'NEW_FACT', 'x', 'general', 'user_claim', ?, ?, 'now')", (eid, status, h))


def test_three_tables_exist(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"experiences", "sleep_cycles", "model_versions"} <= names


def test_bad_status_is_refused(conn):
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "a", status="MAYBE")


def test_two_live_duplicates_are_refused_but_rejected_copy_is_allowed(conn):
    _insert(conn, "a", "PENDING", "same")
    with pytest.raises(sqlite3.IntegrityError):
        _insert(conn, "b", "APPROVED", "same")
    _insert(conn, "c", "REJECTED", "same")  # kept as a record


def test_transaction_rolls_back_on_error(conn):
    with pytest.raises(RuntimeError):
        with transaction(conn):
            _insert(conn, "a")
            raise RuntimeError("boom")
    assert conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0] == 0
