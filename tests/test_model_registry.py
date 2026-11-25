import json
import sqlite3

import pytest

from src.memory.database import get_connection
from src.registry import champion_manager as cm
from src.registry import model_registry as reg
from src.registry.rollback_manager import RollbackError, rollback
from src.sleep.cycle_manager import CycleManager


@pytest.fixture
def conn(tmp_path):
    return get_connection(tmp_path / "t.db")


def add(conn, n, promote=False):
    """Create cycle_00n + adapter candidate, optionally promoted."""
    cycles = CycleManager(conn)
    cycle = cycles.start()
    conn.execute("UPDATE sleep_cycles SET adapter_version=? WHERE cycle_id=?", (f"v{n}", cycle))
    reg.register_candidate(conn, f"v{n}", "Qwen/Qwen3-4B", f"/adapters/v{n}", {"final_val_loss": 0.1})
    if promote:
        cm.promote(conn, f"v{n}", cycle, {"summary": {"accuracy": {"new_knowledge": 1.0}}})
    return cycle


def test_no_champion_means_the_base_model(conn):
    assert reg.champion_version(conn) is None and reg.champion_adapter_path(conn) is None


def test_pending_candidate_is_not_the_champion(conn):
    add(conn, 1)
    assert reg.champion_version(conn) is None
    assert reg.get_version(conn, "v1")["evaluation_status"] == "PENDING"


def test_promotion_makes_champion_and_marks_cycle_successful(conn):
    cycle = add(conn, 1, promote=True)
    assert reg.champion_version(conn) == "v1" and reg.champion_adapter_path(conn) == "/adapters/v1"
    assert conn.execute("SELECT status FROM sleep_cycles WHERE cycle_id=?", (cycle,)).fetchone()[0] == "SUCCESS"


def test_rejected_candidate_never_replaces_the_champion(conn):
    add(conn, 1, promote=True)
    cycle2 = add(conn, 2)
    gate = {"checks": [{"check": "reasoning_drop", "passed": False}], "passed": False}
    cm.reject(conn, "v2", cycle2, {"gate": gate})
    assert reg.champion_version(conn) == "v1"
    assert reg.get_version(conn, "v2")["evaluation_status"] == "REJECTED"
    assert "reasoning_drop" in conn.execute(
        "SELECT evaluation_result FROM sleep_cycles WHERE cycle_id=?", (cycle2,)).fetchone()[0]


def test_candidates_chain_to_the_champion_at_creation_time(conn):
    add(conn, 1, promote=True)
    add(conn, 2)
    assert reg.get_version(conn, "v2")["parent_version"] == "v1"
    assert reg.get_version(conn, "v1")["parent_version"] is None


def test_rollback_returns_to_the_previous_champion_and_keeps_everything(conn):
    add(conn, 1, promote=True)
    cycle2 = add(conn, 2, promote=True)
    assert reg.champion_version(conn) == "v2"
    assert rollback(conn, "reasoning looked worse in manual review") == "v1"
    assert reg.champion_version(conn) == "v1"
    row = reg.get_version(conn, "v2")
    assert row["rolled_back"] == 1 and row["adapter_path"] == "/adapters/v2"      # record kept
    assert "reasoning looked worse" in json.loads(row["metrics"])["rollback"]["reason"]
    # its experiences are no longer counted as learned
    assert conn.execute("SELECT status FROM sleep_cycles WHERE cycle_id=?", (cycle2,)).fetchone()[0] == "REJECTED"


def test_rollback_all_the_way_to_the_base_model_then_refuses(conn):
    add(conn, 1, promote=True)
    assert rollback(conn, "undo") is None
    assert reg.champion_version(conn) is None
    with pytest.raises(RollbackError):
        rollback(conn, "again")


def test_list_versions_flags_the_champion(conn):
    add(conn, 1, promote=True)
    add(conn, 2)
    flags = {v["version_id"]: v["is_champion"] for v in reg.list_versions(conn)}
    assert flags == {"v1": True, "v2": False}


def test_old_databases_are_migrated_when_opened(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE model_versions (version_id TEXT PRIMARY KEY, base_model TEXT NOT NULL, "
                "adapter_path TEXT, parent_version TEXT, created_at TEXT NOT NULL, "
                "evaluation_status TEXT NOT NULL, metrics TEXT)")
    old.execute("INSERT INTO model_versions VALUES ('v0','m','/p',NULL,'now','APPROVED','{}')")
    old.commit(); old.close()
    conn = get_connection(path)
    assert reg.champion_version(conn) == "v0"            # column added, old row still readable
