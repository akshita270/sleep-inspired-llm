"""Bookkeeping for sleep cycles: ids, status, and which experiences each used."""
import sqlite3
from datetime import datetime

from src.memory.database import transaction


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class CycleManager:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def start(self) -> str:
        """Create a RUNNING cycle with the next id: cycle_001, cycle_002, ..."""
        with transaction(self.conn):
            n = self.conn.execute("SELECT COUNT(*) FROM sleep_cycles").fetchone()[0] + 1
            cycle_id = f"cycle_{n:03d}"
            self.conn.execute(
                "INSERT INTO sleep_cycles (cycle_id, start_time, status) VALUES (?, ?, 'RUNNING')",
                (cycle_id, _now()))
        return cycle_id

    def record_experiences(self, cycle_id: str, experience_ids: list[str]) -> None:
        with transaction(self.conn):
            self.conn.executemany(
                "INSERT INTO cycle_experiences (cycle_id, experience_id) VALUES (?, ?)",
                [(cycle_id, e) for e in experience_ids])

    def finish(self, cycle_id: str, status: str, *, experiences: int | None = None,
               examples: int | None = None, dataset_path: str | None = None,
               adapter_version: str | None = None, evaluation_result: str | None = None,
               notes: str | None = None) -> None:
        self.conn.execute(
            """UPDATE sleep_cycles SET end_time=?, status=?,
                 experiences_processed=COALESCE(?, experiences_processed),
                 training_examples=COALESCE(?, training_examples),
                 dataset_path=COALESCE(?, dataset_path),
                 adapter_version=COALESCE(?, adapter_version),
                 evaluation_result=COALESCE(?, evaluation_result),
                 notes=COALESCE(?, notes)
               WHERE cycle_id=?""",
            (_now(), status, experiences, examples, dataset_path, adapter_version,
             evaluation_result, notes, cycle_id))

    def learned_experience_ids(self) -> set[str]:
        """Experiences already learned by a SUCCESSFUL cycle. A rejected or failed
        cycle does not count, so its experiences get another chance."""
        rows = self.conn.execute(
            """SELECT ce.experience_id FROM cycle_experiences ce
               JOIN sleep_cycles sc USING (cycle_id) WHERE sc.status = 'SUCCESS'""").fetchall()
        return {r[0] for r in rows}
