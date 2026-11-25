"""The shared awake/asleep switch.

The state lives in SQLite (table `system_state`), not in a Python variable, so
the chat loop, the API and the dashboard all read the same value. Moving from
AWAKE to another state is one atomic UPDATE, so two sleep cycles can never
start at the same time.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

AWAKE, SLEEPING, TRAINING, EVALUATING = "AWAKE", "SLEEPING", "TRAINING", "EVALUATING"


class SystemState:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self) -> str:
        return self.conn.execute("SELECT value FROM system_state WHERE key='status'").fetchone()[0]

    def set(self, value: str) -> None:
        self.conn.execute("UPDATE system_state SET value=?, updated_at=? WHERE key='status'",
                          (value, datetime.now().isoformat(timespec="seconds")))

    def try_acquire(self, value: str) -> bool:
        """Leave AWAKE for `value`. Returns False if the system is already busy."""
        cur = self.conn.execute(
            "UPDATE system_state SET value=?, updated_at=? WHERE key='status' AND value='AWAKE'",
            (value, datetime.now().isoformat(timespec="seconds")))
        return cur.rowcount == 1

    def force_wake(self) -> None:
        """Recovery tool: if a run crashed hard (power cut), the state can stay stuck."""
        self.set(AWAKE)

    @contextmanager
    def busy(self, value: str = SLEEPING):
        if not self.try_acquire(value):
            raise RuntimeError(f"cannot start: system is {self.get()}")
        try:
            yield self
        finally:
            self.set(AWAKE)  # always wake up, even after an error
