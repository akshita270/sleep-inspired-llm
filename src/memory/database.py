"""SQLite setup: connection, schema, and a transaction helper.

Only text and numbers live here. Model weights never go in SQLite: the
`model_versions` table stores a *path* to an adapter folder, not the adapter.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.utils.config_loader import PROJECT_ROOT, load_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    experience_id   TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    interaction_type TEXT NOT NULL CHECK (interaction_type IN ('NEW_FACT','CORRECTION','FEEDBACK')),
    original_input  TEXT NOT NULL,
    previous_question TEXT,
    previous_answer TEXT,
    corrected_answer TEXT,
    category        TEXT NOT NULL CHECK (category IN ('personal','general')),
    source          TEXT NOT NULL,
    evidence_url    TEXT,
    confidence      REAL,
    status          TEXT NOT NULL CHECK (status IN ('PENDING','APPROVED','REJECTED')),
    learn_requested INTEGER NOT NULL DEFAULT 0,
    requires_explicit_approval INTEGER NOT NULL DEFAULT 0,
    content_hash    TEXT NOT NULL,
    validation_notes TEXT,
    reviewed_by     TEXT,
    reviewed_at     TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exp_status   ON experiences(status);
CREATE INDEX IF NOT EXISTS idx_exp_category ON experiences(category);
-- Database-level duplicate guard: two live (PENDING/APPROVED) rows can never
-- hold the same normalised statement. REJECTED rows are kept as a record.
CREATE UNIQUE INDEX IF NOT EXISTS idx_exp_live_hash
    ON experiences(content_hash) WHERE status IN ('PENDING','APPROVED');

CREATE TABLE IF NOT EXISTS sleep_cycles (
    cycle_id    TEXT PRIMARY KEY,
    start_time  TEXT NOT NULL,
    end_time    TEXT,
    experiences_processed INTEGER NOT NULL DEFAULT 0,
    training_examples     INTEGER NOT NULL DEFAULT 0,
    adapter_version TEXT,
    -- PREPARED = dataset built, no training (dry run); TRAINED = adapter trained, awaiting evaluation;
    -- SUCCESS = candidate promoted; REJECTED = candidate failed evaluation
    status      TEXT NOT NULL CHECK (status IN ('RUNNING','PREPARED','TRAINED','SUCCESS','FAILED','REJECTED')),
    evaluation_result TEXT,
    dataset_path TEXT,
    notes       TEXT
);

-- Which experiences went into which cycle.
CREATE TABLE IF NOT EXISTS cycle_experiences (
    cycle_id      TEXT NOT NULL REFERENCES sleep_cycles(cycle_id),
    experience_id TEXT NOT NULL REFERENCES experiences(experience_id),
    PRIMARY KEY (cycle_id, experience_id)
);

-- One row, key='status': AWAKE / SLEEPING / TRAINING / EVALUATING.
-- Stored in the database so the chat, API and dashboard all see the same truth.
CREATE TABLE IF NOT EXISTS system_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
INSERT OR IGNORE INTO system_state (key, value, updated_at) VALUES ('status', 'AWAKE', datetime('now'));

CREATE TABLE IF NOT EXISTS model_versions (
    version_id  TEXT PRIMARY KEY,
    base_model  TEXT NOT NULL,
    adapter_path TEXT,
    parent_version TEXT REFERENCES model_versions(version_id),
    created_at  TEXT NOT NULL,
    evaluation_status TEXT NOT NULL CHECK (evaluation_status IN ('PENDING','APPROVED','REJECTED')),
    metrics     TEXT,
    rolled_back INTEGER NOT NULL DEFAULT 0   -- 1 = was champion, then withdrawn by a rollback
);
"""


def db_path() -> Path:
    return PROJECT_ROOT / load_config("system_config")["database"]["path"]


def get_connection(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the database (creating tables if needed).

    isolation_level=None means we control transactions ourselves, using the
    `transaction()` helper below, so related writes succeed or fail together.
    """
    path = Path(path) if path else db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring databases created by an older version of the schema up to date."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(model_versions)")}
    if "rolled_back" not in cols:
        conn.execute("ALTER TABLE model_versions ADD COLUMN rolled_back INTEGER NOT NULL DEFAULT 0")


@contextmanager
def transaction(conn: sqlite3.Connection):
    """All-or-nothing block: commits on success, rolls back on any error."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
