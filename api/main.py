"""Small read-only FastAPI service over the real measured results and the SQLite store.

    uvicorn api.main:app --port 8000        # then open http://localhost:8000/docs
"""
import sqlite3

from fastapi import FastAPI, HTTPException

from src.reporting import results_loader as R
from src.utils.config_loader import PROJECT_ROOT, load_config

app = FastAPI(title="Sleep-Inspired Continual Learning: results API", version="1.0")


def _records(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/results/phase8")
def phase8():
    """Final exam (220 questions): original vs episodic memory vs LoRA."""
    return _records(R.phase8_comparison())


@app.get("/results/gate")
def gate():
    return _records(R.phase8_gate_checks())


@app.get("/results/tuning")
def tuning():
    return _records(R.tuning_table())


@app.get("/results/continual")
def continual(arm: str | None = None):
    """Phase 9: one row per (arm, cycle). Optional ?arm=no_replay|general_replay|full_replay"""
    if arm is not None and arm not in R.ARM_ORDER:
        raise HTTPException(404, f"arm must be one of {R.ARM_ORDER}")
    df = R.continual_table()
    if arm:
        df = df[df["arm"] == arm]
    return _records(df.drop(columns=["arm_label"]))


@app.get("/limitations")
def limitations():
    return R.known_limitations()


@app.get("/experiences")
def experiences(status: str | None = None, limit: int = 50):
    """Experiences stored in the real SQLite database (empty until you chat and teach facts)."""
    path = PROJECT_ROOT / load_config("system_config")["database"]["path"]
    if not path.exists():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    q, args = "SELECT * FROM experiences", []
    if status:
        q += " WHERE status = ?"; args.append(status.upper())
    rows = conn.execute(q + " ORDER BY rowid DESC LIMIT ?", (*args, max(1, min(limit, 500)))).fetchall()
    conn.close()
    return [dict(r) for r in rows]
