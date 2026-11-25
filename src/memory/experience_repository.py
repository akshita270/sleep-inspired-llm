"""Read and write experiences. All SQL for the experiences table lives here."""
import sqlite3
from datetime import datetime


class ExperienceRepository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add(self, exp: dict, status: str, confidence: float, notes: str,
            content_hash: str, requires_explicit_approval: bool) -> None:
        """Insert one experience with the validator's decision.

        Raises sqlite3.IntegrityError if the id already exists (re-import) or
        a live duplicate statement exists.
        """
        self.conn.execute(
            """INSERT INTO experiences
               (experience_id, session_id, interaction_type, original_input,
                previous_question, previous_answer, corrected_answer, category,
                source, evidence_url, confidence, status, learn_requested,
                requires_explicit_approval, content_hash, validation_notes, created_at)
               VALUES (:experience_id, :session_id, :interaction_type, :original_input,
                :previous_question, :previous_answer, :corrected_answer, :category,
                :source, :evidence_url, :confidence, :status, :learn_requested,
                :requires_explicit_approval, :content_hash, :validation_notes, :created_at)""",
            {
                "experience_id": exp["experience_id"],
                "session_id": exp["session_id"],
                "interaction_type": exp["interaction_type"],
                "original_input": exp["original_input"],
                "previous_question": exp.get("previous_question"),
                "previous_answer": exp.get("previous_answer"),
                "corrected_answer": exp.get("corrected_answer"),
                "category": exp["category"],
                "source": exp["source"],
                "evidence_url": exp.get("evidence_url"),
                "confidence": confidence,
                "status": status,
                "learn_requested": int(bool(exp.get("learn_requested"))),
                "requires_explicit_approval": int(requires_explicit_approval),
                "content_hash": content_hash,
                "validation_notes": notes,
                "created_at": exp.get("created_at") or datetime.now().isoformat(timespec="seconds"),
            },
        )

    def get(self, experience_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM experiences WHERE experience_id = ?", (experience_id,)).fetchone()

    def list_by_status(self, status: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM experiences WHERE status = ? ORDER BY created_at", (status,)).fetchall()

    def live_statements(self) -> list[dict]:
        """Statements of PENDING and APPROVED rows, used for duplicate/conflict checks."""
        rows = self.conn.execute(
            "SELECT experience_id, status, category, content_hash, "
            "COALESCE(corrected_answer, original_input) AS statement "
            "FROM experiences WHERE status IN ('PENDING','APPROVED')").fetchall()
        return [dict(r) for r in rows]

    def set_status(self, experience_id: str, status: str, reviewer: str, note: str) -> None:
        self.conn.execute(
            """UPDATE experiences
               SET status = ?, reviewed_by = ?, reviewed_at = ?,
                   validation_notes = COALESCE(validation_notes || ' | ', '') || ?
               WHERE experience_id = ?""",
            (status, reviewer, datetime.now().isoformat(timespec="seconds"),
             f"{reviewer}: {note}" if note else f"{reviewer}: {status.lower()}", experience_id))

    def counts(self) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM experiences GROUP BY status").fetchall()
        out = {"PENDING": 0, "APPROVED": 0, "REJECTED": 0}
        out.update({r["status"]: r["n"] for r in rows})
        return out
