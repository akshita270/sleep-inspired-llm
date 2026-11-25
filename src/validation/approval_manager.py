"""The single entry point for putting experiences into the database and
for human approve/reject decisions. Only experiences that end up APPROVED
here can ever be used in the Sleep Phase.
"""
import json
import sqlite3
from pathlib import Path

from src.memory.database import transaction
from src.memory.experience_repository import ExperienceRepository
from src.validation import conflict_detector
from src.validation.experience_validator import (APPROVED, PENDING, REJECTED,
                                                 ExperienceValidator)


class ApprovalError(Exception):
    pass


class ApprovalManager:
    def __init__(self, conn: sqlite3.Connection, validator: ExperienceValidator | None = None):
        self.conn = conn
        self.repo = ExperienceRepository(conn)
        self.validator = validator or ExperienceValidator()

    def submit(self, exp: dict) -> dict:
        """Validate one experience and store it with the decision (atomic)."""
        try:
            with transaction(self.conn):
                result = self.validator.validate(exp, self.repo.live_statements())
                self.repo.add(exp, result.status, result.confidence, "; ".join(result.notes),
                              result.content_hash, result.requires_explicit_approval)
        except sqlite3.IntegrityError as e:
            return {"experience_id": exp["experience_id"], "status": "SKIPPED", "notes": [str(e)]}
        return {"experience_id": exp["experience_id"], "status": result.status, "notes": result.notes}

    def ingest_staging_file(self, path: str | Path) -> list[dict]:
        """Import experiences collected during the Wake Phase. Safe to run twice."""
        results = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    results.append(self.submit(json.loads(line)))
        return results

    def approve(self, experience_id: str, reviewer: str = "human", note: str = "",
                explicit_personal_approval: bool = False, override_conflict: bool = False) -> None:
        """Human approval of a PENDING experience."""
        with transaction(self.conn):
            row = self._pending(experience_id)
            if row["requires_explicit_approval"] and not explicit_personal_approval:
                raise ApprovalError("personal information needs explicit_personal_approval=True")
            if not override_conflict:
                approved = [r for r in self.repo.live_statements() if r["status"] == APPROVED]
                statement = row["corrected_answer"] or row["original_input"]
                clash = conflict_detector.find_conflicts(
                    statement, approved, self.validator.rules["conflict_subject_similarity"])
                if clash:
                    raise ApprovalError(f"conflicts with approved experience(s) {clash}; "
                                        "reject one or pass override_conflict=True")
            self.repo.set_status(experience_id, APPROVED, reviewer, note)

    def reject(self, experience_id: str, reviewer: str = "human", note: str = "") -> None:
        with transaction(self.conn):
            self._pending(experience_id)
            self.repo.set_status(experience_id, REJECTED, reviewer, note)

    def _pending(self, experience_id: str):
        row = self.repo.get(experience_id)
        if row is None:
            raise ApprovalError(f"no experience {experience_id}")
        if row["status"] != PENDING:
            raise ApprovalError(f"experience is {row['status']}, only PENDING can be reviewed")
        return row
