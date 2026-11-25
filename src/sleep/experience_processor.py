"""Choose which APPROVED experiences go into the next sleep cycle.

Safety rules enforced here (the last gate before training data is built):
  - only APPROVED rows, never PENDING or REJECTED
  - personal information is left out unless config says otherwise
  - experiences already learned in a successful cycle are skipped
  - a statement that conflicts with an earlier selected or learned one is held back
"""
import sqlite3

from src.validation import conflict_detector


def select_training_experiences(conn: sqlite3.Connection, learned_ids: set[str],
                                include_personal: bool,
                                conflict_threshold: float) -> tuple[list[dict], list[dict]]:
    """Return (selected, excluded). Each excluded item carries a reason."""
    rows = conn.execute(
        """SELECT * FROM experiences
           WHERE status = 'APPROVED' AND interaction_type IN ('NEW_FACT','CORRECTION')
             AND corrected_answer IS NOT NULL
           ORDER BY created_at, experience_id""").fetchall()

    selected, excluded = [], []
    already = [{"experience_id": r["experience_id"], "statement": r["corrected_answer"]}
               for r in conn.execute(
                   "SELECT experience_id, corrected_answer FROM experiences WHERE experience_id IN (%s)"
                   % ",".join("?" * len(learned_ids)), tuple(learned_ids)).fetchall()
               ] if learned_ids else []

    for r in rows:
        row = dict(r)
        if row["experience_id"] in learned_ids:
            continue  # already part of the model
        if row["category"] == "personal" and not include_personal:
            excluded.append({"experience_id": row["experience_id"], "reason": "personal information"})
            continue
        known = already + [{"experience_id": s["experience_id"], "statement": s["corrected_answer"]}
                           for s in selected]
        clash = conflict_detector.find_conflicts(row["corrected_answer"], known, conflict_threshold)
        if clash:
            excluded.append({"experience_id": row["experience_id"],
                             "reason": f"conflicts with {', '.join(clash)}"})
            continue
        selected.append(row)
    return selected, excluded
