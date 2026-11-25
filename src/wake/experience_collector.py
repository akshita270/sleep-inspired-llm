"""Turn selected interactions into candidate "experiences".

An experience is a *proposal* to learn something. Nothing here trains the
model, and nothing here decides the information is true. Every experience
starts as PENDING. In Phase 4 the validator and a human approve or reject
them, and only APPROVED ones can ever reach the Sleep Phase.

For now experiences are appended to a JSONL staging file (data/raw/) so that
Phase 4 can import them into SQLite. The fields mirror the planned
`experiences` table.
"""
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime

from src.utils.config_loader import PROJECT_ROOT
from src.wake import interaction_classifier as ic

STAGING_FILE = PROJECT_ROOT / "data" / "raw" / "wake_experiences.jsonl"

STATUS_PENDING = "PENDING"


@dataclass
class Experience:
    experience_id: str
    session_id: str
    interaction_type: str          # NEW_FACT / CORRECTION / FEEDBACK
    original_input: str            # exactly what the user typed
    previous_answer: str | None    # what the model said just before (for corrections)
    previous_question: str | None  # the question that led to that answer
    corrected_answer: str | None   # the user's proposed correction / new fact
    category: str                  # "personal" or "general"
    source: str                    # who claims this; "user_claim" = unverified
    confidence: float | None       # filled in by the validator in Phase 4
    status: str                    # PENDING / APPROVED / REJECTED
    learn_requested: bool          # user explicitly asked to learn this
    requires_explicit_approval: bool  # personal info: never train without a clear yes
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ExperienceCollector:
    def __init__(self, session_id: str, staging_file=STAGING_FILE):
        self.session_id = session_id
        self.staging_file = staging_file
        self.collected: list[Experience] = []

    def collect(self, c: ic.Classification, original_input: str,
                last_question: str | None, last_answer: str | None) -> Experience | None:
        """Create a PENDING experience for facts, corrections and feedback.

        Questions and small talk are never collected: not everything the user
        says is worth learning.
        """
        if c.label not in (ic.NEW_FACT, ic.CORRECTION, ic.FEEDBACK):
            return None

        exp = Experience(
            experience_id=uuid.uuid4().hex[:12],
            session_id=self.session_id,
            interaction_type=c.label,
            original_input=original_input,
            previous_answer=last_answer if c.label in (ic.CORRECTION, ic.FEEDBACK) else None,
            previous_question=last_question if c.label in (ic.CORRECTION, ic.FEEDBACK) else None,
            corrected_answer=c.text if c.label != ic.FEEDBACK else None,
            category="personal" if c.is_personal else "general",
            source="user_claim",        # an unverified claim, NOT a verified fact
            confidence=None,
            status=STATUS_PENDING,
            learn_requested=c.learn_requested,
            requires_explicit_approval=c.is_personal,
        )
        self.collected.append(exp)
        self._append_to_staging(exp)
        return exp

    def _append_to_staging(self, exp: Experience) -> None:
        self.staging_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.staging_file, "a") as f:
            f.write(json.dumps(asdict(exp)) + "\n")
