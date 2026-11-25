"""Decide what should happen to a newly collected experience.

IMPORTANT: this validator checks QUALITY and PROVENANCE. It cannot check
whether a claim is TRUE, and nothing here pretends to. A plain user claim is
never auto-approved; it waits for a human (or for a trusted source).

Decision order:
    1. Not a usable statement (too short/long, a question, feedback) -> REJECTED / PENDING
    2. Duplicate of something already stored                         -> REJECTED
    3. Conflicts with something stored                               -> PENDING (human reviews)
    4. Personal information                                          -> PENDING (needs explicit approval)
    5. Trusted source or evidence link                               -> APPROVED
    6. Otherwise (unverified user claim)                             -> PENDING
"""
import re
from dataclasses import dataclass, field

from src.utils.config_loader import load_config
from src.validation import conflict_detector, duplicate_detector

PENDING, APPROVED, REJECTED = "PENDING", "APPROVED", "REJECTED"


@dataclass
class ValidationResult:
    status: str
    confidence: float          # heuristic score, NOT the probability the claim is true
    notes: list = field(default_factory=list)
    requires_explicit_approval: bool = False
    content_hash: str = ""


class ExperienceValidator:
    def __init__(self, rules: dict | None = None):
        self.rules = rules or load_config("system_config")["validation"]
        self._sensitive = [re.compile(p) for p in self.rules["sensitive_patterns"]]

    def validate(self, exp: dict, existing: list[dict]) -> ValidationResult:
        r = self.rules
        statement = (exp.get("corrected_answer") or "").strip()
        h = duplicate_detector.content_hash(statement or exp["original_input"])
        personal = exp.get("category") == "personal" or bool(exp.get("requires_explicit_approval"))

        # 1. Is there a learnable statement at all?
        if exp["interaction_type"] == "FEEDBACK" or not statement:
            return ValidationResult(PENDING, 0.3, ["feedback has no learnable statement; manual review"],
                                    personal, h)
        words = len(statement.split())
        if words < r["min_words"]:
            return ValidationResult(REJECTED, 0.1, [f"incomplete: {words} words (min {r['min_words']})"], personal, h)
        if words > r["max_words"]:
            return ValidationResult(REJECTED, 0.1, [f"too long to be one clear fact: {words} words"], personal, h)
        if statement.endswith("?"):
            return ValidationResult(REJECTED, 0.1, ["this is a question, not information"], personal, h)

        # 2. Duplicate?
        dup = duplicate_detector.find_duplicate(statement, existing, r["duplicate_similarity"])
        if dup:
            kind, other = dup
            return ValidationResult(REJECTED, 0.1, [f"{kind} duplicate of {other}"], personal, h)

        notes, confidence = [], 0.5

        # Private-looking data (email, phone) always needs explicit approval.
        if any(p.search(statement) for p in self._sensitive):
            personal = True
            notes.append("contains email/phone-like data")

        # 3. Conflict?
        conflicts = conflict_detector.find_conflicts(
            statement, existing, r["conflict_subject_similarity"])
        if conflicts:
            notes.append(f"possible conflict with {', '.join(conflicts)}; manual review")
            return ValidationResult(PENDING, 0.2, notes, personal, h)

        # 4. Personal?
        if personal:
            notes.append("personal information: needs explicit approval, kept apart from general knowledge")
            return ValidationResult(PENDING, 0.4, notes, True, h)

        # 5. Trusted source or evidence?
        if exp.get("source") in r["trusted_sources"] or exp.get("evidence_url"):
            notes.append("trusted source or evidence link")
            return ValidationResult(APPROVED, 0.9, notes, False, h)

        # 6. Unverified user claim
        notes.append("unverified user claim: needs evidence or human approval")
        return ValidationResult(PENDING, confidence, notes, False, h)
