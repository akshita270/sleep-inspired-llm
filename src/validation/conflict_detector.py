"""Detect statements that disagree with stored ones.

Method (simple on purpose): split "SUBJECT is VALUE" into two parts. If two
statements talk about the same subject but give different values, flag it.

    "The capital of Zorvania is Kelmirth"  vs  "The capital of Zorvania is Valtara"  -> conflict

Limits: it only understands "X is/was/has ... Y" sentences, and it can
over-flag ("Zorvania is a country" vs "Zorvania is cold"). A flag never
rejects anything. It only sends the experience to a human for review.
"""
import re

from src.validation.duplicate_detector import jaccard, tokens

_ARTICLES = {"the", "a", "an"}
_SPLIT = re.compile(r"^(?P<subj>.+?)\s+(?P<verb>is|are|was|were|has|have|means|equals)\s+(?P<val>.+)$")


def parse_claim(statement: str):
    """Return (subject_words, value_words), or None if the sentence has no 'X is Y' shape."""
    text = statement.lower().strip().rstrip(".! ")
    text = re.sub(r",\s*not\s+.*$", "", text)  # "Canberra, not Sydney" -> keep "Canberra"
    m = _SPLIT.match(text)
    if not m:
        return None
    subj = {w for w in tokens(m["subj"]) if w not in _ARTICLES}
    val = {w for w in tokens(m["val"]) if w not in _ARTICLES}
    return (subj, val) if subj and val else None


def find_conflicts(statement: str, existing: list[dict], subject_threshold: float) -> list[str]:
    """Return ids of stored statements with the same subject but a different value."""
    claim = parse_claim(statement)
    if claim is None:
        return []
    subj, val = claim
    conflicts = []
    for row in existing:
        other = parse_claim(row["statement"])
        if other is None:
            continue
        if jaccard(subj, other[0]) >= subject_threshold and val != other[1]:
            conflicts.append(row["experience_id"])
    return conflicts
