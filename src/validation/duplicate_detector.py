"""Detect statements that are already stored.

Two levels:
  exact - same words after lowercasing and removing punctuation (hash match)
  near  - word overlap (Jaccard) above a threshold, e.g. a reworded repeat
"""
import hashlib
import re

_WORDS = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _WORDS.findall(text.lower())


def normalise(text: str) -> str:
    return " ".join(tokens(text))


def content_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode()).hexdigest()


def jaccard(a: set, b: set) -> float:
    """Share of words the two sets have in common (0 = none, 1 = identical)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_duplicate(statement: str, existing: list[dict], threshold: float):
    """Return (kind, experience_id) of the first duplicate found, or None."""
    h = content_hash(statement)
    words = set(tokens(statement))
    for row in existing:
        if row["content_hash"] == h:
            return "exact", row["experience_id"]
    for row in existing:
        if jaccard(words, set(tokens(row["statement"]))) >= threshold:
            return "near", row["experience_id"]
    return None
