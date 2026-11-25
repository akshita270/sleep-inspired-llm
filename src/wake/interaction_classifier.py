"""Decide what kind of message the user just sent.

This is a deliberately simple, rule-based classifier (regular expressions).
Why not ask the LLM? Rules are fast, free, repeatable, and easy to test and
explain. The trade-off: they miss unusual phrasing. We can swap in an LLM
classifier later behind the same `classify()` function.

Labels:
    QUESTION   - the user wants an answer          -> never stored
    NEW_FACT   - the user states information       -> candidate experience
    CORRECTION - the user says the model was wrong -> candidate experience
    FEEDBACK   - praise / complaint about an answer-> candidate experience
    CHAT       - greetings and small talk          -> never stored
"""
import re
from dataclasses import dataclass

QUESTION = "QUESTION"
NEW_FACT = "NEW_FACT"
CORRECTION = "CORRECTION"
FEEDBACK = "FEEDBACK"
CHAT = "CHAT"

# Words that start a question even without a "?"
_QUESTION_START = re.compile(
    r"^(what|who|whom|whose|when|where|why|how|which|is|are|was|were|do|does|did|"
    r"can|could|will|would|should|tell me|explain|define|name|list)\b", re.I)

# Messages that begin like a correction. Group 1 = the prefix, rest = content.
_CORRECTION_PREFIX = re.compile(
    r"^\s*(correction\s*[:,-]?|actually[,:]?|that'?s (?:wrong|incorrect|not (?:right|correct))[,.:]?|"
    r"that is (?:wrong|incorrect|not (?:right|correct))[,.:]?|no[,.:]|wrong[,.:]|incorrect[,.:])\s*",
    re.I)

# "X is Y, not Z" also signals a correction.
_NOT_PATTERN = re.compile(r",\s*not\s+\w+", re.I)

_FEEDBACK = re.compile(
    r"\b(thanks|thank you|good answer|great answer|helpful|well done|nice|"
    r"bad answer|not helpful|useless|that was (?:wrong|bad|good|great))\b", re.I)

# Words that mean the information is about the user, not the world.
_PERSONAL = re.compile(r"\b(my|mine|i|i'm|i am|i've|i'd|me|we|our|ours)\b", re.I)

# Explicit "please learn this" markers.
_LEARN_MARKER = re.compile(
    r"^\s*(remember|learn|please learn|please remember|note)\s*(this)?\s*[:,-]\s*", re.I)

_STATEMENT_VERB = re.compile(
    r"\b(is|are|was|were|has|have|had|means|equals|called|named|founded|invented|"
    r"discovered|wrote|written|born|located|prefer|prefers|like|likes|use|uses|works)\b", re.I)


@dataclass
class Classification:
    label: str                 # one of the labels above
    text: str                  # message with markers like "Correction:" removed
    is_personal: bool          # about the user rather than general knowledge
    learn_requested: bool      # user explicitly asked the system to learn it


def classify(message: str) -> Classification:
    original = message.strip()
    text = original

    learn_requested = False
    marker = _LEARN_MARKER.match(text)
    if marker:
        learn_requested = True
        text = text[marker.end():].strip()

    is_personal = bool(_PERSONAL.search(text))

    # 1. Corrections (checked before questions: "No, what I meant..." is rare)
    prefix = _CORRECTION_PREFIX.match(text)
    if prefix:
        content = text[prefix.end():].strip()
        if len(content.split()) >= 3:
            return Classification(CORRECTION, content, is_personal, learn_requested)
        return Classification(FEEDBACK, original, is_personal, learn_requested)
    if _NOT_PATTERN.search(text) and _STATEMENT_VERB.search(text) and not text.endswith("?"):
        return Classification(CORRECTION, text, is_personal, learn_requested)

    # 2. Questions
    if text.endswith("?") or _QUESTION_START.match(text):
        return Classification(QUESTION, text, is_personal, learn_requested)

    # 3. Feedback
    if _FEEDBACK.search(text):
        return Classification(FEEDBACK, text, is_personal, learn_requested)

    # 4. Statements that look like information
    if len(text.split()) >= 3 and _STATEMENT_VERB.search(text):
        return Classification(NEW_FACT, text, is_personal, learn_requested)

    return Classification(CHAT, text, is_personal, learn_requested)
