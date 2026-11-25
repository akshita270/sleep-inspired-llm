"""Tests for the wake-phase classifier and collector. No model is loaded."""
import pytest

from src.wake import interaction_classifier as ic
from src.wake.experience_collector import ExperienceCollector


@pytest.mark.parametrize("message,label", [
    ("What is the capital of Australia?", ic.QUESTION),
    ("who wrote Hamlet", ic.QUESTION),
    ("Is Python a compiled language?", ic.QUESTION),
    ("Tell me about black holes", ic.QUESTION),
    ("My preferred programming language is Python.", ic.NEW_FACT),
    ("The capital of Zorvania is Kelmirth.", ic.NEW_FACT),
    ("Correction: The capital of Australia is Canberra, not Sydney.", ic.CORRECTION),
    ("Actually, the Nile is the longest river in Africa.", ic.CORRECTION),
    ("That's wrong, the capital of Canada is Ottawa.", ic.CORRECTION),
    ("The capital of Australia is Canberra, not Sydney.", ic.CORRECTION),
    ("That's wrong.", ic.FEEDBACK),
    ("Thanks, good answer!", ic.FEEDBACK),
    ("Hello", ic.CHAT),
    ("ok", ic.CHAT),
])
def test_labels(message, label):
    assert ic.classify(message).label == label


def test_correction_prefix_is_stripped():
    c = ic.classify("Correction: The capital of Australia is Canberra, not Sydney.")
    assert c.text == "The capital of Australia is Canberra, not Sydney."


def test_personal_vs_general():
    assert ic.classify("My preferred programming language is Python.").is_personal
    assert not ic.classify("The capital of Zorvania is Kelmirth.").is_personal


def test_learn_marker():
    c = ic.classify("Remember: the Calloway Star was launched in 1964.")
    assert c.learn_requested and c.label == ic.NEW_FACT
    assert not ic.classify("The Calloway Star was launched in 1964.").learn_requested


def test_questions_and_chat_are_not_collected(tmp_path):
    col = ExperienceCollector("s1", staging_file=tmp_path / "x.jsonl")
    for msg in ["What is 2+2?", "Hello"]:
        assert col.collect(ic.classify(msg), msg, None, None) is None
    assert col.collected == []


def test_correction_is_collected_as_pending_unverified(tmp_path):
    col = ExperienceCollector("s1", staging_file=tmp_path / "x.jsonl")
    msg = "Correction: The capital of Australia is Canberra, not Sydney."
    exp = col.collect(ic.classify(msg), msg, "What is the capital of Australia?", "Sydney.")
    assert exp.status == "PENDING"
    assert exp.source == "user_claim"
    assert exp.previous_answer == "Sydney."
    assert exp.corrected_answer.startswith("The capital of Australia is Canberra")
    assert (tmp_path / "x.jsonl").read_text().count("\n") == 1


def test_personal_fact_requires_explicit_approval(tmp_path):
    col = ExperienceCollector("s1", staging_file=tmp_path / "x.jsonl")
    msg = "My preferred programming language is Python."
    exp = col.collect(ic.classify(msg), msg, None, None)
    assert exp.category == "personal" and exp.requires_explicit_approval
