import pytest

from src.memory.database import get_connection
from src.validation.approval_manager import ApprovalError, ApprovalManager
from src.validation.conflict_detector import find_conflicts, parse_claim

_n = 0


def exp(text, itype="NEW_FACT", category="general", source="user_claim", **kw):
    global _n
    _n += 1
    d = {"experience_id": f"e{_n}", "session_id": "s", "interaction_type": itype,
         "original_input": text, "corrected_answer": text, "category": category,
         "source": source, "previous_answer": None, "previous_question": None,
         "learn_requested": False, "requires_explicit_approval": category == "personal"}
    d.update(kw)
    return d


@pytest.fixture
def mgr(tmp_path):
    return ApprovalManager(get_connection(tmp_path / "t.db"))


def test_user_claim_stays_pending(mgr):
    assert mgr.submit(exp("The capital of Zorvania is Kelmirth."))["status"] == "PENDING"


def test_trusted_source_is_approved(mgr):
    r = mgr.submit(exp("The capital of Zorvania is Kelmirth.", source="experiment_fixture"))
    assert r["status"] == "APPROVED"


def test_evidence_link_is_approved(mgr):
    r = mgr.submit(exp("The Calloway Star was launched in 1964.", evidence_url="https://example.org"))
    assert r["status"] == "APPROVED"


def test_too_short_is_rejected(mgr):
    assert mgr.submit(exp("Python rocks"))["status"] == "REJECTED"


def test_question_is_rejected(mgr):
    assert mgr.submit(exp("What is the capital of Zorvania?"))["status"] == "REJECTED"


def test_exact_and_near_duplicates_are_rejected(mgr):
    mgr.submit(exp("The capital of Zorvania is Kelmirth."))
    assert mgr.submit(exp("the capital of zorvania is kelmirth"))["status"] == "REJECTED"
    assert mgr.submit(exp("The capital of Zorvania is Kelmirth city."))["status"] == "REJECTED"


def test_conflict_goes_to_manual_review(mgr):
    mgr.submit(exp("The capital of Zorvania is Kelmirth.", source="experiment_fixture"))
    r = mgr.submit(exp("The capital of Zorvania is Valtara."))
    assert r["status"] == "PENDING" and "conflict" in r["notes"][0]


def test_personal_needs_explicit_approval(mgr):
    r = mgr.submit(exp("My preferred programming language is Python.", category="personal"))
    assert r["status"] == "PENDING"
    with pytest.raises(ApprovalError):
        mgr.approve(r["experience_id"])
    mgr.approve(r["experience_id"], explicit_personal_approval=True)
    assert mgr.repo.get(r["experience_id"])["status"] == "APPROVED"


def test_email_in_statement_is_treated_as_personal(mgr):
    r = mgr.submit(exp("The support contact for Plumeria is help@plumeria.example."))
    assert mgr.repo.get(r["experience_id"])["requires_explicit_approval"] == 1


def test_cannot_approve_a_claim_that_conflicts_with_approved(mgr):
    mgr.submit(exp("The capital of Zorvania is Kelmirth.", source="experiment_fixture"))
    r = mgr.submit(exp("The capital of Zorvania is Valtara."))
    with pytest.raises(ApprovalError):
        mgr.approve(r["experience_id"])


def test_only_pending_can_be_reviewed_and_reject_works(mgr):
    r = mgr.submit(exp("The capital of Zorvania is Kelmirth."))
    mgr.reject(r["experience_id"], note="unsure")
    assert mgr.repo.get(r["experience_id"])["status"] == "REJECTED"
    with pytest.raises(ApprovalError):
        mgr.approve(r["experience_id"])


def test_reimporting_same_experience_is_skipped(mgr):
    e = exp("The capital of Zorvania is Kelmirth.")
    mgr.submit(e)
    assert mgr.submit(e)["status"] == "SKIPPED"
    assert sum(mgr.repo.counts().values()) == 1


def test_parse_claim_handles_not_pattern():
    subj, val = parse_claim("The capital of Australia is Canberra, not Sydney.")
    assert "canberra" in val and "sydney" not in val


def test_find_conflicts_ignores_unrelated():
    existing = [{"experience_id": "x", "statement": "The capital of Zorvania is Kelmirth."}]
    assert find_conflicts("Mount Harrowgate is 6412 metres tall.", existing, 0.8) == []
