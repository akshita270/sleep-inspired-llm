import pytest

from src.memory.database import get_connection
from src.sleep.cycle_manager import CycleManager
from src.sleep.experience_processor import select_training_experiences
from src.sleep.sleep_controller import SleepController
from src.sleep.sleep_state import SystemState
from src.validation.approval_manager import ApprovalManager
from src.wake.conversation_manager import ConversationManager

_n = 0


def fx(text, source="experiment_fixture", category="general", prev_q=None):
    global _n
    _n += 1
    return {"experience_id": f"e{_n}", "session_id": "s", "interaction_type": "NEW_FACT",
            "original_input": text, "corrected_answer": text, "category": category,
            "source": source, "previous_answer": None, "previous_question": prev_q,
            "learn_requested": False, "requires_explicit_approval": category == "personal"}


class FakeGen:
    name = "fake"

    def generate(self, statement, n):
        subject = statement.split(" is ")[0].replace("The ", "").lower()
        return [f"Tell me about {subject} number {i}?" for i in range(5)][:n]


@pytest.fixture
def conn(tmp_path):
    return get_connection(tmp_path / "t.db")


@pytest.fixture
def mgr(conn):
    return ApprovalManager(conn)


def select(conn, learned=None, personal=False):
    return select_training_experiences(conn, learned or set(), personal, 0.8)


# ---- state -------------------------------------------------------------
def test_state_starts_awake_and_cannot_be_acquired_twice(conn):
    s = SystemState(conn)
    assert s.get() == "AWAKE"
    assert s.try_acquire("SLEEPING") is True
    assert s.try_acquire("SLEEPING") is False


def test_busy_always_wakes_up_even_after_error(conn):
    s = SystemState(conn)
    with pytest.raises(ValueError):
        with s.busy():
            assert s.get() == "SLEEPING"
            raise ValueError("boom")
    assert s.get() == "AWAKE"


def test_chat_is_blocked_while_asleep_and_works_when_awake(conn):
    s = SystemState(conn)
    chat = ConversationManager(None, system_state=s)   # no model needed for these paths
    s.try_acquire("TRAINING")
    r = chat.handle("What is the capital of France?")
    assert "training" in r["reply"] and r["label"] is None
    s.force_wake()
    assert chat.handle("Hello")["label"] == "CHAT"


# ---- experience selection ---------------------------------------------
def test_only_approved_non_personal_experiences_are_selected(conn, mgr):
    ok = mgr.submit(fx("The capital of Zorvania is Kelmirth."))["experience_id"]
    mgr.submit(fx("The national sport of Zorvania is skarn.", source="user_claim"))      # PENDING
    p = mgr.submit(fx("My favourite language is Python.", category="personal"))["experience_id"]
    mgr.approve(p, explicit_personal_approval=True)                                      # approved personal
    selected, excluded = select(conn)
    assert [e["experience_id"] for e in selected] == [ok]
    assert any(x["reason"] == "personal information" for x in excluded)
    assert [e["experience_id"] for e in select(conn, personal=True)[0]] == [ok, p]


def test_conflicting_approved_facts_are_held_back(conn, mgr):
    a = mgr.submit(fx("The capital of Zorvania is Kelmirth."))["experience_id"]
    b = mgr.submit(fx("The capital of Zorvania is Valtara.", source="user_claim"))["experience_id"]
    mgr.approve(b, override_conflict=True)
    selected, excluded = select(conn)
    assert [e["experience_id"] for e in selected] == [a]
    assert excluded[0]["experience_id"] == b and "conflicts" in excluded[0]["reason"]


def test_already_learned_experiences_are_skipped(conn, mgr):
    a = mgr.submit(fx("The capital of Zorvania is Kelmirth."))["experience_id"]
    assert select(conn, learned={a})[0] == []


# ---- cycles and controller --------------------------------------------
def test_cycle_ids_increment(conn):
    c = CycleManager(conn)
    assert (c.start(), c.start()) == ("cycle_001", "cycle_002")


def test_dry_run_prepares_dataset_and_wakes_up(conn, mgr, tmp_path):
    mgr.submit(fx("The capital of Zorvania is Kelmirth."))
    r = SleepController(conn, FakeGen(), dataset_root=tmp_path / "ds").run()
    assert r["status"] == "PREPARED" and r["counts"]["train"] == 3
    assert (r["dataset_dir"] / "manifest.json").exists()
    assert SystemState(conn).get() == "AWAKE"
    assert conn.execute("SELECT status FROM sleep_cycles").fetchone()[0] == "PREPARED"


def test_nothing_to_learn_creates_no_cycle(conn, tmp_path):
    r = SleepController(conn, FakeGen(), dataset_root=tmp_path).run()
    assert r["status"] == "NOTHING_TO_LEARN"
    assert conn.execute("SELECT COUNT(*) FROM sleep_cycles").fetchone()[0] == 0
    assert SystemState(conn).get() == "AWAKE"


def test_training_runs_in_training_state_and_only_promotion_consumes_experiences(conn, mgr, tmp_path):
    mgr.submit(fx("The capital of Zorvania is Kelmirth."))
    seen = {}

    def train(dataset_dir, cycle_id):
        seen["state"] = SystemState(conn).get()
        return {"adapter_version": "adapter_001"}

    ctl = SleepController(conn, FakeGen(), train_fn=train, dataset_root=tmp_path)
    r = ctl.run()
    assert r["status"] == "TRAINED"                          # trained, not yet approved
    assert seen["state"] == "TRAINING"
    assert SystemState(conn).get() == "AWAKE"
    # until evaluation promotes the candidate, the facts are not counted as learned
    assert CycleManager(conn).learned_experience_ids() == set()
    CycleManager(conn).finish(r["cycle_id"], "SUCCESS")      # Phase 8 will do this after evaluation
    assert ctl.run()["status"] == "NOTHING_TO_LEARN"


def test_failed_training_is_logged_wakes_up_and_experiences_stay_eligible(conn, mgr, tmp_path):
    mgr.submit(fx("The capital of Zorvania is Kelmirth."))

    def broken(dataset_dir, cycle_id):
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        SleepController(conn, FakeGen(), train_fn=broken, dataset_root=tmp_path).run()
    assert SystemState(conn).get() == "AWAKE"
    row = conn.execute("SELECT status, notes FROM sleep_cycles").fetchone()
    assert row["status"] == "FAILED" and "out of memory" in row["notes"]
    assert len(select(conn, CycleManager(conn).learned_experience_ids())[0]) == 1


def test_cannot_start_a_second_sleep_while_busy(conn, mgr, tmp_path):
    mgr.submit(fx("The capital of Zorvania is Kelmirth."))
    SystemState(conn).try_acquire("TRAINING")
    with pytest.raises(RuntimeError):
        SleepController(conn, FakeGen(), dataset_root=tmp_path).run()
