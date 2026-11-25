import json

import pytest

from src.evaluation.benchmark_manager import DEV_BENCHMARKS, load_dev
from src.memory.database import get_connection
from src.sleep import replay_pool
from src.sleep.dataset_builder import eval_questions
from src.sleep.sleep_controller import SleepController
from src.training.lora_trainer import LoRATrainer
from src.validation.approval_manager import ApprovalManager


def dev_questions():
    return [i["question"] for n in DEV_BENCHMARKS for i in load_dev(n)]


def test_replay_dev_and_eval_questions_never_overlap():
    replay = replay_pool.load_questions()
    assert len(replay) == 100 and len(dev_questions()) == 60
    assert replay_pool.find_overlaps(replay, eval_questions(), 0.8) == []
    assert replay_pool.find_overlaps(replay, dev_questions(), 0.8) == []
    assert replay_pool.find_overlaps(dev_questions(), eval_questions(), 0.8) == []


def test_dev_ids_are_unique():
    ids = [i["id"] for n in DEV_BENCHMARKS for i in load_dev(n)]
    assert len(ids) == len(set(ids)) == 60


POOL = [{"question": f"General question number {i} about topic {i * 7}?", "answer": f"Answer {i}."}
        for i in range(10)]


def test_sample_replay_repeats_pool_when_more_is_requested_and_is_deterministic():
    a, _ = replay_pool.sample_replay(POOL, 25, seed=1, eval_questions=[], leak_similarity=0.8)
    b, _ = replay_pool.sample_replay(POOL, 25, seed=1, eval_questions=[], leak_similarity=0.8)
    assert len(a) == 25 and [e.instruction for e in a] == [e.instruction for e in b]
    assert {e.instruction for e in a} == {p["question"] for p in POOL}
    assert all(e.origin == "replay" and e.experience_id == "replay" for e in a)


def test_sample_replay_drops_questions_that_resemble_evaluation_questions():
    pool = POOL + [{"question": "What is the capital of Australia?", "answer": "Canberra."}]
    examples, dropped = replay_pool.sample_replay(pool, 50, 1, ["What is the capital of Australia?"], 0.8)
    assert dropped == 1 and "What is the capital of Australia?" not in {e.instruction for e in examples}


def test_zero_replay_requested_gives_nothing():
    assert replay_pool.sample_replay(POOL, 0, 1, [], 0.8) == ([], 0)


class Gen:
    name = "fake"

    def generate(self, statement, n):
        return [f"Tell me about the Zorvania capital, wording {i}?" for i in range(5)][:n]


@pytest.fixture
def conn_with_fact(tmp_path):
    conn = get_connection(tmp_path / "t.db")
    ApprovalManager(conn).submit({
        "experience_id": "e1", "session_id": "s", "interaction_type": "NEW_FACT",
        "original_input": "The capital of Zorvania is Kelmirth.",
        "corrected_answer": "The capital of Zorvania is Kelmirth.", "category": "general",
        "source": "experiment_fixture", "previous_answer": None, "previous_question": None,
        "learn_requested": False, "requires_explicit_approval": False})
    return conn


def test_replay_is_added_to_train_only_and_recorded(conn_with_fact, tmp_path, monkeypatch):
    monkeypatch.setattr(replay_pool, "load_pool", lambda: POOL)
    r = SleepController(conn_with_fact, Gen(), dataset_root=tmp_path / "ds",
                        dataset_overrides={"replay_ratio": 2.0}).run()
    d = r["dataset_dir"]
    m = json.loads((d / "manifest.json").read_text())
    n_fact_train = 3                                     # 5 wordings -> 3 train, 1 valid, 1 test
    assert m["replay"]["added"] == 2 * n_fact_train == 6
    assert m["counts"] == {"train": 3 + 6, "valid": 1, "test": 1}
    index = [json.loads(l) for l in open(d / "index.jsonl")]
    assert {r["split"] for r in index if r["experience_id"] == "replay"} == {"train"}
    assert m["dataset_config"]["replay_ratio"] == 2.0


def test_no_replay_by_default(conn_with_fact, tmp_path):
    r = SleepController(conn_with_fact, Gen(), dataset_root=tmp_path / "ds").run()
    assert json.loads((r["dataset_dir"] / "manifest.json").read_text())["replay"]["added"] == 0


def test_training_overrides_reach_the_trainer_config(tmp_path):
    t = LoRATrainer(adapter_root=tmp_path, overrides={"epochs": 3, "rank": 8, "num_layers": 8})
    cfg = t.build_config(tmp_path / "d", tmp_path / "o", n_train=100)
    assert cfg["iters"] == 25 * 3 and cfg["lora_parameters"]["rank"] == 8 and cfg["num_layers"] == 8
