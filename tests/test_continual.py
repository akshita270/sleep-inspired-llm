import json

import pandas as pd
import pytest

from experiments.continual_learning import ARMS, cycle_row, fact_group
from src.memory.database import get_connection
from src.sleep import replay_pool
from src.sleep.sleep_controller import SleepController
from experiments.common import add_fixture_facts, fresh_fixture_db
from src.evaluation.benchmark_manager import load_benchmark


def test_fact_groups_are_six_facts_per_cycle():
    assert [fact_group(f"new_{i:03d}") for i in (1, 6, 7, 12, 13, 30)] == [1, 1, 2, 2, 3, 5]


def test_arms_differ_only_in_replay():
    assert ARMS["no_replay"] == {"replay_ratio": 0.0, "previous_replay_ratio": 0.0}
    assert ARMS["general_replay"]["previous_replay_ratio"] == 0.0 and ARMS["general_replay"]["replay_ratio"] == 1.0
    assert ARMS["full_replay"]["previous_replay_ratio"] == 1.0


def test_fresh_db_can_start_empty_and_receive_facts_per_cycle(tmp_path):
    conn, d = fresh_fixture_db(tmp_path / "e.db", ids=[])
    assert sum(d.values()) == 0
    items = load_benchmark("new_knowledge")
    assert add_fixture_facts(conn, items[:6])["APPROVED"] == 6
    assert add_fixture_facts(conn, items[6:12])["APPROVED"] == 6
    assert conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0] == 12


def test_cycle_row_separates_this_cycle_earlier_cycles_and_forgetting():
    ids = [f"new_{i:03d}" for i in range(1, 31)]
    new = pd.DataFrame({"benchmark": "new_knowledge", "id": ids,
                        "correct": [True] * 6 + [False, True, True, False, False, False] + [False] * 18,
                        "model_answer": "", "type": None})
    old = pd.DataFrame({"benchmark": "old_knowledge", "id": ["o1", "o2", "o3", "o4"],
                        "correct": [True, True, True, False], "model_answer": "", "type": None})
    base = pd.concat([new.assign(correct=False), old.assign(correct=[True, True, True, True])])
    details = pd.concat([new, old])
    summary = {"accuracy": {"new_knowledge": 8 / 30, "old_knowledge": 0.75, "reasoning": 1.0},
               "hallucination": {"old_knowledge": {"confidently_wrong": 1}}, "median_latency_s": 1.5,
               "median_tokens_per_s": 7.0, "gpu_peak_memory_gb": 8.3}
    training = {"n_train_examples": 70, "training_seconds": 100.0, "final_val_loss": 0.1, "adapter_size_bytes": 5.8e7}
    manifest = {"replay": {"added": 20, "previous_added": 20}}
    r = cycle_row("full_replay", 2, details, base, {"new_knowledge": 0, "old_knowledge": 1.0, "reasoning": 1.0},
                  summary, training, manifest, ids[6:12], base_verified=True)
    assert r["acc_this_cycle_facts"] == pytest.approx(2 / 6, abs=1e-3)          # cycle 2 facts: 2 of 6 correct
    assert r["acc_earlier_cycles_facts"] == 1.0                       # cycle 1 facts all kept
    assert r["acc_taught_so_far"] == pytest.approx(8 / 12, abs=1e-3)         # 6 + 2 of the 12 taught so far
    assert r["forgetting_rate"] == 0.25 and r["questions_lost"] == 1
    assert r["gate_would_pass"] is False and "forgetting_rate" in r["gate_failed_checks"]
    assert r["acc_by_fact_group"][1] == 1.0 and r["acc_by_fact_group"][5] == 0.0


def test_previous_replay_reuses_only_earlier_fact_wordings(tmp_path, monkeypatch):
    class Gen:
        name = "fake"
        def __init__(self): self.n = 0
        def generate(self, statement, n):
            return [f"{statement[:25]} variant {i}?" for i in range(5)][:n]

    monkeypatch.setattr(replay_pool, "load_pool",
                        lambda: [{"question": f"General q {i} on subject {i * 11}?", "answer": f"A{i}."} for i in range(10)])
    conn, _ = fresh_fixture_db(tmp_path / "e.db", ids=[])
    items = load_benchmark("new_knowledge")
    add_fixture_facts(conn, items[:2])
    r1 = SleepController(conn, Gen(), dataset_root=tmp_path / "ds",
                         dataset_overrides={"replay_ratio": 1.0, "previous_replay_ratio": 1.0,
                                            "previous_datasets": []}).run()
    conn.execute("UPDATE sleep_cycles SET status='SUCCESS'")
    add_fixture_facts(conn, items[2:4])
    r2 = SleepController(conn, Gen(), dataset_root=tmp_path / "ds",
                         dataset_overrides={"replay_ratio": 1.0, "previous_replay_ratio": 1.0,
                                            "previous_datasets": [str(r1["dataset_dir"])]}).run()
    m = json.loads((r2["dataset_dir"] / "manifest.json").read_text())
    assert m["replay"]["previous_added"] == m["replay"]["added"] > 0
    rows = [json.loads(l) for l in open(r2["dataset_dir"] / "index.jsonl")]
    prev = [r for r in rows if r["origin"] == "previous_replay"]
    assert prev and {r["experience_id"] for r in prev} <= {items[0]["id"], items[1]["id"]}
    assert all(r["split"] == "train" for r in prev)
    assert not any(r["origin"] == "previous_replay" for r in map(json.loads, open(r1["dataset_dir"] / "index.jsonl")))
