import pandas as pd
import pytest

from src.evaluation.forgetting_evaluator import forgetting_report
from src.evaluation.knowledge_evaluator import hallucination_stats, is_abstention, taught_accuracy
from src.registry.champion_manager import run_gate

GATE = {"require_base_unchanged": True, "min_taught_accuracy": 0.8, "max_new_knowledge_drop": 0.0,
        "max_old_knowledge_drop": 0.02, "max_reasoning_drop": 0.05, "max_forgetting_rate": 0.05}


def table(rows):
    return pd.DataFrame(rows, columns=["benchmark", "id", "correct", "model_answer"])


def test_forgetting_counts_only_questions_the_base_model_got_right():
    base = table([("old_knowledge", "o1", True, ""), ("old_knowledge", "o2", True, ""),
                  ("old_knowledge", "o3", False, ""), ("reasoning", "r1", True, ""),
                  ("new_knowledge", "n1", False, "")])
    cand = table([("old_knowledge", "o1", True, ""), ("old_knowledge", "o2", False, ""),
                  ("old_knowledge", "o3", True, ""), ("reasoning", "r1", True, ""),
                  ("new_knowledge", "n1", True, "")])
    r = forgetting_report(cand, base)
    assert r["old_knowledge"]["lost_ids"] == ["o2"] and r["old_knowledge"]["gained_ids"] == ["o3"]
    assert r["combined"]["base_correct"] == 3 and r["combined"]["lost"] == 1
    assert r["combined"]["forgetting_rate"] == pytest.approx(0.3333, abs=1e-3)
    assert "new_knowledge" not in r                      # cannot forget what was never known


def test_no_forgetting_when_nothing_is_lost():
    t = table([("old_knowledge", "o1", True, ""), ("reasoning", "r1", True, "")])
    assert forgetting_report(t, t)["combined"]["forgetting_rate"] == 0.0


def test_abstention_detection():
    assert is_abstention("I don't know the capital of Zorvania.")
    assert is_abstention("That country does not exist.")
    assert not is_abstention("The capital of Zorvania is Valtara.")


def test_hallucination_stats_separate_wrong_from_abstained():
    t = table([("new_knowledge", "n1", False, "The capital is Valtara."),
               ("new_knowledge", "n2", False, "I don't know."),
               ("new_knowledge", "n3", True, "Kelmirth.")])
    s = hallucination_stats(t)["new_knowledge"]
    assert (s["correct"], s["abstained"], s["confidently_wrong"]) == (1, 1, 1)


def test_taught_accuracy_uses_only_this_cycles_facts():
    t = table([("new_knowledge", "n1", True, ""), ("new_knowledge", "n2", False, ""),
               ("new_knowledge", "n3", False, ""), ("old_knowledge", "n1", False, "")])
    assert taught_accuracy(t, ["n1", "n2"]) == 0.5
    assert taught_accuracy(t, ["not_an_eval_id"]) is None


CHAMP = {"new_knowledge": 0.0, "old_knowledge": 0.99, "reasoning": 0.93}


def gate(cand, forgetting=0.0, taught=1.0, base_ok=True):
    return run_gate(cand, CHAMP, forgetting, taught, base_ok, GATE)


GOOD = {"new_knowledge": 1.0, "old_knowledge": 0.99, "reasoning": 0.93}


def test_gate_passes_a_good_candidate():
    assert gate(GOOD)["passed"]


@pytest.mark.parametrize("kwargs,failed", [
    ({"cand": {**GOOD, "reasoning": 0.85}}, "reasoning_drop"),
    ({"cand": {**GOOD, "old_knowledge": 0.95}}, "old_knowledge_drop"),
    ({"cand": GOOD, "forgetting": 0.10}, "forgetting_rate"),
    ({"cand": GOOD, "taught": 0.5}, "taught_facts_learned"),
    ({"cand": GOOD, "base_ok": False}, "base_weights_unchanged"),
    ({"cand": GOOD, "base_ok": None}, "base_weights_unchanged"),
])
def test_gate_fails_each_unacceptable_degradation(kwargs, failed):
    r = gate(**kwargs)
    assert not r["passed"]
    assert [c["check"] for c in r["checks"] if not c["passed"]] == [failed]


def test_gate_rejects_losing_previously_learned_facts():
    champion = {**CHAMP, "new_knowledge": 0.5}
    r = run_gate({**GOOD, "new_knowledge": 0.4}, champion, 0.0, 1.0, True, GATE)
    assert not r["passed"]


def test_gate_tolerates_small_changes_inside_the_limits():
    assert gate({**GOOD, "reasoning": 0.9}, forgetting=0.05)["passed"]    # exactly at the limits


def test_gate_skips_taught_check_when_facts_are_not_in_the_eval_set():
    r = gate(GOOD, taught=None)
    assert r["passed"] and "skipped" in [c for c in r["checks"] if c["check"] == "taught_facts_learned"][0]["rule"]
