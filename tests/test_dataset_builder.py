import json

import pytest

from src.sleep import dataset_builder as db
from src.sleep.question_generator import parse_questions


class FakeGen:
    """Returns fixed questions per fact, no model needed."""
    name = "fake"

    def __init__(self, table):
        self.table = table

    def generate(self, statement, n):
        return self.table.get(statement, [])[:n]


def exp(eid, statement, prev_q=None):
    return {"experience_id": eid, "corrected_answer": statement, "previous_question": prev_q}


FIVE = ["Which city is the capital of Zorvania?", "Name the capital city of Zorvania.",
        "Zorvania's seat of government is which city?", "Where does the Zorvanian government sit?",
        "Tell me the capital of the country Zorvania."]
FACT = "The capital of Zorvania is Kelmirth."


def build(table, experiences, leak=0.8, eval_qs=("What is the capital of Australia?",)):
    return db.build_examples(experiences, FakeGen(table), 6, list(eval_qs), leak)


def test_clean_statement_removes_the_wrong_part():
    assert db.clean_statement("The capital of Australia is Canberra, not Sydney.") == \
        "The capital of Australia is Canberra."
    assert db.clean_statement("Water boils at 100 C") == "Water boils at 100 C."


def test_exact_and_near_eval_questions_are_dropped():
    table = {FACT: ["What is the capital of Australia?", "what is the capital of australia",
                    "What is the capital of Australia??", "Which city is the capital of Zorvania?"]}
    examples, report = build(table, [exp("e1", FACT)])
    assert [e.instruction for e in examples] == ["Which city is the capital of Zorvania?"]
    assert report["dropped_eval_leak"] == 3


def test_duplicate_questions_are_dropped_once():
    table = {FACT: ["Which city is the capital of Zorvania?", "which city is the capital of zorvania"]}
    examples, report = build(table, [exp("e1", FACT)])
    assert len(examples) == 1 and report["dropped_duplicate"] == 1


def test_same_question_with_different_answer_is_flagged():
    q = "Which city is the capital of Zorvania?"
    table = {FACT: [q], "The capital of Zorvania is Valtara.": [q]}
    examples, report = build(table, [exp("e1", FACT), exp("e2", "The capital of Zorvania is Valtara.")])
    assert len(examples) == 1 and report["dropped_contradicting_answer"] == 1


def test_users_real_question_becomes_an_example():
    examples, _ = build({}, [exp("e1", "The capital of Zorvania is Kelmirth, not Valtara.",
                                 prev_q="What is Zorvania's capital?")])
    assert examples[0].origin == "user_question"
    assert examples[0].response == "The capital of Zorvania is Kelmirth."


def test_experience_with_no_usable_question_is_reported():
    _, report = build({}, [exp("e1", FACT)])
    assert report["experiences_without_examples"] == ["e1"]


def test_split_five_wordings_gives_3_1_1_without_overlap():
    examples, _ = build({FACT: FIVE}, [exp("e1", FACT)])
    s = db.split_examples(examples, seed=42)
    assert {k: len(v) for k, v in s.items()} == {"train": 3, "valid": 1, "test": 1}
    all_q = [e.instruction for v in s.values() for e in v]
    assert len(all_q) == len(set(all_q)) == 5


def test_split_is_deterministic_and_seed_sensitive():
    examples, _ = build({FACT: FIVE}, [exp("e1", FACT)])
    a = db.split_examples(examples, 42)
    b = db.split_examples(examples, 42)
    assert [e.instruction for e in a["test"]] == [e.instruction for e in b["test"]]
    others = {tuple(e.instruction for e in db.split_examples(examples, s)["test"]) for s in range(10)}
    assert len(others) > 1


def test_few_wordings_all_go_to_train():
    examples, _ = build({FACT: FIVE[:2]}, [exp("e1", FACT)])
    s = db.split_examples(examples, 42)
    assert (len(s["train"]), len(s["valid"]), len(s["test"])) == (2, 0, 0)


def test_dataset_files_manifest_and_no_overwrite(tmp_path):
    examples, _ = build({FACT: FIVE}, [exp("e1", FACT)])
    splits = db.split_examples(examples, 42)
    out = db.write_dataset("cycle_001", splits, "Be brief.", {"seed": 42}, root=tmp_path)
    row = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["counts"] == {"train": 3, "valid": 1, "test": 1}
    assert set(manifest["sha256"]) == {"train.jsonl", "valid.jsonl", "test.jsonl"}
    with pytest.raises(FileExistsError):
        db.write_dataset("cycle_001", splits, "Be brief.", {}, root=tmp_path)


def test_same_inputs_give_identical_files(tmp_path):
    examples, _ = build({FACT: FIVE}, [exp("e1", FACT)])
    a = db.write_dataset("a", db.split_examples(examples, 42), "s", {}, root=tmp_path)
    b = db.write_dataset("b", db.split_examples(examples, 42), "s", {}, root=tmp_path)
    ma, mb = (json.loads((p / "manifest.json").read_text())["sha256"] for p in (a, b))
    assert ma == mb


def test_parse_questions_strips_bullets_and_non_questions():
    text = "1. What is X?\n- Who made Y?\nHere are your questions:\nWhat is X?\nWhere is Z?"
    assert parse_questions(text, 5) == ["What is X?", "Who made Y?", "Where is Z?"]
    assert parse_questions(text, 2) == ["What is X?", "Who made Y?"]


def test_real_eval_questions_are_loaded():
    qs = db.eval_questions()
    assert "What is the capital of Zorvania?" in qs and len(qs) >= 200
