import json
from pathlib import Path

import mlx.core as mx
import pytest

from src.memory.database import get_connection
from src.model import adapter_manager as am
from src.sleep.sleep_controller import SleepController
from src.training.lora_trainer import LoRATrainer
from src.training.training_monitor import parse_log
from src.validation.approval_manager import ApprovalManager

# Real mlx-lm output, colour codes included (copied from an actual training log).
SAMPLE_LOG = (
    "Trainable parameters: 0.365% (14.680M/4022.468M)\n"
    "  \x1b[1miter   train_loss     tok/s     tokens\x1b[0m\n"
    "  \x1b[38;5;244m   1\x1b[0m    \x1b[1;35mval\x1b[0m \x1b[1m5.423\x1b[0m    \x1b[38;5;244m8.07s\x1b[0m\n"
    "  \x1b[38;5;244m   5\x1b[0m    \x1b[1;32m4.044 ▼\x1b[0m    \x1b[1m   35\x1b[0m    \x1b[38;5;244m  0.5k\x1b[0m\n"
    "  \x1b[38;5;244m  10\x1b[0m    \x1b[1;33m4.453 ▲\x1b[0m    \x1b[1m   49\x1b[0m    \x1b[38;5;244m  0.9k\x1b[0m\n"
    "  \x1b[38;5;244m  29\x1b[0m    \x1b[1;35mval\x1b[0m \x1b[1m0.900\x1b[0m    \x1b[38;5;244m7.45s\x1b[0m\n"
    "Test loss 0.850, Test ppl 2.340.\n"
)


def test_parse_log_extracts_losses_speed_and_parameters():
    m = parse_log(SAMPLE_LOG)
    assert m["train_loss"] == [(5, 4.044), (10, 4.453)]
    assert m["first_val_loss"] == 5.423 and m["final_val_loss"] == 0.9 and m["best_val_loss"] == 0.9
    assert m["max_train_loss"] == 4.453 and m["median_tokens_per_s"] == 49
    assert m["test_loss"] == 0.85
    assert m["trainable_percent"] == 0.365 and m["trainable_millions"] == 14.68


def test_parse_log_on_empty_text_does_not_crash():
    assert parse_log("")["final_train_loss"] is None


def test_config_iterations_match_epochs(tmp_path):
    t = LoRATrainer(adapter_root=tmp_path)
    cfg = t.build_config(tmp_path / "data", tmp_path / "out", n_train=113)
    assert cfg["iters"] == 29 * t.lora["epochs"]          # ceil(113/4) steps per epoch
    assert cfg["steps_per_eval"] == 29
    assert cfg["lora_parameters"]["rank"] == t.lora["rank"]
    assert cfg["model"] == "Qwen/Qwen3-4B" and "resume_adapter_file" not in cfg


def test_resume_points_at_previous_adapter(tmp_path):
    t = LoRATrainer(adapter_root=tmp_path, resume_from=tmp_path / "cycle_001")
    cfg = t.build_config(tmp_path / "d", tmp_path / "o", 10)
    assert cfg["resume_adapter_file"].endswith("cycle_001/adapters.safetensors")


def test_base_model_hash_changes_when_a_weight_file_changes(tmp_path):
    from src.training.lora_trainer import hash_base_model
    snap = tmp_path / "models--Org--Model" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (tmp_path / "models--Org--Model" / "refs").mkdir()
    (tmp_path / "models--Org--Model" / "refs" / "main").write_text("abc123\n")
    (snap / "model-1.safetensors").write_bytes(b"weights")
    h1 = hash_base_model("Org/Model", tmp_path)
    assert h1 == hash_base_model("Org/Model", tmp_path)          # stable
    (snap / "model-1.safetensors").write_bytes(b"weightz")
    assert hash_base_model("Org/Model", tmp_path) != h1          # change detected


def test_real_base_model_files_are_found():
    from src.training.lora_trainer import base_model_folder
    assert list(base_model_folder("Qwen/Qwen3-4B").glob("*.safetensors"))


def test_trainer_refuses_to_overwrite_an_existing_adapter(tmp_path):
    (tmp_path / "cycle_001").mkdir()
    with pytest.raises(FileExistsError):
        LoRATrainer(adapter_root=tmp_path)(tmp_path, "cycle_001")


def test_adapter_file_check_detects_non_lora_tensors(tmp_path):
    a = mx.zeros((2, 2))
    mx.save_safetensors(str(tmp_path / "adapters.safetensors"),
                        {"layers.0.q_proj.lora_a": a, "layers.0.q_proj.lora_b": a})
    assert am.non_lora_keys(tmp_path) == []
    mx.save_safetensors(str(tmp_path / "adapters.safetensors"),
                        {"layers.0.q_proj.lora_a": a, "layers.0.q_proj.weight": a})
    assert am.non_lora_keys(tmp_path) == ["layers.0.q_proj.weight"]
    assert am.adapter_size_bytes(tmp_path) > 0



def test_controller_records_candidate_after_training(tmp_path):
    conn = get_connection(tmp_path / "t.db")
    ApprovalManager(conn).submit({
        "experience_id": "e1", "session_id": "s", "interaction_type": "NEW_FACT",
        "original_input": "The capital of Zorvania is Kelmirth.",
        "corrected_answer": "The capital of Zorvania is Kelmirth.", "category": "general",
        "source": "experiment_fixture", "previous_answer": None, "previous_question": None,
        "learn_requested": False, "requires_explicit_approval": False})

    class Gen:
        name = "fake"
        def generate(self, s, n):
            return [f"Tell me about Zorvania capital number {i}?" for i in range(5)][:n]

    def fake_train(dataset_dir, cycle_id):
        return {"adapter_version": f"adapter_{cycle_id}", "adapter_path": tmp_path / cycle_id,
                "metrics": {"final_val_loss": 0.9, "training_seconds": 12.0}}

    r = SleepController(conn, Gen(), train_fn=fake_train, dataset_root=tmp_path / "ds").run()
    assert r["status"] == "TRAINED"
    row = conn.execute("SELECT * FROM model_versions").fetchone()
    assert row["version_id"] == "adapter_cycle_001" and row["evaluation_status"] == "PENDING"
    assert json.loads(row["metrics"])["training"]["final_val_loss"] == 0.9
