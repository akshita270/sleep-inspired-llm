"""Run one sleep cycle: pause chat, prepare data, (later) train, wake up.

    AWAKE -> SLEEPING (pick experiences, build dataset) -> TRAINING (Phase 7) -> AWAKE

Chat is blocked the whole time because the shared state is not AWAKE. The
state always returns to AWAKE at the end, even after an error.
"""
import json
import sqlite3
from pathlib import Path
from typing import Callable

from src.registry import model_registry as reg
from src.sleep import dataset_builder as db
from src.sleep import replay_pool
from src.sleep.cycle_manager import CycleManager
from src.sleep.experience_processor import select_training_experiences
from src.sleep.sleep_state import SLEEPING, TRAINING, SystemState
from src.utils.config_loader import load_config

# A training function receives (dataset_dir, cycle_id) and returns a dict with
# at least {"adapter_version": str}. Phase 7 provides the real one.
TrainFn = Callable[[Path, str], dict]


class SleepController:
    def __init__(self, conn: sqlite3.Connection, generator: db.QuestionGenerator,
                 train_fn: TrainFn | None = None, dataset_root: Path = db.TRAINING_DIR,
                 dataset_overrides: dict | None = None):
        self.conn = conn
        self.generator = generator
        self.train_fn = train_fn
        self.dataset_root = dataset_root
        self.state = SystemState(conn)
        self.cycles = CycleManager(conn)
        # config file values, optionally overridden for one experiment (recorded in the manifest)
        self.cfg = {**load_config("training_config")["dataset"], **(dataset_overrides or {})}
        self.conflict_threshold = load_config("system_config")["validation"]["conflict_subject_similarity"]

    def run(self) -> dict:
        with self.state.busy(SLEEPING):
            selected, excluded = select_training_experiences(
                self.conn, self.cycles.learned_experience_ids(),
                self.cfg["include_personal"], self.conflict_threshold)
            if not selected:
                return {"status": "NOTHING_TO_LEARN", "excluded": excluded}

            cycle_id = self.cycles.start()
            try:
                self.cycles.record_experiences(cycle_id, [e["experience_id"] for e in selected])
                examples, report = db.build_examples(
                    selected, self.generator, self.cfg["questions_per_fact"],
                    db.eval_questions(), self.cfg["eval_leak_similarity"])
                splits = db.split_examples(examples, self.cfg["seed"])
                n_fact_train = len(splits["train"])        # new-fact examples, before any replay is added
                replay_info = {"ratio": self.cfg.get("replay_ratio", 0.0), "added": 0, "dropped_leaky": 0}
                if replay_info["ratio"] > 0:
                    replay, leaky = replay_pool.sample_replay(
                        replay_pool.load_pool(), round(replay_info["ratio"] * n_fact_train),
                        self.cfg["seed"], db.eval_questions(), self.cfg["eval_leak_similarity"])
                    splits["train"] += replay          # replay goes to TRAIN only
                    replay_info.update(added=len(replay), dropped_leaky=leaky)
                prev_ratio, prev_dirs = self.cfg.get("previous_replay_ratio", 0.0), self.cfg.get("previous_datasets", [])
                replay_info["previous_added"] = 0
                if prev_ratio > 0 and prev_dirs:       # re-teach facts from earlier cycles
                    prev = replay_pool.sample_previous(prev_dirs, round(prev_ratio * n_fact_train), self.cfg["seed"])
                    splits["train"] += prev
                    replay_info["previous_added"] = len(prev)
                dataset_dir = db.write_dataset(
                    cycle_id, splits, self.cfg["system_prompt"],
                    {"seed": self.cfg["seed"], "dataset_config": self.cfg, "replay": replay_info,
                     "question_generator": self.generator.name,
                     "experience_ids": [e["experience_id"] for e in selected],
                     "excluded_experiences": excluded, "build_report": report},
                    root=self.dataset_root)
                n_examples = sum(len(v) for v in splits.values())

                if self.train_fn is None:
                    self.cycles.finish(cycle_id, "PREPARED", experiences=len(selected),
                                       examples=n_examples, dataset_path=str(dataset_dir),
                                       notes="dataset built, no training run (dry run)")
                    return {"status": "PREPARED", "cycle_id": cycle_id, "dataset_dir": dataset_dir,
                            "counts": {k: len(v) for k, v in splits.items()}, "report": report,
                            "excluded": excluded}

                self.state.set(TRAINING)
                result = self.train_fn(dataset_dir, cycle_id)
                if "adapter_path" in result:   # record the new adapter as an unapproved candidate
                    m = result.get("metrics", {})
                    keep = ("final_train_loss", "final_val_loss", "test_loss", "training_seconds",
                            "adapter_size_bytes", "cpu_side_rss_gb_UNDERCOUNTS_gpu", "base_unchanged")
                    reg.register_candidate(
                        self.conn, result["adapter_version"],
                        load_config("model_config")["model"]["name"], result["adapter_path"],
                        {k: m.get(k) for k in keep})
                # TRAINED, not SUCCESS: the adapter must pass evaluation (Phase 8) first.
                self.cycles.finish(cycle_id, "TRAINED", experiences=len(selected),
                                   examples=n_examples, dataset_path=str(dataset_dir),
                                   adapter_version=result.get("adapter_version"))
                return {"status": "TRAINED", "cycle_id": cycle_id, "dataset_dir": dataset_dir, **result}
            except Exception as e:
                self.cycles.finish(cycle_id, "FAILED", notes=repr(e))
                raise
