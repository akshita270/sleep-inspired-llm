"""Trigger one sleep cycle by hand.

    python -m scripts.sleep --db PATH --dataset-root DIR                  # dry run: dataset only
    python -m scripts.sleep --db PATH --dataset-root DIR --adapter-root DIR --train

With --train the cycle also fine-tunes a LoRA adapter. The adapter is saved as an
unapproved candidate; evaluation (Phase 8) decides whether it becomes the champion.
"""
import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx

from src.memory.database import get_connection
from src.sleep.question_generator import LLMQuestionGenerator
from src.sleep.sleep_controller import SleepController
from src.training.lora_trainer import LoRATrainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=None, help="default: data/database/sleep_learning.db")
    p.add_argument("--dataset-root", type=Path, default=None, help="default: data/training")
    p.add_argument("--train", action="store_true", help="also run LoRA training")
    p.add_argument("--adapter-root", type=Path, default=None, help="default: models/adapters")
    p.add_argument("--resume-from", type=Path, default=None, help="earlier adapter folder to continue training")
    p.add_argument("--replay-ratio", type=float, default=None, help="replay examples per fact example")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    args = p.parse_args()

    lora_overrides = {k: v for k, v in {"epochs": args.epochs, "learning_rate": args.lr,
                                        "rank": args.rank, "num_layers": args.num_layers}.items() if v is not None}
    dataset_overrides = {} if args.replay_ratio is None else {"replay_ratio": args.replay_ratio}

    conn = get_connection(args.db)
    generator = LLMQuestionGenerator()          # loads Qwen only if a question is not cached

    train_fn = None
    if args.train:
        trainer = LoRATrainer(**({"adapter_root": args.adapter_root} if args.adapter_root else {}),
                              resume_from=args.resume_from, overrides=lora_overrides)

        def train_fn(dataset_dir, cycle_id):
            generator.release()                 # free the model's memory before training
            gc.collect()
            mx.clear_cache()
            return trainer(dataset_dir, cycle_id)

    kwargs = {"dataset_root": args.dataset_root} if args.dataset_root else {}
    kwargs["dataset_overrides"] = dataset_overrides
    result = SleepController(conn, generator, train_fn=train_fn, **kwargs).run()
    result.pop("config", None)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
