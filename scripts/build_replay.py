"""One-off: build the replay pool by asking the ORIGINAL model the replay questions.

    python -m scripts.build_replay
"""
from src.model.model_loader import load_model
from src.sleep import replay_pool
from src.sleep.dataset_builder import eval_questions
from src.evaluation.benchmark_manager import DEV_BENCHMARKS, load_dev


def main() -> None:
    questions = replay_pool.load_questions()
    others = eval_questions() + [i["question"] for n in DEV_BENCHMARKS for i in load_dev(n)]
    overlaps = replay_pool.find_overlaps(questions, others, threshold=0.8)
    if overlaps:
        raise SystemExit(f"replay questions overlap evaluation/dev questions: {overlaps}")
    print(f"{len(questions)} replay questions, none overlap the {len(others)} evaluation/dev questions")

    pool = replay_pool.build_pool(load_model())
    print(f"saved {len(pool)} replay examples to {replay_pool.POOL_FILE}")
    for p in pool[:3]:
        print(" ", p["question"], "->", p["answer"][:80])


if __name__ == "__main__":
    main()
