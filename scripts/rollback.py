"""Withdraw the current champion and fall back to the previous one.

    python -m scripts.rollback --db PATH --reason "why"
"""
import argparse
from pathlib import Path

from src.memory.database import get_connection
from src.registry import model_registry as reg
from src.registry.rollback_manager import RollbackError, rollback


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--reason", required=True)
    args = p.parse_args()
    conn = get_connection(args.db)
    try:
        before = reg.champion_version(conn)
        after = rollback(conn, args.reason)
        print(f"rolled back {before}; champion is now {after or 'the original base model'}")
    except RollbackError as e:
        print(f"Not done: {e}")


if __name__ == "__main__":
    main()
