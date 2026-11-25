"""Command-line review tool for the experience database.

    python -m scripts.experiences ingest            # import Wake Phase staging file
    python -m scripts.experiences list [STATUS]     # PENDING | APPROVED | REJECTED
    python -m scripts.experiences approve ID [--personal-ok]
    python -m scripts.experiences reject ID [--note "why"]
"""
import argparse

from src.memory.database import get_connection
from src.validation.approval_manager import ApprovalError, ApprovalManager
from src.wake.experience_collector import STAGING_FILE


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    ls = sub.add_parser("list"); ls.add_argument("status", nargs="?", default="PENDING")
    ap = sub.add_parser("approve"); ap.add_argument("id"); ap.add_argument("--personal-ok", action="store_true")
    ap.add_argument("--note", default="")
    rj = sub.add_parser("reject"); rj.add_argument("id"); rj.add_argument("--note", default="")
    args = p.parse_args()

    conn = get_connection()
    mgr = ApprovalManager(conn)
    try:
        if args.cmd == "ingest":
            for r in mgr.ingest_staging_file(STAGING_FILE):
                print(f"{r['experience_id']}  {r['status']:<8} {'; '.join(r['notes'])}")
        elif args.cmd == "list":
            for r in mgr.repo.list_by_status(args.status.upper()):
                print(f"{r['experience_id']}  [{r['category']}] {r['interaction_type']}: "
                      f"{r['corrected_answer'] or r['original_input']}\n    notes: {r['validation_notes']}")
        elif args.cmd == "approve":
            mgr.approve(args.id, note=args.note, explicit_personal_approval=args.personal_ok)
            print("approved")
        elif args.cmd == "reject":
            mgr.reject(args.id, note=args.note)
            print("rejected")
        print("counts:", mgr.repo.counts())
    except ApprovalError as e:
        print(f"Not done: {e}")


if __name__ == "__main__":
    main()
