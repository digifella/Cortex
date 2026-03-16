#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cortex_engine.maintenance_runner import CortexMaintenanceRunner  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scheduled local Cortex maintenance tasks.")
    parser.add_argument("--db-path", default="", help="Knowledge-base root path or knowledge_hub_db path")
    parser.add_argument(
        "--profile",
        default="daily",
        choices=["daily", "weekly", "full"],
        help="Maintenance profile to run",
    )
    parser.add_argument("--window-start", default="11:00", help="Local maintenance window start (HH:MM)")
    parser.add_argument("--window-end", default="14:00", help="Local maintenance window end (HH:MM)")
    parser.add_argument(
        "--window-only",
        action="store_true",
        help="Skip execution outside the configured local time window",
    )
    parser.add_argument(
        "--run-if-due",
        action="store_true",
        help="Run only if the daily/weekly marker has not already been completed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if the profile was already completed for the current day/week",
    )
    parser.add_argument(
        "--real-dedup",
        action="store_true",
        help="Allow actual Chroma duplicate removal instead of dry-run analysis only",
    )
    parser.add_argument(
        "--keep-backups",
        type=int,
        default=10,
        help="Number of most recent backups to retain during weekly cleanup",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON report",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runner = CortexMaintenanceRunner(db_path=args.db_path or None)
    report = runner.run(
        profile=args.profile,
        window_start=args.window_start,
        window_end=args.window_end,
        enforce_window=args.window_only,
        run_if_due=args.run_if_due,
        force=args.force,
        real_dedup=args.real_dedup,
        keep_backups=args.keep_backups,
    )
    print(json.dumps(report, indent=2 if args.pretty else None))
    return 1 if report.get("status") == "completed_with_errors" else 0


if __name__ == "__main__":
    raise SystemExit(main())
