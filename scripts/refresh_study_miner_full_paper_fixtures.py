#!/usr/bin/env python3
"""
Refresh the Study Miner full-paper regression fixtures from exported CSVs.

Usage:
    python scripts/refresh_study_miner_full_paper_fixtures.py \
        --source-dir /home/longboardfella/cortex_suite

The script looks for the latest `*_table_<n>_study_miner_export.csv` files for
tables 2, 3, 4, and 5, validates that all four are present, and then copies
them into `tests/fixtures/study_miner_full_paper/` as stable fixture names.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT
DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "study_miner_full_paper"
REQUIRED_TABLES = (2, 3, 4, 5)


def find_latest_table_exports(
    source_dir: Path,
    table_numbers: tuple[int, ...] = REQUIRED_TABLES,
) -> dict[int, Path]:
    exports: dict[int, Path] = {}
    for table_number in table_numbers:
        pattern = f"*_table_{table_number}_study_miner_export.csv"
        matches = sorted(source_dir.glob(pattern), key=lambda item: item.name)
        if not matches:
            raise FileNotFoundError(
                f"Missing export for table {table_number} under {source_dir} matching {pattern}"
            )
        exports[table_number] = matches[-1]
    return exports


def copy_exports_to_fixture_dir(
    exports: dict[int, Path],
    fixture_dir: Path,
    *,
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[Path, Path]] = []
    for table_number in sorted(exports):
        source = exports[table_number]
        destination = fixture_dir / f"table_{table_number}.csv"
        copied.append((source, destination))
        if not dry_run:
            shutil.copy2(source, destination)
    return copied


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing Study Miner CSV exports. Defaults to the repo root.",
    )
    parser.add_argument(
        "--fixture-dir",
        default=str(DEFAULT_FIXTURE_DIR),
        help="Destination fixture directory. Defaults to tests/fixtures/study_miner_full_paper.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected files without copying them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    source_dir = Path(args.source_dir).expanduser().resolve()
    fixture_dir = Path(args.fixture_dir).expanduser().resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        parser.error(f"Source directory does not exist or is not a directory: {source_dir}")

    exports = find_latest_table_exports(source_dir)
    copied = copy_exports_to_fixture_dir(exports, fixture_dir, dry_run=bool(args.dry_run))

    action = "Would copy" if args.dry_run else "Copied"
    for source, destination in copied:
        print(f"{action}: {source.name} -> {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
