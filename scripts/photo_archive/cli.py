"""Subcommand dispatch for the P: drive pipeline.

Every writing command requires an explicit --apply.
"""
import argparse
import sys

from . import config, db, dupes, execute, exif, hashing, journal, organise, walk

STAGES = ("walk", "exif", "date-conflicts", "hash", "plan-dupes",
          "quarantine", "plan-organise", "organise", "undo")


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="photo_archive")
    sub = parser.add_subparsers(dest="command", required=True)
    for stage in STAGES:
        p = sub.add_parser(stage)
        p.add_argument("--db", required=True)
        p.add_argument("--drive", default=config.DRIVE)
        p.add_argument("--plan", default=None)
        p.add_argument("--journal", default="photo_archive_journal.csv")
        p.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.command not in STAGES:
        parser.error(f"unknown stage: {args.command}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    conn = db.connect(args.db)
    db.init_schema(conn)
    jrn = journal.Journal(args.journal)

    if args.command == "walk":
        print(walk.walk_scope(conn, args.drive, config.SCOPE_ROOTS))
    elif args.command == "exif":
        print(exif.read_exif_into_index(conn))
    elif args.command == "date-conflicts":
        print(exif.report_date_conflicts(conn, args.plan))
    elif args.command == "hash":
        print(hashing.hash_candidates(conn))
    elif args.command == "plan-dupes":
        print(dupes.plan_tier1(conn, args.plan))
    elif args.command == "quarantine":
        print(execute.quarantine(conn, args.plan, args.drive, jrn, args.apply))
    elif args.command == "plan-organise":
        print(organise.plan_organise(conn, args.drive, args.plan))
    elif args.command == "organise":
        print(organise.apply_organise(conn, args.plan, jrn, args.apply))
    elif args.command == "undo":
        print(journal.undo(args.journal, args.apply))
    else:
        print(f"stage not yet wired: {args.command}", file=sys.stderr)
        return 2
    conn.commit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
