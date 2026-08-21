# scripts/photo_archive/execute.py
"""Stage 5 - quarantine duplicates by same-volume rename.

Nothing is deleted. Provenance is preserved in the target path so the
original location is recoverable without consulting the journal.
"""
import csv
import os

from .paths import win_long

QUARANTINE_DIR = "_DUPES"


def safe_move(src: str, dst: str) -> str:
    """Move src to dst, suffixing -2, -3 ... rather than ever overwriting."""
    os.makedirs(win_long(os.path.dirname(dst)), exist_ok=True)
    stem, ext = os.path.splitext(dst)
    candidate, n = dst, 2
    while os.path.exists(win_long(candidate)):
        candidate = f"{stem}-{n}{ext}"
        n += 1
    os.replace(win_long(src), win_long(candidate))
    return candidate


def quarantine(conn, plan_csv: str, drive_root: str, journal,
               apply: bool = False) -> dict:
    stats = {"moved": 0, "would_move": 0, "missing": 0, "errors": 0}
    with open(plan_csv, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["role"] == "quarantine"]

    for row in rows:
        src = row["path"]
        if not os.path.exists(win_long(src)):
            stats["missing"] += 1
            continue
        dst = os.path.join(drive_root, QUARANTINE_DIR, row["top_folder"],
                           row["rel_dir"], os.path.basename(src))
        if not apply:
            stats["would_move"] += 1
            continue
        try:
            final = safe_move(src, dst)
        except OSError as exc:
            stats["errors"] += 1
            journal.append("quarantine", "move", src, dst, row["size"],
                           row["sha256"], f"error:{exc.errno}")
            continue
        journal.append("quarantine", "move", src, final, row["size"],
                       row["sha256"], "ok")
        conn.execute("UPDATE files SET path = ?, state = 'quarantined' "
                     "WHERE path = ?", (final, src))
        stats["moved"] += 1
    conn.commit()
    return stats
