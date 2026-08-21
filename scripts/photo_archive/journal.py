# scripts/photo_archive/journal.py
"""Append-only operation journal and reverse-replay undo.

The journal lives on C:, so a P: or L: dropout cannot destroy the record of
what was moved.
"""
import csv
import os
import time

from .paths import win_long
from .hashing import full_hash

FIELDS = ["timestamp", "stage", "operation", "src", "dst", "size",
          "sha256", "status"]


class Journal:
    def __init__(self, path: str):
        self.path = path
        self._ensure_header()

    def _ensure_header(self) -> None:
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        if not exists:
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    def append(self, stage, operation, src, dst, size, sha256, status) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "stage": stage, "operation": operation, "src": src,
                "dst": dst, "size": size, "sha256": sha256, "status": status,
            })


def undo(journal_path: str, apply: bool = False) -> dict:
    stats = {"restored": 0, "would_restore": 0, "refused": 0,
             "missing": 0, "order": []}
    with open(journal_path, newline="", encoding="utf-8") as fh:
        entries = [r for r in csv.DictReader(fh)
                   if r["operation"] == "move" and r["status"] == "ok"]

    for entry in reversed(entries):
        src, dst = entry["src"], entry["dst"]
        if not os.path.exists(win_long(dst)):
            stats["missing"] += 1
            continue
        if entry["sha256"]:
            try:
                if full_hash(dst) != entry["sha256"]:
                    stats["refused"] += 1
                    continue
            except OSError:
                stats["refused"] += 1
                continue
        stats["order"].append(dst)
        if not apply:
            stats["would_restore"] += 1
            continue
        os.makedirs(win_long(os.path.dirname(src)), exist_ok=True)
        os.replace(win_long(dst), win_long(src))
        stats["restored"] += 1
    return stats
