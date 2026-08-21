# scripts/photo_archive/organise.py
"""Stage 8 - rename to the house convention and file under YYYY/YYYY-MM."""
import csv
import os

from .exif import sanitise_model
from .execute import safe_move
from .paths import win_long

UNDATED_DIR = "_UNDATED"
PLAN_FIELDS = ["id", "src", "dst", "kind", "reason"]


def target_name(exif_dt: str, model, ext: str) -> str:
    date, time_part = exif_dt.split("T")
    stamp = f"{date} {time_part.replace(':', '-')}"
    clean = sanitise_model(model)
    suffix = f"-{clean}" if clean else ""
    return f"{stamp}{suffix}{ext.lower()}"


def target_dir(drive_root: str, exif_dt: str) -> str:
    year, month = exif_dt[:4], exif_dt[5:7]
    return os.path.join(drive_root, year, f"{year}-{month}")


def plan_organise(conn, drive_root: str, out_csv: str) -> dict:
    stats = {"planned": 0, "undated": 0}
    rows = conn.execute(
        "SELECT * FROM files WHERE state NOT IN "
        "('quarantined', 'evacuated', 'organised')").fetchall()
    by_id = {r["id"]: r for r in rows}

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        for row in rows:
            if row["kind"] == "sidecar":
                parent = by_id.get(row["sidecar_of"])
                if parent is None or not parent["exif_dt"]:
                    dst = os.path.join(drive_root, UNDATED_DIR,
                                       row["top_folder"], row["rel_dir"],
                                       row["filename"])
                    reason = "undated"
                    stats["undated"] += 1
                else:
                    name = target_name(parent["exif_dt"],
                                       parent["camera_model"], row["ext"])
                    dst = os.path.join(target_dir(drive_root, parent["exif_dt"]),
                                       name)
                    reason = "sidecar_follows_parent"
                    stats["planned"] += 1
            elif row["exif_dt"]:
                name = target_name(row["exif_dt"], row["camera_model"], row["ext"])
                dst = os.path.join(target_dir(drive_root, row["exif_dt"]), name)
                reason = "dated"
                stats["planned"] += 1
            else:
                dst = os.path.join(drive_root, UNDATED_DIR, row["top_folder"],
                                   row["rel_dir"], row["filename"])
                reason = "undated"
                stats["undated"] += 1
            writer.writerow({"id": row["id"], "src": row["path"], "dst": dst,
                             "kind": row["kind"], "reason": reason})
    return stats


def apply_organise(conn, plan_csv: str, journal, apply: bool = False) -> dict:
    stats = {"moved": 0, "would_move": 0, "missing": 0, "errors": 0}
    with open(plan_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        src, dst = row["src"], row["dst"]
        if not os.path.exists(win_long(src)):
            stats["missing"] += 1
            continue
        if not apply:
            stats["would_move"] += 1
            continue
        try:
            final = safe_move(src, dst)
        except OSError as exc:
            stats["errors"] += 1
            journal.append("organise", "move", src, dst, 0, "",
                           f"error:{exc.errno}")
            continue
        journal.append("organise", "move", src, final, 0, "", "ok")
        conn.execute("UPDATE files SET path = ?, state = 'organised' "
                     "WHERE id = ?", (final, row["id"]))
        stats["moved"] += 1
    conn.commit()
    return stats
