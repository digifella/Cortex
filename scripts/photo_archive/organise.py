# scripts/photo_archive/organise.py
"""Stage 8 - rename to the house convention and file under YYYY/YYYY-MM."""
import collections
import csv
import os
import re

from .exif import sanitise_model
from .execute import safe_move
from .paths import win_long

UNDATED_DIR = "_UNDATED"

# Film scans carry batch-assigned timestamps - 151 different photographs can
# share "1985-01-01 00-00-00". A bare -2..-151 counter would discard the only
# thing telling them apart, which lives in the current filename (_WA_,
# _Holiday_, Sophie_Wedding, pixel dimensions). On collision we keep that.
_LEADING_DT = re.compile(r"^\d{4}[-:]\d{2}[-:]\d{2}[ _T-]\d{2}[-:]\d{2}[-:]\d{2}")
_UNSAFE_STEM = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _distinguisher(filename: str, model) -> str:
    """The part of the original name that the new name does not already say."""
    stem = os.path.splitext(filename)[0]
    stem = _LEADING_DT.sub("", stem)
    if model:
        stem = stem.replace(str(model), "")
    stem = _UNSAFE_STEM.sub("-", stem)
    stem = re.sub(r"[-_\s]{2,}", "-", stem).strip("-_ .")
    return stem[:60].strip("-_ .")
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


def _unique(dst: str, taken: set) -> tuple[str, bool]:
    """Reserve dst, suffixing -2, -3 ... if already taken. Mirrors safe_move."""
    if dst not in taken:
        taken.add(dst)
        return dst, False
    stem, ext = os.path.splitext(dst)
    n = 2
    while f"{stem}-{n}{ext}" in taken:
        n += 1
    final = f"{stem}-{n}{ext}"
    taken.add(final)
    return final, True


def plan_organise(conn, drive_root: str, out_csv: str) -> dict:
    """Two passes: real files claim their names first, then sidecars follow
    their parent's FINAL name so a collision suffix propagates to the sidecar."""
    stats = {"planned": 0, "undated": 0, "collisions": 0}
    rows = conn.execute(
        "SELECT * FROM files WHERE state NOT IN "
        "('quarantined', 'evacuated', 'organised') "
        "AND kind != 'other' ORDER BY path").fetchall()
    taken: set = set()
    assigned: dict = {}

    # Pass 0 - which target names are contested? If a name is claimed by more
    # than one file, EVERY claimant gets its distinguisher. Letting the first
    # arrival keep the clean name would arbitrarily discard that one file's
    # information while preserving everyone else's.
    contested: collections.Counter = collections.Counter()
    for row in rows:
        if row["kind"] == "sidecar" or not row["exif_dt"]:
            continue
        contested[os.path.join(
            target_dir(drive_root, row["exif_dt"]),
            target_name(row["exif_dt"], row["camera_model"], row["ext"]))] += 1

    def undated_dst(row):
        return os.path.join(drive_root, UNDATED_DIR, row["top_folder"],
                            row["rel_dir"], row["filename"])

    for row in rows:
        if row["kind"] == "sidecar":
            continue
        if row["exif_dt"]:
            name = target_name(row["exif_dt"], row["camera_model"], row["ext"])
            tdir = target_dir(drive_root, row["exif_dt"])
            candidate = os.path.join(tdir, name)
            collided = contested[candidate] > 1
            if collided:
                extra = _distinguisher(row["filename"], row["camera_model"])
                if extra:
                    stem, ext = os.path.splitext(name)
                    candidate = os.path.join(tdir, f"{stem}-{extra}{ext}")
            dst, _ = _unique(candidate, taken)
            reason = "dated"
            stats["planned"] += 1
        else:
            dst, collided = _unique(undated_dst(row), taken)
            reason = "undated"
            stats["undated"] += 1
        stats["collisions"] += int(collided)
        assigned[row["id"]] = (dst, reason)

    for row in rows:
        if row["kind"] != "sidecar":
            continue
        parent = assigned.get(row["sidecar_of"])
        if parent is None or parent[1] == "undated":
            dst, collided = _unique(undated_dst(row), taken)
            reason = "undated"
            stats["undated"] += 1
        else:
            dst, collided = _unique(
                os.path.splitext(parent[0])[0] + row["ext"].lower(), taken)
            reason = "sidecar_follows_parent"
            stats["planned"] += 1
        stats["collisions"] += int(collided)
        assigned[row["id"]] = (dst, reason)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        for row in rows:
            dst, reason = assigned[row["id"]]
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
