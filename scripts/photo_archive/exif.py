# scripts/photo_archive/exif.py
"""Stage 2 - EXIF capture date and camera model via exiftool -stay_open.

Per-file exiftool invocations run about 7s each on these drives; batch mode
is the only workable approach at this scale.
"""
import csv
import json
import os
import re
import subprocess

from .paths import strip_long

DATE_TAGS = ("SubSecDateTimeOriginal", "DateTimeOriginal", "CreateDate")
_DATE_RE = re.compile(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def parse_dt(raw) -> str | None:
    """Parse one exiftool date to ISO, rejecting nulls like 0000:00:00."""
    if not raw:
        return None
    m = _DATE_RE.match(str(raw))
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    if y == "0000" or mo == "00" or d == "00":
        return None
    return f"{y}-{mo}-{d}T{h}:{mi}:{s}"


def pick_datetime(tags: dict) -> tuple[str | None, str | None]:
    for tag in DATE_TAGS:
        parsed = parse_dt(tags.get(tag))
        if parsed:
            return parsed, tag
    return None, None


def sanitise_model(model) -> str:
    if not model:
        return ""
    cleaned = _UNSAFE.sub("-", str(model)).strip()
    return re.sub(r"-{2,}", "-", cleaned).strip("-. ")


def _exiftool_args(paths: list[str]) -> list[str]:
    """Build the exiftool argument list, stripping the \\?\\ long-path prefix.

    exiftool 12.85 REJECTS \\?\\ paths outright - it parses the '?' as a
    wildcard and answers "Wildcards don't work in the directory specification",
    even for a short path. walk.py stores every Windows path with that prefix,
    so passing stored paths through unmodified makes every read fail SILENTLY:
    read_many returns {}, every photo is marked undated, and the entire archive
    routes to _UNDATED without a single error. Verified against real exiftool
    under Windows Python, 2026-08-21.

    Safe because no path on P: exceeds 260 characters (longest measured: 194),
    so the plain form always reaches the file.
    """
    return ["-json", "-fast2", "-charset", "filename=utf8",
            "-DateTimeOriginal", "-SubSecDateTimeOriginal",
            "-CreateDate", "-Model", "-Make"] + [strip_long(p) for p in paths]


class ExifReader:
    """Persistent exiftool process. Always close() it."""

    def __init__(self, exiftool: str = "exiftool"):
        self.proc = subprocess.Popen(
            [exiftool, "-stay_open", "True", "-@", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
        )

    def read_many(self, paths: list[str]) -> dict[str, dict]:
        if not paths:
            return {}
        args = _exiftool_args(paths)
        self.proc.stdin.write("\n".join(args) + "\n-execute\n")
        self.proc.stdin.flush()

        chunks = []
        for line in self.proc.stdout:
            if line.strip() == "{ready}":
                break
            chunks.append(line)
        try:
            records = json.loads("".join(chunks) or "[]")
        except json.JSONDecodeError:
            return {}
        return {rec.get("SourceFile", ""): rec for rec in records}

    def close(self) -> None:
        try:
            self.proc.stdin.write("-stay_open\nFalse\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()


def _norm(path: str) -> str:
    """exiftool echoes SourceFile with forward slashes and no \\?\\ prefix.

    Without normalising both sides, the lookup misses every record and every
    photo silently comes back undated.
    """
    return strip_long(path).replace("\\", "/").lower()


def read_exif_into_index(conn, reader=None, exiftool: str = "exiftool",
                         batch_size: int = 200) -> dict:
    """Batch capture date and camera model from disk into the index.

    Resumable: only rows still in state 'walked' are read, and every row
    touched is marked 'exif_read' whether or not a date was found.
    """
    stats = {"read": 0, "dated": 0, "undated": 0}
    rows = conn.execute(
        "SELECT id, path FROM files WHERE state = 'walked' "
        "AND kind IN ('image', 'raw', 'video') ORDER BY id").fetchall()
    if not rows:
        return stats

    owned = reader is None
    reader = reader or ExifReader(exiftool)
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            got = reader.read_many([r["path"] for r in batch])
            lookup = {_norm(k): v for k, v in got.items()}
            for row in batch:
                tags = lookup.get(_norm(row["path"]), {})
                dt, src = pick_datetime(tags)
                model = sanitise_model(tags.get("Model")) or None
                conn.execute(
                    "UPDATE files SET exif_dt = ?, exif_dt_source = ?, "
                    "exif_create_dt = ?, camera_model = ?, state = 'exif_read' "
                    "WHERE id = ?",
                    (dt, src, parse_dt(tags.get("CreateDate")), model, row["id"]))
                stats["read"] += 1
                stats["dated" if dt else "undated"] += 1
            conn.commit()
    finally:
        if owned:
            reader.close()
    return stats


CONFLICT_FIELDS = ["reason", "path", "top_folder", "exif_dt", "exif_dt_source",
                   "exif_create_dt", "camera_model"]

# Photography predates 1826, and a capture date in the future is a dead camera
# clock. Real example: 2426-03-07 from an Olympus u1050SW, where BOTH tags
# agreed - so the year-mismatch test could never have caught it.
PLAUSIBLE_MIN_YEAR = 1826


def report_date_conflicts(conn, out_csv: str) -> dict:
    """List files whose DateTimeOriginal and CreateDate disagree on the YEAR.

    Neither tag can be trusted blindly, and the two failure modes are opposite:

      * SMS_Photos: DateTimeOriginal is corrupt (2003 for a true 2023 iPhone
        shot) and CreateDate is right.
      * Scans: DateTimeOriginal is right (1986, when the photo was taken) and
        CreateDate is merely when it was digitised (2000).

    So no precedence rule serves both. The chosen date is left alone -
    DateTimeOriginal wins, which is correct for scans - and the disagreement is
    reported here for a human to adjudicate before organising.
    """
    import datetime
    next_year = str(datetime.date.today().year + 1)
    cols = ("path", "top_folder", "exif_dt", "exif_dt_source",
            "exif_create_dt", "camera_model")
    sel = "SELECT " + ", ".join(cols) + " FROM files "

    mismatched = conn.execute(
        sel + "WHERE exif_dt IS NOT NULL AND exif_create_dt IS NOT NULL "
        "AND substr(exif_dt, 1, 4) != substr(exif_create_dt, 1, 4) "
        "ORDER BY path").fetchall()
    implausible = conn.execute(
        sel + "WHERE exif_dt IS NOT NULL AND (substr(exif_dt, 1, 4) < ? "
        "OR substr(exif_dt, 1, 4) > ?) ORDER BY path",
        (str(PLAUSIBLE_MIN_YEAR), next_year)).fetchall()

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CONFLICT_FIELDS)
        writer.writeheader()
        for reason, rows in (("year_mismatch", mismatched),
                             ("implausible_year", implausible)):
            for row in rows:
                rec = {k: row[k] for k in cols}
                rec["reason"] = reason
                writer.writerow(rec)
    return {"conflicts": len(mismatched),
            "implausible": len(implausible),
            "folders": len({r["top_folder"] for r in mismatched + implausible})}
