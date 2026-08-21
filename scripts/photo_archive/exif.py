# scripts/photo_archive/exif.py
"""Stage 2 - EXIF capture date and camera model via exiftool -stay_open.

Per-file exiftool invocations run about 7s each on these drives; batch mode
is the only workable approach at this scale.
"""
import json
import os
import re
import subprocess

from .paths import strip_long

DATE_TAGS = ("SubSecDateTimeOriginal", "DateTimeOriginal", "CreateDate")
_DATE_RE = re.compile(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def pick_datetime(tags: dict) -> tuple[str | None, str | None]:
    for tag in DATE_TAGS:
        raw = tags.get(tag)
        if not raw:
            continue
        m = _DATE_RE.match(str(raw))
        if not m:
            continue
        y, mo, d, h, mi, s = m.groups()
        if y == "0000" or mo == "00" or d == "00":
            continue
        return f"{y}-{mo}-{d}T{h}:{mi}:{s}", tag
    return None, None


def sanitise_model(model) -> str:
    if not model:
        return ""
    cleaned = _UNSAFE.sub("-", str(model)).strip()
    return re.sub(r"-{2,}", "-", cleaned).strip("-. ")


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
        args = ["-json", "-fast2", "-charset", "filename=utf8",
                "-DateTimeOriginal", "-SubSecDateTimeOriginal",
                "-CreateDate", "-Model", "-Make"]
        args += paths
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
                    "camera_model = ?, state = 'exif_read' WHERE id = ?",
                    (dt, src, model, row["id"]))
                stats["read"] += 1
                stats["dated" if dt else "undated"] += 1
            conn.commit()
    finally:
        if owned:
            reader.close()
    return stats
