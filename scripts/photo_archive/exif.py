# scripts/photo_archive/exif.py
"""Stage 2 - EXIF capture date and camera model via exiftool -stay_open.

Per-file exiftool invocations run about 7s each on these drives; batch mode
is the only workable approach at this scale.
"""
import json
import os
import re
import subprocess

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
