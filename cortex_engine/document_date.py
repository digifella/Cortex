"""Conservative document-date provenance for knowledge and client ingestion.

An inferred date is a ranking hint, not evidence for excluding search results.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import os
import re
import shutil
import subprocess
from pathlib import Path

MONTHS = {
    name.lower(): i for i, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"), 1)
}
MONTHS.update({name[:3]: value for name, value in list(MONTHS.items())})
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
FULL_DATE = re.compile(
    rf"(?<!\d)((?:19|20)\d{{2}})[-/.]([01]?\d)[-/.]([0-3]?\d)(?!\d)"
    rf"|(?<!\d)([0-3]?\d)[ /.-]+({MONTH_RE})[ /,.-]+((?:19|20)\d{{2}})(?!\d)"
    rf"|\b({MONTH_RE})[ /.-]+([0-3]?\d)(?:st|nd|rd|th)?[, /.-]+((?:19|20)\d{{2}})\b"
    r"|(?<!\d)([0-3]?\d)[/-]([01]?\d)[/-]((?:19|20)\d{2})(?!\d)",
    re.IGNORECASE,
)
YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
MONTH_YEAR = re.compile(rf"\b({MONTH_RE})[ ,/-]+((?:19|20)\d{{2}})\b", re.IGNORECASE)
LABEL = re.compile(r"\b(?:published|publication date|date of publication|issued|document date|dated)\b",
                   re.IGNORECASE)


def _parse_match(match: re.Match) -> str | None:
    g = match.groups()
    try:
        if g[0]:
            y, mo, day = map(int, g[:3])
        elif g[3]:
            day, mo, y = int(g[3]), MONTHS[g[4].lower()], int(g[5])
        elif g[6]:
            mo, day, y = MONTHS[g[6].lower()], int(g[7]), int(g[8])
        else:
            day, mo, y = int(g[9]), int(g[10]), int(g[11])
        dt = datetime(y, mo, day).date()
        if 1900 <= y <= datetime.now().year + 1:
            return dt.isoformat()
    except (ValueError, TypeError):
        pass
    return None


def _date_in(text: str) -> str | None:
    for match in FULL_DATE.finditer(text):
        date = _parse_match(match)
        if date:
            return date
    return None


def _partial_date_in(text: str) -> tuple[str, str] | None:
    month = MONTH_YEAR.search(text)
    if month:
        year = int(month.group(2))
        if 1900 <= year <= datetime.now().year + 1:
            return f"{year:04d}-{MONTHS[month.group(1).lower()]:02d}-01", "month"
    year = YEAR.search(text)
    if year and 1900 <= int(year.group(1)) <= datetime.now().year + 1:
        return year.group(1) + "-01-01", "year"
    return None


def _file_created(path: str) -> str | None:
    """Get actual birth time, including NTFS CreationTime through WSL if needed."""
    try:
        stat = os.stat(path)
        birth = getattr(stat, "st_birthtime", None)
        if birth:
            return datetime.fromtimestamp(birth, timezone.utc).date().isoformat()
    except OSError:
        return None
    if not path.startswith("/mnt/"):
        return None
    try:
        windows = subprocess.run(["wslpath", "-w", path], text=True,
                                 capture_output=True, timeout=5, check=True).stdout.strip()
        encoded_path = base64.b64encode(windows.encode("utf-8")).decode("ascii")
        script = ("$p=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('"
                  + encoded_path + "')); (Get-Item -LiteralPath $p).CreationTimeUtc.ToString('yyyy-MM-dd')")
        command = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        powershell = (shutil.which("powershell.exe") or
                      "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe")
        result = subprocess.run([powershell, "-NoProfile", "-NonInteractive",
                                 "-EncodedCommand", command], text=True,
                                capture_output=True, timeout=15, check=True)
        candidate = result.stdout.strip().splitlines()[-1]
        return candidate if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate) else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def infer_document_date(text: str, path: str) -> dict[str, str]:
    """Return ISO date, provenance and precision without treating guesses as facts."""
    lines = [line.strip() for line in (text or "").splitlines()
             if line.strip() and not line.lstrip().lower().startswith("filename:")]
    # Headers/footers are much stronger evidence than dates deep in the body,
    # which are often citations, references, or events described by the article.
    for region, source in ((lines[:20], "header"), (lines[-20:], "footer")):
        for line in region:
            if LABEL.search(line):
                date = _date_in(line)
                if date:
                    return {"source_date": date, "source_date_source": source + "_publication",
                            "source_date_precision": "day"}
                partial = _partial_date_in(line)
                if partial:
                    return {"source_date": partial[0],
                            "source_date_source": source + "_publication",
                            "source_date_precision": partial[1]}
    for region, source in ((lines[:8], "header"), (lines[-8:], "footer")):
        for line in region:
            date = _date_in(line)
            if date:
                return {"source_date": date, "source_date_source": source,
                        "source_date_precision": "day"}
            if len(line) <= 80:
                month = MONTH_YEAR.search(line)
                if month:
                    year = int(month.group(2))
                    if 1900 <= year <= datetime.now().year + 1:
                        return {"source_date": f"{year:04d}-{MONTHS[month.group(1).lower()]:02d}-01",
                                "source_date_source": source,
                                "source_date_precision": "month"}
    name = Path(path).stem
    date = _date_in(name)
    if date:
        return {"source_date": date, "source_date_source": "filename",
                "source_date_precision": "day"}
    month = MONTH_YEAR.search(name)
    if month:
        year = int(month.group(2))
        if 1900 <= year <= datetime.now().year + 1:
            return {"source_date": f"{year:04d}-{MONTHS[month.group(1).lower()]:02d}-01",
                    "source_date_source": "filename", "source_date_precision": "month"}
    year = YEAR.search(name)
    if year and 1900 <= int(year.group(1)) <= datetime.now().year + 1:
        return {"source_date": year.group(1) + "-01-01", "source_date_source": "filename",
                "source_date_precision": "year"}
    date = _file_created(path)
    if date:
        return {"source_date": date, "source_date_source": "file_created",
                "source_date_precision": "day"}
    try:
        date = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).date().isoformat()
        return {"source_date": date, "source_date_source": "file_modified_fallback",
                "source_date_precision": "day"}
    except OSError:
        return {}
