from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from cortex_engine.handoff_contract import validate_csv_profile_import_input

CSV_COMMENT_PREFIX = "#"
CSV_MAX_ROWS = 500
_SMART_QUOTES = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
})
_CSV_SUBJECT_RE = re.compile(r"\bcsv\b", re.IGNORECASE)


class CsvProfileImportError(RuntimeError):
    pass


def _normalize_header(value: str) -> str:
    text = str(value or "").strip().lstrip("\ufeff")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text


def _clean_cell(value: Any) -> str:
    text = str(value or "").strip()
    text = text.translate(_SMART_QUOTES)
    return text


def _normalize_watch(value: Any) -> str:
    return "yes" if str(value or "").strip().lower() in {"yes", "y", "true", "1", "x"} else "no"


def _normalize_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    return status if status in {"active", "archived"} else "active"


def _attachment_is_csv(item: Dict[str, Any]) -> bool:
    filename = str(item.get("filename") or "").strip().lower()
    mime_type = str(item.get("mime_type") or "").strip().lower()
    return filename.endswith(".csv") or mime_type == "text/csv"


def detect_csv_profile_import(subject: str, attachments: Sequence[Dict[str, Any]]) -> bool:
    return subject_looks_like_csv_profile_import(subject) and any(_attachment_is_csv(item) for item in attachments or [])


def subject_looks_like_csv_profile_import(subject: str) -> bool:
    lowered = str(subject or "").strip().lower()
    if not lowered:
        return False
    if "profiles" in lowered:
        return True
    words = set(re.findall(r"[a-z0-9]+", lowered))
    if {"profile", "import"}.issubset(words):
        return True
    return bool(_CSV_SUBJECT_RE.search(lowered))


def _find_first_csv_attachment(attachments: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for item in attachments or []:
        if _attachment_is_csv(item):
            return dict(item)
    return None


def _build_import_url(explicit_url: str, callback_url: str, queue_server_url: str) -> str:
    for candidate in (explicit_url, callback_url, queue_server_url):
        parsed = urlparse(str(candidate or "").strip())
        if parsed.scheme and parsed.netloc:
            break
    else:
        return ""
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["action"] = "bulk_import_profiles"
    new_path = "/lab/market_radar_api.php"
    return urlunparse((parsed.scheme, parsed.netloc, new_path, "", urlencode(query), ""))


def _row_is_comment(row: Sequence[Any]) -> bool:
    if not row:
        return False
    first = str(row[0] or "").strip()
    return bool(first) and first.startswith(CSV_COMMENT_PREFIX)


def _detect_target_type(row: Dict[str, str]) -> str:
    for key in ("website_url", "industry", "address_street"):
        if str(row.get(key) or "").strip():
            return "organisation"
    return "person"


def _normalize_row(row: Dict[str, str]) -> Dict[str, Any]:
    normalized = {key: _clean_cell(value) for key, value in row.items()}
    normalized["canonical_name"] = normalized.get("canonical_name", "")
    normalized["watch"] = _normalize_watch(normalized.get("watch", ""))
    normalized["status"] = _normalize_status(normalized.get("status", ""))
    normalized["target_type"] = _detect_target_type(normalized)
    if normalized["target_type"] == "organisation" and not normalized.get("address_country"):
        normalized["address_country"] = "Australia"
    return {key: value for key, value in normalized.items() if value != ""}


def parse_csv_profile_rows(attachment_path: str) -> Dict[str, Any]:
    path = Path(str(attachment_path or "").strip())
    if not path.exists():
        raise CsvProfileImportError("CSV attachment path not accessible from Cortex worker")

    raw_text = path.read_text(encoding="utf-8-sig", errors="ignore")
    reader = csv.reader(io.StringIO(raw_text))

    headers: List[str] = []
    prepared_rows: List[Dict[str, Any]] = []
    skipped_rows: List[str] = []
    data_rows_seen = 0

    for row_index, row in enumerate(reader, start=1):
        if not any(str(cell or "").strip() for cell in row):
            continue
        if _row_is_comment(row):
            continue
        if not headers:
            headers = [_normalize_header(cell) for cell in row]
            continue

        data_rows_seen += 1
        if data_rows_seen > CSV_MAX_ROWS:
            raise CsvProfileImportError("CSV exceeds 500-row limit. Split into multiple files.")

        record: Dict[str, str] = {}
        for idx, header in enumerate(headers):
            if not header:
                continue
            record[header] = _clean_cell(row[idx] if idx < len(row) else "")

        canonical_name = str(record.get("canonical_name") or "").strip()
        if not canonical_name:
            skipped_rows.append(f"Row {row_index}: canonical_name is required")
            continue

        prepared_rows.append(_normalize_row(record))

    if not headers:
        raise CsvProfileImportError("CSV appears empty — no header row found.")
    if data_rows_seen == 0:
        raise CsvProfileImportError("CSV appears empty — no data rows found.")
    if not prepared_rows:
        raise CsvProfileImportError("All rows were skipped — canonical_name is required.")

    return {
        "filename": path.name,
        "row_count": data_rows_seen,
        "rows": prepared_rows,
        "skipped_errors": skipped_rows,
    }


class CsvProfileImportProcessor:
    def __init__(
        self,
        import_url: str,
        queue_secret: str,
        timeout: int = 30,
    ):
        self.import_url = str(import_url or "").strip()
        self.queue_secret = str(queue_secret or "").strip()
        self.timeout = max(5, int(timeout or 30))

    @classmethod
    def from_config(
        cls,
        explicit_url: str,
        callback_url: str,
        queue_server_url: str,
        queue_secret: str,
        timeout: int = 30,
    ) -> "CsvProfileImportProcessor":
        return cls(
            import_url=_build_import_url(explicit_url, callback_url, queue_server_url),
            queue_secret=queue_secret,
            timeout=timeout,
        )

    def _call_api(self, rows: List[Dict[str, Any]], org_name: str, on_behalf_of: str, dry_run: bool) -> Dict[str, Any]:
        if not self.import_url:
            raise CsvProfileImportError("CSV profile import URL is not configured")
        payload = validate_csv_profile_import_input(
            {
                "rows": rows,
                "org_name": org_name,
                "on_behalf_of": on_behalf_of,
                "dry_run": dry_run,
            }
        )
        request_body = dict(payload)
        request_body["queue_secret"] = self.queue_secret

        headers = {"Content-Type": "application/json"}
        if self.queue_secret:
            headers["X-Queue-Key"] = self.queue_secret

        response = requests.post(self.import_url, headers=headers, json=request_body, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise CsvProfileImportError("Website bulk import returned a non-object response")
        if body.get("error"):
            raise CsvProfileImportError(str(body["error"]))
        return body

    @staticmethod
    def _reply_subject(original_subject: str, dry_run: bool, ok: bool) -> str:
        subject = str(original_subject or "PROFILES").strip()
        if dry_run:
            return f"Re: {subject} — Preview (not saved)"
        if ok:
            return f"Re: {subject} — Import complete"
        return f"Re: {subject} — Import failed"

    @staticmethod
    def _reply_body(filename: str, row_count: int, api_result: Dict[str, Any], skipped_errors: List[str], dry_run: bool) -> str:
        created = int(api_result.get("created") or 0)
        updated = int(api_result.get("updated") or 0)
        skipped = int(api_result.get("skipped") or 0) + len(skipped_errors)
        all_errors = list(skipped_errors) + [str(item) for item in api_result.get("errors") or [] if str(item).strip()]

        if dry_run:
            lines = [
                "Dry run complete — no changes were made.",
                "",
                f"  File         : {filename} ({row_count} rows)",
                f"  Would create : {created}",
                f"  Would update : {updated}",
                f"  Would skip   : {skipped}",
            ]
            if all_errors:
                lines.extend(["", "Skipped rows:"])
                lines.extend(f"  - {item}" for item in all_errors[:25])
            lines.extend(["", 'To apply, re-send without "DRY RUN" in the subject.'])
            return "\n".join(lines)

        lines = [
            "Profile CSV import complete",
            "",
            f"  File     : {filename} ({row_count} rows)",
            f"  Created  : {created}",
            f"  Updated  : {updated}",
            f"  Skipped  : {skipped}",
        ]
        if all_errors:
            lines.extend(["", "Skipped rows:"])
            lines.extend(f"  - {item}" for item in all_errors[:25])
        lines.extend(["", "Profiles are now visible in Market Radar -> Profiles tab."])
        return "\n".join(lines)

    def process_message(self, message: Dict[str, Any], persisted: Dict[str, Any], org_name: str) -> Dict[str, Any]:
        csv_attachment = _find_first_csv_attachment(persisted.get("attachments") or [])
        if not csv_attachment:
            raise CsvProfileImportError("No CSV attachment found. Please attach a .csv file.")

        dry_run = "dry run" in str(message.get("subject") or "").lower()
        parsed = parse_csv_profile_rows(csv_attachment.get("stored_path") or "")
        api_result = self._call_api(
            rows=parsed["rows"],
            org_name=org_name,
            on_behalf_of=str(message.get("from_email") or "").strip().lower(),
            dry_run=dry_run,
        )
        reply_subject = self._reply_subject(message.get("subject", ""), dry_run=dry_run, ok=True)
        reply_body = self._reply_body(
            filename=parsed["filename"],
            row_count=int(parsed["row_count"] or 0),
            api_result=api_result,
            skipped_errors=list(parsed["skipped_errors"] or []),
            dry_run=dry_run,
        )
        return {
            "status": "processed",
            "result_type": "csv_profile_import_result",
            "filename": parsed["filename"],
            "row_count": parsed["row_count"],
            "rows": parsed["rows"],
            "dry_run": dry_run,
            "created": int(api_result.get("created") or 0),
            "updated": int(api_result.get("updated") or 0),
            "skipped": int(api_result.get("skipped") or 0) + len(parsed["skipped_errors"]),
            "errors": list(parsed["skipped_errors"] or []) + [str(item) for item in api_result.get("errors") or [] if str(item).strip()],
            "api_result": api_result,
            "reply_subject": reply_subject,
            "reply_body": reply_body,
        }
