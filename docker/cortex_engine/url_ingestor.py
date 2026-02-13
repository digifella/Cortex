"""
URL ingestor for open-access PDF discovery/download with optional PDF->Markdown conversion.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .textifier import DocumentTextifier
from .utils import get_logger

logger = get_logger(__name__)


def normalize_url_list(raw_text: str) -> List[str]:
    urls: List[str] = []
    seen = set()
    for line in (raw_text or "").splitlines():
        u = line.strip()
        if not u:
            continue
        if not re.match(r"^https?://", u, flags=re.IGNORECASE):
            continue
        if u not in seen:
            urls.append(u)
            seen.add(u)
    return urls


def safe_filename(text: str, suffix: str) -> str:
    base = (text or "document").strip().lower()
    base = re.sub(r"https?://", "", base)
    base = re.sub(r"[^a-z0-9._-]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("._-")
    if not base:
        base = "document"
    return f"{base[:120]}{suffix}"


@dataclass
class URLIngestResult:
    input_url: str
    final_url: str = ""
    page_title: str = ""
    status: str = "failed"
    reason: str = ""
    http_code: str = ""
    open_access_pdf_found: bool = False
    pdf_url: str = ""
    pdf_path: str = ""
    md_path: str = ""
    converted_to_md: bool = False
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class URLIngestor:
    def __init__(self, output_root: Path, timeout: int = 25):
        self.output_root = Path(output_root)
        self.timeout = timeout
        self.pdf_dir = self.output_root / "pdfs"
        self.md_dir = self.output_root / "markdown"
        self.report_dir = self.output_root / "reports"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.md_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "CortexSuiteURLIngestor/1.0 (+open-access-pdf-discovery)",
                "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
            }
        )

    @staticmethod
    def _is_pdf_content(content_type: str, url: str) -> bool:
        ctype = (content_type or "").lower()
        if "application/pdf" in ctype:
            return True
        parsed = urlparse(url or "")
        return parsed.path.lower().endswith(".pdf")

    def _extract_pdf_candidates(self, html: str, base_url: str) -> Tuple[str, List[str]]:
        soup = BeautifulSoup(html or "", "html.parser")
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")
        candidates: List[str] = []

        meta_pdf = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if meta_pdf and meta_pdf.get("content"):
            candidates.append(urljoin(base_url, meta_pdf.get("content").strip()))

        for link in soup.find_all("link"):
            href = (link.get("href") or "").strip()
            link_type = (link.get("type") or "").lower()
            if not href:
                continue
            if "pdf" in link_type or href.lower().endswith(".pdf"):
                candidates.append(urljoin(base_url, href))

        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            hlow = href.lower()
            text_low = (a.get_text(" ", strip=True) or "").lower()
            if hlow.endswith(".pdf") or "download" in text_low and "pdf" in text_low or "/pdf" in hlow:
                candidates.append(urljoin(base_url, href))

        deduped = list(dict.fromkeys(candidates))
        return title, deduped[:25]

    def _download_pdf(self, pdf_url: str, title_hint: str) -> Tuple[bool, str, str, str]:
        try:
            resp = self.session.get(pdf_url, stream=True, timeout=self.timeout, allow_redirects=True)
        except Exception as e:
            return False, "", "", f"request_failed: {e}"

        status = str(resp.status_code)
        if resp.status_code >= 400:
            reason = "paywalled_or_forbidden" if resp.status_code in (401, 402, 403) else "http_error"
            return False, status, "", reason

        content_type = resp.headers.get("Content-Type", "")
        if not self._is_pdf_content(content_type, resp.url):
            return False, status, "", "not_pdf_content"

        filename = safe_filename(title_hint or resp.url, ".pdf")
        out_path = self.pdf_dir / filename
        try:
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
            return True, status, str(out_path), ""
        except Exception as e:
            return False, status, "", f"save_failed: {e}"

    def _convert_pdf_to_md(self, pdf_path: str, use_vision: bool = False) -> Tuple[bool, str, str]:
        try:
            textifier = DocumentTextifier(use_vision=use_vision)
            md_content = textifier.textify_file(pdf_path)
            md_name = Path(pdf_path).with_suffix(".md").name
            md_path = self.md_dir / md_name
            md_path.write_text(md_content, encoding="utf-8")
            return True, str(md_path), ""
        except Exception as e:
            logger.warning(f"PDF->MD conversion failed for {pdf_path}: {e}")
            return False, "", str(e)

    def process_urls(
        self,
        urls: List[str],
        convert_to_md: bool = False,
        use_vision_for_md: bool = False,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[URLIngestResult]:
        results: List[URLIngestResult] = []
        total = len(urls)
        for idx, input_url in enumerate(urls, 1):
            start = time.time()
            result = URLIngestResult(input_url=input_url)
            if progress_cb:
                progress_cb(idx - 1, total, f"Processing {input_url}")
            try:
                resp = self.session.get(input_url, timeout=self.timeout, allow_redirects=True)
                result.final_url = resp.url
                result.http_code = str(resp.status_code)
                if resp.status_code >= 400:
                    result.status = "failed"
                    result.reason = "paywalled_or_forbidden" if resp.status_code in (401, 402, 403) else "http_error"
                    results.append(result)
                    continue

                content_type = resp.headers.get("Content-Type", "")
                title_hint = ""
                pdf_candidates: List[str] = []

                if self._is_pdf_content(content_type, resp.url):
                    title_hint = Path(urlparse(resp.url).path).stem or "downloaded_pdf"
                    pdf_candidates = [resp.url]
                else:
                    html = resp.text or ""
                    title_hint, pdf_candidates = self._extract_pdf_candidates(html, resp.url)
                    result.page_title = title_hint

                if not pdf_candidates:
                    result.status = "failed"
                    result.reason = "no_open_access_pdf_found"
                    results.append(result)
                    continue

                downloaded = False
                last_reason = "pdf_download_failed"
                for candidate in pdf_candidates:
                    ok, http_code, pdf_path, reason = self._download_pdf(candidate, title_hint or input_url)
                    if http_code:
                        result.http_code = http_code
                    if ok:
                        result.open_access_pdf_found = True
                        result.pdf_url = candidate
                        result.pdf_path = pdf_path
                        result.status = "downloaded"
                        downloaded = True
                        break
                    last_reason = reason or last_reason

                if not downloaded:
                    result.status = "failed"
                    result.reason = last_reason
                    results.append(result)
                    continue

                if convert_to_md and result.pdf_path:
                    ok, md_path, conv_reason = self._convert_pdf_to_md(result.pdf_path, use_vision=use_vision_for_md)
                    if ok:
                        result.converted_to_md = True
                        result.md_path = md_path
                    else:
                        result.reason = f"md_conversion_failed: {conv_reason}"

                results.append(result)
            except Exception as e:
                result.status = "failed"
                result.reason = f"exception: {e}"
                results.append(result)
            finally:
                results[-1].elapsed_seconds = round(time.time() - start, 2)
                if progress_cb:
                    progress_cb(idx, total, f"Completed {idx}/{total}")
        return results

    def build_reports(self, results: List[URLIngestResult]) -> Tuple[Path, Path]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self.report_dir / f"url_ingest_report_{timestamp}.csv"
        json_path = self.report_dir / f"url_ingest_report_{timestamp}.json"

        rows = [r.to_dict() for r in results]
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            csv_path.write_text("", encoding="utf-8")

        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return csv_path, json_path

    @staticmethod
    def build_zip_bytes(results: List[URLIngestResult], csv_path: Path, json_path: Path) -> bytes:
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if csv_path.exists():
                zf.write(csv_path, csv_path.name)
            if json_path.exists():
                zf.write(json_path, json_path.name)
            for r in results:
                if r.pdf_path and Path(r.pdf_path).exists():
                    zf.write(r.pdf_path, f"pdfs/{Path(r.pdf_path).name}")
                if r.md_path and Path(r.md_path).exists():
                    zf.write(r.md_path, f"markdown/{Path(r.md_path).name}")
        return buf.getvalue()
