from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from cortex_engine.document_registry import build_content_fingerprint, build_document_meta
from cortex_engine.org_chart_extractor import extract_org_chart_structured
from cortex_engine.stakeholder_signal_store import StakeholderSignalStore, orgs_compatible
from cortex_engine.stakeholder_signal_matcher import normalize_lookup
from cortex_engine.strategic_doc_analyser import analyse_strategic_documents
from cortex_engine.url_ingestor import URLIngestor

logger = logging.getLogger(__name__)

_SOURCE_TYPE_ORDER = {
    "annual_report": 0,
    "strategic_plan": 1,
    "org_chart": 2,
    "about_page": 3,
    "other": 4,
}
_COMMON_DISCOVERY_PATHS = (
    "/about",
    "/about-us",
    "/our-organisation",
    "/our-organization",
    "/reports",
    "/annual-reports",
    "/publications",
    "/publications-and-reports",
    "/strategy",
    "/our-strategy",
    "/strategic-plan",
    "/corporate-plan",
    "/leadership",
    "/leadership-team",
    "/executive-team",
    "/board",
    "/governance",
)
_SITEMAP_PATHS = (
    "/sitemap.xml",
    "/sitemap_index.xml",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalise_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not re.match(r"^https?://", text, flags=re.IGNORECASE):
        text = f"https://{text.lstrip('/')}"
    return text


def _extract_year(*values: str) -> str:
    for value in values:
        match = re.search(r"\b(20\d{2})\b", str(value or ""))
        if match:
            return match.group(1)
    return ""


def _org_acronym(value: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", str(value or "")) if token]
    if len(tokens) < 2:
        return ""
    acronym = "".join(token[0].upper() for token in tokens if token and token[0].isalpha())
    return acronym if 2 <= len(acronym) <= 8 else ""


def _same_host(left: str, right: str) -> bool:
    return (urlparse(left).netloc or "").lower() == (urlparse(right).netloc or "").lower()


def _source_targets_org(source: Dict[str, Any], target_org_name: str) -> bool:
    target = str(target_org_name or "").strip()
    if not target:
        return True
    title = unquote(str(source.get("title") or ""))
    label = unquote(str(source.get("source_label") or source.get("label") or ""))
    url = unquote(str(source.get("url") or ""))
    candidates = [title, label]
    if any(orgs_compatible(candidate, target) for candidate in candidates if str(candidate).strip()):
        return True
    haystack = normalize_lookup(" ".join([title, label, url]))
    target_key = normalize_lookup(target)
    if target_key and target_key in haystack:
        return True
    acronym = normalize_lookup(_org_acronym(target))
    if acronym:
        tokens = set(re.findall(r"[a-z0-9]+", haystack))
        if acronym in tokens:
            return True
    return False


def _source_sort_key(source: Dict[str, Any], target_org_name: str) -> tuple[int, int, int, int, str]:
    source_type = str(source.get("type") or "")
    title = unquote(str(source.get("title") or ""))
    url = unquote(str(source.get("url") or ""))
    year_text = str(source.get("year") or _extract_year(title, url) or "")
    try:
        year_value = int(year_text)
    except Exception:
        year_value = 0
    direct_document = 1 if url.lower().endswith(".pdf") else 0
    targeted = 1 if _source_targets_org(source, target_org_name) else 0
    return (
        _SOURCE_TYPE_ORDER.get(source_type, 9),
        -targeted,
        -year_value,
        -direct_document,
        normalize_lookup(title or url),
    )


def _limit_relevant_sources(
    sources: List[Dict[str, Any]],
    target_org_name: str,
    max_sources: int,
) -> List[Dict[str, Any]]:
    if not sources:
        return []
    per_type_limits = {
        "annual_report": 2,
        "strategic_plan": 2,
        "org_chart": 2,
        "about_page": 1,
    }
    ordered = sorted(sources, key=lambda item: _source_sort_key(item, target_org_name))
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    fallback: List[Dict[str, Any]] = []
    for source in ordered:
        source_type = str(source.get("type") or "")
        limit = per_type_limits.get(source_type, 1)
        if _source_targets_org(source, target_org_name):
            if counts.get(source_type, 0) >= limit:
                continue
            selected.append(source)
            counts[source_type] = counts.get(source_type, 0) + 1
        else:
            fallback.append(source)
        if len(selected) >= max_sources:
            return selected[:max_sources]
    for source in fallback:
        source_type = str(source.get("type") or "")
        limit = per_type_limits.get(source_type, 1)
        if counts.get(source_type, 0) >= limit:
            continue
        selected.append(source)
        counts[source_type] = counts.get(source_type, 0) + 1
        if len(selected) >= max_sources:
            break
    return selected[:max_sources]


def _attachment_filename(source: Dict[str, Any], result: Any) -> str:
    title = unquote(str(source.get("title") or source.get("source_label") or "")).strip()
    url_name = Path(unquote(urlparse(str(source.get("url") or "")).path)).name
    suffix = Path(url_name).suffix or Path(str(getattr(result, "pdf_path", "") or getattr(result, "md_path", ""))).suffix
    base = title or Path(url_name).stem or "document"
    base = re.sub(r"[\\/:*?\"<>|]+", "-", base)
    base = re.sub(r"\s+", " ", base).strip(" .-_")
    if suffix and not base.lower().endswith(suffix.lower()):
        return f"{base}{suffix}"
    return base or "document"


def _discover_source_type(text: str, url: str = "") -> str:
    haystack = normalize_lookup(" ".join([str(text or ""), str(url or "")]))
    if any(token in haystack for token in ("annual report", "financial report")):
        return "annual_report"
    if any(token in haystack for token in ("strategic plan", "strategic direction", "strategy", "our plan", "corporate plan", "business plan")):
        return "strategic_plan"
    if any(token in haystack for token in ("org chart", "organisational chart", "organizational chart", "leadership team", "executive team", "board of directors", "our people", "leadership")):
        return "org_chart"
    if any(token in haystack for token in ("about us", "about", "who we are", "our organisation", "our organization")):
        return "about_page"
    return "other"


def _field_value(
    value: Any,
    source: Dict[str, Any],
    confidence: str = "medium",
    evidence_excerpt: str = "",
) -> Dict[str, Any]:
    return {
        "value": value,
        "source_url": str(source.get("url") or "").strip(),
        "source_label": str(source.get("source_label") or source.get("title") or "").strip(),
        "confidence": confidence,
        "evidence_excerpt": str(evidence_excerpt or "").strip()[:500],
        "as_of_date": str(source.get("year") or ""),
    }


def _normalise_comparable(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(_normalise_comparable(item) for item in value if _normalise_comparable(item))
    if isinstance(value, dict):
        pieces = []
        for key in sorted(value.keys()):
            rendered = _normalise_comparable(value.get(key))
            if rendered:
                pieces.append(f"{key}:{rendered}")
        return " | ".join(pieces)
    return " ".join(str(value or "").strip().lower().split())


def _is_changed(snapshot: Dict[str, Any], field_name: str, new_value: Any) -> bool:
    return _normalise_comparable(snapshot.get(field_name)) != _normalise_comparable(new_value)


def _read_text(path_text: str) -> str:
    path = Path(str(path_text or "").strip())
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _clean_text_summary(value: str, limit: int = 1400) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:limit]


def _build_manual_document_prompt(
    target_org_name: str,
    requested_docs: List[str],
    blocked_urls: List[str],
) -> Dict[str, Any]:
    doc_labels = {
        "annual_report": "annual report",
        "strategic_plan": "strategic plan",
        "org_chart": "org chart",
        "about_page": "about page",
    }
    requested = [doc_labels.get(item, item.replace("_", " ")) for item in requested_docs if item]
    requested_text = ", ".join(requested) if requested else "relevant organisation documents"
    if blocked_urls:
        message = (
            f"Cortex could not access public source material for {target_org_name}. "
            f"The discovered source URLs appear blocked, unavailable, or hostile to automated access. "
            f"To continue, obtain the {requested_text} manually and email or upload them into Cortex."
        )
    else:
        message = (
            f"Cortex could not find usable public source material for {target_org_name}. "
            f"To continue, obtain the {requested_text} manually and email or upload them into Cortex."
        )
    return {
        "requires_manual_documents": True,
        "user_message": message,
        "recommended_next_step": f"Upload or email the {requested_text} for {target_org_name} so Cortex can extract and apply the profile updates.",
        "blocked_source_urls": blocked_urls,
    }


class OfficialSourceDiscoverer:
    def __init__(self, timeout: int = 25):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "CortexSuiteOrgProfileRefresh/1.0",
                "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
            }
        )

    def _fetch_html(self, url: str, raise_on_error: bool = True) -> tuple[str, str]:
        response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
        if raise_on_error:
            response.raise_for_status()
        elif not response.ok:
            return response.url, ""
        if "html" not in (response.headers.get("Content-Type") or "").lower():
            return response.url, ""
        return response.url, response.text or ""

    def _fetch_text(self, url: str, raise_on_error: bool = True) -> tuple[str, str, str]:
        response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
        if raise_on_error:
            response.raise_for_status()
        elif not response.ok:
            return response.url, "", str(response.headers.get("Content-Type") or "")
        return response.url, response.text or "", str(response.headers.get("Content-Type") or "")

    @staticmethod
    def _title_from_html(html: str, fallback: str = "") -> str:
        soup = BeautifulSoup(html or "", "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            if title:
                return title
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(" ", strip=True)
        return fallback

    def _extract_links(self, html: str, base_url: str) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html or "", "html.parser")
        links: List[Dict[str, str]] = []
        for anchor in soup.find_all("a"):
            href = str(anchor.get("href") or "").strip()
            if not href:
                continue
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme not in {"http", "https"}:
                continue
            label = anchor.get_text(" ", strip=True)
            links.append(
                {
                    "url": absolute,
                    "label": label,
                    "title": label,
                    "type": _discover_source_type(f"{label} {href}", absolute),
                }
            )
        return links

    def _extract_sitemap_links(self, sitemap_url: str, homepage_url: str, max_urls: int = 40) -> List[Dict[str, str]]:
        try:
            final_url, body, content_type = self._fetch_text(sitemap_url, raise_on_error=False)
        except Exception:
            return []
        if not body or ("xml" not in content_type.lower() and "<loc>" not in body.lower()):
            return []
        soup = BeautifulSoup(body, "xml")
        links: List[Dict[str, str]] = []
        for loc in soup.find_all("loc"):
            url = str(loc.get_text(strip=True) or "").strip()
            if not url or not _same_host(homepage_url, url):
                continue
            source_type = _discover_source_type(url, url)
            if source_type == "other" and not url.lower().endswith(".pdf"):
                continue
            links.append(
                {
                    "url": url,
                    "label": url,
                    "title": url,
                    "type": source_type,
                }
            )
            if len(links) >= max_urls:
                break
        return links

    def discover_sources(
        self,
        website_url: str,
        requested_docs: List[str],
        target_org_name: str = "",
        max_sources: int = 6,
    ) -> List[Dict[str, Any]]:
        homepage = _normalise_url(website_url)
        if not homepage:
            return []

        discovered: Dict[str, Dict[str, Any]] = {}

        def remember(source: Dict[str, Any]) -> None:
            url = str(source.get("url") or "").strip()
            if not url or url in discovered:
                return
            discovered[url] = source

        try:
            final_homepage, homepage_html = self._fetch_html(homepage, raise_on_error=False)
        except Exception:
            # Homepage unreachable (timeout, DNS, etc.) — return empty
            return []

        if not homepage_html:
            # 403/blocked/non-HTML — still record the URL but can't discover links
            return [
                {
                    "type": "about_page",
                    "title": homepage,
                    "url": homepage,
                    "year": "",
                    "source_label": "Organisation website (blocked or unavailable)",
                }
            ]

        homepage_title = self._title_from_html(homepage_html, fallback=homepage)
        site_root = f"{urlparse(final_homepage).scheme}://{urlparse(final_homepage).netloc}"
        remember(
            {
                "type": "about_page",
                "title": homepage_title or homepage,
                "url": final_homepage,
                "year": _extract_year(homepage_title, final_homepage),
                "source_label": homepage_title or "Organisation website",
            }
        )

        requested = set(requested_docs or [])
        homepage_links = self._extract_links(homepage_html, final_homepage)
        candidate_pages: List[Dict[str, Any]] = []
        for link in homepage_links:
            link_type = link["type"]
            if link_type == "other":
                continue
            if requested and link_type not in requested and link_type != "about_page":
                continue
            if not _same_host(final_homepage, link["url"]):
                continue
            source = {
                "type": link_type,
                "title": link["title"] or link["url"],
                "url": link["url"],
                "year": _extract_year(link["title"], link["url"]),
                "source_label": link["label"] or link["title"] or link["url"],
            }
            remember(source)
            if not link["url"].lower().endswith(".pdf"):
                candidate_pages.append(source)

        for sitemap_path in _SITEMAP_PATHS:
            sitemap_url = urljoin(site_root, sitemap_path)
            for link in self._extract_sitemap_links(sitemap_url, final_homepage):
                link_type = link["type"]
                if link_type == "other":
                    continue
                if requested and link_type not in requested and link_type != "about_page":
                    continue
                source = {
                    "type": link_type,
                    "title": link["title"] or link["url"],
                    "url": link["url"],
                    "year": _extract_year(link["title"], link["url"]),
                    "source_label": link["label"] or link["title"] or link["url"],
                }
                remember(source)
                if not link["url"].lower().endswith(".pdf"):
                    candidate_pages.append(source)

        for path in _COMMON_DISCOVERY_PATHS:
            url = urljoin(site_root, path)
            source_type = _discover_source_type(path, url)
            if requested and source_type not in requested and source_type != "about_page":
                continue
            try:
                final_url, page_html = self._fetch_html(url, raise_on_error=False)
            except Exception:
                continue
            if not page_html:
                continue
            source = {
                "type": source_type,
                "title": self._title_from_html(page_html, fallback=final_url),
                "url": final_url,
                "year": _extract_year(final_url),
                "source_label": self._title_from_html(page_html, fallback=final_url),
            }
            remember(source)
            candidate_pages.append(source)

        ordered_candidate_pages = sorted(
            {str(item.get("url") or ""): item for item in candidate_pages if str(item.get("url") or "")}.values(),
            key=lambda item: (_SOURCE_TYPE_ORDER.get(str(item.get("type") or ""), 9), str(item.get("title") or "")),
        )
        for source in ordered_candidate_pages[:10]:
            try:
                page_url, page_html = self._fetch_html(source["url"])
            except Exception:
                continue
            source["url"] = page_url
            source["title"] = source.get("title") or self._title_from_html(page_html, fallback=page_url)
            for link in self._extract_links(page_html, page_url):
                if not _same_host(final_homepage, link["url"]):
                    continue
                link_type = link["type"]
                if link_type == "other":
                    continue
                if requested and link_type not in requested and link_type != "about_page":
                    continue
                remember(
                    {
                        "type": link_type,
                        "title": link["title"] or link["url"],
                        "url": link["url"],
                        "year": _extract_year(link["title"], link["url"]),
                        "source_label": link["label"] or link["title"] or link["url"],
                    }
                )

        return _limit_relevant_sources(list(discovered.values()), target_org_name=target_org_name, max_sources=max_sources)


def _merge_people(people: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for person in people:
        name = str(person.get("name") or "").strip()
        if not name:
            continue
        key = normalize_lookup(name)
        current = merged.get(key)
        candidate = {
            "name": name,
            "current_role": str(person.get("current_role") or "").strip(),
            "current_employer": str(person.get("current_employer") or "").strip(),
            "source_url": str(person.get("source_url") or "").strip(),
            "source_label": str(person.get("source_label") or "").strip(),
            "confidence": str(person.get("confidence") or "medium"),
            "evidence_excerpt": str(person.get("evidence_excerpt") or person.get("evidence") or "").strip()[:400],
        }
        if current is None or (candidate["current_role"] and not current.get("current_role")):
            merged[key] = candidate
    return sorted(merged.values(), key=lambda item: item["name"])


def _build_org_structure_summary(people: List[Dict[str, Any]], target_org_name: str) -> str:
    if not people:
        return ""
    leaders = [f"{item['name']} ({item['current_role']})" for item in people[:6] if item.get("current_role")]
    if leaders:
        return f"{target_org_name} leadership/public structure identified: " + ", ".join(leaders) + "."
    return f"{target_org_name} leadership/public structure identified from discovered organisation materials."


def run_org_profile_refresh(
    payload: Dict[str, Any],
    run_dir: Path,
    progress_cb: Optional[Callable[[float, str, Optional[str]], None]] = None,
    is_cancelled_cb: Optional[Callable[[], bool]] = None,
    signal_store: Optional[StakeholderSignalStore] = None,
) -> Dict[str, Any]:
    signal_store = signal_store or StakeholderSignalStore()
    target_org_name = str(payload.get("target_org_name") or "").strip()
    snapshot = dict(payload.get("current_profile_snapshot") or {})
    requested_docs = list(payload.get("requested_docs") or [])
    warnings: List[str] = []
    debug_info: Dict[str, Any] = {
        "website_url": "",
        "requested_docs": requested_docs,
        "discovered_sources": [],
        "ingest_results": [],
        "processed_sources": [],
        "warning_count": 0,
    }

    website_url = str(payload.get("website_url") or snapshot.get("website_url") or "").strip()
    if not website_url:
        raise ValueError("org_profile_refresh requires website_url in payload or current_profile_snapshot for official-source discovery")
    debug_info["website_url"] = website_url
    logger.info(
        "org_profile_refresh starting target=%s org=%s website=%s requested_docs=%s",
        target_org_name,
        str(payload.get("org_name") or "").strip(),
        website_url,
        requested_docs,
    )

    if progress_cb:
        progress_cb(10, "Discovering official organisation sources", "discover")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before source discovery")

    discoverer = OfficialSourceDiscoverer(timeout=int(payload.get("timeout_seconds") or 25))
    discovered_sources = discoverer.discover_sources(
        website_url=website_url,
        requested_docs=requested_docs,
        target_org_name=target_org_name,
        max_sources=int(payload.get("max_sources") or 6),
    )
    debug_info["discovered_sources"] = [
        {
            "type": str(item.get("type") or ""),
            "title": str(item.get("title") or ""),
            "url": str(item.get("url") or ""),
        }
        for item in discovered_sources
    ]
    logger.info(
        "org_profile_refresh discovered %d source(s) for %s: %s",
        len(discovered_sources),
        target_org_name,
        [
            f"{str(item.get('type') or '')}:{str(item.get('url') or '')}"
            for item in discovered_sources
        ],
    )
    if not discovered_sources:
        warnings.append("No official sources discovered from the supplied website URL")
    else:
        discovered_types = {str(item.get("type") or "") for item in discovered_sources}
        missing_types = [doc_type for doc_type in requested_docs if doc_type not in discovered_types]
        if missing_types:
            warnings.append("Requested source types not found: " + ", ".join(missing_types))
            logger.warning(
                "org_profile_refresh missing requested source types for %s: %s",
                target_org_name,
                missing_types,
            )

    if progress_cb:
        progress_cb(35, f"Capturing {len(discovered_sources)} discovered source(s)", "capture")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before source capture")

    ingestor = URLIngestor(run_dir, timeout=int(payload.get("timeout_seconds") or 25))
    urls = [str(item.get("url") or "").strip() for item in discovered_sources if str(item.get("url") or "").strip()]
    ingested = ingestor.process_urls(
        urls=urls,
        convert_to_md=True,
        use_vision_for_md=bool(payload.get("use_vision", True)),
        textify_options={"pdf_strategy": "hybrid"},
        capture_web_md_on_no_pdf=True,
    ) if urls else []

    source_by_url = {str(item.get("url") or "").strip(): dict(item) for item in discovered_sources}
    attachments: List[Dict[str, Any]] = []
    processed_sources: List[Dict[str, Any]] = []
    org_chart_texts: List[str] = []
    for result in ingested:
        debug_info["ingest_results"].append(
            {
                "input_url": str(result.input_url or ""),
                "final_url": str(result.final_url or ""),
                "status": str(result.status or ""),
                "md_path": str(result.md_path or ""),
                "pdf_path": str(result.pdf_path or ""),
                "reason": str(getattr(result, "reason", "") or ""),
            }
        )
        logger.info(
            "org_profile_refresh ingest result target=%s status=%s input=%s final=%s md=%s pdf=%s reason=%s",
            target_org_name,
            str(result.status or ""),
            str(result.input_url or ""),
            str(result.final_url or ""),
            str(result.md_path or ""),
            str(result.pdf_path or ""),
            str(getattr(result, "reason", "") or ""),
        )
        source = dict(source_by_url.get(result.input_url) or source_by_url.get(result.final_url) or {})
        if not source:
            logger.warning(
                "org_profile_refresh could not match ingest result back to discovered source target=%s input=%s final=%s",
                target_org_name,
                str(result.input_url or ""),
                str(result.final_url or ""),
            )
            continue
        md_text = _read_text(result.md_path) if result.md_path else ""
        if not md_text and result.pdf_path:
            md_text = _clean_text_summary(Path(result.pdf_path).name)
        if not md_text:
            warnings.append(f"Could not extract text from {source.get('url')}")
            logger.warning(
                "org_profile_refresh extracted no text for target=%s source=%s status=%s",
                target_org_name,
                str(source.get("url") or ""),
                str(result.status or ""),
            )
            continue
        filename = _attachment_filename(source, result)
        attachments.append(
            {
                "filename": filename,
                "stored_path": result.md_path or result.pdf_path,
                "excerpt": md_text,
                "status": "processed",
            }
        )
        processed_source = {
            **source,
            "resolved_url": result.final_url or result.input_url,
            "status": result.status,
            "md_path": result.md_path,
            "pdf_path": result.pdf_path,
            "content_fingerprint": build_content_fingerprint(
                text=md_text,
                source_url=str(result.final_url or result.input_url or ""),
            ),
        }
        document_meta = build_document_meta(
            doc_type=str(source.get("type") or ""),
            target_org_name=target_org_name,
            title=str(source.get("title") or ""),
            period_label=str(source.get("year") or ""),
            published_at="",
            content_fingerprint=str(processed_source.get("content_fingerprint") or ""),
            source_url=str(source.get("url") or ""),
            source_label=str(source.get("source_label") or ""),
        )
        document_meta = signal_store.classify_document_meta(str(payload.get("org_name") or "").strip(), document_meta)
        signal_store.register_document_meta(
            str(payload.get("org_name") or "").strip(),
            {key: value for key, value in document_meta.items() if key != "existing_record"},
        )
        processed_source["document_meta"] = {
            key: value for key, value in document_meta.items() if key != "existing_record"
        }
        processed_sources.append(processed_source)
        debug_info["processed_sources"].append(
            {
                "type": str(source.get("type") or ""),
                "url": str(source.get("url") or ""),
                "resolved_url": str(processed_source.get("resolved_url") or ""),
                "status": str(processed_source.get("status") or ""),
                "filename": filename,
                "document_status": str(processed_source.get("document_meta", {}).get("status") or ""),
            }
        )
        if str(source.get("type") or "") == "org_chart":
            org_chart_texts.append(md_text)

    if not processed_sources:
        warnings.append("No discovered sources yielded usable text for analysis")
        logger.warning(
            "org_profile_refresh no usable text target=%s discovered=%d ingested=%d",
            target_org_name,
            len(discovered_sources),
            len(ingested),
        )
    else:
        logger.info(
            "org_profile_refresh prepared %d attachment(s) for analysis target=%s",
            len(attachments),
            target_org_name,
        )

    if progress_cb:
        progress_cb(60, f"Analysing {len(attachments)} captured source(s)", "analyse")
    if is_cancelled_cb and is_cancelled_cb():
        raise RuntimeError("Cancelled before analysis")

    strategic_analysis = analyse_strategic_documents(
        attachments=attachments,
        extracted_summary="",
        subject=target_org_name,
        raw_text="",
    )
    org_chart_analysis = extract_org_chart_structured(
        attachment_texts=org_chart_texts,
        attachment_summaries=attachments,
        employer_hint=target_org_name,
    )
    logger.info(
        "org_profile_refresh analysis target=%s strategic_doc_type=%s strategic_org=%s leaders=%d org_chart_people=%d",
        target_org_name,
        str(strategic_analysis.get("doc_type") or ""),
        str(strategic_analysis.get("org_name") or ""),
        len(strategic_analysis.get("leadership_people") or []),
        len(org_chart_analysis.get("people") or []),
    )

    source_lookup_by_type: Dict[str, Dict[str, Any]] = {}
    for source in processed_sources:
        source_lookup_by_type.setdefault(str(source.get("type") or ""), source)

    auto_apply: Dict[str, Any] = {}
    proposed: Dict[str, Any] = {}

    effective_website_url = _normalise_url(website_url)
    if effective_website_url and _is_changed(snapshot, "website_url", effective_website_url):
        auto_apply["website_url"] = _field_value(
            effective_website_url,
            source_lookup_by_type.get("about_page") or {"url": effective_website_url, "source_label": "Organisation website"},
            confidence="high",
            evidence_excerpt="Official organisation website used for source discovery.",
        )

    mission = str(strategic_analysis.get("mission") or "").strip()
    if mission and _is_changed(snapshot, "mission_statement", mission):
        proposed["mission_statement"] = _field_value(
            mission,
            source_lookup_by_type.get("strategic_plan") or source_lookup_by_type.get("about_page") or {},
            confidence="high",
            evidence_excerpt=mission,
        )

    priorities = [
        str(item).strip()
        for item in (
            list(strategic_analysis.get("priorities") or [])
            + list(strategic_analysis.get("themes") or [])
            + list(strategic_analysis.get("initiatives") or [])
        )
        if str(item).strip()
    ]
    unique_priorities = list(dict.fromkeys(priorities))
    if unique_priorities and _is_changed(snapshot, "strategic_priorities", unique_priorities):
        proposed["strategic_priorities"] = _field_value(
            unique_priorities,
            source_lookup_by_type.get("strategic_plan") or source_lookup_by_type.get("annual_report") or {},
            confidence="high",
            evidence_excerpt=", ".join(unique_priorities[:6]),
        )

    leadership_candidates = _merge_people(
        [
            {
                "name": item.get("name"),
                "current_role": item.get("current_role"),
                "current_employer": item.get("current_employer") or target_org_name,
                "source_url": str(source_lookup_by_type.get("annual_report", {}).get("url") or source_lookup_by_type.get("strategic_plan", {}).get("url") or ""),
                "source_label": str(source_lookup_by_type.get("annual_report", {}).get("source_label") or source_lookup_by_type.get("strategic_plan", {}).get("source_label") or ""),
                "confidence": "high",
                "evidence_excerpt": item.get("evidence"),
            }
            for item in strategic_analysis.get("leadership_people") or []
        ]
        + [
            {
                "name": item.get("name"),
                "current_role": item.get("current_role"),
                "current_employer": item.get("current_employer") or target_org_name,
                "source_url": str(source_lookup_by_type.get("org_chart", {}).get("url") or ""),
                "source_label": str(source_lookup_by_type.get("org_chart", {}).get("source_label") or ""),
                "confidence": "medium",
                "evidence_excerpt": item.get("evidence"),
            }
            for item in org_chart_analysis.get("people") or []
        ]
    )

    leadership_summary = [f"{item['name']} ({item['current_role']})" for item in leadership_candidates[:8] if item.get("current_role")]
    if leadership_summary and _is_changed(snapshot, "leadership_team", leadership_summary):
        proposed["leadership_team"] = _field_value(
            leadership_summary,
            source_lookup_by_type.get("org_chart") or source_lookup_by_type.get("annual_report") or {},
            confidence="medium" if source_lookup_by_type.get("org_chart") else "high",
            evidence_excerpt=", ".join(leadership_summary[:6]),
        )

    org_structure_summary = _build_org_structure_summary(leadership_candidates, target_org_name)
    if org_structure_summary and _is_changed(snapshot, "org_structure_summary", org_structure_summary):
        proposed["org_structure_summary"] = _field_value(
            org_structure_summary,
            source_lookup_by_type.get("org_chart") or source_lookup_by_type.get("annual_report") or {},
            confidence="medium",
            evidence_excerpt=org_structure_summary,
        )

    recent_developments = [
        str(item.get("headline") or "").strip()
        for item in strategic_analysis.get("strategic_signals") or []
        if str(item.get("headline") or "").strip()
    ]
    if recent_developments and _is_changed(snapshot, "recent_developments", recent_developments):
        proposed["recent_developments"] = _field_value(
            recent_developments,
            source_lookup_by_type.get("annual_report") or source_lookup_by_type.get("strategic_plan") or {},
            confidence="medium",
            evidence_excerpt=", ".join(recent_developments[:5]),
        )

    risk_factors = [
        str(item.get("label") or "").strip()
        for item in strategic_analysis.get("performance_indicators") or []
        if str(item.get("label") or "").strip()
    ]
    if risk_factors and _is_changed(snapshot, "risk_factors", risk_factors):
        proposed["risk_factors"] = _field_value(
            risk_factors,
            source_lookup_by_type.get("annual_report") or source_lookup_by_type.get("strategic_plan") or {},
            confidence="medium",
            evidence_excerpt=", ".join(risk_factors[:5]),
        )

    strategic_profile_candidate = {
        "description": _clean_text_summary(strategic_analysis.get("strategic_summary") or ""),
        "industries": [str(snapshot.get("industry_classification") or "").strip()] if str(snapshot.get("industry_classification") or "").strip() else [],
        "key_themes": list(strategic_analysis.get("themes") or [])[:6],
        "strategic_objectives": unique_priorities[:6],
        "updated_at": _utc_now_iso(),
    }

    blocked_urls = [
        str(item.get("final_url") or item.get("input_url") or "").strip()
        for item in debug_info.get("ingest_results", [])
        if str(item.get("status") or "").strip().lower() == "failed"
        and "forbidden" in str(item.get("reason") or "").lower()
    ]
    manual_document_prompt: Dict[str, Any] = {}
    if not auto_apply and not proposed and not leadership_candidates:
        manual_document_prompt = _build_manual_document_prompt(
            target_org_name=target_org_name,
            requested_docs=requested_docs,
            blocked_urls=blocked_urls,
        )
        logger.warning(
            "org_profile_refresh manual document prompt target=%s blocked_urls=%s",
            target_org_name,
            blocked_urls,
        )

    sync_payload = {
        "org_name": str(payload.get("org_name") or "").strip(),
        "profiles": [
            {
                "canonical_name": item["name"],
                "target_type": "person",
                "current_employer": item.get("current_employer") or target_org_name,
                "current_role": item.get("current_role") or "",
            }
            for item in leadership_candidates
        ],
        "org_strategic_profile": strategic_profile_candidate,
        "source_system": "org_profile_refresh",
    }

    if progress_cb:
        progress_cb(100, "Organisation refresh analysis complete", "done")
    debug_info["warning_count"] = len(warnings)
    logger.info(
        "org_profile_refresh complete target=%s auto_apply=%d proposed=%d leadership_candidates=%d discovered_sources=%d warnings=%d",
        target_org_name,
        len(auto_apply),
        len(proposed),
        len(leadership_candidates),
        len(processed_sources or discovered_sources),
        len(warnings),
    )
    if warnings:
        logger.warning("org_profile_refresh warnings target=%s: %s", target_org_name, warnings)

    return {
        "status": "refreshed",
        "profile_id": str(payload.get("profile_id") or "").strip(),
        "org_name": str(payload.get("org_name") or "").strip(),
        "target_org_name": target_org_name,
        "discovery_mode": str(payload.get("discovery_mode") or "official_sources_first"),
        "auto_apply": auto_apply,
        "proposed": proposed,
        "leadership_candidates": leadership_candidates,
        "discovered_sources": [
            {
                "type": str(item.get("type") or ""),
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "year": str(item.get("year") or ""),
                "source_label": str(item.get("source_label") or ""),
                "document_meta": dict(item.get("document_meta") or {}),
            }
            for item in processed_sources or discovered_sources
        ],
        "warnings": warnings,
        "source_count": len(processed_sources or discovered_sources),
        "discovery_debug": debug_info,
        **manual_document_prompt,
        "strategic_profile_candidate": strategic_profile_candidate,
        "sync_payload": sync_payload,
    }
