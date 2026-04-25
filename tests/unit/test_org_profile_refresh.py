from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cortex_engine.document_registry import build_document_meta
from cortex_engine.org_profile_refresh import OfficialSourceDiscoverer, _limit_relevant_sources, run_org_profile_refresh


class _FakeDocumentSignalStore:
    def __init__(self):
        self.document_records = {}

    def get_document_record(self, org_name, canonical_doc_key):
        return dict(self.document_records.get((org_name.lower(), str(canonical_doc_key or "").strip()), {})) or None

    def classify_document_meta(self, org_name, document_meta):
        meta = dict(document_meta or {})
        existing = self.get_document_record(org_name, meta.get("canonical_doc_key"))
        if existing:
            meta["status"] = "known_same" if str(existing.get("content_fingerprint") or "") == str(meta.get("content_fingerprint") or "") else "changed_document"
            meta["ingest_recommendation"] = "skip" if meta["status"] == "known_same" and str(meta.get("ingest_policy") or "") == "strict" else "ingest"
            meta["existing_record"] = existing
        else:
            meta["status"] = "new_document"
            meta["ingest_recommendation"] = "ingest"
            meta["existing_record"] = {}
        return meta

    def register_document_meta(self, org_name, document_meta, latest_trace_id="", latest_intel_id=""):
        meta = dict(document_meta or {})
        meta["org_name"] = org_name
        meta["latest_trace_id"] = latest_trace_id
        meta["latest_intel_id"] = latest_intel_id
        self.document_records[(org_name.lower(), str(meta.get("canonical_doc_key") or "").strip())] = meta
        return meta


def test_official_source_discoverer_uses_sitemap_links(monkeypatch):
    discoverer = OfficialSourceDiscoverer(timeout=5)

    def _fake_fetch_html(url, raise_on_error=True):
        if url == "https://barwonwater.vic.gov.au":
            return url, "<html><head><title>Barwon Water</title></head><body><a href='/about'>About</a></body></html>"
        if url.endswith("/about"):
            return url, "<html><head><title>About Barwon Water</title></head><body>About</body></html>"
        return url, ""

    def _fake_fetch_text(url, raise_on_error=True):
        if url.endswith("/sitemap.xml"):
            return url, """<?xml version='1.0' encoding='UTF-8'?>
            <urlset>
              <url><loc>https://barwonwater.vic.gov.au/reports/annual-report-2025.pdf</loc></url>
              <url><loc>https://barwonwater.vic.gov.au/our-strategy</loc></url>
            </urlset>""", "application/xml"
        return url, "", "text/html"

    monkeypatch.setattr(discoverer, "_fetch_html", _fake_fetch_html)
    monkeypatch.setattr(discoverer, "_fetch_text", _fake_fetch_text)

    sources = discoverer.discover_sources(
        website_url="https://barwonwater.vic.gov.au",
        requested_docs=["annual_report", "strategic_plan", "org_chart"],
        max_sources=6,
    )

    assert any(item["type"] == "annual_report" for item in sources)
    assert any(item["type"] == "strategic_plan" for item in sources)


def test_org_profile_refresh_builds_structured_output(monkeypatch, tmp_path):
    strategic_md = tmp_path / "barwon-strategy.md"
    strategic_md.write_text(
        "\n".join(
            [
                "# Barwon Water Strategic Plan 2025",
                "Mission",
                "Deliver reliable and sustainable water services for the region.",
                "Strategic priorities",
                "Drought resilience",
                "Customer stewardship",
                "Shaun Cumming",
                "Managing Director",
            ]
        ),
        encoding="utf-8",
    )
    org_chart_md = tmp_path / "barwon-org-chart.md"
    org_chart_md.write_text(
        "\n".join(
            [
                "Leadership Team",
                "Shaun Cumming",
                "Managing Director",
                "Anna Murray",
                "General Manager Strategy",
            ]
        ),
        encoding="utf-8",
    )

    class _FakeDiscoverer:
        def __init__(self, timeout: int = 25):
            self.timeout = timeout

        def discover_sources(self, website_url: str, requested_docs: list[str], target_org_name: str = "", max_sources: int = 6):
            return [
                {
                    "type": "strategic_plan",
                    "title": "Barwon Water Strategic Plan 2025",
                    "url": "https://barwonwater.vic.gov.au/strategy",
                    "year": "2025",
                    "source_label": "Strategic Plan",
                },
                {
                    "type": "org_chart",
                    "title": "Leadership Team",
                    "url": "https://barwonwater.vic.gov.au/leadership",
                    "year": "2025",
                    "source_label": "Leadership Team",
                },
            ]

    def _fake_process_urls(self, urls, convert_to_md, use_vision_for_md, textify_options, capture_web_md_on_no_pdf):
        return [
            SimpleNamespace(
                input_url="https://barwonwater.vic.gov.au/strategy",
                final_url="https://barwonwater.vic.gov.au/strategy",
                md_path=str(strategic_md),
                pdf_path="",
                status="web_markdown",
            ),
            SimpleNamespace(
                input_url="https://barwonwater.vic.gov.au/leadership",
                final_url="https://barwonwater.vic.gov.au/leadership",
                md_path=str(org_chart_md),
                pdf_path="",
                status="web_markdown",
            ),
        ]

    monkeypatch.setattr("cortex_engine.org_profile_refresh.OfficialSourceDiscoverer", _FakeDiscoverer)
    monkeypatch.setattr("cortex_engine.org_profile_refresh.URLIngestor.process_urls", _fake_process_urls)

    output = run_org_profile_refresh(
        payload={
            "profile_id": "123",
            "org_name": "Escient",
            "target_org_name": "Barwon Water",
            "website_url": "https://barwonwater.vic.gov.au",
            "current_profile_snapshot": {"website_url": ""},
            "requested_docs": ["strategic_plan", "org_chart"],
            "max_sources": 4,
            "timeout_seconds": 10,
            "use_vision": True,
            "discovery_mode": "official_sources_first",
        },
        run_dir=Path(tmp_path),
        signal_store=_FakeDocumentSignalStore(),
    )

    assert output["status"] == "refreshed"
    assert output["auto_apply"]["website_url"]["value"] == "https://barwonwater.vic.gov.au"
    assert output["proposed"]["mission_statement"]["value"] == "Deliver reliable and sustainable water services for the region."
    assert "strategic_priorities" in output["proposed"]
    assert any(item["name"] == "Shaun Cumming" for item in output["leadership_candidates"])
    assert len(output["discovered_sources"]) == 2


def test_org_profile_refresh_returns_manual_document_prompt_when_sources_blocked(monkeypatch, tmp_path):
    class _FakeDiscoverer:
        def __init__(self, timeout: int = 25):
            self.timeout = timeout

        def discover_sources(self, website_url: str, requested_docs: list[str], target_org_name: str = "", max_sources: int = 6):
            return [
                {
                    "type": "about_page",
                    "title": "Barwon Water",
                    "url": "https://www.barwonwater.vic.gov.au/",
                    "year": "",
                    "source_label": "Organisation website",
                }
            ]

    def _fake_process_urls(self, urls, convert_to_md, use_vision_for_md, textify_options, capture_web_md_on_no_pdf):
        return [
            SimpleNamespace(
                input_url="https://www.barwonwater.vic.gov.au/",
                final_url="https://www.barwonwater.vic.gov.au/",
                md_path="",
                pdf_path="",
                status="failed",
                reason="paywalled_or_forbidden",
            )
        ]

    monkeypatch.setattr("cortex_engine.org_profile_refresh.OfficialSourceDiscoverer", _FakeDiscoverer)
    monkeypatch.setattr("cortex_engine.org_profile_refresh.URLIngestor.process_urls", _fake_process_urls)

    output = run_org_profile_refresh(
        payload={
            "profile_id": "123",
            "org_name": "Escient",
            "target_org_name": "Barwon Water",
            "website_url": "https://www.barwonwater.vic.gov.au/",
            "current_profile_snapshot": {"website_url": "https://www.barwonwater.vic.gov.au/"},
            "requested_docs": ["annual_report", "strategic_plan", "org_chart"],
            "max_sources": 4,
            "timeout_seconds": 10,
            "use_vision": True,
            "discovery_mode": "official_sources_first",
        },
        run_dir=Path(tmp_path),
        signal_store=_FakeDocumentSignalStore(),
    )

    assert output["requires_manual_documents"] is True
    assert "email or upload" in output["user_message"].lower()
    assert output["blocked_source_urls"] == ["https://www.barwonwater.vic.gov.au/"]


def test_org_profile_refresh_prefers_current_target_sources_over_predecessor_reports(monkeypatch, tmp_path):
    filtered = _limit_relevant_sources(
        [
            {
                "type": "annual_report",
                "title": "Greater Western Water Annual Report 2024-25",
                "url": "https://www.gww.com.au/sites/default/files/2025-11/GWW_Annual_Report_2024-25.pdf",
                "year": "2025",
                "source_label": "Greater Western Water Annual Report 2024-25",
            },
            {
                "type": "annual_report",
                "title": "Annual reports | Greater Western Water",
                "url": "https://www.gww.com.au/about/corporate-information/our-strategies-plans-reports/annual-reports",
                "year": "2025",
                "source_label": "Annual reports | Greater Western Water",
            },
            {
                "type": "annual_report",
                "title": "City West Water Annual Report 2015",
                "url": "https://www.gww.com.au/sites/default/files/2022-09/City_West_Water_Annual_Report_2015.pdf",
                "year": "2015",
                "source_label": "City West Water Annual Report 2015",
            },
        ],
        target_org_name="Greater Western Water",
        max_sources=6,
    )
    filtered_urls = [item["url"] for item in filtered]

    assert all("City_West_Water" not in url for url in filtered_urls)
    assert any("GWW_Annual_Report_2024-25.pdf" in url for url in filtered_urls)

    current_md = tmp_path / "greater-western-water-annual-report.md"
    current_md.write_text(
        "\n".join(
            [
                "Greater Western Water Annual Report 2024-25",
                "Greater Western Water is a Victorian Government water corporation.",
                "Lisa Neville",
                "Chair",
                "Greater Western Water",
            ]
        ),
        encoding="utf-8",
    )
    listing_md = tmp_path / "annual-reports-page.md"
    listing_md.write_text(
        "\n".join(
            [
                "Annual reports | Greater Western Water",
                "Greater Western Water's Annual Report for 2024-25",
            ]
        ),
        encoding="utf-8",
    )

    class _FakeDiscoverer:
        def __init__(self, timeout: int = 25):
            self.timeout = timeout

        def discover_sources(self, website_url: str, requested_docs: list[str], target_org_name: str = "", max_sources: int = 6):
            return [
                {
                    "type": "annual_report",
                    "title": "Greater Western Water Annual Report 2024-25",
                    "url": "https://www.gww.com.au/sites/default/files/2025-11/GWW_Annual_Report_2024-25.pdf",
                    "year": "2025",
                    "source_label": "Greater Western Water Annual Report 2024-25",
                },
                {
                    "type": "annual_report",
                    "title": "Annual reports | Greater Western Water",
                    "url": "https://www.gww.com.au/about/corporate-information/our-strategies-plans-reports/annual-reports",
                    "year": "2025",
                    "source_label": "Annual reports | Greater Western Water",
                },
            ]

    def _fake_process_urls(self, urls, convert_to_md, use_vision_for_md, textify_options, capture_web_md_on_no_pdf):
        return [
            SimpleNamespace(
                input_url="https://www.gww.com.au/sites/default/files/2025-11/GWW_Annual_Report_2024-25.pdf",
                final_url="https://www.gww.com.au/sites/default/files/2025-11/GWW_Annual_Report_2024-25.pdf",
                md_path=str(current_md),
                pdf_path="",
                status="downloaded",
            ),
            SimpleNamespace(
                input_url="https://www.gww.com.au/about/corporate-information/our-strategies-plans-reports/annual-reports",
                final_url="https://www.gww.com.au/about/corporate-information/our-strategies-plans-reports/annual-reports",
                md_path=str(listing_md),
                pdf_path="",
                status="web_markdown",
            ),
        ]

    monkeypatch.setattr("cortex_engine.org_profile_refresh.OfficialSourceDiscoverer", _FakeDiscoverer)
    monkeypatch.setattr("cortex_engine.org_profile_refresh.URLIngestor.process_urls", _fake_process_urls)

    output = run_org_profile_refresh(
        payload={
            "profile_id": "456",
            "org_name": "Escient",
            "target_org_name": "Greater Western Water",
            "website_url": "https://www.gww.com.au/",
            "current_profile_snapshot": {"website_url": ""},
            "requested_docs": ["annual_report", "strategic_plan", "org_chart"],
            "max_sources": 6,
            "timeout_seconds": 10,
            "use_vision": True,
            "discovery_mode": "official_sources_first",
        },
        run_dir=Path(tmp_path),
        signal_store=_FakeDocumentSignalStore(),
    )

    discovered_urls = [item["url"] for item in output["discovered_sources"]]
    assert any("GWW_Annual_Report_2024-25.pdf" in url for url in discovered_urls)
    assert output["discovery_debug"]["processed_sources"][0]["filename"].startswith("Greater Western Water Annual Report 2024-25")


def test_org_profile_refresh_marks_known_same_documents_on_second_run(monkeypatch, tmp_path):
    annual_md = tmp_path / "barwon-annual-report.md"
    annual_md.write_text(
        "\n".join(
            [
                "Barwon Water Annual Report 2025",
                "Revenue for the year increased to $344.2 million, up from $292.8 million.",
                "Shaun Cumming",
                "Managing Director",
            ]
        ),
        encoding="utf-8",
    )

    class _FakeDiscoverer:
        def __init__(self, timeout: int = 25):
            self.timeout = timeout

        def discover_sources(self, website_url: str, requested_docs: list[str], target_org_name: str = "", max_sources: int = 6):
            return [
                {
                    "type": "annual_report",
                    "title": "Barwon Water Annual Report 2025",
                    "url": "https://barwonwater.vic.gov.au/reports/annual-report-2025.pdf",
                    "year": "2025",
                    "source_label": "Annual Report 2025",
                }
            ]

    def _fake_process_urls(self, urls, convert_to_md, use_vision_for_md, textify_options, capture_web_md_on_no_pdf):
        return [
            SimpleNamespace(
                input_url="https://barwonwater.vic.gov.au/reports/annual-report-2025.pdf",
                final_url="https://barwonwater.vic.gov.au/reports/annual-report-2025.pdf",
                md_path=str(annual_md),
                pdf_path="",
                status="downloaded",
            )
        ]

    signal_store = _FakeDocumentSignalStore()
    monkeypatch.setattr("cortex_engine.org_profile_refresh.OfficialSourceDiscoverer", _FakeDiscoverer)
    monkeypatch.setattr("cortex_engine.org_profile_refresh.URLIngestor.process_urls", _fake_process_urls)

    payload = {
        "profile_id": "123",
        "org_name": "Escient",
        "target_org_name": "Barwon Water",
        "website_url": "https://barwonwater.vic.gov.au",
        "current_profile_snapshot": {"website_url": ""},
        "requested_docs": ["annual_report"],
        "max_sources": 2,
        "timeout_seconds": 10,
        "use_vision": True,
        "discovery_mode": "official_sources_first",
    }

    first = run_org_profile_refresh(payload=payload, run_dir=Path(tmp_path / "run1"), signal_store=signal_store)
    second = run_org_profile_refresh(payload=payload, run_dir=Path(tmp_path / "run2"), signal_store=signal_store)

    assert first["discovered_sources"][0]["document_meta"]["status"] == "new_document"
    assert first["discovered_sources"][0]["document_meta"]["ingest_recommendation"] == "ingest"
    assert second["discovered_sources"][0]["document_meta"]["status"] == "known_same"
    assert second["discovered_sources"][0]["document_meta"]["ingest_recommendation"] == "skip"
