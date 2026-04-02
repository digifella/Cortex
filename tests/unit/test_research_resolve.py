from __future__ import annotations

from cortex_engine.research_resolve import (
    ResearchResolver,
    build_research_preferred_url_list,
    parse_research_spreadsheet_text,
)


class _FakeResolver(ResearchResolver):
    def __init__(self, *, responses, options=None):
        super().__init__(options=options or {}, sleep_fn=lambda _seconds: None)
        self.responses = responses

    def _request_json(self, url, *, params=None, source):  # type: ignore[override]
        key = (url, tuple(sorted((params or {}).items())), source)
        return self.responses.get(key)


class _FakeHtmlResponse:
    def __init__(self, *, url, status_code=200, headers=None, text=""):
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def close(self):
        return None


class _FakePdfResponse(_FakeHtmlResponse):
    pass


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None, allow_redirects=True, stream=False, headers=None):
        self.calls.append(
            {
                "url": url,
                "params": params,
                "timeout": timeout,
                "allow_redirects": allow_redirects,
                "stream": stream,
                "headers": headers or {},
            }
        )
        assert self.responses, f"Unexpected GET for {url}"
        response = self.responses.pop(0)
        return response


def test_research_resolve_doi_direct_builds_enriched_output(monkeypatch):
    monkeypatch.setattr(
        "cortex_engine.research_resolve.classify_journal_authority",
        lambda title, text: {
            "journal_title": "Current Hematology Malignancy Reports",
            "journal_issn": "1943-0108",
            "journal_quartile": "Q2",
            "journal_sjr": 0.876,
            "journal_rank_global": 2341,
        },
    )

    doi = "10.1007/s11899-020-00562-9"
    responses = {
        (
            f"https://api.crossref.org/works/10.1007%2Fs11899-020-00562-9",
            (),
            "crossref",
        ): {
            "message": {
                "DOI": doi,
                "URL": f"https://doi.org/{doi}",
                "title": ["Patient-Reported Outcomes in Myelodysplastic Syndromes"],
                "publisher": "Springer",
                "type": "journal-article",
                "container-title": ["Current Hematology Malignancy Reports"],
                "ISSN": ["1943-0108"],
            }
        },
        (
            f"https://api.unpaywall.org/v2/10.1007%2Fs11899-020-00562-9",
            (("email", "paul@example.com"),),
            "unpaywall",
        ): {
            "is_oa": True,
            "oa_status": "green",
            "best_oa_location": {"url_for_pdf": "https://europepmc.org/articles/PMC123/pdf"},
        },
    }
    resolver = _FakeResolver(
        responses=responses,
        options={
            "check_open_access": True,
            "enrich_sjr": True,
            "unpaywall_email": "paul@example.com",
        },
    )

    result = resolver.resolve_one(
        {
            "row_id": 1,
            "title": "Patient-Reported Outcomes in Myelodysplastic Syndromes",
            "authors": "Barot, S. V.; Patel, B. J.",
            "year": "2020",
            "doi": doi,
            "journal": "Curr Hematol Malig Rep",
        }
    )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert payload["source_api"] == "crossref_doi"
    assert payload["resolution_method"] == "doi_direct"
    assert payload["confidence"] == "high"
    assert payload["open_access"]["is_oa"] is True
    assert payload["open_access"]["pdf_url"] == "https://europepmc.org/articles/PMC123/pdf"
    assert payload["journal"]["sjr_quartile"] == "Q2"
    assert payload["journal"]["sjr_rank"] == 2341
    assert payload["credibility_tier"] == "peer-reviewed"


def test_research_resolve_title_search_can_return_low_confidence(monkeypatch):
    monkeypatch.setattr(
        "cortex_engine.research_resolve._title_similarity",
        lambda left, right: 0.82 if "Candidate Match" in right else 0.42,
    )
    monkeypatch.setattr(
        "cortex_engine.research_resolve.classify_journal_authority",
        lambda title, text: {
            "journal_title": title,
            "journal_issn": "1234-5678",
            "journal_quartile": "",
            "journal_sjr": 0.0,
            "journal_rank_global": 0,
        },
    )

    responses = {
        (
            "https://api.crossref.org/works",
            (("query.author", "Barot"), ("query.title", "Target title"), ("rows", 5)),
            "crossref",
        ): {
            "message": {
                "items": [
                    {
                        "DOI": "10.1000/candidate",
                        "URL": "https://doi.org/10.1000/candidate",
                        "title": ["Candidate Match for Target title"],
                        "publisher": "Example Publisher",
                        "type": "journal-article",
                        "container-title": ["Journal of Testing"],
                        "ISSN": ["1234-5678"],
                        "issued": {"date-parts": [[2020]]},
                    }
                ]
            }
        }
    }
    resolver = _FakeResolver(
        responses=responses,
        options={"check_open_access": False, "enrich_sjr": True},
    )

    result = resolver.resolve_one(
        {
            "row_id": 2,
            "title": "Target title",
            "authors": "Barot, S. V.",
            "year": "2020",
            "journal": "Journal of Testing",
        }
    )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert payload["source_api"] == "crossref_title_author"
    assert payload["resolution_method"] == "title_author_search"
    assert payload["confidence"] == "low"
    assert payload["title_similarity"] == 0.85


def test_research_resolve_returns_unresolved_with_candidates(monkeypatch):
    monkeypatch.setattr(
        "cortex_engine.research_resolve._title_similarity",
        lambda left, right: 0.65 if "Nearest Candidate" in right else 0.31,
    )

    responses = {
        (
            "https://api.crossref.org/works",
            (("query.author", "Smith"), ("query.title", "Unmatched title"), ("rows", 5)),
            "crossref",
        ): {"message": {"items": []}},
        (
            "https://api.crossref.org/works",
            (("query", "Unmatched title"), ("rows", 5)),
            "crossref",
        ): {
            "message": {
                "items": [
                    {"title": ["Nearest Candidate"], "DOI": "10.1000/nearest"},
                    {"title": ["Other Candidate"], "DOI": "10.1000/other"},
                ]
            }
        },
        (
            "https://api.crossref.org/works",
            (("query", "Unmatched title"), ("rows", 3)),
            "crossref",
        ): {
            "message": {
                "items": [
                    {"title": ["Nearest Candidate"], "DOI": "10.1000/nearest"},
                    {"title": ["Other Candidate"], "DOI": "10.1000/other"},
                ]
            }
        },
    }
    resolver = _FakeResolver(responses=responses, options={"check_open_access": False, "enrich_sjr": False})

    result = resolver.resolve_one(
        {
            "row_id": 7,
            "title": "Unmatched title",
            "authors": "Smith, J.",
            "year": "2021",
        }
    )

    payload = result["payload"]
    assert result["status"] == "unresolved"
    assert payload["reason"] == "No CrossRef match found"
    assert payload["best_candidates"][0]["title"] == "Nearest Candidate"
    assert payload["best_candidates"][0]["similarity"] == 0.65


def test_parse_research_spreadsheet_text_detects_columns_and_dedupes_doi():
    parsed = parse_research_spreadsheet_text(
        "\n".join(
            [
                "Article Title\tAuthor(s)\tPublished Year\tDOI\tJournal\tStudy",
                "Paper One\tSmith, J.\t2021\t10.1000/example\tJournal A\tBarot 2021",
                "Paper Two\tJones, A.\t2022\t10.1000/example\tJournal B\tBarot 2022",
                "Paper Three\t\t2023\t\tJournal C\tBarot 2023",
            ]
        )
    )

    assert parsed["detected_fields"][:5] == ["title", "authors", "year", "doi", "journal"]
    assert len(parsed["citations"]) == 2
    assert parsed["citations"][0]["notes"] == "Barot 2021"
    assert parsed["citations"][0]["preview_confidence"] == "green"
    assert parsed["citations"][1]["preview_confidence"] == "red"
    assert "duplicate DOI" in parsed["warnings"][0]


def test_build_research_preferred_url_list_prefers_pdf_urls():
    urls = build_research_preferred_url_list(
        [
            {
                "resolved_url": "https://doi.org/10.1000/one",
                "open_access": {"pdf_url": "https://example.org/one.pdf"},
            },
            {
                "resolved_url": "https://doi.org/10.1000/two",
                "open_access": {"pdf_url": ""},
            },
        ]
    )

    assert urls == ["https://example.org/one.pdf", "https://doi.org/10.1000/two"]


def test_open_access_falls_back_to_publisher_pdf_when_unpaywall_is_closed():
    doi = "10.3390/curroncol32050265"
    resolver = _FakeResolver(
        responses={
            (
                f"https://api.unpaywall.org/v2/{doi.replace('/', '%2F')}",
                (("email", "paul@example.com"),),
                "unpaywall",
            ): {
                "is_oa": False,
                "oa_status": "closed",
                "best_oa_location": None,
            }
        },
        options={"check_open_access": True, "unpaywall_email": "paul@example.com"},
    )
    resolver._publisher_page_open_access = lambda resolved_url: {  # type: ignore[method-assign]
        "is_oa": True,
        "oa_status": "publisher_free_pdf",
        "pdf_url": "https://www.mdpi.com/1718-7729/32/5/265/pdf?version=1746086728",
        "oa_source": "publisher_page",
    }

    oa = resolver._open_access_info(
        doi,
        "https://www.mdpi.com/1718-7729/32/5/265",
        {"publisher": "MDPI AG", "ISSN": ["1718-7729"], "volume": "32", "issue": "5", "page": "265"},
        {"volume": "32", "issue": "5", "pages": ""},
    )

    assert oa["is_oa"] is True
    assert oa["pdf_url"] == "https://www.mdpi.com/1718-7729/32/5/265/pdf?version=1746086728"
    assert oa["oa_status"] == "publisher_free_pdf"
    assert oa["oa_source"] == "unpaywall+publisher_page"


def test_publisher_page_probe_extracts_pdf_from_citation_meta():
    session = _FakeSession(
        [
            _FakeHtmlResponse(
                url="https://www.mdpi.com/1718-7729/32/5/265",
                headers={"Content-Type": "text/html; charset=utf-8"},
                text="<html></html>",
            ),
            _FakeHtmlResponse(
                url="https://www.mdpi.com/1718-7729/32/5/265",
                headers={"Content-Type": "text/html; charset=utf-8"},
                text="""
                <html><head>
                <meta name="citation_pdf_url" content="/1718-7729/32/5/265/pdf?version=1746086728">
                </head><body></body></html>
                """,
            ),
            _FakePdfResponse(
                url="https://www.mdpi.com/1718-7729/32/5/265/pdf?version=1746086728",
                headers={"Content-Type": "application/pdf"},
            ),
        ]
    )
    resolver = ResearchResolver(
        options={"check_open_access": True},
        session=session,
        sleep_fn=lambda _seconds: None,
    )

    oa = resolver._publisher_page_open_access("https://www.mdpi.com/1718-7729/32/5/265")

    assert oa["is_oa"] is True
    assert oa["oa_status"] == "publisher_free_pdf"
    assert oa["pdf_url"] == "https://www.mdpi.com/1718-7729/32/5/265/pdf?version=1746086728"
    assert oa["oa_source"] == "publisher_page"
    assert "text/html" in session.calls[1]["headers"].get("Accept", "")
    assert "application/pdf" in session.calls[2]["headers"].get("Accept", "")


def test_publisher_page_probe_accepts_explicit_oa_page_when_pdf_verification_fails():
    session = _FakeSession(
        [
            _FakeHtmlResponse(
                url="https://www.mdpi.com/1718-7729/32/5/265",
                headers={"Content-Type": "text/html; charset=utf-8"},
                text="<html></html>",
            ),
            _FakeHtmlResponse(
                url="https://www.mdpi.com/1718-7729/32/5/265",
                headers={"Content-Type": "text/html; charset=utf-8"},
                text="""
                <html><body>
                <a href="/1718-7729/32/5/265/pdf?version=1746086728">Download PDF</a>
                <div>This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license.</div>
                </body></html>
                """,
            ),
            _FakeHtmlResponse(
                url="https://www.mdpi.com/1718-7729/32/5/265/pdf?version=1746086728",
                status_code=429,
                headers={"Content-Type": "text/html; charset=utf-8"},
                text="rate limited",
            ),
        ]
    )
    resolver = ResearchResolver(
        options={"check_open_access": True},
        session=session,
        sleep_fn=lambda _seconds: None,
    )

    oa = resolver._publisher_page_open_access("https://www.mdpi.com/1718-7729/32/5/265")

    assert oa["is_oa"] is True
    assert oa["oa_status"] == "publisher_page_open_access"
    assert oa["pdf_url"] == "https://www.mdpi.com/1718-7729/32/5/265/pdf?version=1746086728"
    assert oa["oa_source"] == "publisher_page"


def test_open_access_uses_mdpi_synthesized_landing_page_when_doi_url_is_unhelpful():
    doi = "10.3390/curroncol32050265"
    resolver = _FakeResolver(
        responses={
            (
                f"https://api.unpaywall.org/v2/{doi.replace('/', '%2F')}",
                (("email", "paul@example.com"),),
                "unpaywall",
            ): {
                "is_oa": None,
                "oa_status": "unknown",
                "best_oa_location": None,
            }
        },
        options={"check_open_access": True, "unpaywall_email": "paul@example.com"},
    )
    seen = []

    def fake_publisher_page(url):
        seen.append(url)
        if url == "https://www.mdpi.com/1718-7729/32/5/265":
            return {
                "is_oa": True,
                "oa_status": "publisher_page_open_access",
                "pdf_url": "https://www.mdpi.com/1718-7729/32/5/265/pdf?version=1746086728",
                "oa_source": "publisher_page",
            }
        return {
            "is_oa": None,
            "oa_status": "unknown",
            "pdf_url": "",
            "oa_source": "",
        }

    resolver._publisher_page_open_access = fake_publisher_page  # type: ignore[method-assign]

    oa = resolver._open_access_info(
        doi,
        "https://doi.org/10.3390/curroncol32050265",
        {
            "publisher": "MDPI AG",
            "ISSN": ["1718-7729"],
            "volume": "32",
            "issue": "5",
            "page": "265",
        },
        {"volume": "32", "issue": "5", "pages": ""},
    )

    assert "https://doi.org/10.3390/curroncol32050265" in seen
    assert "https://www.mdpi.com/1718-7729/32/5/265" in seen
    assert oa["is_oa"] is True
    assert oa["oa_status"] == "publisher_page_open_access"
    assert oa["pdf_url"] == "https://www.mdpi.com/1718-7729/32/5/265/pdf?version=1746086728"


def test_open_access_uses_mdpi_publisher_policy_as_last_fallback():
    doi = "10.3390/curroncol32050265"
    resolver = _FakeResolver(
        responses={
            (
                f"https://api.unpaywall.org/v2/{doi.replace('/', '%2F')}",
                (("email", "paul@example.com"),),
                "unpaywall",
            ): {
                "is_oa": None,
                "oa_status": "unknown",
                "best_oa_location": None,
            }
        },
        options={"check_open_access": True, "unpaywall_email": "paul@example.com"},
    )
    resolver._publisher_page_open_access = lambda url: {  # type: ignore[method-assign]
        "is_oa": None,
        "oa_status": "unknown",
        "pdf_url": "",
        "oa_source": "",
    }

    oa = resolver._open_access_info(
        doi,
        "https://doi.org/10.3390/curroncol32050265",
        {
            "publisher": "MDPI AG",
            "ISSN": ["1718-7729"],
        },
        {"volume": "", "issue": "", "pages": ""},
    )

    assert oa["is_oa"] is True
    assert oa["oa_status"] == "publisher_policy_open_access"
    assert oa["pdf_url"] == "https://www.mdpi.com/1718-7729/32/5/265/pdf"
    assert oa["oa_source"] == "unpaywall+publisher_policy"


def test_open_access_uses_mdpi_policy_when_crossref_omits_issn_but_journal_name_exists(monkeypatch):
    doi = "10.3390/curroncol32050265"
    monkeypatch.setattr(
        "cortex_engine.research_resolve.classify_journal_authority",
        lambda title, text: {
            "journal_title": "Current Oncology",
            "journal_issn": "1718-7729",
            "journal_quartile": "",
            "journal_sjr": 0.0,
            "journal_rank_global": 0,
        },
    )
    resolver = _FakeResolver(
        responses={
            (
                f"https://api.unpaywall.org/v2/{doi.replace('/', '%2F')}",
                (("email", "paul@example.com"),),
                "unpaywall",
            ): {
                "is_oa": None,
                "oa_status": "unknown",
                "best_oa_location": None,
            }
        },
        options={"check_open_access": True, "unpaywall_email": "paul@example.com"},
    )
    resolver._publisher_page_open_access = lambda url: {  # type: ignore[method-assign]
        "is_oa": None,
        "oa_status": "unknown",
        "pdf_url": "",
        "oa_source": "",
    }

    oa = resolver._open_access_info(
        doi,
        "https://doi.org/10.3390/curroncol32050265",
        {
            "publisher": "MDPI AG",
            "container-title": ["Current Oncology"],
        },
        {"journal": "Current Oncology", "volume": "", "issue": "", "pages": ""},
    )

    assert oa["is_oa"] is True
    assert oa["oa_status"] == "publisher_policy_open_access"
    assert oa["pdf_url"] == "https://www.mdpi.com/1718-7729/32/5/265/pdf"


def test_open_access_uses_mdpi_policy_without_unpaywall_email():
    doi = "10.3390/curroncol32050265"
    resolver = _FakeResolver(
        responses={},
        options={"check_open_access": True, "unpaywall_email": ""},
    )
    resolver._publisher_page_open_access = lambda url: {  # type: ignore[method-assign]
        "is_oa": None,
        "oa_status": "unknown",
        "pdf_url": "",
        "oa_source": "",
    }

    oa = resolver._open_access_info(
        doi,
        "https://doi.org/10.3390/curroncol32050265",
        {
            "publisher": "MDPI AG",
            "ISSN": ["1718-7729"],
            "container-title": ["Current Oncology"],
        },
        {"journal": "Current Oncology", "volume": "", "issue": "", "pages": ""},
    )

    assert oa["is_oa"] is True
    assert oa["oa_status"] == "publisher_policy_open_access"
    assert oa["pdf_url"] == "https://www.mdpi.com/1718-7729/32/5/265/pdf"
