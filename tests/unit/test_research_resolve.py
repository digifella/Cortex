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
