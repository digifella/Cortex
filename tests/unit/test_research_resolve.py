from __future__ import annotations

import json

from cortex_engine.research_resolve import (
    ResearchResolver,
    build_research_preferred_url_list,
    parse_research_spreadsheet_text,
    run_research_resolve,
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


def test_research_resolve_accepts_bibliography_resolved_rows_without_crossref():
    resolver = _FakeResolver(
        responses={},
        options={"check_open_access": False, "enrich_sjr": False},
    )

    result = resolver.resolve_one(
        {
            "row_id": 5,
            "title": "Study evaluating the safety and efficacy of KTE-C19 in adult participants with refractory aggressive non-Hodgkin lymphoma (ZUMA-1)",
            "authors": "ClinicalTrials.gov",
            "year": "2015",
            "journal": "ClinicalTrials.gov",
            "extra_fields": {
                "reference_number": "70",
                "bibliography_match_method": "reference_number",
                "bibliography_entry_text": "ClinicalTrials.gov. Study evaluating the safety and efficacy of KTE-C19 in adult participants with refractory aggressive non-Hodgkin lymphoma (ZUMA-1). 2015. https://clinicaltrials.gov/ct2/show/NCT02348216",
            },
        }
    )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert payload["source_api"] == "review_bibliography"
    assert payload["resolution_method"] == "bibliography_reference_number"
    assert payload["confidence"] == "high"
    assert payload["matched_title"].startswith("Study evaluating the safety and efficacy of KTE-C19")
    assert payload["resolved_url"] == "https://clinicaltrials.gov/ct2/show/NCT02348216"
    assert payload["type"] == "registry"


def test_research_resolve_uses_doi_url_for_bibliography_resolved_rows():
    resolver = _FakeResolver(
        responses={},
        options={"check_open_access": False, "enrich_sjr": False},
    )

    result = resolver.resolve_one(
        {
            "row_id": 2,
            "title": "Health-related quality of life and utility outcomes with selinexor in relapsed/refractory diffuse large B-cell lymphoma",
            "authors": "Shah J, Shacham S, Kauffman M, et al.",
            "year": "2021",
            "journal": "Future Oncology",
            "doi": "10.2217/fon-2020-1264",
            "extra_fields": {
                "reference_number": "38",
                "bibliography_match_method": "reference_number",
                "bibliography_entry_text": "Shah J, Shacham S, Kauffman M, et al. Health-related quality of life and utility outcomes with selinexor in relapsed/refractory diffuse large B-cell lymphoma. Future Oncol. 2021;17(11):1295-310.",
            },
        }
    )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert payload["source_api"] == "review_bibliography"
    assert payload["resolved_doi"] == "10.2217/fon-2020-1264"
    assert payload["resolved_url"] == "https://doi.org/10.2217/fon-2020-1264"


def test_research_resolve_normalizes_wrapped_bibliography_text_and_urls():
    resolver = _FakeResolver(
        responses={},
        options={"check_open_access": False, "enrich_sjr": False},
    )

    result = resolver.resolve_one(
        {
            "row_id": 8,
            "title": "Health-related qual- ity of life and utility outcomes with selinexor in relapsed/ refractory diffuse large B-cell lymphoma",
            "authors": "Shah J",
            "year": "2021",
            "journal": "Health-related qual- ity of life and utility outcomes with selinexor in relapsed/ refractory diffuse large B-cell lymphoma",
            "extra_fields": {
                "reference_number": "70",
                "bibliography_match_method": "reference_number",
                "bibliography_entry_text": "ClinicalTrials.gov. Study evaluating the safety and efficacy of KTE-C19 in adult participants with refractory aggressive non-\nHodgkin lymphoma (ZUMA-1 (NCT02348216). 2015. https://\nclini\u200bcaltr\u200bials.\u200bgov/\u200bct2/\u200bshow/\u200bNCT02\u200b348216. Accessed 13 Dec 2022.",
            },
        }
    )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert payload["matched_title"].startswith("Study evaluating the safety and efficacy of KTE-C19")
    assert "non-Hodgkin lymphoma" in payload["matched_title"]
    assert payload["resolved_url"] == "https://clinicaltrials.gov/ct2/show/NCT02348216"


def test_research_resolve_normalizes_inline_wrapped_word_artifacts():
    resolver = _FakeResolver(
        responses={},
        options={"check_open_access": False, "enrich_sjr": False},
    )

    result = resolver.resolve_one(
        {
            "row_id": 9,
            "title": "Health-related qual- ity of life and utility outcomes with selinexor in relapsed/ refractory diffuse large B-cell lymphoma",
            "authors": "Shah J",
            "year": "2021",
            "journal": "Health-related qual- ity of life and utility outcomes with selinexor in relapsed/ refractory diffuse large B-cell lymphoma",
            "extra_fields": {
                "reference_number": "38",
                "bibliography_match_method": "reference_number",
                "bibliography_entry_text": "Shah J, Shacham S, Kauffman M, et al. Health-related qual- ity of life and utility outcomes with selinexor in relapsed/ refractory diffuse large B-cell lymphoma. Future Oncol. 2021;17(11):1295-310.",
            },
        }
    )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert "qual- ity" not in payload["matched_title"]
    assert "relapsed/ refractory" not in payload["matched_title"]
    assert payload["matched_title"] == "Health-related quality of life and utility outcomes with selinexor in relapsed/refractory diffuse large B-cell lymphoma"


def test_research_resolve_repairs_mojibake_punctuation_in_bibliography_entries():
    resolver = _FakeResolver(
        responses={},
        options={"check_open_access": False, "enrich_sjr": False},
    )

    result = resolver.resolve_one(
        {
            "row_id": 9,
            "title": "Casasnovas 2020 [69]",
            "authors": "Casasnovas RO",
            "year": "2020",
            "journal": "",
            "extra_fields": {
                "reference_number": "69",
                "bibliography_match_method": "reference_number",
                "bibliography_entry_text": "Casasnovas RO, Daniele P, Tremblay G, et al. PCN325 Health utility in relapsed/refractory diffuse large B-cell lymphoma (RR-DLBCL) patientsâ€”results of a phase II trial with ORAL selinexor. Value Health. 2020;23(suppl 2):S479-80.",
            },
        }
    )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert "â€”" not in payload["matched_title"]
    assert payload["matched_title"] == "PCN325 Health utility in relapsed/refractory diffuse large B-cell lymphoma (RR-DLBCL) patients-results of a phase II trial with ORAL selinexor"
    assert payload["journal"]["name"] == "Value Health"


def test_research_resolve_bibliography_parsing_keeps_vs_titles_intact():
    resolver = _FakeResolver(
        responses={},
        options={"check_open_access": False, "enrich_sjr": False},
    )

    result = resolver.resolve_one(
        {
            "row_id": 11,
            "title": "Li 2022 [54]",
            "authors": "Li Y",
            "year": "2022",
            "journal": "",
            "extra_fields": {
                "reference_number": "54",
                "bibliography_match_method": "reference_number",
                "bibliography_entry_text": "Li Y, et al. Cost-effectiveness analysis of axicabtagene ciloleucel vs. salvage chemotherapy for relapsed or refractory adult diffuse large B-cell lymphoma in China. Front Public Health. 2022;10:123.",
            },
        }
    )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert payload["matched_title"] == "Cost-effectiveness analysis of axicabtagene ciloleucel vs salvage chemotherapy for relapsed or refractory adult diffuse large B-cell lymphoma in China"
    assert payload["journal"]["name"] == "Front Public Health"


def test_research_resolve_backfills_doi_url_for_bibliography_rows_without_identifiers(monkeypatch):
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
            (("query.author", "Shah"), ("query.title", "Health-related quality of life and utility outcomes with selinexor in relapsed/refractory diffuse large B-cell lymphoma"), ("rows", 5)),
            "crossref",
        ): {
            "message": {
                "items": [
                    {
                        "DOI": "10.2217/fon-2020-1264",
                        "URL": "https://doi.org/10.2217/fon-2020-1264",
                        "title": ["Health-related quality of life and utility outcomes with selinexor in relapsed/refractory diffuse large B-cell lymphoma"],
                        "publisher": "Future Oncol",
                        "type": "journal-article",
                        "container-title": ["Future Oncol"],
                        "ISSN": ["1234-5678"],
                        "issued": {"date-parts": [[2021]]},
                        "author": [{"family": "Shah", "given": "J."}],
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
            "row_id": 10,
            "title": "Shah 2021 [38]",
            "authors": "Shah J",
            "year": "2021",
            "journal": "",
            "extra_fields": {
                "reference_number": "38",
                "bibliography_match_method": "reference_number",
                "bibliography_entry_text": "Shah J, Shacham S, Kauffman M, et al. Health-related quality of life and utility outcomes with selinexor in relapsed/refractory diffuse large B-cell lymphoma. Future Oncol. 2021;17(11):1295-310.",
            },
        }
    )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert payload["source_api"] == "review_bibliography"
    assert payload["resolved_doi"] == "10.2217/fon-2020-1264"
    assert payload["resolved_url"] == "https://doi.org/10.2217/fon-2020-1264"
    assert payload["resolution_method"] == "bibliography_reference_number+title_author_search"
    assert payload["backfill_source_api"] == "crossref_title_author"


def test_research_resolve_rejects_backfill_with_wrong_journal(monkeypatch):
    monkeypatch.setattr(
        "cortex_engine.research_resolve.classify_journal_authority",
        lambda title, text: {
            "journal_title": title,
            "journal_issn": "",
            "journal_quartile": "",
            "journal_sjr": 0.0,
            "journal_rank_global": 0,
        },
    )

    responses = {
        (
            "https://api.crossref.org/works",
            (("query.author", "Lin"), ("query.title", "Cost-effectiveness of chimeric antigen receptor T-cell therapy in multiply relapsed or refractory adult large B-cell lymphoma"), ("rows", 5)),
            "crossref",
        ): {
            "message": {
                "items": [
                    {
                        "DOI": "10.1016/j.carbpol.2019.04.053",
                        "URL": "https://doi.org/10.1016/j.carbpol.2019.04.053",
                        "title": ["Cost-effectiveness of chimeric antigen receptor T-cell therapy in multiply relapsed or refractory adult large B-cell lymphoma"],
                        "publisher": "Elsevier BV",
                        "type": "journal-article",
                        "container-title": ["Carbohydrate Polymers"],
                        "ISSN": ["0008-6215"],
                        "issued": {"date-parts": [[2019]]},
                        "author": [{"family": "Lin", "given": "VW"}],
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
            "row_id": 12,
            "title": "Lin 2019 [50]",
            "authors": "Lin VW",
            "year": "2019",
            "journal": "",
            "extra_fields": {
                "reference_number": "50",
                "bibliography_match_method": "reference_number",
                "bibliography_entry_text": "Lin VW, et al. Cost-effectiveness of chimeric antigen receptor T-cell therapy in multiply relapsed or refractory adult large B-cell lymphoma. J Med Econ. 2019;22(1):100-110.",
            },
        }
    )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert payload["source_api"] == "review_bibliography"
    assert payload["resolved_doi"] == ""
    assert payload["resolved_url"] == ""
    assert payload["journal"]["name"] == "J Med Econ"
    assert payload["resolution_method"] == "bibliography_reference_number"


def test_research_resolve_rejects_backfill_without_plausible_type(monkeypatch):
    monkeypatch.setattr(
        "cortex_engine.research_resolve.classify_journal_authority",
        lambda title, text: {
            "journal_title": title,
            "journal_issn": "",
            "journal_quartile": "",
            "journal_sjr": 0.0,
            "journal_rank_global": 0,
        },
    )

    responses = {
        (
            "https://api.crossref.org/works",
            (("query.author", "Roth"), ("query.title", "Cost-effectiveness of axicabtagene ciloleucel for adult patients with relapsed or refractory large B-cell lymphoma in the United States"), ("rows", 5)),
            "crossref",
        ): {
            "message": {
                "items": [
                    {
                        "DOI": "10.4324/9780429494208",
                        "URL": "https://doi.org/10.4324/9780429494208",
                        "title": ["Cost-effectiveness of axicabtagene ciloleucel for adult patients with relapsed or refractory large B-cell lymphoma in the United States"],
                        "publisher": "Routledge",
                        "type": "book",
                        "container-title": ["J Med Econ"],
                        "issued": {"date-parts": [[2018]]},
                        "author": [{"family": "Roth", "given": "JA"}],
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
            "row_id": 13,
            "title": "Roth 2018 [51]",
            "authors": "Roth JA",
            "year": "2018",
            "journal": "",
            "extra_fields": {
                "reference_number": "51",
                "bibliography_match_method": "reference_number",
                "bibliography_entry_text": "Roth JA, et al. Cost-effectiveness of axicabtagene ciloleucel for adult patients with relapsed or refractory large B-cell lymphoma in the United States. J Med Econ. 2018;21(12):1234-1240.",
            },
        }
    )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert payload["resolved_doi"] == ""
    assert payload["resolved_url"] == ""
    assert payload["journal"]["name"] == "J Med Econ"
    assert payload["resolution_method"] == "bibliography_reference_number"


def test_research_resolve_author_year_citation_label_uses_bibliographic_search(monkeypatch):
    monkeypatch.setattr(
        "cortex_engine.research_resolve._title_similarity",
        lambda left, right: 0.18,
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
            (
                ("filter", "from-pub-date:2019-01-01,until-pub-date:2019-12-31"),
                ("query.author", "Zinzani"),
                ("query.bibliographic", "Zinzani 2019 CheckMate 436"),
                ("rows", 5),
            ),
            "crossref",
        ): {
            "message": {
                "items": [
                    {
                        "DOI": "10.1000/checkmate436",
                        "URL": "https://doi.org/10.1000/checkmate436",
                        "title": ["Nivolumab plus brentuximab vedotin for relapsed/refractory PMBCL: CheckMate 436"],
                        "publisher": "Example Publisher",
                        "type": "journal-article",
                        "container-title": ["Journal of Testing"],
                        "ISSN": ["1234-5678"],
                        "issued": {"date-parts": [[2019]]},
                        "author": [{"family": "Zinzani", "given": "P. L."}],
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
            "row_id": 11,
                "title": "Zinzani 2019 [45]",
                "authors": "Zinzani et al.",
                "year": "2019",
                "extra_fields": {"citation_display": "Zinzani 2019 [45]", "trial_label": "CheckMate 436"},
            }
        )

    payload = result["payload"]
    assert result["status"] == "resolved"
    assert payload["source_api"] == "crossref_bibliographic"
    assert payload["resolution_method"] == "author_year_search"
    assert payload["confidence"] == "high"


def test_research_resolve_preserves_source_review_metadata():
    responses = {
        (
            "https://api.crossref.org/works/10.1000%2Fexample",
            (),
            "crossref",
        ): {
            "message": {
                "DOI": "10.1000/example",
                "URL": "https://doi.org/10.1000/example",
                "title": ["Example Trial"],
                "publisher": "Example Publisher",
                "type": "journal-article",
                "container-title": ["Journal of Testing"],
                "ISSN": ["1234-5678"],
            }
        }
    }
    resolver = _FakeResolver(responses=responses, options={"check_open_access": False, "enrich_sjr": False})

    result = resolver.resolve_one(
        {
            "row_id": 9,
            "title": "Example Trial",
            "doi": "10.1000/example",
            "source_review": "review_a.md",
            "source_review_title": "Review A",
            "extra_fields": {
                "source_review": "review_a.md",
                "source_review_title": "Review A",
                "source_doc_id": "1",
            },
        }
    )

    payload = result["payload"]
    assert payload["source_review"] == "review_a.md"
    assert payload["source_review_title"] == "Review A"
    assert payload["source_doc_id"] == "1"
    assert payload["extra_fields"]["source_review"] == "review_a.md"


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


def test_run_research_resolve_builds_bundle_and_optional_retrieval(monkeypatch, tmp_path):
    class _FakeIngestResult:
        def __init__(self, input_url: str, pdf_url: str, pdf_path: str, md_path: str):
            self.input_url = input_url
            self.final_url = input_url
            self.page_title = "Example"
            self.status = "downloaded"
            self.reason = ""
            self.http_code = "200"
            self.open_access_pdf_found = True
            self.pdf_url = pdf_url
            self.pdf_path = pdf_path
            self.md_path = md_path
            self.converted_to_md = True
            self.web_captured = False
            self.elapsed_seconds = 0.2

        def to_dict(self):
            return {
                "input_url": self.input_url,
                "final_url": self.final_url,
                "status": self.status,
                "pdf_url": self.pdf_url,
                "pdf_path": self.pdf_path,
                "md_path": self.md_path,
                "converted_to_md": self.converted_to_md,
                "web_captured": self.web_captured,
            }

    class _FakeURLIngestor:
        def __init__(self, output_root, timeout=25):
            self.output_root = output_root
            self.output_root.mkdir(parents=True, exist_ok=True)
            (self.output_root / "pdfs").mkdir(exist_ok=True)
            (self.output_root / "markdown").mkdir(exist_ok=True)

        def process_urls(self, urls, convert_to_md=False, use_vision_for_md=False, textify_options=None, capture_web_md_on_no_pdf=True):
            pdf_path = self.output_root / "pdfs" / "example.pdf"
            md_path = self.output_root / "markdown" / "example.md"
            pdf_path.write_bytes(b"%PDF-1.4 test")
            md_path.write_text("# Example", encoding="utf-8")
            return [_FakeIngestResult(urls[0], urls[0], str(pdf_path), str(md_path))]

        def build_reports(self, results):
            csv_path = self.output_root / "reports.csv"
            json_path = self.output_root / "reports.json"
            csv_path.write_text("input_url,status\nhttps://example.org/paper.pdf,downloaded\n", encoding="utf-8")
            json_path.write_text("{}", encoding="utf-8")
            return csv_path, json_path

    monkeypatch.setattr("cortex_engine.research_resolve.URLIngestor", _FakeURLIngestor)
    monkeypatch.setattr(
        ResearchResolver,
        "resolve_all",
        lambda self, citations: {
            "status": "completed",
            "resolved": [
                {
                    "row_id": 1,
                    "input_title": "Example title",
                    "matched_title": "Example title",
                    "resolved_doi": "10.1000/example",
                    "resolved_url": "https://doi.org/10.1000/example",
                    "source_api": "review_bibliography",
                    "resolution_method": "bibliography_reference_number",
                    "confidence": "high",
                    "publisher": "Example Publisher",
                    "journal": {"name": "Example Journal", "issn": "", "sjr_quartile": "", "sjr_score": 0.0, "sjr_rank": 0},
                    "open_access": {"is_oa": True, "oa_status": "gold", "pdf_url": "https://example.org/paper.pdf", "oa_source": "test"},
                    "extra_fields": {"citation_display": "Example 2024 [1]", "reference_number": "1"},
                    "credibility_tier": "peer-reviewed",
                    "retraction": {"is_retracted": False, "source": "", "note": ""},
                }
            ],
            "unresolved": [],
            "stats": {"total": 1, "resolved_high": 1, "resolved_low": 0, "unresolved": 0, "open_access": 1, "closed_access": 0},
        },
    )

    payload = {
        "citations": [{"title": "Example title", "extra_fields": {"citation_display": "Example 2024 [1]", "reference_number": "1"}}],
        "options": {"check_open_access": False, "enrich_sjr": False, "unpaywall_email": ""},
        "retrieval_options": {
            "retrieve_after_resolve": True,
            "timeout_seconds": 25,
            "ingest_options": {"convert_to_md": True, "use_vision": False, "capture_web_md_on_no_pdf": True},
            "textify_options": {"pdf_strategy": "hybrid", "use_vision": True, "docling_timeout_seconds": 240.0, "image_description_timeout_seconds": 20.0, "image_enrich_max_seconds": 120.0},
        },
    }

    output = run_research_resolve(payload=payload, run_dir=tmp_path / "run")

    assert output["preferred_urls"] == ["https://example.org/paper.pdf"]
    assert output["retrieval"]["performed"] is True
    assert (tmp_path / "run" / "research_resolve_bundle.zip").exists()
    result_json = json.loads((tmp_path / "run" / "research_resolve_result.json").read_text(encoding="utf-8"))
    assert result_json["preferred_urls"] == ["https://example.org/paper.pdf"]
    assert result_json["retrieval"]["performed"] is True
    assert result_json["artifacts"]["bundle_zip"] == "research_resolve_bundle.zip"


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
