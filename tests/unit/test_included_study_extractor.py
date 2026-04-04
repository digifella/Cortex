from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError

from cortex_engine.included_study_extractor import (
    IncludedStudyExtractorQuotaError,
    get_gemini_api_key,
    parse_included_study_extraction_response,
    run_included_study_extractor,
    run_included_study_extractor_with_fallback,
)


def test_parse_included_study_extraction_response_handles_fenced_json():
    payload = parse_included_study_extraction_response(
        """```json
        {
          "tables": [
            {
              "table_number": "2",
              "table_title": "Overview of Included Studies on HRQOL Measures",
              "groups": []
            }
          ],
          "warnings": []
        }
        ```"""
    )

    assert payload["tables"][0]["table_number"] == "2"


def test_get_gemini_api_key_falls_back_to_worker_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "cortex_engine.included_study_extractor._worker_env_value",
        lambda name: "AI-test-key" if name == "GEMINI_API_KEY" else "",
    )

    assert get_gemini_api_key() == "AI-test-key"


def test_run_included_study_extractor_gemini_normalizes_grouped_tables(monkeypatch, tmp_path):
    pdf_path = tmp_path / "review.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    monkeypatch.setattr("cortex_engine.included_study_extractor.get_gemini_api_key", lambda: "AI-test-key")
    monkeypatch.setattr(
        "cortex_engine.included_study_extractor._gemini_generate_content",
        lambda model, payload, api_key: {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": """
                                {
                                  "tables": [
                                    {
                                      "table_number": "2",
                                      "table_title": "Overview of Included Studies on HRQOL Measures",
                                      "grouping_basis": "Grouped by instrument",
                                      "groups": [
                                        {
                                          "group_label": "EORTC QLQ-C30",
                                          "trial_label": "TRANSCEND NHL 001",
                                          "citations": [
                                            {
                                              "display": "Patrick 2021 [17]",
                                              "authors": "Patrick",
                                              "year": "2021",
                                              "reference_number": "17",
                                              "resolved_title": "Health-related quality of life with lisocabtagene maraleucel",
                                              "resolved_authors": "Patrick D",
                                              "resolved_year": "2021",
                                              "resolved_journal": "Blood Advances",
                                              "resolved_doi": "10.1000/example",
                                              "notes": "",
                                              "needs_review": false
                                            }
                                          ]
                                        }
                                      ]
                                    }
                                  ],
                                  "warnings": []
                                }
                                """
                            }
                        ]
                    }
                }
            ]
        },
    )

    result = run_included_study_extractor(
        pdf_path=str(pdf_path),
        provider="gemini",
        model="gemini-2.5-pro",
        review_title="Review",
    )

    assert result["provider"] == "gemini"
    assert result["tables"][0]["table_number"] == "2"
    citation = result["tables"][0]["groups"][0]["citations"][0]
    assert citation["reference_number"] == "17"
    assert citation["resolved_doi"] == "10.1000/example"


def test_run_included_study_extractor_anthropic_normalizes_tables(monkeypatch, tmp_path):
    pdf_path = tmp_path / "review.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    class _FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        text="""
                        {
                          "tables": [
                            {
                              "table_number": "4",
                              "table_title": "Overview of Economic Studies Reporting Health State Utility Values",
                              "grouping_basis": "Economic studies",
                              "groups": [
                                {
                                  "group_label": "Included Economic Studies",
                                  "trial_label": "",
                                  "citations": [
                                    {
                                      "display": "Li 2022 [54]",
                                      "authors": "Li",
                                      "year": "2022",
                                      "reference_number": "54",
                                      "resolved_title": "Cost-utility analysis of axi-cel in China",
                                      "resolved_authors": "Li et al.",
                                      "resolved_year": "2022",
                                      "resolved_journal": "",
                                      "resolved_doi": "",
                                      "notes": "",
                                      "needs_review": false
                                    }
                                  ]
                                }
                              ]
                            }
                          ],
                          "warnings": []
                        }
                        """
                    )
                ]
            )

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("cortex_engine.included_study_extractor._anthropic_client", lambda: _FakeClient())

    result = run_included_study_extractor(
        pdf_path=str(pdf_path),
        provider="anthropic",
        model="claude-sonnet-4-6",
        review_title="Review",
    )

    assert result["provider"] == "anthropic"
    assert result["tables"][0]["table_number"] == "4"
    assert result["tables"][0]["groups"][0]["citations"][0]["reference_number"] == "54"


def test_run_included_study_extractor_gemini_raises_quota_error(monkeypatch, tmp_path):
    pdf_path = tmp_path / "review.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    monkeypatch.setattr("cortex_engine.included_study_extractor.get_gemini_api_key", lambda: "AI-test-key")

    def _raise_quota(*_args, **_kwargs):
        raise HTTPError(
            url="https://example.test",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=BytesIO(
                b'{"error":{"code":429,"message":"Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count"}}'
            ),
        )

    monkeypatch.setattr("cortex_engine.included_study_extractor._gemini_generate_content", _raise_quota)

    try:
        run_included_study_extractor(
            pdf_path=str(pdf_path),
            provider="gemini",
            model="gemini-2.5-pro",
            review_title="Review",
        )
        assert False, "Expected IncludedStudyExtractorQuotaError"
    except IncludedStudyExtractorQuotaError as exc:
        assert exc.status_code == 429
        assert "quota/rate limit" in str(exc).lower()


def test_run_included_study_extractor_with_fallback_uses_anthropic(monkeypatch, tmp_path):
    pdf_path = tmp_path / "review.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    def _fake_run(*, pdf_path, provider, model, review_title):
        if provider == "gemini":
            raise IncludedStudyExtractorQuotaError("gemini", 429, "Gemini quota/rate limit exceeded", body="quota")
        return {
            "provider": "anthropic",
            "model": model or "claude-sonnet-4-6",
            "tables": [
                {
                    "table_number": "2",
                    "table_title": "Overview",
                    "grouping_basis": "Grouped by trial",
                    "groups": [
                        {
                            "group_label": "Global",
                            "trial_label": "JULIET",
                            "notes": "",
                            "citations": [
                                {
                                    "display": "Maziarz 2020 [19]",
                                    "authors": "Maziarz",
                                    "year": "2020",
                                    "reference_number": "19",
                                    "resolved_title": "Title",
                                    "resolved_authors": "Maziarz",
                                    "resolved_year": "2020",
                                    "resolved_journal": "",
                                    "resolved_doi": "",
                                    "notes": "",
                                    "needs_review": False,
                                }
                            ],
                        }
                    ],
                }
            ],
            "warnings": [],
            "raw_response": "{}",
        }

    monkeypatch.setattr("cortex_engine.included_study_extractor.run_included_study_extractor", _fake_run)
    monkeypatch.setattr("cortex_engine.included_study_extractor.included_study_extractor_available", lambda provider: provider == "anthropic")

    result = run_included_study_extractor_with_fallback(
        pdf_path=str(pdf_path),
        provider="gemini",
        model="gemini-2.5-pro",
        review_title="Review",
        fallback_provider="anthropic",
        fallback_model="claude-sonnet-4-6",
    )

    assert result["provider"] == "anthropic"
    assert result["requested_provider"] == "gemini"
    assert result["requested_model"] == "gemini-2.5-pro"
    assert "fell back to anthropic" in result["warnings"][0].lower()
