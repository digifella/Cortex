from __future__ import annotations

from types import SimpleNamespace

from cortex_engine.included_study_extractor import (
    get_gemini_api_key,
    parse_included_study_extraction_response,
    run_included_study_extractor,
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
