from __future__ import annotations

from cortex_engine.review_table_rescue import get_anthropic_api_key, parse_review_table_rescue_response


def test_parse_review_table_rescue_response_handles_fenced_json():
    payload = parse_review_table_rescue_response(
        """```json
        {
          "candidates": [
            {
              "title": "Maziarz 2020 trial",
              "reference_number": "19",
              "meets_criteria": true
            }
          ],
          "warnings": ["check rotation"]
        }
        ```"""
    )

    assert payload["candidates"][0]["reference_number"] == "19"
    assert payload["warnings"] == ["check rotation"]


def test_parse_review_table_rescue_response_handles_embedded_json():
    payload = parse_review_table_rescue_response(
        'Here is the result {"candidates":[{"title":"Study A","needs_review":true}],"warnings":[]}'
    )

    assert payload["candidates"][0]["title"] == "Study A"
    assert payload["candidates"][0]["needs_review"] is True


def test_get_anthropic_api_key_falls_back_to_worker_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("cortex_engine.review_table_rescue._worker_env_value", lambda name: "test-key" if name == "ANTHROPIC_API_KEY" else "")

    assert get_anthropic_api_key() == "test-key"
