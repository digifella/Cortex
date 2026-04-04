from __future__ import annotations

from types import SimpleNamespace

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


def test_run_claude_table_rescue_includes_table_context_and_snapshot_hints(monkeypatch):
    captured = {}

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text='{"candidates":[],"warnings":[]}')])

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("cortex_engine.review_table_rescue._anthropic_client", lambda: _FakeClient())

    from cortex_engine.review_table_rescue import run_claude_table_rescue

    run_claude_table_rescue(
        review_title="Supportive care review",
        design_query="randomized trial",
        outcome_query="quality of life",
        table_snapshots=[
            {
                "table_index": 1,
                "page_number": 3,
                "text_sample": "Continuation of characteristics of included studies",
                "image_bytes": b"fake-image",
            }
        ],
        table_blocks=[
            {
                "table_index": 1,
                "header": ["Study", "Design", "Outcome"],
                "context_text": "Before table:\nTable 2. Characteristics of included studies\n\nAfter table:\nOutcomes reported at 12 months.",
                "markdown": "| Study | Design | Outcome |\n| --- | --- | --- |\n| Smith 2020 | Randomized trial | Quality of life |",
            }
        ],
        references_text="[19] Smith J. Trial paper. Journal. 2020.",
    )

    prompt = captured["messages"][0]["content"][0]["text"]
    assert "Nearby table context" in prompt
    assert "Table 2. Characteristics of included studies" in prompt
    assert "Snapshot hints" in prompt
    assert "Continuation of characteristics of included studies" in prompt


def test_run_claude_table_rescue_attaches_full_pdf_when_available(monkeypatch, tmp_path):
    captured = {}

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text='{"candidates":[],"warnings":[]}')])

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    monkeypatch.setattr("cortex_engine.review_table_rescue._anthropic_client", lambda: _FakeClient())

    pdf_path = tmp_path / "review.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n")

    from cortex_engine.review_table_rescue import run_claude_table_rescue

    result = run_claude_table_rescue(
        review_title="Supportive care review",
        design_query="randomized trial",
        outcome_query="quality of life",
        table_snapshots=[
            {
                "table_index": 1,
                "page_number": 3,
                "text_sample": "Characteristics of included studies",
                "image_bytes": b"fake-image",
            }
        ],
        table_blocks=[
            {
                "table_index": 1,
                "header": ["Study", "Design", "Outcome"],
                "context_text": "Before table:\nTable 2. Characteristics of included studies",
                "markdown": "| Study | Design | Outcome |\n| --- | --- | --- |\n| Smith 2020 | Randomized trial | Quality of life |",
            }
        ],
        references_text="[19] Smith J. Trial paper. Journal. 2020.",
        pdf_path=str(pdf_path),
    )

    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["source"]["media_type"] == "application/pdf"
    assert content[1]["type"] == "text"
    assert "Primary evidence is the attached full PDF" in content[1]["text"]
    assert result["used_full_pdf"] is True
