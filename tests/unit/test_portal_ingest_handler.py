from __future__ import annotations

from worker.handlers.portal_ingest import chunk_text


def test_chunk_text_splits_and_preserves_metadata():
    text = (
        "# Heading One\n\n"
        + ("Paragraph one sentence. " * 80)
        + "\n\n## Heading Two\n\n"
        + ("Paragraph two sentence. " * 80)
    )
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    for idx, item in enumerate(chunks):
        assert "text" in item
        assert "metadata" in item
        assert item["metadata"]["chunk_index"] == idx
        assert item["metadata"]["char_count"] == len(item["text"])


def test_chunk_text_empty_input():
    assert chunk_text("") == []
