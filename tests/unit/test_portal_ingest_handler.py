from __future__ import annotations

import pytest

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


def test_chunk_text_respects_max_chunks():
    text = "\n\n".join([f"Paragraph {i} " + ("x " * 200) for i in range(20)])
    chunks = chunk_text(text, chunk_target_chars=500, chunk_min_chars=100, max_chunks=3)
    assert len(chunks) == 3


def test_chunk_text_honors_cancellation():
    def _cancel() -> bool:
        return True

    with pytest.raises(RuntimeError, match="Cancelled during chunking"):
        chunk_text(
            "Paragraph one\n\nParagraph two\n\nParagraph three",
            chunk_target_chars=100,
            chunk_min_chars=10,
            max_chunks=10,
            is_cancelled_cb=_cancel,
        )
