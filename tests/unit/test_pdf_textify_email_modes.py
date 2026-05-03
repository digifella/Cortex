from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from worker.handlers import pdf_textify


@pytest.fixture
def fake_input(tmp_path: Path) -> Path:
    p = tmp_path / "report.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def _patch_textifier(markdown: str = "# Title\n\nBody.\n"):
    """Patch DocumentTextifier so we don't actually run Docling."""
    class _FakeTextifier:
        @classmethod
        def from_options(cls, options, on_progress=None):
            return cls()

        def textify_file(self, _path):
            return markdown

        @staticmethod
        def markdown_to_plaintext(text: str, width: int = 80) -> str:
            # Strip the leading '# ' so we can assert it was called.
            import re
            return re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)

    return patch.object(pdf_textify, "DocumentTextifier", _FakeTextifier)


def test_legacy_invocation_writes_only_md(fake_input):
    """Without email_textify_mode, behaviour matches the original handler."""
    with _patch_textifier():
        result = pdf_textify.handle(
            input_path=fake_input,
            input_data={"textify_options": {}},
            job={"id": 1},
        )
    assert result["output_file"].suffix == ".md"
    assert result["output_file"].exists()
    assert not fake_input.with_suffix(".txt").exists()


def test_email_text_mode_writes_only_txt(fake_input):
    with _patch_textifier(markdown="# Heading\n\n- bullet\n"):
        result = pdf_textify.handle(
            input_path=fake_input,
            input_data={
                "textify_options": {},
                "email_textify_mode": "text",
            },
            job={"id": 2},
        )
    assert result["output_file"].suffix == ".txt"
    assert result["output_file"].exists()
    # No .md sibling — we only write what the worker will upload.
    assert not fake_input.with_suffix(".md").exists()
    txt_content = result["output_file"].read_text(encoding="utf-8")
    # markdown_to_plaintext stripped the '# '
    assert "Heading" in txt_content
    assert not txt_content.lstrip().startswith("#")


def test_email_markdown_mode_writes_only_md(fake_input):
    with _patch_textifier(markdown="# Heading\n"):
        result = pdf_textify.handle(
            input_path=fake_input,
            input_data={
                "textify_options": {},
                "email_textify_mode": "markdown",
            },
            job={"id": 3},
        )
    assert result["output_file"].suffix == ".md"
    assert result["output_file"].exists()
    assert not fake_input.with_suffix(".txt").exists()
    md_content = result["output_file"].read_text(encoding="utf-8")
    # Content should be raw markdown (still has the '#').
    assert md_content.lstrip().startswith("#")
