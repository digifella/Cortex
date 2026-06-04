import builtins

import pytest

from cortex_engine.pdf_export import markdown_to_pdf_bytes


SAMPLE_MD = """# Proposal: Example Tender

**Workspace:** Demo

---

## Section One

Some body text that should flow into a paragraph.

| Field | Value |
| --- | --- |
| ABN | 123 |
| Name | Acme |
"""


def test_returns_pdf_bytes_for_markdown_with_table():
    result = markdown_to_pdf_bytes(SAMPLE_MD, title="Example Tender")
    assert isinstance(result, bytes)
    assert result[:5] == b"%PDF-"


def test_returns_none_when_markdown_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "markdown":
            raise ImportError("simulated missing markdown")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert markdown_to_pdf_bytes(SAMPLE_MD, title="x") is None


def test_returns_none_when_weasyprint_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "weasyprint":
            raise ImportError("simulated missing weasyprint")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert markdown_to_pdf_bytes(SAMPLE_MD, title="x") is None
