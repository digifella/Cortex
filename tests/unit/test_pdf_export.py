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


def test_engine_generate_export_pdf_wraps_markdown(monkeypatch):
    from cortex_engine import proposal_export_engine as pee

    captured = {}

    def fake_md(self, workspace_id, include_citations=False, flag_incomplete=True):
        captured["workspace_id"] = workspace_id
        captured["include_citations"] = include_citations
        captured["flag_incomplete"] = flag_incomplete
        return "# Proposal: T\n\nbody\n"

    monkeypatch.setattr(pee.ProposalExportEngine, "generate_export_markdown", fake_md)

    engine = pee.ProposalExportEngine.__new__(pee.ProposalExportEngine)
    result = engine.generate_export_pdf("ws-1", include_citations=True, flag_incomplete=False)

    assert captured == {"workspace_id": "ws-1", "include_citations": True, "flag_incomplete": False}
    assert isinstance(result, bytes)
    assert result[:5] == b"%PDF-"


def test_idea_exporter_writes_pdf(tmp_path):
    from cortex_engine.idea_generator.export import IdeaExporter

    exporter = IdeaExporter()
    phase_results = {"phase_1": {"summary": "An idea", "items": ["a", "b"]}}

    exported = exporter.export_results(
        phase_results, output_dir=str(tmp_path), filename_prefix="sess"
    )

    assert "pdf" in exported
    pdf_path = tmp_path / exported["pdf"].split("/")[-1]
    assert pdf_path.exists()
    assert pdf_path.read_bytes()[:5] == b"%PDF-"
