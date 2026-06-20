from __future__ import annotations

import sys
import types
from pathlib import Path

from cortex_engine.textifier import DocumentTextifier


def test_opendataloader_pdf_strategy_uses_markdown_output(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    module = types.SimpleNamespace()

    def fake_convert(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sample.md").write_text(
            "# Extracted\n\n| Col A | Col B |\n|---|---|\n| Alpha | Beta |\n\n"
            + ("This paragraph has enough text to be treated as useful parser output. " * 20)
            + "\n",
            encoding="utf-8",
        )

    module.convert = fake_convert
    monkeypatch.setitem(sys.modules, "opendataloader_pdf", module)

    textifier = DocumentTextifier(use_vision=False, pdf_strategy="opendataloader")
    output = textifier.textify_pdf(str(pdf_path))

    assert "# Extracted" in output
    assert "| Alpha | Beta |" in output


def test_opendataloader_pdf_strategy_falls_back_on_weak_output(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    module = types.SimpleNamespace()

    def fake_convert(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sample.md").write_text("x\n", encoding="utf-8")

    module.convert = fake_convert
    monkeypatch.setitem(sys.modules, "opendataloader_pdf", module)

    textifier = DocumentTextifier(use_vision=False, pdf_strategy="opendataloader")
    monkeypatch.setattr(textifier, "_try_docling", lambda *args, **kwargs: "# Docling fallback\n")

    output = textifier.textify_pdf(str(pdf_path))

    assert output == "# Docling fallback\n"
