# Proposal PDF Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PDF as an export format for completed proposals and idea-generation reports, via one shared Markdown→PDF helper.

**Architecture:** A single helper `cortex_engine/pdf_export.py::markdown_to_pdf_bytes()` converts Markdown → HTML (`markdown` lib) → PDF (WeasyPrint) with one embedded stylesheet. Each existing export point passes its already-assembled Markdown to the helper. The helper returns `None` if either dependency is unavailable, so callers degrade gracefully (matching the existing DOCX pattern).

**Tech Stack:** Python 3.11, `markdown`, `weasyprint` (v68.1, already installed), Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-06-01-proposal-pdf-export-design.md`

---

## File Structure

- **Create** `cortex_engine/pdf_export.py` — the shared Markdown→PDF helper + CSS. One responsibility: render Markdown text to PDF bytes.
- **Create** `tests/unit/test_pdf_export.py` — unit tests for the helper.
- **Modify** `requirements.txt` and `docker/requirements.txt` — pin `markdown` and `weasyprint`.
- **Modify** `cortex_engine/proposal_export_engine.py` — add `generate_export_pdf()` (wraps existing `generate_export_markdown()`).
- **Modify** `pages/13_Proposal_Manager.py` — add `"PDF"` to format selector + download branch.
- **Modify** `pages/Proposal_Intelligent_Completion.py` — add "Download PDF" button beside "Export All".
- **Modify** `cortex_engine/idea_generator/export.py` — write a `.pdf` file alongside JSON/MD.

---

## Task 1: Dependencies

**Files:**
- Modify: `requirements.txt:118` (after `docx2txt==0.8`)
- Modify: `docker/requirements.txt:119` (after `docx2txt==0.8`)

- [ ] **Step 1: Add deps to `requirements.txt`**

In `requirements.txt`, after the line `docx2txt==0.8` (line 118), add:

```
markdown==3.7
weasyprint==68.1
```

- [ ] **Step 2: Add deps to `docker/requirements.txt`**

In `docker/requirements.txt`, after the line `docx2txt==0.8` (line 119), add the same two lines:

```
markdown==3.7
weasyprint==68.1
```

- [ ] **Step 3: Verify both import in the venv**

Run: `source venv/bin/activate && python -c "import markdown, weasyprint; print(markdown.__version__, weasyprint.__version__)"`
Expected: prints `3.x 68.1` (markdown may already need install — if `ModuleNotFoundError`, run `pip install markdown==3.7` then re-run).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt docker/requirements.txt
git commit -m "build: pin markdown and weasyprint for PDF export"
```

> **Note (Docker):** WeasyPrint needs native libs (Pango/cairo/gdk-pixbuf) present in the image. They are present in the current WSL venv. If a future Docker build cannot render, the helper returns `None` and callers degrade gracefully — adding the apt packages to the Dockerfile is out of scope for this plan.

---

## Task 2: The shared `markdown_to_pdf_bytes` helper

**Files:**
- Create: `cortex_engine/pdf_export.py`
- Test: `tests/unit/test_pdf_export.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_pdf_export.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && python -m pytest tests/unit/test_pdf_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cortex_engine.pdf_export'`

- [ ] **Step 3: Write the helper**

Create `cortex_engine/pdf_export.py`:

```python
"""
PDF Export Helper

Converts Markdown text to a paginated PDF (Markdown -> HTML -> PDF).

Both `markdown` and `weasyprint` are imported lazily; if either is
unavailable the function returns ``None`` so callers can degrade gracefully
(mirroring the DOCX export's None-degrades pattern).
"""

from __future__ import annotations

from html import escape
from typing import Optional

from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Clean, minimal print stylesheet. Single source of truth for PDF styling.
_PDF_CSS = """
@page {
    size: A4;
    margin: 2.2cm 2cm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #888;
    }
}
body {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.5;
    color: #1a1a1a;
}
h1 { font-size: 20pt; margin: 0 0 0.4em; }
h2 { font-size: 15pt; margin: 1.2em 0 0.3em; border-bottom: 1px solid #ddd; padding-bottom: 2px; }
h3 { font-size: 12.5pt; margin: 1em 0 0.2em; }
p { margin: 0 0 0.6em; }
hr { border: none; border-top: 1px solid #ccc; margin: 1em 0; }
table { border-collapse: collapse; width: 100%; margin: 0.6em 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #f2f2f2; }
code { font-family: "Courier New", monospace; font-size: 9.5pt; background: #f4f4f4; padding: 0 2px; }
pre { background: #f4f4f4; padding: 8px; overflow-wrap: break-word; white-space: pre-wrap; }
"""


def markdown_to_pdf_bytes(markdown_text: str, *, title: str = "") -> Optional[bytes]:
    """Render Markdown text to PDF bytes.

    Returns ``None`` if `markdown` or `weasyprint` is not importable, or if
    rendering fails for any reason.
    """
    try:
        import markdown as _markdown
    except ImportError:
        logger.warning("markdown not installed — PDF export unavailable")
        return None
    try:
        from weasyprint import HTML
    except ImportError:
        logger.warning("weasyprint not installed — PDF export unavailable")
        return None

    try:
        body_html = _markdown.markdown(
            markdown_text or "",
            extensions=["tables", "fenced_code"],
        )
        document = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{escape(title or 'Document')}</title>"
            f"<style>{_PDF_CSS}</style></head><body>{body_html}</body></html>"
        )
        return HTML(string=document).write_pdf()
    except Exception as exc:
        logger.warning(f"PDF rendering failed: {exc}")
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest tests/unit/test_pdf_export.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add cortex_engine/pdf_export.py tests/unit/test_pdf_export.py
git commit -m "feat: add markdown_to_pdf_bytes helper for PDF export"
```

---

## Task 3: `ProposalExportEngine.generate_export_pdf()`

**Files:**
- Modify: `cortex_engine/proposal_export_engine.py` (add method after `generate_export_docx`, which ends at line 476)
- Test: `tests/unit/test_pdf_export.py` (add one test)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_pdf_export.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && python -m pytest tests/unit/test_pdf_export.py::test_engine_generate_export_pdf_wraps_markdown -v`
Expected: FAIL with `AttributeError: ... has no attribute 'generate_export_pdf'`

- [ ] **Step 3: Add the method**

In `cortex_engine/proposal_export_engine.py`, immediately after the end of `generate_export_docx` (after line 476, `return buffer.getvalue()`), add this method (same indentation level — a method of `ProposalExportEngine`):

```python
    def generate_export_pdf(
        self,
        workspace_id: str,
        include_citations: bool = False,
        flag_incomplete: bool = True
    ) -> Optional[bytes]:
        """
        Generate a PDF export of the proposal.

        Reuses the Markdown export and renders it to PDF. Returns the PDF bytes,
        or None if the PDF dependencies (markdown/weasyprint) are unavailable.
        """
        from cortex_engine.pdf_export import markdown_to_pdf_bytes

        markdown_text = self.generate_export_markdown(
            workspace_id,
            include_citations=include_citations,
            flag_incomplete=flag_incomplete,
        )

        title = ""
        workspace = self.workspace_manager.get_workspace(workspace_id)
        if workspace:
            title = workspace.metadata.tender_name or ""

        return markdown_to_pdf_bytes(markdown_text, title=title)
```

> `Optional` is already imported at the top of this file (`from typing import List, Dict, Optional, Any, Tuple`).

- [ ] **Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && python -m pytest tests/unit/test_pdf_export.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add cortex_engine/proposal_export_engine.py tests/unit/test_pdf_export.py
git commit -m "feat: add generate_export_pdf to ProposalExportEngine"
```

---

## Task 4: PDF option in Proposal Manager

**Files:**
- Modify: `pages/13_Proposal_Manager.py:1298` (selectbox) and `:1318-1334` (download branches)

> No unit test — this is Streamlit UI glue. Verified by import + manual smoke check.

- [ ] **Step 1: Add `"PDF"` to the format selectbox**

In `pages/13_Proposal_Manager.py`, change the line at 1298:

```python
        export_format = st.selectbox("Format", ["Markdown", "DOCX"], key="pm_export_fmt")
```

to:

```python
        export_format = st.selectbox("Format", ["Markdown", "DOCX", "PDF"], key="pm_export_fmt")
```

- [ ] **Step 2: Add the PDF download branch**

In the same file, the export branches currently end with the DOCX `else:` block (lines 1318–1334). Change the `else:` at line 1318 to `elif export_format == "DOCX":`, then append a new `else:` PDF branch after the DOCX block (after line 1334, the `st.warning(...)` line). The result reads:

```python
    elif export_format == "DOCX":
        docx_bytes = export_engine.generate_export_docx(
            st.session_state.pm_workspace_id,
            include_citations=include_citations,
            flag_incomplete=flag_incomplete
        )
        if docx_bytes:
            st.download_button(
                "Download DOCX",
                data=docx_bytes,
                file_name=f"proposal_{ws.metadata.workspace_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                key="pm_dl_docx"
            )
        else:
            st.warning("DOCX export requires python-docx. Install with: pip install python-docx")
    else:
        pdf_bytes = export_engine.generate_export_pdf(
            st.session_state.pm_workspace_id,
            include_citations=include_citations,
            flag_incomplete=flag_incomplete
        )
        if pdf_bytes:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"proposal_{ws.metadata.workspace_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                type="primary",
                key="pm_dl_pdf"
            )
        else:
            st.warning("PDF export requires markdown and weasyprint. Install with: pip install markdown weasyprint")
```

- [ ] **Step 3: Verify the page imports cleanly**

Run: `source venv/bin/activate && python -c "import ast; ast.parse(open('pages/13_Proposal_Manager.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add pages/13_Proposal_Manager.py
git commit -m "feat: add PDF export option to Proposal Manager"
```

---

## Task 5: "Download PDF" button in Intelligent Completion

**Files:**
- Modify: `pages/Proposal_Intelligent_Completion.py:1123-1129` (after the "Export All" button)

- [ ] **Step 1: Add the PDF download button**

In `pages/Proposal_Intelligent_Completion.py`, directly after the existing "Export All" `st.download_button(...)` call (which ends at line 1129 with `)`), and still inside the `with exp_col2:` block (same indentation as the `st.download_button` above it), add:

```python
            from cortex_engine.pdf_export import markdown_to_pdf_bytes
            pdf_bytes = markdown_to_pdf_bytes(export_text, title=selected_workspace_name)
            if pdf_bytes:
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"proposal_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    key="ic_dl_pdf"
                )
```

- [ ] **Step 2: Verify the page parses**

Run: `source venv/bin/activate && python -c "import ast; ast.parse(open('pages/Proposal_Intelligent_Completion.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add pages/Proposal_Intelligent_Completion.py
git commit -m "feat: add PDF download to Intelligent Completion export"
```

---

## Task 6: Write a `.pdf` alongside JSON/MD in IdeaExporter

**Files:**
- Modify: `cortex_engine/idea_generator/export.py:64-65` (after the Markdown file is written)
- Test: `tests/unit/test_pdf_export.py` (add one test)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_pdf_export.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && python -m pytest tests/unit/test_pdf_export.py::test_idea_exporter_writes_pdf -v`
Expected: FAIL with `KeyError: 'pdf'`

- [ ] **Step 3: Add the PDF write**

In `cortex_engine/idea_generator/export.py`, after the Markdown file is written (after line 64, `f.write(markdown_content)`, and before the `# Export summary` comment at line 66), insert:

```python

            # Export as PDF (skipped silently if markdown/weasyprint unavailable)
            from cortex_engine.pdf_export import markdown_to_pdf_bytes
            pdf_bytes = markdown_to_pdf_bytes(markdown_content, title=filename_prefix)
            if pdf_bytes:
                pdf_file = output_path / f"{filename_prefix}_{timestamp}.pdf"
                with open(pdf_file, 'wb') as f:
                    f.write(pdf_bytes)
                exported_files["pdf"] = str(pdf_file)
```

> `markdown_content` and `timestamp` are already in scope at this point (defined at lines 62 and 47 respectively). `title` is keyword-only in the helper signature, so it is passed as `title=`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source venv/bin/activate && python -m pytest tests/unit/test_pdf_export.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add cortex_engine/idea_generator/export.py tests/unit/test_pdf_export.py
git commit -m "feat: write PDF report in IdeaExporter export_results"
```

---

## Task 7: Full-suite sanity check

- [ ] **Step 1: Run the new test file plus a quick import of touched modules**

Run:
```bash
source venv/bin/activate && \
python -m pytest tests/unit/test_pdf_export.py -v && \
python -c "import cortex_engine.proposal_export_engine, cortex_engine.idea_generator.export, cortex_engine.pdf_export; print('imports OK')"
```
Expected: 5 passed, then `imports OK`

- [ ] **Step 2: Confirm no stray uncommitted changes**

Run: `git status --short`
Expected: clean (all PDF-export changes already committed in Tasks 1–6)
