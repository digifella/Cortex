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
