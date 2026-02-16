from __future__ import annotations

from typing import Callable, Dict

from .pdf_anonymise import handle as pdf_anonymise_handle
from .portal_ingest import handle as portal_ingest_handle
from .pdf_textify import handle as pdf_textify_handle
from .url_ingest import handle as url_ingest_handle

HandlerFunc = Callable[..., dict]

HANDLERS: Dict[str, HandlerFunc] = {
    "pdf_anonymise": pdf_anonymise_handle,
    "portal_ingest": portal_ingest_handle,
    "pdf_textify": pdf_textify_handle,
    "url_ingest": url_ingest_handle,
}
