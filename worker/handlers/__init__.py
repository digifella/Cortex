from __future__ import annotations

from typing import Callable, Dict

from .pdf_anonymise import handle as pdf_anonymise_handle
from .pdf_textify import handle as pdf_textify_handle

HandlerFunc = Callable[..., dict]

HANDLERS: Dict[str, HandlerFunc] = {
    "pdf_anonymise": pdf_anonymise_handle,
    "pdf_textify": pdf_textify_handle,
}

