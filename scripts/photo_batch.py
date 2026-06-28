#!/usr/bin/env python3
"""photo_batch.py — headless batch photo tagging + Lightroom catalog sync.

Drives the existing cortex_engine tagging (DocumentTextifier.keyword_image) and
reconciliation (cortex_engine.llm_metadata_sync) engines over a directory of
JPGs, so 5000+ photos can be processed without the Streamlit page's per-batch
upload ceiling.

See docs/superpowers/specs/2026-06-28-photo-batch-harness-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_OWNERSHIP = (
    "All rights (c) Longboardfella. Contact longboardfella.com for info on use of photos."
)
DEFAULT_MIN_DESC_LEN = 40
CHECKPOINT_NAME = ".photo_batch_tag.json"

# Lower-cased prefixes that signal a hallucinated / refusal / meta "description"
# rather than a real caption. Matched case-insensitively, independent of length.
REFUSAL_PREFIXES = (
    "i must",
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "sure,",
    "here is a description",
    "here's a description",
    "i will describe",
    "i'd be happy",
)


def description_is_bad(text, min_len: int = DEFAULT_MIN_DESC_LEN) -> bool:
    """Return True when an existing description should be regenerated.

    Bad = empty/whitespace, the engine's "[Image:" placeholder, a refusal/meta
    prefix, or shorter than min_len characters.
    """
    s = (text or "").strip()
    if not s:
        return True
    low = s.lower()
    if low.startswith("[image:"):
        return True
    if low.startswith(REFUSAL_PREFIXES):
        return True
    if len(s) < min_len:
        return True
    return False
