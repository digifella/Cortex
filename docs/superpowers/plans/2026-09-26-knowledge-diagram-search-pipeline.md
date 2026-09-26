# Knowledge-Base Diagram Search Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make diagrams/charts embedded as pictures in Paul's PPTX/PDF knowledge base (both `04_Knowledge` and `01_Clients/Active`) findable through Hermes/kb-query, by giving Docling real PPTX support, replacing the broken Ollama figure-captioning path with Claude Haiku (size-filtered, deduplicated), cleaning up chunk text quality, and making the two-stage ingest pipeline resilient to a single pathological file — then backfilling both collections and adding a Hermes routing hint.

**Architecture:** Two existing production stages stay in place and get modified, not replaced: Stage 1 (`nemoclaw-kdocs-extract.py` / `nemoclaw-cdocs-extract.py`, Cortex venv, text+figure extraction) and Stage 2 (`nemoclaw-kdocs-ingest.py` / `nemoclaw-cdocs-ingest.py`, vault-rag venv, chunk+embed+Chroma upsert). Stage 1 changes from "call Docling once for the whole batch" to "spawn one isolated subprocess per file with a hard timeout," closing a real gap where a single hung file currently blocks an entire batch of ~30-60 files forever. VLM figure-captioning moves from an always-on, miswired local-Ollama call inside `docling_reader.py` to a single, deliberately-gated path in `enhanced_ingest_cortex.py` that filters by figure size, checks a persistent cross-file dedup cache, and calls Claude Haiku. A `PIPELINE_VERSION` folded into Stage 2's existing resumability key makes the full backfill happen automatically through the existing resumable `todo`-list mechanism — no bespoke backfill runner needed.

**Tech Stack:** Python 3.11 (Cortex venv: `~/cortex_suite/venv`, vault-rag venv: `~/venvs/vault-rag`), Docling, `anthropic` SDK (Claude Haiku vision), SQLite (dedup cache), Chroma (`~/vault-rag-db`), LibreOffice headless (legacy `.doc`/`.ppt` conversion).

**Spec:** This plan's own "Problem being solved" / "Already decided" sections below serve as the spec — captured directly from a same-session design conversation with the user (full corpus sizing scan already run: 7,040 files scanned, 127,126 raw pictures found, 10% area threshold chosen — see numbers below). No separate spec file exists; the user explicitly approved proceeding straight from design to plan to implementation in one instruction ("write it all as a resumable plan ... then you can proceed. 10% sounds fine.").

## Problem being solved

Hermes/kb-query cannot find diagrams/charts embedded as pictures in the knowledge base (concrete case: a "heat map" prioritization matrix in an old consulting report) because of five compounding issues, all confirmed by direct investigation of the live code and a live search test, not assumption:

1. Docling is restricted to `allowed_formats=[InputFormat.PDF]` in `cortex_suite/cortex_engine/docling_reader.py` with no comment explaining why — Docling supports PPTX natively (confirmed live: adding `InputFormat.PPTX` and converting the actual heatmap file worked cleanly). Every `.pptx` in the whole knowledge base today silently falls back to a much weaker legacy `PptxReader` with zero figure/table intelligence.
2. VLM figure-captioning that *is* wired up today is broken twice over: `docling_reader.py`'s own `_generate_vlm_descriptions_for_figures` runs **unconditionally by default** (its caller in `enhanced_ingest_cortex.py` never passes `skip_vlm_processing`, so it stays `False`; a separate `DOCLING_VLM_ENABLED` config flag also defaults to `"true"`) and calls `describe_image_with_vlm_for_ingestion`, which hits **local Ollama** — the exact path project memory already flags as unreliable (emits reasoning-scaffolding text instead of real captions). Worse: even when it produces a caption, that caption is stored only in `document.metadata['docling_figures'][i]['vlm_description']`, which `nemoclaw-kdocs-extract.py`'s output loop **never reads** (`text = getattr(d, "text", "") or ""` — metadata is discarded at the Stage 1→2 JSON boundary). So today's VLM captioning is both low-quality *and* invisible to search regardless of quality — pure wasted Ollama calls.
3. Chunk text embeds poorly even when present: the heatmap's diagram-marker numbers ("1a", "29", "9"...) sit between the two meaningful axis-label phrases and dilute the embedding, and the filename (often the most distinctive vocabulary — "Heat Map" — appears nowhere else) never enters the embedded text, only the document's internal title does.
4. 1,532 legacy binary `.doc`/`.ppt` files (1,462 in `04_Knowledge`, 70 in `01_Clients/Active`) can't be read by Docling at all (OOXML-only).
5. General `kb-query` never reaches the `knowledge_docs`/`client_docs` Chroma collections unless the query is scoped with the literal words "knowledge base" — confirmed live: an unscoped query for the heatmap returned seven completely unrelated results from the private email/travel corpus.

## Already decided (do not re-litigate)

- VLM backend: **Claude Haiku** (`claude-haiku-4-5-20251001` — the exact model ID already used elsewhere in this codebase, e.g. `cortex_engine/textifier.py`; follow that precedent rather than a bare/undated ID). One-time backfill cost estimated ~$50-65 pre-dedup, affordable; user explicitly chose this over local GPU/LM Studio.
- Size filter: only caption pictures whose bounding-box area is **≥10%** of their page/slide area (`area_frac >= 0.10`) — confirmed against real bounding-box data today to separate genuine diagrams/charts/photos from logos/icons/decorative graphics.
- Dedup: **corpus-wide** (not just per-document) SHA256-of-image-bytes → caption cache, checked before every Haiku call, persisted permanently (useful for future incremental ingestion too).
- Legacy `.doc`/`.ppt`: convert via headless LibreOffice to `.docx`/`.pptx` first, then run through the same fixed pipeline.
- Chunk-quality fix: (a) strip runs of ≥4 consecutive bare short reference-marker lines into a compact trailing note; (b) prepend the cleaned filename into the extracted text.
- Hermes routing hint: same `config.yaml` pattern as the already-shipped-and-twice-verified `kb-travel` fix from earlier this session.
- Full corpus backfill, both `knowledge_docs` (~4,705 files) and `client_docs` (~2,340 files) — not a narrowed pilot.
- Resilience requirement (explicit user instruction): **no single corrupt/pathological file may block the pipeline for hours.** This has already happened in this exact codebase before — `nemoclaw-kdocs-ingest.py` carries a comment: *"a pathological file once hung docling for 7.5h and blocked the whole ingest (no timeout)"* — and happened again, independently, during today's sizing scan (3h25m hang, fixed with a chunk+isolate pattern validated in this session). The existing mitigation (`EXTRACT_TIMEOUT = 1200`, a *per-batch* subprocess timeout in Stage 2) only bounds a single run's damage — it skips the *whole* batch of 30-60 files on timeout, none of which get an updated resumability key, so they're retried in the same doomed grouping forever. This plan closes that gap with **per-file** isolation.

## Numbers from the completed sizing scan (both roots fully scanned, real data)

- 7,040 files scanned OK (5 errors: 2 permanently-hung/skipped, 2 memory-allocation, 1 other — none require special handling, see Scope Boundaries)
- 86.3% of files have ≥1 embedded picture; 127,126 raw pictures total
- At the chosen 10% threshold: **32,887 pictures** need captioning pre-dedup (25.9% of raw)

## Scope Boundaries

- Two Chroma collections only: `knowledge_docs` and `client_docs`. Not touching `vault_private`/`vault_public`/other collections.
- Not touching the photo pipeline (`photo_batch.py`) — separate, already-working system.
- The two files that failed with `OSError: Cannot allocate memory` during today's scan need no special handling — they'll be retried naturally when the backfill runs (that was transient memory pressure from running two scans in parallel, not a property of the files).
- Not rewriting the FTS5 mirror (`nemoclaw_fts.py`), the delta-prune logic, or the embedding model choice (`bge-base-en-v1.5`) — working fine, out of scope.
- Not merging `nemoclaw-kdocs-*` and `nemoclaw-cdocs-*` into one script — keep them as parallel copies as today; a shared helper module is used where logic must be identical (text cleanup, legacy conversion, single-file extraction) to avoid duplicating that logic twice, but the two orchestrator/driver scripts themselves stay separate per existing project convention ("kept separate so the live 04_Knowledge ingest is never touched").

## Global Constraints

- Cortex venv (`~/cortex_suite/venv`) runs Stage 1 (Docling, figure extraction, VLM captioning). Vault-rag venv (`~/venvs/vault-rag`) runs Stage 2 (chunking, embedding, Chroma). A module needed by both stages must not import anything only available in the other venv's site-packages — `anthropic` and `sqlite3` (stdlib) are available in both; verify before assuming.
- `ANTHROPIC_API_KEY` lives in `~/cortex_suite/.env` (double-quoted value) — must be loaded explicitly in-process (no ambient dotenv loading exists on this path today); reuse the exact loader pattern from `cortex_suite/scripts/photo_batch.py::load_dotenv_keys()`.
- Follow the existing throttle convention (`taskset -c 0-3`, `nice -n 19`, 4-thread BLAS/OMP caps) in both resume shell scripts — do not remove it for speed; it exists because an all-core run once crashed WSL.
- `cdocs` and `kdocs` ingests must not run concurrently (existing "STAGED" comment in `cdocs-ingest-resume.sh` — they share the same 4 pinned cores).
- Every new/modified script must be resumable: a kill or crash at any point, followed by re-running the same command, must continue from where it left off, not restart from scratch or silently skip un-ingested work.
- No task may write a step that hangs indefinitely on external I/O (Docling conversion, LibreOffice conversion, an API call) without a hard timeout that the caller can observe and recover from.

## Review Focus

- **A file identical in every byte to one already ingested under the old pipeline must still get reprocessed once `PIPELINE_VERSION` bumps** — the resumability key must include the version, not just size+mtime, or the whole backfill silently no-ops. (Task 11)
- **A picture that is large (passes the 10% filter) but whose Haiku call fails (network error, rate limit) must not crash the whole file's extraction** — a single figure's caption failure degrades gracefully (no caption for that figure) rather than losing the rest of the document's text. (Task 3)
- **Two different pictures with the same content (e.g. a letterhead reused across 40 files in one engagement) must hit the dedup cache on the second-and-later occurrence** — this is the dedup requirement the user explicitly asked for, not just an internal detail. (Task 2, tested directly)
- **A file whose legacy `.doc`/`.ppt` LibreOffice conversion itself fails or hangs must not block extraction of the rest of the batch** — same isolation requirement extends to the conversion step, not just the Docling step. (Task 6, Task 8)
- **The numeric-marker-stripping regex must not eat a legitimate short number that happens to appear alone** (a copyright year, a single statistic on its own line) — already validated against a synthetic case during design, must be pinned as an actual test, not just remembered. (Task 4)

---

## File Structure

**New files:**
- `cortex_suite/cortex_engine/image_caption_cache.py` — SQLite dedup cache (hash → caption), used by Stage 1.
- `cortex_suite/cortex_engine/haiku_figure_caption.py` — Claude Haiku vision captioning for document figures (distinct prompt from the photo-captioning one in `textifier.py`, which is written for photographs, not diagrams).
- `~/nemoclaw_doc_text_cleanup.py` — shared text-quality helper (marker-run stripping + filename prefix), imported by both extract drivers.
- `~/nemoclaw_legacy_office_convert.py` — shared LibreOffice `.doc`/`.ppt` → `.docx`/`.pptx` converter, imported by both extract drivers.
- `~/nemoclaw_doc_extract_one.py` — single-file extraction worker (the new isolation unit); invoked as a subprocess per file by both extract drivers.

**Modified files:**
- `cortex_suite/cortex_engine/docling_reader.py` — PPTX in `allowed_formats`; page-size/`area_frac` added to figure metadata.
- `cortex_suite/cortex_engine/enhanced_ingest_cortex.py` — `skip_vlm_processing=True` passed explicitly (kills the redundant always-on Ollama path); `_enrich_docling_figures`/`_summarize_figure_with_vlm` rewritten to filter by size, check the dedup cache, and call Haiku.
- `~/nemoclaw-kdocs-extract.py` / `~/nemoclaw-cdocs-extract.py` — become thin per-file-isolating drivers (spawn `nemoclaw_doc_extract_one.py` per file with a timeout + permanent skip-list) instead of one `enhanced_load_documents()` call per batch.
- `~/nemoclaw-kdocs-ingest.py` / `~/nemoclaw-cdocs-ingest.py` — `PIPELINE_VERSION` folded into the stored `fkey`.
- `~/.hermes/config.yaml` (on host `sp4`, via `ssh`) — routing-hint line added.

---

### Task 1: Docling PPTX support + figure area_frac

**Files:**
- Modify: `cortex_suite/cortex_engine/docling_reader.py` (`_create_converter`, `_extract_docling_figures`)
- Test: `cortex_suite/tests/unit/test_docling_reader_pptx.py`

**Interfaces:**
- Produces: `_extract_docling_figures(...)` now returns `figure_entry` dicts with an additional `area_frac: Optional[float]` key (bbox area / page area, `None` if either is unavailable), and `figure_payloads` dicts also carry `area_frac` (same value, needed by the VLM gating step in Task 3, which reads payloads not entries).

- [ ] **Step 1: Write the failing test**

```python
# cortex_suite/tests/unit/test_docling_reader_pptx.py
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

from cortex_engine.docling_reader import DoclingDocumentReader

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_findings_heatmap.pptx"


def test_docling_converts_pptx_and_reports_figure_area():
    reader = DoclingDocumentReader(ocr_enabled=False, table_structure_recognition=False,
                                    skip_vlm_processing=True)
    assert reader.is_available, "Docling must be available for this test"
    docs = reader.load_data(str(FIXTURE))
    assert len(docs) == 1
    doc = docs[0]
    assert "Ease of Implementation" in doc.text
    figures = doc.metadata.get("docling_figures") or []
    assert figures, "expected at least one figure entry from the fixture"
    # every figure_entry must carry an area_frac key (value may be None, but key must exist)
    for entry in figures:
        assert "area_frac" in entry
```

- [ ] **Step 2: Create the test fixture**

The fixture is a copy of the real heatmap file already confirmed to exercise this exact code path — python-pptx text extraction was validated against it earlier this session (axis labels "Degree of Benefit" / "Ease of Implementation", 20+ finding text boxes, a run of bare reference-marker numbers).

```bash
mkdir -p /home/longboardfella/cortex_suite/tests/fixtures
cp "/mnt/c/Users/paul/OneDrive - VentraIP Australia/04_Knowledge/SMS knowledge/SMS Knowledge Base/Fireservices- Emergency Services/In Progress/Major Findings Heat Map - A3 ra.pptx" \
   /home/longboardfella/cortex_suite/tests/fixtures/sample_findings_heatmap.pptx
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/longboardfella/cortex_suite && venv/bin/python -m pytest tests/unit/test_docling_reader_pptx.py -v`
Expected: FAIL — either Docling raises `File format not allowed` (PPTX not in `allowed_formats` yet) or the `area_frac` key is missing.

- [ ] **Step 4: Add PPTX to allowed_formats**

In `cortex_suite/cortex_engine/docling_reader.py`, find `_create_converter` (the block starting `if "format_options" in init_params:`). Change:

```python
            return DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pdf_pipeline_options
                    )
                },
            )
```

to:

```python
            return DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.PPTX],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pdf_pipeline_options
                    )
                },
            )
```

(No PPTX-specific `format_options` entry is needed — verified directly against the real heatmap file this session: Docling parses PPTX cleanly with default options once it's in `allowed_formats`.)

- [ ] **Step 5: Add area_frac to figure extraction**

In `_extract_docling_figures` (same file), the method receives `conv_result` which exposes `conv_result.document.pages` (a dict keyed by page number, each with a `.size.width` / `.size.height`) — the same API already validated this session for computing bbox-area-as-fraction-of-page-area. Add a page-size lookup near the top of the method (after the existing `rendered_figures` try/except block) and use it when building both `figure_entry` and `payload`:

```python
        page_sizes = {}
        try:
            for page_no, page in (conv_result.document.pages or {}).items():
                sz = getattr(page, "size", None)
                if sz:
                    page_sizes[page_no] = (sz.width, sz.height)
        except Exception as page_size_error:
            logger.debug(f"Could not read page sizes for area_frac: {page_size_error}")
```

Then, inside the `for idx in range(total_figures):` loop, after `bbox = (prov or {}).get('bbox')` is computed (it's already assigned into `figure_entry['bbox']`), compute the fraction:

```python
            area_frac = None
            page_no = (prov or {}).get('page')
            if bbox and page_no in page_sizes:
                try:
                    bw = abs(bbox.get('r', 0) - bbox.get('l', 0))
                    bh = abs(bbox.get('t', 0) - bbox.get('b', 0))
                    pw, ph = page_sizes[page_no]
                    if pw and ph:
                        area_frac = round((bw * bh) / (pw * ph), 4)
                except Exception as area_error:
                    logger.debug(f"Could not compute area_frac for figure {idx}: {area_error}")
```

Add `'area_frac': area_frac,` to the `figure_entry` dict literal, and add the same key to the `payload` dict a few lines below (where `payload = {'index': idx, 'image_base64': ..., ...}` is built) — add `'area_frac': area_frac,` there too, since Task 3's VLM gating reads `payload`, not `figure_entry`.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /home/longboardfella/cortex_suite && venv/bin/python -m pytest tests/unit/test_docling_reader_pptx.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
cd /home/longboardfella/cortex_suite
git add cortex_engine/docling_reader.py tests/unit/test_docling_reader_pptx.py tests/fixtures/sample_findings_heatmap.pptx
git commit -m "$(cat <<'EOF'
Enable Docling PPTX support and figure area_frac

allowed_formats was PDF-only with no comment explaining why, despite
Docling supporting PPTX natively — every .pptx in the knowledge base
silently fell back to a much weaker legacy reader. Also adds a
page-relative area fraction to each extracted figure, needed to filter
real diagrams from logos/icons in the next task.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UX4euTs9dwAssN2DjzH6hK
EOF
)"
```

---

### Task 2: Image caption dedup cache

**Files:**
- Create: `cortex_suite/cortex_engine/image_caption_cache.py`
- Test: `cortex_suite/tests/unit/test_image_caption_cache.py`

**Interfaces:**
- Produces: `ImageCaptionCache(db_path: str = "~/.cortex_image_caption_cache.db")` with `.get(image_bytes: bytes) -> Optional[str]` and `.put(image_bytes: bytes, caption: str) -> None`. Hashing (SHA256) happens inside the class — callers never compute or pass a hash directly, so Task 3 always calls `.get(raw_png_bytes)` / `.put(raw_png_bytes, caption)`.

- [ ] **Step 1: Write the failing test**

```python
# cortex_suite/tests/unit/test_image_caption_cache.py
import os
import tempfile

from cortex_engine.image_caption_cache import ImageCaptionCache


def test_cache_miss_then_hit(tmp_path):
    db_path = str(tmp_path / "cache.db")
    cache = ImageCaptionCache(db_path=db_path)

    image_bytes = b"fake-png-bytes-for-testing"
    assert cache.get(image_bytes) is None

    cache.put(image_bytes, "A bar chart showing quarterly revenue.")
    assert cache.get(image_bytes) == "A bar chart showing quarterly revenue."


def test_different_bytes_do_not_collide(tmp_path):
    db_path = str(tmp_path / "cache.db")
    cache = ImageCaptionCache(db_path=db_path)

    cache.put(b"image-one", "First caption")
    cache.put(b"image-two", "Second caption")

    assert cache.get(b"image-one") == "First caption"
    assert cache.get(b"image-two") == "Second caption"


def test_cache_persists_across_instances(tmp_path):
    db_path = str(tmp_path / "cache.db")
    ImageCaptionCache(db_path=db_path).put(b"persist-me", "Persisted caption")

    reopened = ImageCaptionCache(db_path=db_path)
    assert reopened.get(b"persist-me") == "Persisted caption"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/longboardfella/cortex_suite && venv/bin/python -m pytest tests/unit/test_image_caption_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cortex_engine.image_caption_cache'`

- [ ] **Step 3: Write the implementation**

```python
# cortex_suite/cortex_engine/image_caption_cache.py
"""Corpus-wide, permanent cache mapping an image's content hash to its VLM
caption. Two different files that embed the same picture (a letterhead, a
logo, a repeated template graphic) hit the cache on the second and later
occurrence instead of paying for another VLM call.
"""
import hashlib
import os
import sqlite3
from typing import Optional

DEFAULT_DB_PATH = os.path.expanduser("~/.cortex_image_caption_cache.db")


class ImageCaptionCache:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS captions ("
            "  image_hash TEXT PRIMARY KEY,"
            "  caption TEXT NOT NULL,"
            "  created_at TEXT DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def _hash(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def get(self, image_bytes: bytes) -> Optional[str]:
        row = self._conn.execute(
            "SELECT caption FROM captions WHERE image_hash = ?",
            (self._hash(image_bytes),),
        ).fetchone()
        return row[0] if row else None

    def put(self, image_bytes: bytes, caption: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO captions (image_hash, caption) VALUES (?, ?)",
            (self._hash(image_bytes), caption),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/longboardfella/cortex_suite && venv/bin/python -m pytest tests/unit/test_image_caption_cache.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/longboardfella/cortex_suite
git add cortex_engine/image_caption_cache.py tests/unit/test_image_caption_cache.py
git commit -m "$(cat <<'EOF'
Add corpus-wide image caption dedup cache

SQLite, keyed by SHA256 of image bytes. Persisted permanently at
~/.cortex_image_caption_cache.db so repeated template graphics (a
letterhead reused across many client-engagement files) are captioned
once, not once per occurrence.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UX4euTs9dwAssN2DjzH6hK
EOF
)"
```

---

### Task 3: Haiku figure captioning

**Files:**
- Create: `cortex_suite/cortex_engine/haiku_figure_caption.py`
- Test: `cortex_suite/tests/unit/test_haiku_figure_caption.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `caption_figure_with_haiku(image_bytes: bytes, context_hint: str = "") -> str`. Returns `""` on any error (missing API key, network failure, rate limit) — callers must treat an empty string as "no caption available," never raise, matching the Review Focus item about a single figure's failure not crashing the whole file.

- [ ] **Step 1: Write the failing test**

This test makes one real, cheap Haiku call (a single tiny synthetic image) rather than mocking the SDK — the existing codebase precedent (`textifier.py`) calls the real API directly with no mock layer, and a 1x1-pixel test image costs a negligible fraction of a cent.

```python
# cortex_suite/tests/unit/test_haiku_figure_caption.py
import io
import os

import pytest
from PIL import Image

from cortex_engine.haiku_figure_caption import caption_figure_with_haiku


def _tiny_red_square_png() -> bytes:
    img = Image.new("RGB", (200, 200), color=(220, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/cortex_suite/.env")),
    reason="requires cortex_suite/.env with ANTHROPIC_API_KEY",
)
def test_captions_a_real_image():
    caption = caption_figure_with_haiku(_tiny_red_square_png())
    assert isinstance(caption, str)
    assert len(caption) > 0


def test_returns_empty_string_on_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    caption = caption_figure_with_haiku(_tiny_red_square_png())
    assert caption == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/longboardfella/cortex_suite && venv/bin/python -m pytest tests/unit/test_haiku_figure_caption.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cortex_engine.haiku_figure_caption'`

- [ ] **Step 3: Write the implementation**

```python
# cortex_suite/cortex_engine/haiku_figure_caption.py
"""Caption a document figure (chart, diagram, screenshot, table image) using
Claude Haiku vision. Distinct from textifier.py's photo-captioning prompt,
which is written for photographs and explicitly tells the model to omit
logos/icons — this prompt is written for the opposite case: we've already
filtered out small decorative graphics by bounding-box area before this
function is ever called (see enhanced_ingest_cortex.py), so every image
reaching here is presumed to be genuine document content worth describing
in full, including any text, axis labels, or data visible in it.
"""
import base64
import os

MODEL = "claude-haiku-4-5-20251001"

_PROMPT = (
    "This image is a figure extracted from a business/consulting document "
    "(a chart, diagram, table screenshot, matrix, or similar). Describe it "
    "for a search index: what kind of figure it is, any axis labels or "
    "headings visible, and the key data or relationships it shows. "
    "Transcribe any short text labels verbatim if legible. "
    "Write 2-4 plain sentences. Do not use markdown, headings, or bullet points."
)


def _load_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return api_key
    env_path = os.path.expanduser("~/cortex_suite/.env")
    if not os.path.exists(env_path):
        return ""
    for line in open(env_path, encoding="utf-8").read().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key == "ANTHROPIC_API_KEY" and val:
            return val
    return ""


def caption_figure_with_haiku(image_bytes: bytes, context_hint: str = "") -> str:
    api_key = _load_api_key()
    if not api_key:
        return ""
    try:
        import anthropic
    except ImportError:
        return ""

    prompt = _PROMPT
    if context_hint:
        prompt += " " + context_hint.strip()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/png",
                                                  "data": encoded}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        for block in response.content or []:
            if hasattr(block, "text") and block.text:
                return block.text.strip()
        return ""
    except Exception:
        return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/longboardfella/cortex_suite && venv/bin/python -m pytest tests/unit/test_haiku_figure_caption.py -v`
Expected: PASS (both tests — the real-API test costs a fraction of a cent)

- [ ] **Step 5: Commit**

```bash
cd /home/longboardfella/cortex_suite
git add cortex_engine/haiku_figure_caption.py tests/unit/test_haiku_figure_caption.py
git commit -m "$(cat <<'EOF'
Add Haiku-based document figure captioning

Replaces the local-Ollama VLM path for document figures. That path
(describe_image_with_vlm_for_ingestion) is already known-unreliable
per project memory (emits reasoning-scaffolding instead of captions).
Returns "" on any error so a single figure's caption failure never
crashes the rest of a document's extraction.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UX4euTs9dwAssN2DjzH6hK
EOF
)"
```

---

### Task 4: Text cleanup helper (marker-run stripping + filename prefix)

**Files:**
- Create: `nemoclaw_doc_text_cleanup.py` (home directory, alongside its siblings — not inside cortex_suite, since it's imported by the home-directory extract drivers in Task 8, not by cortex_engine)
- Test: `test_nemoclaw_doc_text_cleanup.py` (home directory)

**Interfaces:**
- Produces: `strip_marker_runs(text: str, min_run: int = 4) -> str` and `prefix_filename(text: str, file_path: str) -> str`. Both are pure string→string functions with no I/O, so both take plain strings/paths and return a string — no shared state with any other task.

- [ ] **Step 1: Write the failing test**

```python
# /home/longboardfella/test_nemoclaw_doc_text_cleanup.py
from nemoclaw_doc_text_cleanup import strip_marker_runs, prefix_filename


def test_strips_a_run_of_bare_reference_markers():
    text = (
        "Title: Major Findings/Recommendations for Action\n"
        "**Degree of Benefit**\n"
        "**Ease of Implementation**\n"
        "**1a**\n"
        "**29**\n"
        "**9**\n"
        "**1b**\n"
        "Legend: Items in the Green zone offer the greatest benefit"
    )
    result = strip_marker_runs(text)
    assert "**1a**" not in result
    assert "**29**" not in result
    assert "Degree of Benefit" in result
    assert "Legend: Items in the Green zone" in result
    assert "Reference markers on this diagram: 1a, 29, 9, 1b" in result


def test_does_not_strip_a_lone_isolated_number():
    text = "Annual Report\n**2011**\nPrepared by SMS Management"
    result = strip_marker_runs(text)
    assert result == text


def test_prefix_filename_adds_cleaned_filename():
    text = "Title: Major Findings/Recommendations for Action\nSome body text"
    result = prefix_filename(text, "/some/path/Major Findings Heat Map - A3 ra.pptx")
    assert result.startswith("Filename: Major Findings Heat Map - A3 ra\n")
    assert "Some body text" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/longboardfella && venv/bin/python -m pytest test_nemoclaw_doc_text_cleanup.py -v 2>&1 || python3 -m pytest test_nemoclaw_doc_text_cleanup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nemoclaw_doc_text_cleanup'`

- [ ] **Step 3: Write the implementation**

This is the exact logic already prototyped and validated against the real heatmap text and a synthetic false-positive case earlier this session.

```python
# /home/longboardfella/nemoclaw_doc_text_cleanup.py
"""Shared text-quality fixes applied to extracted document text before it's
chunked and embedded (see nemoclaw-kdocs-extract.py / nemoclaw-cdocs-extract.py,
via nemoclaw_doc_extract_one.py). Two problems, one module:

1. A diagram's bare reference-marker numbers (each its own text box/line in
   the source file, e.g. "1a", "29", "9") sit between meaningful phrases and
   dilute the embedding. strip_marker_runs() only touches a RUN of >=4 such
   lines in a row -- the actual signature of diagram-label noise -- so a lone
   number (a year, a page number, a single stat) is never touched.
2. The filename is often the most distinctive vocabulary a document has (a
   file called "Major Findings Heat Map - A3 ra.pptx" whose internal slide
   title is just "Major Findings/Recommendations for Action") but never
   enters the embedded text today -- only the internal title does.
   prefix_filename() fixes that.
"""
import os
import re

_MARKER_LINE_RE = re.compile(r"^(\*+\d{1,3}[a-z]?\*+)+$")
_MARKER_TOKEN_RE = re.compile(r"\d{1,3}[a-z]?")


def strip_marker_runs(text: str, min_run: int = 4) -> str:
    lines = text.split("\n")
    out: list[str] = []
    markers: list[str] = []
    buf: list[str] = []

    def flush():
        nonlocal buf
        if len(buf) >= min_run:
            for stripped in buf:
                markers.extend(_MARKER_TOKEN_RE.findall(stripped))
        else:
            out.extend(buf)
        buf = []

    for line in lines:
        stripped = line.strip()
        if _MARKER_LINE_RE.match(stripped):
            buf.append(stripped)
        else:
            flush()
            out.append(line)
    flush()

    result = "\n".join(out)
    if markers:
        result += f"\n\nReference markers on this diagram: {', '.join(markers)}"
    return result


def prefix_filename(text: str, file_path: str) -> str:
    base = os.path.splitext(os.path.basename(file_path))[0]
    return f"Filename: {base}\n{text}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/longboardfella && python3 -m pytest test_nemoclaw_doc_text_cleanup.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/longboardfella
git add nemoclaw_doc_text_cleanup.py test_nemoclaw_doc_text_cleanup.py
git commit -m "$(cat <<'EOF'
Add doc-extract text cleanup: marker-run stripping + filename prefix

Strips runs of >=4 bare diagram reference-marker lines into a compact
trailing note (tested to leave a lone isolated number untouched), and
prefixes the cleaned filename into extracted text so distinctive
filename vocabulary (often more searchable than the internal document
title) becomes embeddable.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UX4euTs9dwAssN2DjzH6hK
EOF
)"
```

---

### Task 5: Wire size filter + dedup + Haiku into figure enrichment

**Files:**
- Modify: `cortex_suite/cortex_engine/enhanced_ingest_cortex.py` (`_enrich_docling_figures`, `_summarize_figure_with_vlm`, `__init__`)
- Test: `cortex_suite/tests/unit/test_enrich_docling_figures.py`

**Interfaces:**
- Consumes: `ImageCaptionCache` from Task 2 (`.get(bytes) -> Optional[str]`, `.put(bytes, str) -> None`), `caption_figure_with_haiku` from Task 3 (`bytes -> str`), `area_frac` key on each figure payload from Task 1.
- Produces: `_enrich_docling_figures(document, skip_image_processing, min_area_frac=0.10)` — a new `min_area_frac` parameter (default `0.10`, matching the chosen threshold). Figures below the threshold are left untouched (no VLM call, no "Figure Intelligence" text for them) — the existing behavior of skipping ALL figures when `skip_image_processing=True` is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# cortex_suite/tests/unit/test_enrich_docling_figures.py
import base64
import io
from unittest.mock import patch

from llama_index.core import Document
from PIL import Image

from cortex_engine.enhanced_ingest_cortex import EnhancedDocumentProcessor


def _png_b64(size=(400, 400)) -> str:
    img = Image.new("RGB", size, color=(10, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_doc_with_figures(area_fracs):
    doc = Document(text="Some body text")
    figures = []
    payloads = []
    for i, frac in enumerate(area_fracs):
        figures.append({"index": i, "caption": "", "area_frac": frac})
        payloads.append({"index": i, "image_base64": _png_b64(), "area_frac": frac})
    doc.metadata["docling_figures"] = figures
    doc.metadata["docling_figures_payload"] = payloads
    return doc


def test_small_figure_below_threshold_is_never_captioned():
    processor = EnhancedDocumentProcessor(enable_docling=False)
    doc = _make_doc_with_figures([0.02])  # 2% -- below the 10% threshold
    with patch("cortex_engine.enhanced_ingest_cortex.caption_figure_with_haiku") as mock_caption:
        processor._enrich_docling_figures(doc, skip_image_processing=False)
        mock_caption.assert_not_called()
    assert "Figure Intelligence" not in doc.text


def test_large_figure_above_threshold_is_captioned():
    processor = EnhancedDocumentProcessor(enable_docling=False)
    doc = _make_doc_with_figures([0.35])  # 35% -- above the 10% threshold
    with patch("cortex_engine.enhanced_ingest_cortex.caption_figure_with_haiku",
               return_value="A blue rectangle.") as mock_caption:
        processor._enrich_docling_figures(doc, skip_image_processing=False)
        mock_caption.assert_called_once()
    assert "Figure Intelligence" in doc.text
    assert "A blue rectangle." in doc.text


def test_dedup_cache_prevents_second_haiku_call(tmp_path, monkeypatch):
    from cortex_engine.image_caption_cache import ImageCaptionCache
    cache_path = str(tmp_path / "dedup.db")
    monkeypatch.setattr("cortex_engine.enhanced_ingest_cortex.DEFAULT_CACHE_PATH", cache_path)

    processor = EnhancedDocumentProcessor(enable_docling=False)
    same_png = _png_b64()
    doc1 = Document(text="Doc one")
    doc1.metadata["docling_figures"] = [{"index": 0, "caption": "", "area_frac": 0.5}]
    doc1.metadata["docling_figures_payload"] = [{"index": 0, "image_base64": same_png, "area_frac": 0.5}]
    doc2 = Document(text="Doc two")
    doc2.metadata["docling_figures"] = [{"index": 0, "caption": "", "area_frac": 0.5}]
    doc2.metadata["docling_figures_payload"] = [{"index": 0, "image_base64": same_png, "area_frac": 0.5}]

    with patch("cortex_engine.enhanced_ingest_cortex.caption_figure_with_haiku",
               return_value="A shared logo.") as mock_caption:
        processor._enrich_docling_figures(doc1, skip_image_processing=False)
        processor._enrich_docling_figures(doc2, skip_image_processing=False)
        assert mock_caption.call_count == 1, "second identical image must hit the dedup cache, not call Haiku again"
    assert "A shared logo." in doc1.text
    assert "A shared logo." in doc2.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/longboardfella/cortex_suite && venv/bin/python -m pytest tests/unit/test_enrich_docling_figures.py -v`
Expected: FAIL — `_enrich_docling_figures` doesn't filter by `area_frac` yet, doesn't call `caption_figure_with_haiku`, and `DEFAULT_CACHE_PATH` doesn't exist in `enhanced_ingest_cortex.py` yet.

- [ ] **Step 3: Wire the imports and cache instance**

In `cortex_suite/cortex_engine/enhanced_ingest_cortex.py`, near the top with the other imports, add:

```python
from .haiku_figure_caption import caption_figure_with_haiku
from .image_caption_cache import ImageCaptionCache, DEFAULT_DB_PATH as DEFAULT_CACHE_PATH
```

In `EnhancedDocumentProcessor.__init__` (the same `__init__` read in Task 1's investigation), after the existing `self.enable_ocr = enable_ocr` line, add:

```python
        self._caption_cache = ImageCaptionCache(db_path=DEFAULT_CACHE_PATH)
```

- [ ] **Step 4: Kill the redundant always-on Ollama path**

Still in `__init__`, find the `create_docling_reader(...)` call (identified during investigation as never passing `skip_vlm_processing`, so it silently defaulted to `False` and ran an independent, broken, invisible-to-search Ollama captioning pass on every PDF). Change:

```python
                self.docling_reader = create_docling_reader(
                    ocr_enabled=enable_ocr,
                    table_structure_recognition=True
                )
```

to:

```python
                self.docling_reader = create_docling_reader(
                    ocr_enabled=enable_ocr,
                    table_structure_recognition=True,
                    skip_vlm_processing=True,  # figure captioning happens once, in
                                                # _enrich_docling_figures below, where
                                                # its output actually reaches document.text
                )
```

- [ ] **Step 5: Rewrite the figure enrichment to filter, dedup, and call Haiku**

Replace the body of `_enrich_docling_figures` and `_summarize_figure_with_vlm` (found during Task 1's investigation at the locations documented in this plan's problem statement). Full replacement:

```python
    def _enrich_docling_figures(self, document: Document, skip_image_processing: bool,
                                 min_area_frac: float = 0.10) -> None:
        """Convert Docling figure payloads into VLM summaries when allowed.

        Only figures whose bounding-box area covers >= min_area_frac of their
        page/slide are captioned -- below that, a figure is presumed to be a
        logo/icon/decorative graphic, not real content (verified against real
        bounding-box data: at a 10% threshold, real diagrams/charts/photos are
        reliably separated from template decoration).
        """
        figures = document.metadata.get('docling_figures') or []
        payloads = document.metadata.pop('docling_figures_payload', None)

        if not figures or not payloads:
            return

        if skip_image_processing:
            for figure in figures:
                figure['vlm_status'] = 'skipped'
            return

        payload_map = {
            payload.get('index'): payload
            for payload in payloads
            if isinstance(payload, dict) and payload.get('index') is not None
        }

        figure_blocks: List[str] = []
        for figure in figures:
            idx = figure.get('index')
            if idx is None or idx not in payload_map:
                continue
            payload = payload_map[idx]
            area_frac = payload.get('area_frac')
            if area_frac is None or area_frac < min_area_frac:
                figure['vlm_status'] = 'skipped_small'
                continue

            summary = self._summarize_figure_with_vlm(payload)
            if summary:
                figure['vlm_summary'] = summary
                figure['vlm_status'] = 'processed'
                heading = figure.get('caption') or f"Figure {idx + 1}"
                figure_blocks.append(f"### Figure {idx + 1}: {heading}\n{summary}")
            else:
                figure['vlm_status'] = 'error'

        if figure_blocks:
            document.text += "\n\n## Figure Intelligence\n" + "\n\n".join(figure_blocks)

    def _summarize_figure_with_vlm(self, payload: Dict[str, Any]) -> Optional[str]:
        """Decode the Docling figure payload, check the dedup cache, and call Haiku."""
        image_b64 = payload.get('image_base64')
        if not image_b64:
            return None
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as decode_error:
            logger.warning(f"Invalid Docling figure payload: {decode_error}")
            return None

        cached = self._caption_cache.get(image_bytes)
        if cached is not None:
            return cached

        caption = caption_figure_with_haiku(image_bytes)
        if caption:
            self._caption_cache.put(image_bytes, caption)
            return caption
        return None
```

This removes the old `tempfile`/`vlm_fn` plumbing entirely (Haiku takes raw bytes directly, no temp PNG file needed) — the `_summarize_figure_with_vlm` signature drops its old `vlm_fn` parameter since there's now exactly one VLM backend, not a pluggable one.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /home/longboardfella/cortex_suite && venv/bin/python -m pytest tests/unit/test_enrich_docling_figures.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Run the full unit test suite to check for regressions**

Run: `cd /home/longboardfella/cortex_suite && venv/bin/python -m pytest tests/unit -v`
Expected: PASS — no other test should reference the old `_summarize_figure_with_vlm(payload, vlm_fn)` two-argument signature or the old `create_docling_reader` call without `skip_vlm_processing`. If any test fails on those grounds, update that test's call site to match the new signature (do not revert this task's change).

- [ ] **Step 8: Commit**

```bash
cd /home/longboardfella/cortex_suite
git add cortex_engine/enhanced_ingest_cortex.py tests/unit/test_enrich_docling_figures.py
git commit -m "$(cat <<'EOF'
Filter figures by size, dedup, and caption with Haiku

_enrich_docling_figures now only sends figures >=10% of page area to
VLM captioning (logos/icons never reach Haiku), checks the corpus-wide
dedup cache before every call, and uses Haiku instead of local Ollama.
Also passes skip_vlm_processing=True to create_docling_reader, which
was previously omitted -- that silently ran an independent, broken,
invisible-to-search Ollama captioning pass on every PDF (its output
went only to metadata, which the extract stage discards before it
ever reaches the search index).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UX4euTs9dwAssN2DjzH6hK
EOF
)"
```

---

### Task 6: LibreOffice legacy-format conversion helper

**Files:**
- Create: `nemoclaw_legacy_office_convert.py` (home directory)
- Test: `test_nemoclaw_legacy_office_convert.py` (home directory)

**Interfaces:**
- Produces: `convert_legacy_office(path: str, out_dir: str, timeout: int = 60) -> str` — converts a `.doc`/`.ppt` file to `.docx`/`.pptx` in `out_dir`, returns the converted file's path. Raises `LegacyConvertError` (a new, small exception class this module defines) on failure or timeout — callers (Task 8) must catch this specific exception, not a bare `Exception`, so the per-file isolation logic can distinguish "conversion failed" from other failure modes in its skip-list reason.

- [ ] **Step 1: Confirm LibreOffice is installed**

```bash
which soffice libreoffice 2>&1
```

Expected: at least one of `soffice` or `libreoffice` resolves to a path. If neither is installed, install it first: `sudo apt install -y libreoffice` (ask the user to run this via `! sudo apt install -y libreoffice` if a password prompt is required — do not run `sudo` for a package install without the user's explicit go-ahead in this session).

- [ ] **Step 2: Write the failing test**

```python
# /home/longboardfella/test_nemoclaw_legacy_office_convert.py
import glob
import os

import pytest

from nemoclaw_legacy_office_convert import convert_legacy_office, LegacyConvertError

FIXTURE_PPT = "/mnt/c/Users/paul/OneDrive - VentraIP Australia/04_Knowledge/SMS knowledge/SMS Knowledge Base/Fireservices- Emergency Services/In Progress/COP Conceptual Diagram_041111.ppt"


@pytest.mark.skipif(not os.path.exists(FIXTURE_PPT), reason="fixture file not present on this machine")
def test_converts_legacy_ppt_to_pptx(tmp_path):
    out_path = convert_legacy_office(FIXTURE_PPT, str(tmp_path))
    assert out_path.endswith(".pptx")
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_raises_on_missing_file(tmp_path):
    with pytest.raises(LegacyConvertError):
        convert_legacy_office("/nonexistent/path/does-not-exist.ppt", str(tmp_path))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/longboardfella && python3 -m pytest test_nemoclaw_legacy_office_convert.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nemoclaw_legacy_office_convert'`

- [ ] **Step 4: Write the implementation**

```python
# /home/longboardfella/nemoclaw_legacy_office_convert.py
"""Convert legacy binary .doc/.ppt files to .docx/.pptx via headless
LibreOffice, so they can go through the same Docling pipeline as every
other document. Docling only reads OOXML formats -- pre-2007 binary Office
files are invisible to it otherwise.
"""
import glob
import os
import subprocess

_TARGET_FILTER = {".doc": "docx", ".ppt": "pptx"}


class LegacyConvertError(Exception):
    pass


def _soffice_binary() -> str:
    for candidate in ("soffice", "libreoffice"):
        found = subprocess.run(["which", candidate], capture_output=True, text=True)
        if found.returncode == 0 and found.stdout.strip():
            return candidate
    raise LegacyConvertError("neither 'soffice' nor 'libreoffice' found on PATH")


def convert_legacy_office(path: str, out_dir: str, timeout: int = 60) -> str:
    if not os.path.exists(path):
        raise LegacyConvertError(f"source file does not exist: {path}")

    ext = os.path.splitext(path)[1].lower()
    target_ext = _TARGET_FILTER.get(ext)
    if not target_ext:
        raise LegacyConvertError(f"unsupported legacy extension: {ext}")

    os.makedirs(out_dir, exist_ok=True)
    binary = _soffice_binary()

    try:
        result = subprocess.run(
            [binary, "--headless", "--convert-to", target_ext, "--outdir", out_dir, path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise LegacyConvertError(f"LibreOffice conversion timed out after {timeout}s") from e

    if result.returncode != 0:
        raise LegacyConvertError(f"LibreOffice conversion failed (rc={result.returncode}): {result.stderr[:300]}")

    stem = os.path.splitext(os.path.basename(path))[0]
    matches = glob.glob(os.path.join(out_dir, f"{stem}.{target_ext}"))
    if not matches:
        raise LegacyConvertError(f"conversion reported success but no output file found for {path}")
    return matches[0]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/longboardfella && python3 -m pytest test_nemoclaw_legacy_office_convert.py -v`
Expected: PASS (the missing-file test always runs; the real-conversion test runs if the fixture exists on this machine, which it does)

- [ ] **Step 6: Commit**

```bash
cd /home/longboardfella
git add nemoclaw_legacy_office_convert.py test_nemoclaw_legacy_office_convert.py
git commit -m "$(cat <<'EOF'
Add LibreOffice legacy .doc/.ppt conversion helper

Docling only reads OOXML formats. Converts to .docx/.pptx first via
headless LibreOffice so the 1,532 legacy binary files across both
knowledge roots go through the same pipeline as everything else,
instead of being permanently out of scope.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UX4euTs9dwAssN2DjzH6hK
EOF
)"
```

---

### Task 7: Single-file extraction worker

**Files:**
- Create: `nemoclaw_doc_extract_one.py` (home directory)
- Test: `test_nemoclaw_doc_extract_one.py` (home directory)

**Interfaces:**
- Consumes: `strip_marker_runs`/`prefix_filename` (Task 4), `convert_legacy_office`/`LegacyConvertError` (Task 6), and `enhanced_load_documents` from `cortex_engine.enhanced_ingest_cortex` (existing, now safe to call with `skip_image_processing=False` after Tasks 1-6).
- Produces: a CLI script — `python3 nemoclaw_doc_extract_one.py <absolute_file_path>` prints exactly one JSON line to stdout: `{"source_file": <original path>, "text": <cleaned text>, "size": int, "mtime": int, "ok": true}` on success, or `{"source_file": ..., "ok": false, "error": <message>}` on failure. This is the isolation unit Task 8's driver spawns per file under a hard subprocess timeout.

- [ ] **Step 1: Write the failing test**

```python
# /home/longboardfella/test_nemoclaw_doc_extract_one.py
import json
import subprocess
import sys

FIXTURE = "/home/longboardfella/cortex_suite/tests/fixtures/sample_findings_heatmap.pptx"


def test_extracts_the_heatmap_fixture_with_cleaned_text():
    result = subprocess.run(
        [sys.executable, "/home/longboardfella/nemoclaw_doc_extract_one.py", FIXTURE],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    rec = json.loads(result.stdout.strip().splitlines()[-1])
    assert rec["ok"] is True
    assert rec["source_file"] == FIXTURE
    assert "Filename: sample_findings_heatmap" in rec["text"]
    assert "Ease of Implementation" in rec["text"]
    # the marker-run cleanup must have fired: no bare "**1a**"-style line left
    assert "**1a**" not in rec["text"]


def test_reports_ok_false_for_missing_file():
    result = subprocess.run(
        [sys.executable, "/home/longboardfella/nemoclaw_doc_extract_one.py", "/nonexistent/file.pdf"],
        capture_output=True, text=True, timeout=30,
    )
    rec = json.loads(result.stdout.strip().splitlines()[-1])
    assert rec["ok"] is False
    assert "error" in rec
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/longboardfella && python3 -m pytest test_nemoclaw_doc_extract_one.py -v`
Expected: FAIL — `nemoclaw_doc_extract_one.py` doesn't exist yet.

- [ ] **Step 3: Write the implementation**

```python
#!/home/longboardfella/cortex_suite/venv/bin/python3
"""Extract text (+ Haiku-captioned figures) from exactly ONE file. Invoked as
a subprocess, one call per file, by nemoclaw-kdocs-extract.py / -cdocs-. This
is the isolation unit: a hang or crash on this one file costs at most the
caller's per-file timeout, never the rest of a batch.
"""
import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME", "/mnt/f/hf-home")
sys.path.insert(0, "/home/longboardfella/cortex_suite")
sys.path.insert(0, "/home/longboardfella")

from nemoclaw_doc_text_cleanup import strip_marker_runs, prefix_filename
from nemoclaw_legacy_office_convert import convert_legacy_office, LegacyConvertError

LEGACY_EXTS = (".doc", ".ppt")


def extract_one(path: str) -> dict:
    rec = {"source_file": path}
    if not os.path.exists(path):
        rec["ok"] = False
        rec["error"] = "file does not exist"
        return rec

    try:
        size = os.path.getsize(path)
        mtime = int(os.path.getmtime(path))
    except OSError as e:
        rec["ok"] = False
        rec["error"] = f"stat failed: {e}"
        return rec

    actual_path = path
    tmp_dir = None
    if path.lower().endswith(LEGACY_EXTS):
        tmp_dir = tempfile.mkdtemp(prefix="nemoclaw_legacy_")
        try:
            actual_path = convert_legacy_office(path, tmp_dir)
        except LegacyConvertError as e:
            rec["ok"] = False
            rec["error"] = f"legacy conversion failed: {e}"
            return rec

    try:
        from cortex_engine.enhanced_ingest_cortex import enhanced_load_documents
        docs = enhanced_load_documents([actual_path], skip_image_processing=False)
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"extraction failed: {type(e).__name__}: {e}"
        return rec
    finally:
        if tmp_dir:
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    text = "\n\n".join(getattr(d, "text", "") or "" for d in docs).strip()
    if text:
        text = strip_marker_runs(text)
        text = prefix_filename(text, path)

    rec["ok"] = True
    rec["text"] = text
    rec["size"] = size
    rec["mtime"] = mtime
    return rec


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error": "usage: nemoclaw_doc_extract_one.py <file_path>"}))
        sys.exit(1)
    print(json.dumps(extract_one(sys.argv[1]), ensure_ascii=False))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/longboardfella && python3 -m pytest test_nemoclaw_doc_extract_one.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/longboardfella
git add nemoclaw_doc_extract_one.py test_nemoclaw_doc_extract_one.py
git commit -m "$(cat <<'EOF'
Add single-file document extraction worker

The isolation unit for Stage 1: extracts exactly one file per process
invocation (legacy conversion, Docling+Haiku figure captioning, text
cleanup), so a per-file subprocess timeout in the driver script can
bound a single pathological file's cost without losing its batch-mates.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UX4euTs9dwAssN2DjzH6hK
EOF
)"
```

---

### Task 8: Per-file isolation in both Stage-1 extract drivers

**Files:**
- Modify: `nemoclaw-kdocs-extract.py`, `nemoclaw-cdocs-extract.py` (home directory)
- Test: `test_nemoclaw_extract_isolation.py` (home directory)

**Interfaces:**
- Consumes: `nemoclaw_doc_extract_one.py` (Task 7) as a subprocess, one call per file.
- Produces: both extract scripts keep their existing external contract unchanged (stdin: newline-separated file paths; stdout: one JSON line per file) — Task 9 (Stage 2) needs no changes to how it invokes them. What changes internally: instead of one `enhanced_load_documents()` call for the whole batch, each file gets its own subprocess with a hard timeout; a file that exceeds the timeout is permanently recorded in a shared skip-list and reported as an error, never retried automatically.

- [ ] **Step 1: Write the failing test**

This test proves the actual resilience property the user asked for: one bad file in a batch must not block its neighbors, and must not be retried forever.

```python
# /home/longboardfella/test_nemoclaw_extract_isolation.py
import json
import os
import subprocess
import sys
import tempfile

EXTRACT = "/home/longboardfella/nemoclaw-kdocs-extract.py"
GOOD_FIXTURE = "/home/longboardfella/cortex_suite/tests/fixtures/sample_findings_heatmap.pptx"


def _run_extract(paths, skiplist_path, timeout=30):
    env = dict(os.environ)
    env["NEMOCLAW_EXTRACT_SKIPLIST"] = skiplist_path
    env["NEMOCLAW_EXTRACT_PER_FILE_TIMEOUT"] = "2"  # short, for a fast test
    result = subprocess.run(
        [EXTRACT], input="\n".join(paths), capture_output=True, text=True,
        timeout=timeout, env=env,
    )
    return result


def test_a_timing_out_file_does_not_block_its_batch_mates(tmp_path, monkeypatch):
    skiplist_path = str(tmp_path / "skiplist.json")
    # a "sleep" script standing in for a pathological file: nemoclaw_doc_extract_one.py
    # can't sleep on command, so this test exercises the driver's timeout handling
    # directly by pointing NEMOCLAW_EXTRACT_WORKER at a fake worker for one path.
    fake_worker = tmp_path / "fake_slow_worker.py"
    fake_worker.write_text(
        "import sys, json, time\n"
        "path = sys.argv[1]\n"
        "if 'SLOWFILE' in path:\n"
        "    time.sleep(30)\n"
        "print(json.dumps({'source_file': path, 'ok': True, 'text': 'fast ok', "
        "'size': 1, 'mtime': 1}))\n"
    )
    monkeypatch.setenv("NEMOCLAW_EXTRACT_WORKER", str(fake_worker))
    result = _run_extract(["/tmp/GOODFILE_a.pdf", "/tmp/SLOWFILE_b.pdf", "/tmp/GOODFILE_c.pdf"],
                           skiplist_path)
    lines = [json.loads(l) for l in result.stdout.strip().splitlines() if l.strip()]
    by_path = {r["source_file"]: r for r in lines}

    assert by_path["/tmp/GOODFILE_a.pdf"]["ok"] is True
    assert by_path["/tmp/GOODFILE_c.pdf"]["ok"] is True
    assert by_path["/tmp/SLOWFILE_b.pdf"]["ok"] is False

    skiplist = json.loads(open(skiplist_path).read())
    assert "/tmp/SLOWFILE_b.pdf" in skiplist


def test_a_file_already_on_the_skiplist_is_skipped_without_running(tmp_path, monkeypatch):
    skiplist_path = str(tmp_path / "skiplist.json")
    with open(skiplist_path, "w") as f:
        json.dump({"/tmp/KNOWN_BAD.pdf": "previously hung"}, f)

    fake_worker = tmp_path / "worker_that_should_not_run.py"
    fake_worker.write_text(
        "import sys\nraise SystemExit('worker should never be invoked for a skiplisted file')\n"
    )
    monkeypatch.setenv("NEMOCLAW_EXTRACT_WORKER", str(fake_worker))
    result = _run_extract(["/tmp/KNOWN_BAD.pdf"], skiplist_path)
    lines = [json.loads(l) for l in result.stdout.strip().splitlines() if l.strip()]
    assert lines[0]["ok"] is False
    assert "skiplist" in lines[0]["error"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/longboardfella && python3 -m pytest test_nemoclaw_extract_isolation.py -v`
Expected: FAIL — `nemoclaw-kdocs-extract.py` doesn't yet honor `NEMOCLAW_EXTRACT_SKIPLIST`, `NEMOCLAW_EXTRACT_PER_FILE_TIMEOUT`, or `NEMOCLAW_EXTRACT_WORKER`.

- [ ] **Step 3: Rewrite nemoclaw-kdocs-extract.py as a per-file-isolating driver**

Full replacement (keeps the same stdin/stdout contract Stage 2 already relies on — verified against `nemoclaw-kdocs-ingest.py`'s call site during investigation: `subprocess.run([EXTRACT], input="\n".join(batch), ...)`, reading `proc.stdout.splitlines()` as JSON lines):

```python
#!/usr/bin/env python3
"""Stage 1 driver (04_Knowledge root): reads absolute file paths on stdin (one
per line), spawns nemoclaw_doc_extract_one.py as a SEPARATE SUBPROCESS PER
FILE with a hard timeout, and prints one JSON line per file to stdout -- same
contract as before. This is the fix for a real, twice-experienced failure
mode in this codebase: a single pathological file hanging the whole batch
(once for 7.5h before this driver existed, once for 3h25m during this
project's own sizing scan). A file that exceeds the timeout is permanently
recorded in a shared skip-list (not retried on future runs) instead of
silently dooming its entire batch forever.

Environment overrides (for testing; production uses the defaults):
  NEMOCLAW_EXTRACT_WORKER          -- path to the worker script
  NEMOCLAW_EXTRACT_SKIPLIST        -- path to the skip-list JSON file
  NEMOCLAW_EXTRACT_PER_FILE_TIMEOUT -- seconds
"""
import json
import os
import subprocess
import sys

WORKER = os.environ.get("NEMOCLAW_EXTRACT_WORKER", "/home/longboardfella/nemoclaw_doc_extract_one.py")
SKIPLIST_PATH = os.environ.get("NEMOCLAW_EXTRACT_SKIPLIST",
                                os.path.expanduser("~/.nemoclaw-doc-extract-skiplist.json"))
PER_FILE_TIMEOUT = int(os.environ.get("NEMOCLAW_EXTRACT_PER_FILE_TIMEOUT", "120"))


def load_skiplist() -> dict:
    if os.path.exists(SKIPLIST_PATH):
        try:
            return json.loads(open(SKIPLIST_PATH).read())
        except Exception:
            return {}
    return {}


def save_skiplist(skiplist: dict) -> None:
    os.makedirs(os.path.dirname(SKIPLIST_PATH) or ".", exist_ok=True)
    with open(SKIPLIST_PATH, "w") as f:
        json.dump(skiplist, f, indent=2)


def main():
    paths = [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
    skiplist = load_skiplist()

    for path in paths:
        if path in skiplist:
            print(json.dumps({"source_file": path, "ok": False,
                               "error": f"on skiplist: {skiplist[path]}"}), flush=True)
            continue

        try:
            proc = subprocess.run(
                [sys.executable, WORKER, path],
                capture_output=True, text=True, timeout=PER_FILE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            skiplist[path] = f"exceeded {PER_FILE_TIMEOUT}s timeout"
            save_skiplist(skiplist)
            print(json.dumps({"source_file": path, "ok": False,
                               "error": f"HUNG: exceeded {PER_FILE_TIMEOUT}s, skipped permanently"}),
                  flush=True)
            continue

        out = (proc.stdout or "").strip()
        last_line = out.splitlines()[-1] if out else ""
        try:
            rec = json.loads(last_line)
        except Exception:
            rec = {"source_file": path, "ok": False,
                   "error": f"worker produced no valid JSON (rc={proc.returncode}): {proc.stderr[:200]}"}
        print(json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create the cdocs variant**

`nemoclaw-cdocs-extract.py` is identical except its default worker path is unchanged (same worker handles both roots — the root-specific logic lives in the ingest/Stage-2 scripts, not the extract driver). Copy the same file verbatim to `/home/longboardfella/nemoclaw-cdocs-extract.py` (the two scripts are kept as parallel copies per this project's existing convention, confirmed during investigation) — only the module docstring's first line differs ("01_Clients/Active root" instead of "04_Knowledge root").

```bash
cp /home/longboardfella/nemoclaw-kdocs-extract.py /home/longboardfella/nemoclaw-cdocs-extract.py
sed -i '0,/04_Knowledge root/s//01_Clients\/Active root/' /home/longboardfella/nemoclaw-cdocs-extract.py
chmod +x /home/longboardfella/nemoclaw-kdocs-extract.py /home/longboardfella/nemoclaw-cdocs-extract.py
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/longboardfella && python3 -m pytest test_nemoclaw_extract_isolation.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Run a real end-to-end smoke test against the actual heatmap fixture**

```bash
echo "/home/longboardfella/cortex_suite/tests/fixtures/sample_findings_heatmap.pptx" | \
  /home/longboardfella/nemoclaw-kdocs-extract.py
```

Expected: one JSON line, `"ok": true`, `"text"` containing `"Filename: sample_findings_heatmap"` and `"Ease of Implementation"`.

- [ ] **Step 7: Commit**

```bash
cd /home/longboardfella
git add nemoclaw-kdocs-extract.py nemoclaw-cdocs-extract.py test_nemoclaw_extract_isolation.py
git commit -m "$(cat <<'EOF'
Isolate Stage-1 extraction to one subprocess per file

Previously the whole batch (30-60 files) was processed inside a
single enhanced_load_documents() call -- a pathological file hung the
process for 7.5h once before (see the EXTRACT_TIMEOUT comment in
nemoclaw-kdocs-ingest.py) and again during this project's own sizing
scan (3h25m). Now each file gets its own subprocess and a per-file
timeout; a file that times out is permanently recorded in a shared
skip-list instead of dooming its whole batch on every future run.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UX4euTs9dwAssN2DjzH6hK
EOF
)"
```

---

### Task 9: PIPELINE_VERSION forces the backfill through existing resumability

**Files:**
- Modify: `nemoclaw-kdocs-ingest.py`, `nemoclaw-cdocs-ingest.py` (home directory)
- Test: `test_nemoclaw_ingest_pipeline_version.py` (home directory)

**Interfaces:**
- Produces: a `PIPELINE_VERSION` module-level constant in both ingest scripts, folded into the `fkey` string used for both the stored-metadata comparison (`load_existing_keys`) and the newly-computed comparison in the `todo`-building loop. No other behavior changes — Stage 2's existing batching, timeout, and Chroma-upsert logic (already investigated and confirmed correct) are untouched.

- [ ] **Step 1: Write the failing test**

This test exercises the actual bug this task closes: without a version in the key, a file whose size/mtime hasn't changed is invisible to a re-run even after the extraction code changes underneath it.

```python
# /home/longboardfella/test_nemoclaw_ingest_pipeline_version.py
import re


def test_kdocs_ingest_fkey_includes_pipeline_version():
    src = open("/home/longboardfella/nemoclaw-kdocs-ingest.py").read()
    assert re.search(r'^PIPELINE_VERSION\s*=', src, re.M), \
        "expected a module-level PIPELINE_VERSION constant"
    # the fkey computed for the todo list must include PIPELINE_VERSION, not just size_mtime
    todo_fkey_line = re.search(r'fk\s*=\s*f".*"', src)
    assert todo_fkey_line is not None
    assert "PIPELINE_VERSION" in todo_fkey_line.group(0)


def test_cdocs_ingest_fkey_includes_pipeline_version():
    src = open("/home/longboardfella/nemoclaw-cdocs-ingest.py").read()
    assert re.search(r'^PIPELINE_VERSION\s*=', src, re.M)
    todo_fkey_line = re.search(r'fk\s*=\s*f".*"', src)
    assert todo_fkey_line is not None
    assert "PIPELINE_VERSION" in todo_fkey_line.group(0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/longboardfella && python3 -m pytest test_nemoclaw_ingest_pipeline_version.py -v`
Expected: FAIL — no `PIPELINE_VERSION` constant exists yet in either file.

- [ ] **Step 3: Add PIPELINE_VERSION to nemoclaw-kdocs-ingest.py**

Near the top, alongside the other module constants (`CHUNK_MAX, OVERLAP, MIN_CHUNK = 3000, 200, 80`), add:

```python
# Bump this whenever the extraction/chunking pipeline changes in a way that
# should reprocess every file, even ones whose size+mtime are unchanged on
# disk (e.g. today: PPTX Docling support, Haiku figure captioning, chunk
# text cleanup). The version rides inside the resumability key itself, so
# bumping it makes the existing todo-list mechanism do a full backfill with
# no separate backfill runner needed.
PIPELINE_VERSION = "2026-09-26-diagram-pipeline-v1"
```

In the `todo` construction loop, change:

```python
            fk = f"{os.path.getsize(f)}_{int(os.path.getmtime(f))}"
```

to:

```python
            fk = f"{os.path.getsize(f)}_{int(os.path.getmtime(f))}_{PIPELINE_VERSION}"
```

(The `fkey` written into Chroma metadata after a successful ingest already reuses `rec.get("size")`/`rec.get("mtime")` from the Stage-1 JSON output combined the same way in the ingest loop — `fk = f"{rec.get('size')}_{rec.get('mtime')}"` — that line must also gain the `_{PIPELINE_VERSION}` suffix so the stored key matches the comparison key on the *next* run.)

- [ ] **Step 4: Apply the same change to nemoclaw-cdocs-ingest.py**

Identical edit, same `PIPELINE_VERSION` value (both collections should backfill together under one version marker, since they share the same underlying pipeline change).

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/longboardfella && python3 -m pytest test_nemoclaw_ingest_pipeline_version.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
cd /home/longboardfella
git add nemoclaw-kdocs-ingest.py nemoclaw-cdocs-ingest.py test_nemoclaw_ingest_pipeline_version.py
git commit -m "$(cat <<'EOF'
Fold PIPELINE_VERSION into the ingest resumability key

Stage 2's resumability is keyed purely on disk size+mtime, so shipping
the extraction/chunking fixes alone would never reprocess existing
files -- their stored key already matches. Bumping PIPELINE_VERSION
makes the existing resumable todo-list mechanism do the full backfill
with no separate backfill runner needed, and the same lever is
reusable for any future pipeline change.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UX4euTs9dwAssN2DjzH6hK
EOF
)"
```

---

### Task 10: Small-sample dry run before the full backfill

**Files:** none created/modified — this is a verification-only task using everything built in Tasks 1-9.

- [ ] **Step 1: Pick a verification sample covering every new code path**

```bash
cat > /tmp/backfill_dry_run_sample.txt <<'EOF'
/mnt/c/Users/paul/OneDrive - VentraIP Australia/04_Knowledge/SMS knowledge/SMS Knowledge Base/Fireservices- Emergency Services/In Progress/Major Findings Heat Map - A3 ra.pptx
/mnt/c/Users/paul/OneDrive - VentraIP Australia/04_Knowledge/SMS knowledge/SMS Knowledge Base/Fireservices- Emergency Services/Project Outputs/4. Project report/COP Findings.pptx
/mnt/c/Users/paul/OneDrive - VentraIP Australia/04_Knowledge/SMS knowledge/SMS Knowledge Base/Fireservices- Emergency Services/In Progress/COP Conceptual Diagram_041111.ppt
/mnt/c/Users/paul/OneDrive - VentraIP Australia/04_Knowledge/Paul work on wellbeing assessment.pdf
EOF
```

This covers: the standalone heatmap PPTX (native shapes, no real embedded picture — proves the pipeline doesn't regress on text-only content), the final 27-slide deck (contains the pasted-picture ranked-breakdown table on slides 24-25 — proves a genuine embedded picture gets captioned), a legacy `.ppt` (proves LibreOffice conversion), and `Paul work on wellbeing assessment.pdf` (confirmed during today's sizing scan to have 67 real pictures including at least one large, full-page-height image — proves the size filter and real-PDF path together).

- [ ] **Step 2: Run Stage 1 against the sample**

```bash
cat /tmp/backfill_dry_run_sample.txt | /home/longboardfella/nemoclaw-kdocs-extract.py > /tmp/backfill_dry_run_results.jsonl
cat /tmp/backfill_dry_run_results.jsonl | python3 -m json.tool --json-lines 2>/dev/null || \
  python3 -c "
import json
for line in open('/tmp/backfill_dry_run_results.jsonl'):
    r = json.loads(line)
    print(r['source_file'].split('/')[-1], '-> ok=', r.get('ok'), 'text_len=', len(r.get('text','')))
    if not r.get('ok'):
        print('   ERROR:', r.get('error'))
"
```

Expected: all 4 files `ok: true`, the legacy `.ppt` succeeded (proving LibreOffice conversion worked), and `Paul work on wellbeing assessment.pdf`'s text contains a `## Figure Intelligence` section (proving Haiku captioning actually fired on its large embedded picture).

- [ ] **Step 3: Manually inspect the wellbeing-assessment figure caption for quality**

```bash
python3 -c "
import json
for line in open('/tmp/backfill_dry_run_results.jsonl'):
    r = json.loads(line)
    if 'wellbeing' in r['source_file'] and r.get('ok'):
        idx = r['text'].find('Figure Intelligence')
        print(r['text'][idx:idx+800] if idx >= 0 else 'NO FIGURE INTELLIGENCE SECTION FOUND')
"
```

Expected: readable prose describing an actual chart/diagram — not reasoning-scaffolding text (e.g. not "Let's analyze this image..."), confirming the Haiku path (not a leftover Ollama path) actually ran. If this looks wrong, stop and re-check Task 5's `skip_vlm_processing=True` wiring before proceeding — do not continue to the full backfill on unverified output.

- [ ] **Step 4: Confirm dedup actually engages on a real repeat**

```bash
sqlite3 ~/.cortex_image_caption_cache.db "SELECT COUNT(*) FROM captions;"
```

Run the same 4-file sample through Stage 1 a second time (after temporarily clearing just those 4 files' entries from `~/.nemoclaw-doc-extract-skiplist.json` if any landed there, which they should not have):

```bash
cat /tmp/backfill_dry_run_sample.txt | /home/longboardfella/nemoclaw-kdocs-extract.py > /tmp/backfill_dry_run_results_2.jsonl
diff <(python3 -c "import json; [print(json.loads(l)['text']) for l in open('/tmp/backfill_dry_run_results.jsonl')]") \
     <(python3 -c "import json; [print(json.loads(l)['text']) for l in open('/tmp/backfill_dry_run_results_2.jsonl')]")
```

Expected: no diff (identical captions both times — the second run's Haiku calls, if any fired again since this reruns extraction not just captioning, should hit the dedup cache and return the same cached text either way).

- [ ] **Step 5: Run Stage 2 against just this sample and confirm the heatmap becomes searchable**

```bash
cd /home/longboardfella
venvs/vault-rag/bin/python3 nemoclaw-kdocs-ingest.py --limit 4
```

Then query the live bridge (matching the exact realistic phrasing already confirmed to fail before this project started):

```bash
TOKEN=$(cat ~/.kb-query-token)
curl -sS -G 'http://127.0.0.1:7333/query' -H "Authorization: Bearer $TOKEN" \
  --data-urlencode "q=knowledge base heatmap ease of implementation versus expected benefit" \
  --data-urlencode "mode=rapid" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for r in d.get('results', [])[:7]:
    print(round(r.get('score', 0), 3), r.get('source_file'))
"
```

Expected: `Major Findings Heat Map - A3 ra.pptx` or `COP Findings.pptx` now appears in the top results — this is the concrete, original success criterion for the whole project. If it doesn't appear, stop and debug before running the full backfill (do not scale up an unverified fix).

- [ ] **Step 6: No commit for this task** — it's verification only, using code already committed in Tasks 1-9. Note the confirmed working state in the next task's PR/commit message instead.

---

### Task 11: Full backfill

**Files:** none created/modified — running existing (now-fixed) production scripts at scale.

- [ ] **Step 1: Confirm no ingest is currently running**

```bash
ls /home/longboardfella/kdocs-ingest.lock /home/longboardfella/cdocs-ingest.lock 2>/dev/null
flock -n /home/longboardfella/kdocs-ingest.lock true && echo "kdocs free" || echo "kdocs BUSY -- wait"
flock -n /home/longboardfella/cdocs-ingest.lock true && echo "cdocs free" || echo "cdocs BUSY -- wait"
```

- [ ] **Step 2: Run the kdocs backfill (04_Knowledge) to completion**

Respect the existing throttle (do not remove it):

```bash
/home/longboardfella/kdocs-ingest-resume.sh
```

This may need to run multiple times if the cron's own throttled pace doesn't finish in one sitting — it's resumable by design (confirmed in Task 9), so re-running the same command picks up where it left off. Watch `~/kdocs-ingest.log` for the `DONE` line.

- [ ] **Step 3: Only after kdocs finishes, run the cdocs backfill (01_Clients/Active)**

Per the existing "STAGED — do NOT run concurrently" comment (both share the same 4 pinned cores):

```bash
/home/longboardfella/cdocs-ingest-resume.sh
```

Watch `~/cdocs-ingest.log` for the `DONE` line.

- [ ] **Step 4: Check the skip-list for anything that needs a human look**

```bash
cat ~/.nemoclaw-doc-extract-skiplist.json 2>/dev/null | python3 -m json.tool
```

A handful of entries is expected and fine (this project's own sizing scan found 2 genuinely pathological files across 7,040). A large number (dozens+) would indicate a systemic problem worth investigating before considering the backfill complete — report the count either way rather than silently continuing.

- [ ] **Step 5: No commit** — this task runs existing scripts against production data; nothing in the repo changes.

---

### Task 12: Bridge index refresh

**Files:** none created/modified.

- [ ] **Step 1: Restart the bridge so it picks up the new embeddings**

`kb-query-server.py` caches the Chroma HNSW index at process startup (same precedent as the photo-index refresh procedure already used in this environment).

```bash
~/kb-query-server-ensure.sh
```

- [ ] **Step 2: Confirm the bridge is back up**

```bash
TOKEN=$(cat ~/.kb-query-token)
curl -sS -G 'http://127.0.0.1:7333/query' -H "Authorization: Bearer $TOKEN" \
  --data-urlencode "q=knowledge base heatmap ease of implementation versus expected benefit" \
  --data-urlencode "mode=rapid" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('ok:', d.get('ok'))
for r in d.get('results', [])[:5]:
    print(round(r.get('score', 0), 3), r.get('source_file'))
"
```

Expected: the heatmap file appears near the top, this time from the FULL backfilled corpus, not just the 4-file dry-run sample.

- [ ] **Step 3: No commit.**

---

### Task 13: Hermes routing hint

**Files:**
- Modify: `~/.hermes/config.yaml` on host `sp4` (via `ssh sp4`, or the local `~/.hermes/config.yaml` on this WSL host if Hermes reads a local copy — confirm which by checking `ssh -o RemoteCommand=none sp4 "grep -c 'FOR TRAVEL AND' ~/.hermes/config.yaml"` first, matching the exact procedure already used successfully twice this session for the `kb-travel` fix)

**Interfaces:** none — this is a standalone prompt-text change, independent of every other task in this plan.

- [ ] **Step 1: Pull the current config.yaml to a local scratch copy**

```bash
mkdir -p /tmp/knowledge-routing-hint-scratch
scp -q sp4:/home/paul/.hermes/config.yaml /tmp/knowledge-routing-hint-scratch/config.yaml.orig
cp /tmp/knowledge-routing-hint-scratch/config.yaml.orig /tmp/knowledge-routing-hint-scratch/config.yaml
```

- [ ] **Step 2: Locate the existing PHOTO SEARCH / SOURCE HINTS section and add the routing hint immediately after it**

Find the line containing `PRESERVE these words VERBATIM in the text you pass to kb-ask/kb-query` (already read this session — it's the block explaining the "knowledge base" scope-routing keyword). Add a new paragraph directly after it in the same style as every other guidance block in this file:

```
      FOR "FIND ME A DOCUMENT/REPORT/DIAGRAM/DECK ABOUT X" QUESTIONS: run /home/paul/.local/bin/kb-query "knowledge base <what the document is about>" FIRST (or "client docs <...>" if Paul names a client/engagement). Without the literal words "knowledge base" or "client docs" the search never reaches those collections at all -- it gets drowned by the much larger private-vault/email corpus. This was directly measured on 2026-09-26: an unscoped query for a real diagram returned seven completely unrelated results (emails, travel blogs), while the same query scoped with "knowledge base" found the right document. If the scoped search comes back empty, THEN broaden to an unscoped kb-query as a fallback, not the other way around.
```

- [ ] **Step 3: Validate YAML and check the diff**

```bash
python3 -c "
import yaml
with open('/tmp/knowledge-routing-hint-scratch/config.yaml') as f:
    d = yaml.safe_load(f)
print('YAML OK')
"
diff -u /tmp/knowledge-routing-hint-scratch/config.yaml.orig /tmp/knowledge-routing-hint-scratch/config.yaml
```

Expected: YAML parses; diff shows exactly one clean insertion, matching the pattern of the `kb-travel` fix's diff earlier this session.

- [ ] **Step 4: Push back to sp4, preserving permissions**

```bash
scp -q /tmp/knowledge-routing-hint-scratch/config.yaml sp4:/home/paul/.hermes/config.yaml.new
ssh -o RemoteCommand=none sp4 "
set -e
chmod 600 /home/paul/.hermes/config.yaml.new
python3 -c \"import yaml; yaml.safe_load(open('/home/paul/.hermes/config.yaml.new'))\" && echo 'remote YAML parse OK'
cp /home/paul/.hermes/config.yaml.new /home/paul/.hermes/config.yaml
rm /home/paul/.hermes/config.yaml.new
grep -c 'FIND ME A DOCUMENT' /home/paul/.hermes/config.yaml
"
```

Expected: `remote YAML parse OK`, grep count `1`.

- [ ] **Step 5: No git commit** (Hermes config lives on sp4, not in a repo this session tracks) — instead, verify live in Discord as the closing check for the whole plan: ask Hermes a "find me a document/report about X" style question (e.g. about the fixed heatmap) and confirm it now routes through `kb-query "knowledge base ..."` and returns the right file, the same live-verification pattern already used twice this session for the `kb-travel` fix.

---

## Self-Review

**1. Spec coverage:** All five problem-statement items are covered — PPTX support (Task 1), broken/invisible VLM path replaced with Haiku + size filter + dedup (Tasks 1,2,3,5), chunk-quality fix (Task 4), legacy `.doc`/`.ppt` (Task 6), search routing (Task 13). All "already decided" items are implemented as specified (10% threshold in Task 5's default, corpus-wide dedup in Task 2, LibreOffice in Task 6, ≥4-run marker stripping in Task 4). The resilience requirement gets its own dedicated tasks (7, 8) rather than being folded in as an afterthought. Backfill execution (10) and verification (10, 12) both present. No gaps found.

**2. Placeholder scan:** No "TBD"/"TODO"/"handle appropriately" language present. Every code block is complete, runnable code, not a sketch.

**3. Type consistency:** `ImageCaptionCache.get(bytes) -> Optional[str]` / `.put(bytes, str) -> None` (Task 2) matches its usage in Task 5. `caption_figure_with_haiku(bytes, str="") -> str` (Task 3) matches its usage in Task 5. `strip_marker_runs(str, int=4) -> str` / `prefix_filename(str, str) -> str` (Task 4) matches usage in Task 7. `convert_legacy_office(str, str, int=60) -> str`, raising `LegacyConvertError` (Task 6), matches the `try/except LegacyConvertError` in Task 7. The `nemoclaw_doc_extract_one.py` JSON contract (`ok`/`source_file`/`text`/`size`/`mtime`/`error`) is identical to what Task 8's driver expects to parse from its stdout, and identical in shape to the pre-existing Stage-1→Stage-2 JSON contract Task 9 relies on unchanged.

**4. Review Focus:** All five listed items have an owning task with a test: PIPELINE_VERSION forcing reprocessing (Task 9, tested directly against both files' source), a single figure's Haiku failure not crashing extraction (Task 3 returns `""` on any error, never raises; Task 5's `_summarize_figure_with_vlm` returns `None` on empty caption rather than propagating an exception), dedup cache hit on a real repeat (Task 5, `test_dedup_cache_prevents_second_haiku_call`, and Task 10 Step 4 against real data), a hung LibreOffice conversion not blocking the batch (Task 6's `timeout` parameter + `LegacyConvertError`, consumed by Task 7's `extract_one` which returns `ok: false` rather than raising, so Task 8's per-file subprocess isolation still bounds the total cost even when the hang is inside conversion rather than Docling itself), and the marker-stripping false-positive case (Task 4, `test_does_not_strip_a_lone_isolated_number`).
