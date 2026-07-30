from __future__ import annotations

import io
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz

from cortex_engine.textifier import DocumentTextifier
from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]

_SKIP_BOOKMARK_TITLES = {
    "cover",
    "front cover",
    "june 2025 cover",
    "table of contents",
    "contents",
}
_CATEGORY_LABEL_RE = re.compile(r"^[A-Z][A-Z &/\-]{3,}$")
_BYLINE_RE = re.compile(r"^\s*BY\s+[A-Z][A-Z .'\-]+$", re.IGNORECASE)


@dataclass
class TextBlock:
    page_number: int
    bbox: fitz.Rect
    text: str
    max_font: float
    is_bold: bool


@dataclass
class ArticleCandidate:
    index: int
    title: str
    bookmark_title: str
    start_page: int
    end_page: int
    next_start_page: int
    anchor_terms: List[str]


def _slugify(value: str, fallback: str = "article") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    if len(text) > 96:
        text = text[:96].rstrip("._-")
    return text or fallback


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _canonicalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _bookmark_anchor_terms(bookmark_title: str) -> List[str]:
    title = _normalize_text(bookmark_title)
    if not title:
        return []
    terms = [title]
    if " - " in title:
        tail = title.split(" - ")[-1].strip()
        if tail:
            terms.insert(0, tail)
    return [term for term in terms if term]


def _keep_bookmark(title: str) -> bool:
    clean = _normalize_text(title)
    if not clean:
        return False
    lowered = clean.lower()
    if lowered in _SKIP_BOOKMARK_TITLES:
        return False
    return True


def _iter_text_blocks(page: fitz.Page, page_number: int) -> Iterable[TextBlock]:
    try:
        text_dict = page.get_text("dict")
    except Exception as exc:
        logger.debug("Article extractor block read failed on page %s: %s", page_number, exc)
        return []

    blocks = text_dict.get("blocks", []) if isinstance(text_dict, dict) else []
    items: List[TextBlock] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        lines = block.get("lines", [])
        parts: List[str] = []
        max_font = 0.0
        is_bold = False
        for line in lines:
            spans = line.get("spans", [])
            if not spans:
                continue
            line_text = "".join((span.get("text", "") or "") for span in spans)
            line_text = _normalize_text(line_text)
            if not line_text:
                continue
            parts.append(line_text)
            for span in spans:
                try:
                    max_font = max(max_font, float(span.get("size", 0) or 0))
                except Exception:
                    pass
                if "bold" in str(span.get("font", "")).lower():
                    is_bold = True
        text = "\n".join(parts).strip()
        if not text:
            continue
        bbox = fitz.Rect(block.get("bbox", page.rect))
        items.append(
            TextBlock(
                page_number=page_number,
                bbox=bbox,
                text=text,
                max_font=max_font,
                is_bold=is_bold,
            )
        )
    items.sort(key=lambda item: (round(item.bbox.y0, 1), round(item.bbox.x0, 1)))
    return items


def _score_block_for_terms(block: TextBlock, terms: Sequence[str]) -> float:
    if not block.text.strip():
        return 0.0
    block_text = _normalize_text(block.text)
    block_canon = _canonicalize(block_text)
    score = 0.0
    for term in terms:
        canon = _canonicalize(term)
        if not canon:
            continue
        if block_canon == canon:
            score += 10.0
        elif canon in block_canon:
            score += 7.0
        elif block_canon in canon and len(block_canon) >= 6:
            score += 4.0
    if _CATEGORY_LABEL_RE.match(block_text):
        score += 1.0
    if block.is_bold:
        score += 0.5
    score += min(block.max_font / 20.0, 1.5)
    return score


def _find_anchor_block(page: fitz.Page, page_number: int, terms: Sequence[str]) -> Optional[TextBlock]:
    best: Optional[TextBlock] = None
    best_score = 0.0
    for block in _iter_text_blocks(page, page_number):
        score = _score_block_for_terms(block, terms)
        if score > best_score:
            best = block
            best_score = score
    return best if best_score >= 4.0 else None


def _extract_title_from_blocks(page: fitz.Page, page_number: int, anchor: Optional[TextBlock]) -> str:
    blocks = list(_iter_text_blocks(page, page_number))
    if anchor is None:
        for block in blocks[:8]:
            text = _normalize_text(block.text.replace("\n", " "))
            if len(text) < 8:
                continue
            if _BYLINE_RE.match(text):
                continue
            if block.max_font >= 15 or (block.is_bold and len(text) <= 120):
                return text
        return ""

    try:
        anchor_idx = blocks.index(anchor)
    except ValueError:
        anchor_idx = -1
    anchor_lines = [_normalize_text(line) for line in str(anchor.text or "").splitlines() if _normalize_text(line)]
    anchor_is_category = bool(anchor_lines and _CATEGORY_LABEL_RE.match(anchor_lines[0]))
    if anchor_is_category and anchor_idx >= 0:
        title_parts: List[str] = []
        for block in blocks[anchor_idx + 1 : anchor_idx + 5]:
            text = _normalize_text(block.text.replace("\n", " "))
            if not text:
                continue
            if _BYLINE_RE.match(text):
                break
            if _CATEGORY_LABEL_RE.match(text):
                continue
            if len(text) > 120:
                break
            if block.max_font >= max(anchor.max_font + 2.0, 14.0) or block.is_bold:
                title_parts.append(text)
                if len(title_parts) >= 3:
                    break
            elif title_parts:
                break
        if title_parts:
            return _normalize_text(" ".join(title_parts))

    nearby = [anchor]
    for block in blocks:
        if block is anchor:
            continue
        if abs(block.bbox.y0 - anchor.bbox.y0) <= max(anchor.bbox.height * 1.2, 22):
            nearby.append(block)
    nearby.sort(key=lambda item: (round(item.bbox.y0, 1), round(item.bbox.x0, 1)))
    merged = " ".join(_normalize_text(item.text.replace("\n", " ")) for item in nearby if item.text.strip())
    merged = _normalize_text(merged)
    if merged and len(merged) <= 180 and not _CATEGORY_LABEL_RE.match(merged):
        return merged
    if len(anchor_lines) >= 2 and _CATEGORY_LABEL_RE.match(anchor_lines[0]):
        return anchor_lines[1]
    return _normalize_text(anchor.text.replace("\n", " "))


def _normalize_article_title(title: str, bookmark_title: str) -> str:
    clean = _normalize_text(title)
    if not clean:
        return bookmark_title
    if clean.startswith("©") or clean.lower().startswith("scientific american"):
        return bookmark_title
    if len(clean) < 4:
        return bookmark_title
    return clean


def _structured_lines_for_clip(page: fitz.Page, clip_rect: fitz.Rect) -> List[str]:
    lines_out: List[str] = []
    try:
        text_dict = page.get_text("dict", clip=clip_rect)
    except Exception as exc:
        logger.debug("Article extractor structured clip failed on page %s: %s", page.number + 1, exc)
        return lines_out

    font_sizes: List[float] = []
    blocks = text_dict.get("blocks", []) if isinstance(text_dict, dict) else []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                try:
                    size = float(span.get("size", 0) or 0)
                except Exception:
                    size = 0.0
                if size > 0:
                    font_sizes.append(size)
    base_font = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else 10.0
    heading_threshold = max(base_font * 1.25, base_font + 1.5)

    for block in blocks:
        if block.get("type") != 0:
            continue
        added = 0
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join((span.get("text", "") or "") for span in spans).strip()
            text = _normalize_text(text)
            if not text:
                continue
            max_font = max(float(span.get("size", 0) or 0) for span in spans)
            is_bold = any("bold" in str(span.get("font", "")).lower() for span in spans)
            if _BYLINE_RE.match(text):
                text = f"*{text.title()}*"
            elif (
                (max_font >= heading_threshold and len(text) <= 120)
                or (is_bold and len(text) <= 80 and text[:1].isupper())
                or _CATEGORY_LABEL_RE.match(text)
            ):
                text = f"### {text}"
            lines_out.append(text)
            added += 1
        if added:
            lines_out.append("")
    return lines_out


def _clip_markdown_for_pages(
    doc: fitz.Document,
    *,
    start_page: int,
    end_page: int,
    start_y: Optional[float],
    end_y: Optional[float],
) -> str:
    parts: List[str] = []
    for page_number in range(start_page, end_page + 1):
        page = doc.load_page(page_number - 1)
        top = 0.0
        bottom = page.rect.y1
        if page_number == start_page and start_y is not None:
            top = max(0.0, start_y)
        if page_number == end_page and end_y is not None:
            bottom = min(page.rect.y1, end_y)
        if bottom <= top + 2:
            continue
        clip = fitz.Rect(page.rect.x0, top, page.rect.x1, bottom)
        lines = _structured_lines_for_clip(page, clip)
        if not lines:
            text = page.get_text("text", clip=clip) or ""
            lines = [_normalize_text(line) for line in text.splitlines() if _normalize_text(line)]
        text = "\n".join(lines).strip()
        if not text:
            continue
        parts.append(text)
        if page_number < end_page:
            parts.append("\n---\n")
    return DocumentTextifier()._normalize_markdown_output("\n".join(parts))


def _build_candidates(doc: fitz.Document) -> List[ArticleCandidate]:
    toc = list(doc.get_toc() or [])
    if not toc:
        raise RuntimeError("PDF does not expose bookmarks/table-of-contents entries; article extraction needs PDF bookmarks")

    raw: List[Tuple[str, int]] = []
    for entry in toc:
        if len(entry) < 3:
            continue
        title = _normalize_text(str(entry[1] or ""))
        page_number = int(entry[2] or 0)
        if page_number <= 0 or not _keep_bookmark(title):
            continue
        raw.append((title, page_number))

    if not raw:
        raise RuntimeError("No usable article bookmarks found in PDF")

    deduped: List[Tuple[str, int]] = []
    seen: set[Tuple[str, int]] = set()
    for title, page_number in raw:
        key = (title.lower(), page_number)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((title, page_number))

    candidates: List[ArticleCandidate] = []
    total_pages = len(doc)
    for idx, (title, start_page) in enumerate(deduped):
        next_start = deduped[idx + 1][1] if idx + 1 < len(deduped) else total_pages + 1
        end_page = max(start_page, next_start - 1)
        candidates.append(
            ArticleCandidate(
                index=idx + 1,
                title=title,
                bookmark_title=title,
                start_page=start_page,
                end_page=end_page,
                next_start_page=next_start,
                anchor_terms=_bookmark_anchor_terms(title),
            )
        )
    return candidates


def extract_pdf_articles_to_bundle(
    pdf_path: str | Path,
    *,
    output_dir: str | Path = "",
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    path = Path(str(pdf_path or "").strip())
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Article markdown extraction requires a PDF input file")

    root = Path(output_dir) if str(output_dir or "").strip() else Path(tempfile.mkdtemp(prefix="article_md_extract_"))
    root.mkdir(parents=True, exist_ok=True)
    articles_dir = root / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(str(path)) as doc:
        candidates = _build_candidates(doc)
        article_records: List[Dict[str, Any]] = []

        for idx, candidate in enumerate(candidates, start=1):
            if progress_cb:
                progress_cb((idx - 1) / max(len(candidates), 1), f"Extracting article {idx}/{len(candidates)}")

            start_page_obj = doc.load_page(candidate.start_page - 1)
            anchor = _find_anchor_block(start_page_obj, candidate.start_page, candidate.anchor_terms)
            next_anchor: Optional[TextBlock] = None
            effective_end_page = candidate.end_page
            if idx < len(candidates):
                next_candidate = candidates[idx]
                if next_candidate.start_page == candidate.start_page:
                    next_anchor = _find_anchor_block(start_page_obj, candidate.start_page, next_candidate.anchor_terms)
                elif next_candidate.start_page <= len(doc):
                    next_page_obj = doc.load_page(next_candidate.start_page - 1)
                    possible_next_anchor = _find_anchor_block(
                        next_page_obj,
                        next_candidate.start_page,
                        next_candidate.anchor_terms,
                    )
                    if possible_next_anchor and possible_next_anchor.bbox.y0 >= 80:
                        next_anchor = possible_next_anchor
                        effective_end_page = next_candidate.start_page

            title = _normalize_article_title(
                _extract_title_from_blocks(start_page_obj, candidate.start_page, anchor) or candidate.title,
                candidate.title,
            )
            start_y = max(anchor.bbox.y0 - 18, 0.0) if anchor else None
            end_y = max(next_anchor.bbox.y0 - 12, 0.0) if next_anchor else None
            markdown_body = _clip_markdown_for_pages(
                doc,
                start_page=candidate.start_page,
                end_page=effective_end_page,
                start_y=start_y,
                end_y=end_y,
            )
            markdown = "\n".join(
                [
                    f"# {title}",
                    "",
                    f"- Source PDF: `{path.name}`",
                    f"- Bookmark: `{candidate.bookmark_title}`",
                    f"- Pages: {candidate.start_page}-{candidate.end_page}",
                    "",
                    markdown_body.strip(),
                    "",
                ]
            ).strip() + "\n"

            slug = f"{idx:03d}_{_slugify(title, f'article_{idx:03d}')}"
            md_path = articles_dir / f"{slug}.md"
            md_path.write_text(markdown, encoding="utf-8")
            article_records.append(
                {
                    "index": idx,
                    "title": title,
                    "bookmark_title": candidate.bookmark_title,
                    "start_page": candidate.start_page,
                    "end_page": effective_end_page,
                    "output_file": f"articles/{md_path.name}",
                }
            )

    manifest = {
        "source_pdf": path.name,
        "article_count": len(article_records),
        "articles": article_records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    zip_path = root / f"{path.stem}_articles_markdown.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname="manifest.json")
        for article in article_records:
            article_path = root / article["output_file"]
            zf.write(article_path, arcname=article["output_file"])

    if progress_cb:
        progress_cb(1.0, f"Extracted {len(article_records)} article markdown files")

    return {
        "output_dir": str(root),
        "zip_path": str(zip_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }
