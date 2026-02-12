"""
Document Textifier Module
Converts PDF, DOCX, and PPTX documents to rich Markdown with optional
vision-model descriptions of images and tables.
"""

import os
import re
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DocumentTextifier:
    """Converts documents to Markdown with optional VLM image descriptions."""

    # Preferred vision models in order of priority
    VISION_MODELS = ["qwen3-vl:8b", "qwen3-vl:4b", "qwen3-vl"]

    def __init__(self, use_vision: bool = True, on_progress: Optional[Callable[[float, str], None]] = None):
        self.use_vision = use_vision
        self.on_progress = on_progress
        self._vlm_client = None
        self._vlm_model = None

    def _report(self, fraction: float, message: str):
        """Send a progress update if a callback is registered."""
        if self.on_progress:
            self.on_progress(fraction, message)

    def _init_vlm(self):
        """Lazy-init the Ollama VLM client, preferring Qwen3-VL models."""
        if self._vlm_client is not None:
            return
        try:
            import ollama
            client = ollama.Client(timeout=120)

            # Try preferred Qwen3-VL models first
            for model in self.VISION_MODELS:
                try:
                    client.show(model)
                    self._vlm_client = client
                    self._vlm_model = model
                    logger.info(f"Textifier using vision model: {model}")
                    return
                except Exception:
                    continue

            # Fall back to configured VLM_MODEL (e.g. llava:7b)
            from cortex_engine.config import VLM_MODEL
            client.show(VLM_MODEL)
            self._vlm_client = client
            self._vlm_model = VLM_MODEL
            logger.info(f"Textifier falling back to VLM_MODEL: {VLM_MODEL}")
        except Exception as e:
            logger.warning(f"No vision model available: {e}")
            self._vlm_client = None
            self._vlm_model = None

    @staticmethod
    def _looks_like_logo_icon_description(text: str) -> bool:
        """Heuristic: detect short logo/icon-only descriptions that add metadata noise."""
        t = (text or "").strip().lower()
        if not t:
            return False
        logo_markers = [
            "logo", "icon", "icons", "emblem", "symbol", "badge", "watermark",
            "seal", "silhouette", "initials", "letters",
        ]
        visual_markers = [
            "left icon", "right icon", "circular", "circle", "black and white",
            "background", "strip", "main colors", "main colours",
        ]
        marker_hits = sum(1 for m in logo_markers if m in t)
        visual_hits = sum(1 for m in visual_markers if m in t)
        return (marker_hits >= 1 and visual_hits >= 1) or marker_hits >= 2

    def describe_image(self, image_bytes: bytes) -> str:
        """Describe an image using the VLM. Returns placeholder on failure."""
        if not self.use_vision:
            return "[Image: vision model disabled]"
        self._init_vlm()
        if self._vlm_client is None:
            return "[Image: could not be described — vision model unavailable]"
        try:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            response = self._vlm_client.chat(
                model=self._vlm_model,
                messages=[{
                    "role": "user",
                    "content": (
                        "Describe this photograph in 2-4 plain sentences. Identify main subject, setting, and colours. "
                        "If the image is primarily a logo/icon/watermark or tiny decorative graphic, "
                        "return exactly: [Image: logo/icon omitted]. "
                        "Do not use markdown, headings, or bullet points. /no_think"
                    ),
                    "images": [encoded],
                }],
                options={"temperature": 0.1, "num_predict": 600},
            )
            result = response["message"]["content"].strip()
            # Qwen3-VL may wrap output in <think>...</think> tags
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            if not result:
                logger.warning(f"VLM returned empty description (model: {self._vlm_model})")
                return "[Image: vision model returned empty description]"
            if self._looks_like_logo_icon_description(result):
                return "[Image: logo/icon omitted]"
            return result
        except Exception as e:
            logger.warning(f"VLM describe failed: {e}")
            return "[Image: could not be described — vision model error]"

    @staticmethod
    def table_to_markdown(rows: List[List[str]]) -> str:
        """Convert a list-of-lists table to Markdown."""
        if not rows:
            return ""
        # Ensure all rows have same column count
        max_cols = max(len(r) for r in rows)
        normalised = [r + [""] * (max_cols - len(r)) for r in rows]
        header = normalised[0]
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in normalised[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    @staticmethod
    def _clean_table_rows(rows: List[List[str]]) -> List[List[str]]:
        cleaned_rows: List[List[str]] = []
        for row in rows or []:
            cleaned_row = []
            for cell in row or []:
                text = re.sub(r"\s+", " ", str(cell or "")).strip()
                cleaned_row.append(text)
            cleaned_rows.append(cleaned_row)
        return cleaned_rows

    @staticmethod
    def _is_simple_table_quality(rows: List[List[str]]) -> bool:
        if not rows or len(rows) < 2:
            return False
        col_count = max((len(r) for r in rows), default=0)
        if col_count < 2:
            return False
        if col_count > 12:
            return False

        total_cells = len(rows) * col_count
        filled_cells = 0
        long_cell_count = 0
        for row in rows:
            normalized = row + [""] * (col_count - len(row))
            for cell in normalized:
                if cell:
                    filled_cells += 1
                if len(cell) > 140:
                    long_cell_count += 1

        fill_ratio = (filled_cells / total_cells) if total_cells else 0.0
        if fill_ratio < 0.35:
            return False
        if long_cell_count > max(2, int(total_cells * 0.15)):
            return False

        header = rows[0] + [""] * (col_count - len(rows[0]))
        header_nonempty = sum(1 for c in header if c)
        if header_nonempty < 1:
            return False
        return True

    def _extract_pdf_tables(self, page) -> List[Dict[str, object]]:
        """Extract simple tables from a PDF page, returning parsed/failed statuses."""
        results: List[Dict[str, object]] = []
        if not hasattr(page, "find_tables"):
            return results
        try:
            found = page.find_tables()
            table_objs = list(getattr(found, "tables", []) or [])
        except Exception as e:
            logger.debug(f"PDF table detection failed on page {page.number + 1}: {e}")
            return results

        for idx, table_obj in enumerate(table_objs, 1):
            try:
                raw_rows = table_obj.extract() or []
                rows = self._clean_table_rows(raw_rows)
                if self._is_simple_table_quality(rows):
                    results.append({"status": "parsed", "index": idx, "rows": rows})
                else:
                    results.append({"status": "unreliable", "index": idx, "rows": []})
            except Exception as e:
                logger.debug(f"Table extraction failed for table {idx} on page {page.number + 1}: {e}")
                results.append({"status": "unreliable", "index": idx, "rows": []})
        return results

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def textify_file(self, file_path: str) -> str:
        """Dispatch to the correct converter based on file extension."""
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self.textify_pdf(file_path)
        elif ext == ".docx":
            return self.textify_docx(file_path)
        elif ext == ".pptx":
            return self.textify_pptx(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"):
            return self.textify_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def textify_pdf(self, file_path: str) -> str:
        """Convert a PDF to Markdown using PyMuPDF."""
        self._report(0.0, "Checking Docling availability...")
        docling_md = self._try_docling(file_path)
        if docling_md:
            self._report(1.0, "Converted via Docling")
            return docling_md

        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        total_pages = len(doc)
        md_parts: List[str] = []
        page_lines: List[List[str]] = []
        header_footer_counts: Dict[str, int] = {}

        # Pass 1: collect lines and detect repeated header/footer boilerplate.
        for page_num in range(total_pages):
            page = doc[page_num]
            raw_text = page.get_text("text") or ""
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw_text.splitlines() if ln.strip()]
            page_lines.append(lines)

            candidates = lines[:4] + lines[-4:]
            for line in candidates:
                if len(line) < 4:
                    continue
                if re.fullmatch(r"[\d\W]+", line):
                    continue
                header_footer_counts[line] = header_footer_counts.get(line, 0) + 1

        repeated_boilerplate = {line for line, count in header_footer_counts.items() if count >= 2}

        for page_num in range(total_pages):
            self._report(page_num / total_pages, f"Page {page_num + 1}/{total_pages}")
            page = doc[page_num]
            md_parts.append(f"## Page {page_num + 1}\n")

            lines = page_lines[page_num] if page_num < len(page_lines) else []
            filtered_lines: List[str] = []
            for idx, line in enumerate(lines):
                in_header_footer_zone = idx < 5 or idx >= max(len(lines) - 5, 0)
                if in_header_footer_zone and line in repeated_boilerplate:
                    continue
                filtered_lines.append(line)

            text = "\n".join(filtered_lines).strip()
            if text:
                md_parts.append(text)
                md_parts.append("")

            table_results = self._extract_pdf_tables(page)
            for t in table_results:
                table_idx = int(t.get("index", 0))
                if t.get("status") == "parsed":
                    rows = t.get("rows", [])
                    md_parts.append(f"**Table {table_idx}:**")
                    md_parts.append(self.table_to_markdown(rows))
                    md_parts.append("")
                else:
                    md_parts.append(f"> **[Table {table_idx}]**: unable to be reliably parsed.")
                    md_parts.append("")

            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    if base_image and base_image.get("image"):
                        self._report(
                            (page_num + (img_idx + 1) / max(len(image_list), 1)) / total_pages,
                            f"Page {page_num + 1} — scanning figure {img_idx + 1}/{len(image_list)}",
                        )
                        md_parts.append(f"> **[Figure {img_idx + 1}]**: unable to be reliably parsed.")
                        md_parts.append("")
                except Exception as e:
                    logger.debug(f"Could not extract image xref={xref}: {e}")

            # Explicit page break marker for downstream parsers.
            if page_num < total_pages - 1:
                md_parts.append("\n---\n")

        doc.close()
        self._report(1.0, "PDF conversion complete")
        return "\n".join(md_parts)

    # ------------------------------------------------------------------
    # DOCX
    # ------------------------------------------------------------------

    def textify_docx(self, file_path: str) -> str:
        """Convert a DOCX to Markdown."""
        from docx import Document as DocxDocument
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
        import io

        doc = DocxDocument(file_path)
        md_parts: List[str] = []

        total_paras = len(doc.paragraphs)
        self._report(0.0, "Processing paragraphs...")

        for p_idx, para in enumerate(doc.paragraphs):
            if total_paras > 0 and p_idx % 20 == 0:
                self._report(p_idx / total_paras * 0.5, f"Paragraph {p_idx}/{total_paras}")
            style_name = (para.style.name or "").lower()
            text = para.text.strip()
            if not text:
                md_parts.append("")
                continue
            if "heading 1" in style_name:
                md_parts.append(f"# {text}")
            elif "heading 2" in style_name:
                md_parts.append(f"## {text}")
            elif "heading 3" in style_name:
                md_parts.append(f"### {text}")
            elif "heading" in style_name:
                md_parts.append(f"#### {text}")
            elif "list" in style_name:
                md_parts.append(f"- {text}")
            else:
                md_parts.append(text)
            md_parts.append("")

        self._report(0.5, "Processing tables...")
        for table_idx, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(cells)
            if rows:
                md_parts.append(f"**Table {table_idx + 1}:**")
                md_parts.append(self.table_to_markdown(rows))
                md_parts.append("")

        # Count images first for progress
        image_rels = [r for r in doc.part.rels.values() if "image" in r.reltype]
        total_images = len(image_rels)
        self._report(0.7, f"Processing {total_images} embedded images...")

        img_idx = 0
        for rel_i, rel in enumerate(image_rels):
            try:
                image_bytes = rel.target_part.blob
                self._report(0.7 + 0.3 * (rel_i / max(total_images, 1)),
                             f"Describing image {rel_i + 1}/{total_images}")
                desc = self.describe_image(image_bytes)
                img_idx += 1
                md_parts.append(f"> **[Image {img_idx}]**: {desc}")
                md_parts.append("")
            except Exception as e:
                logger.debug(f"Could not extract DOCX image: {e}")

        return "\n".join(md_parts)

    # ------------------------------------------------------------------
    # PPTX
    # ------------------------------------------------------------------

    def textify_pptx(self, file_path: str) -> str:
        """Convert a PPTX to Markdown."""
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        prs = Presentation(file_path)
        total_slides = len(prs.slides)
        md_parts: List[str] = []

        for slide_num, slide in enumerate(prs.slides, 1):
            self._report((slide_num - 1) / total_slides, f"Slide {slide_num}/{total_slides}")
            md_parts.append(f"## Slide {slide_num}\n")

            img_count = 0
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            md_parts.append(text)
                    md_parts.append("")

                if shape.has_table:
                    rows = []
                    for row in shape.table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        rows.append(cells)
                    if rows:
                        md_parts.append(self.table_to_markdown(rows))
                        md_parts.append("")

                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        img_count += 1
                        self._report(
                            (slide_num - 1) / total_slides,
                            f"Slide {slide_num}/{total_slides} — describing image {img_count}",
                        )
                        image_bytes = shape.image.blob
                        desc = self.describe_image(image_bytes)
                        md_parts.append(f"> **[Image]**: {desc}")
                        md_parts.append("")
                    except Exception as e:
                        logger.debug(f"Could not extract PPTX image: {e}")

            md_parts.append("---\n")

        self._report(1.0, "PPTX conversion complete")

        return "\n".join(md_parts)

    # ------------------------------------------------------------------
    # Image files (PNG, JPG, etc.)
    # ------------------------------------------------------------------

    def textify_image(self, file_path: str) -> str:
        """Convert an image file to Markdown with a VLM description."""
        self._report(0.0, "Reading image...")
        file_name = Path(file_path).name

        with open(file_path, "rb") as f:
            image_bytes = f.read()

        md_parts = [f"# {file_name}\n"]

        self._report(0.3, "Describing image with vision model...")
        description = self.describe_image(image_bytes)
        md_parts.append(f"> **[Image]**: {description}")
        md_parts.append("")

        # Try OCR extraction via pytesseract if available
        try:
            from PIL import Image
            import pytesseract
            self._report(0.7, "Extracting text via OCR...")
            img = Image.open(file_path)
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                md_parts.append("## Extracted Text (OCR)\n")
                md_parts.append(ocr_text)
                md_parts.append("")
        except ImportError:
            logger.debug("pytesseract not available — skipping OCR")
        except Exception as e:
            logger.debug(f"OCR extraction failed: {e}")

        self._report(1.0, "Image conversion complete")
        return "\n".join(md_parts)

    # ------------------------------------------------------------------
    # Photo Keywords — describe, extract keywords, write EXIF
    # ------------------------------------------------------------------

    # Text models for keyword extraction (VLMs need images, so use a text LLM)
    TEXT_MODELS = ["mistral:latest", "mistral-small3.2", "llama3:latest", "gemma:latest"]

    def extract_keywords(self, description: str) -> List[str]:
        """Extract flat keywords from an image description using a text LLM.

        Uses a text model (Mistral etc.) rather than the VLM, because VLMs
        like Qwen3-VL return empty responses for text-only prompts.
        """
        # Don't hallucinate keywords from empty/failed descriptions
        if not description or description.startswith("[Image:"):
            logger.warning("No valid description available — skipping keyword extraction")
            return []
        try:
            import ollama
            client = ollama.Client(timeout=30)

            # Find an available text model
            text_model = None
            for model in self.TEXT_MODELS:
                try:
                    client.show(model)
                    text_model = model
                    break
                except Exception:
                    continue

            if not text_model:
                logger.warning("No text model available for keyword extraction")
                return self._extract_keywords_simple(description)

            response = client.chat(
                model=text_model,
                messages=[{
                    "role": "user",
                    "content": (
                        "Extract 10-15 photo tags from this image description. "
                        "Return ONLY a comma-separated list. Each tag should be 1-2 "
                        "simple lowercase words (no underscores, no hyphens). "
                        "Good tags: specific subjects, species, colours, season, "
                        "location type, weather. "
                        "Do NOT include photography jargon (bokeh, depth of field, "
                        "backlit, composition, close up) or vague words (atmosphere, "
                        "mood, scene, tones). Maximum 15 tags.\n\n"
                        f"Description: {description}"
                    ),
                }],
                options={"temperature": 0.1, "num_predict": 200},
            )
            raw = response["message"]["content"].strip()
            # Parse comma-separated keywords, clean up
            keywords = [k.strip().lower().strip('"\'') for k in raw.split(",")]
            # Replace underscores with spaces
            keywords = [k.replace("_", " ") for k in keywords]
            # Filter out photography jargon and empty/too-long entries
            _jargon = {
                "bokeh", "bokeh effect", "depth of field", "shallow depth of field",
                "backlit", "backlighting", "side lit", "composition", "close up",
                "blurred background", "out of focus", "warm tones", "cool tones",
                "atmosphere", "mood", "ambiance", "scene", "tones", "texture",
                "glossy texture", "gradient",
            }
            keywords = [
                k for k in keywords
                if k and len(k) > 1 and len(k) < 50 and k not in _jargon
            ]
            # Cap at 15
            keywords = keywords[:15]
            if keywords:
                return keywords
            # If LLM returned something unparseable, fall back
            return self._extract_keywords_simple(description)
        except Exception as e:
            logger.warning(f"LLM keyword extraction failed: {e}")
            return self._extract_keywords_simple(description)

    @staticmethod
    def _extract_keywords_simple(description: str) -> List[str]:
        """Fallback keyword extraction using simple NLP."""
        import re
        # Remove bracketed placeholders
        text = re.sub(r'\[.*?\]', '', description).lower()
        # Common stop words
        stop = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'of', 'in', 'to', 'for',
            'with', 'on', 'at', 'from', 'by', 'about', 'as', 'into', 'through',
            'and', 'but', 'or', 'so', 'if', 'that', 'this', 'it', 'its', 'not',
            'what', 'which', 'who', 'how', 'there', 'their', 'they', 'them',
            'image', 'shows', 'appears', 'visible', 'seen', 'also', 'very',
        }
        words = re.findall(r'[a-z]+', text)
        keywords = []
        seen = set()
        for w in words:
            if w not in stop and len(w) > 2 and w not in seen:
                seen.add(w)
                keywords.append(w)
        return keywords[:20]

    @staticmethod
    def read_exif_keywords(file_path: str) -> List[str]:
        """Read existing XMP Subject / IPTC Keywords from image."""
        import shutil
        import json
        exiftool_path = shutil.which("exiftool")
        if not exiftool_path:
            return []
        try:
            result = subprocess.run(
                [exiftool_path, "-json", "-XMP-dc:Subject", "-IPTC:Keywords",
                 file_path],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []
            data = json.loads(result.stdout)
            if not data:
                return []
            existing = set()
            for field in ("Subject", "Keywords"):
                val = data[0].get(field, [])
                if isinstance(val, str):
                    val = [val]
                for v in val:
                    existing.add(v.strip().lower())
            return sorted(existing)
        except Exception as e:
            logger.warning(f"Read existing keywords failed: {e}")
            return []

    @staticmethod
    def write_exif_keywords(file_path: str, keywords: List[str],
                            description: str = "") -> Dict[str, any]:
        """Write keywords and description to image EXIF/XMP using exiftool.

        Merges with existing keywords (uses += to append, not replace).

        Writes to:
          - XMP:dc:subject (Keywords — read by Lightroom, Bridge, etc.)
          - IPTC:Keywords (broader compatibility)
          - XMP:dc:description (Caption)
          - EXIF:ImageDescription (legacy caption)

        Returns dict with 'success', 'message', and 'keywords_written'.
        """
        import shutil
        exiftool_path = shutil.which("exiftool")
        if not exiftool_path:
            return {
                "success": False,
                "message": "exiftool not found on system PATH",
                "keywords_written": 0,
            }

        cmd = [exiftool_path, "-overwrite_original"]

        # Use += to ADD to existing keywords rather than replacing them.
        # XMP-dc:Subject is the keyword list that Lightroom Classic,
        # Bridge, Capture One, and other DAMs read as "Keywords".
        # IPTC:Keywords provides legacy compatibility.
        for kw in keywords:
            cmd.append(f"-XMP-dc:Subject+={kw}")
            cmd.append(f"-IPTC:Keywords+={kw}")

        # Write description/caption to the correct fields:
        # - XMP-dc:Description → LRC "Caption" in metadata panel
        # - IPTC:Caption-Abstract → legacy caption
        # - EXIF:ImageDescription → EXIF-level caption
        if description:
            caption = description[:2000]
            cmd.append(f"-XMP-dc:Description={caption}")
            cmd.append(f"-IPTC:Caption-Abstract={caption}")
            cmd.append(f"-EXIF:ImageDescription={caption}")

        cmd.append(file_path)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                logger.info(f"Wrote {len(keywords)} keywords to {Path(file_path).name}")
                return {
                    "success": True,
                    "message": result.stdout.strip(),
                    "keywords_written": len(keywords),
                }
            else:
                logger.warning(f"exiftool error: {result.stderr}")
                return {
                    "success": False,
                    "message": result.stderr.strip(),
                    "keywords_written": 0,
                }
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "exiftool timed out", "keywords_written": 0}
        except Exception as e:
            return {"success": False, "message": str(e), "keywords_written": 0}

    @staticmethod
    def clear_exif_keywords(file_path: str) -> bool:
        """Remove all existing XMP Subject, IPTC Keywords, and description fields."""
        import shutil
        exiftool_path = shutil.which("exiftool")
        if not exiftool_path:
            return False
        try:
            result = subprocess.run(
                [exiftool_path, "-overwrite_original",
                 "-XMP-dc:Subject=", "-IPTC:Keywords=",
                 "-XMP-dc:Description=", "-IPTC:Caption-Abstract=",
                 "-EXIF:ImageDescription=",
                 file_path],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Clear keywords failed: {e}")
            return False

    @staticmethod
    def clear_exif_location(file_path: str) -> bool:
        """Remove existing IPTC/XMP location fields (Country, State, City)."""
        import shutil
        exiftool_path = shutil.which("exiftool")
        if not exiftool_path:
            return False
        try:
            result = subprocess.run(
                [exiftool_path, "-overwrite_original",
                 "-IPTC:Country-PrimaryLocationName=",
                 "-XMP-photoshop:Country=",
                 "-IPTC:Province-State=",
                 "-XMP-photoshop:State=",
                 "-IPTC:City=",
                 "-XMP-photoshop:City=",
                 file_path],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Clear location failed: {e}")
            return False

    @staticmethod
    def read_gps(file_path: str) -> Optional[Tuple[float, float]]:
        """Read GPS coordinates from image EXIF. Returns (lat, lon) or None."""
        import shutil
        exiftool_path = shutil.which("exiftool")
        if not exiftool_path:
            return None
        try:
            import json
            result = subprocess.run(
                [exiftool_path, "-json", "-n", "-GPSLatitude", "-GPSLongitude",
                 file_path],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return None
            data = json.loads(result.stdout)
            if not data:
                return None
            lat = data[0].get("GPSLatitude")
            lon = data[0].get("GPSLongitude")
            if lat is not None and lon is not None:
                return (float(lat), float(lon))
        except Exception as e:
            logger.warning(f"GPS read failed for {file_path}: {e}")
        return None

    @staticmethod
    def reverse_geocode(lat: float, lon: float) -> Dict[str, str]:
        """Reverse-geocode GPS coordinates to country/state/city.

        Returns dict with 'country', 'state', 'city' keys (empty string if
        not found).
        """
        try:
            from geopy.geocoders import Nominatim
            g = Nominatim(user_agent="cortex_suite", timeout=10)
            loc = g.reverse((lat, lon), exactly_one=True, language="en")
            if loc is None:
                return {"country": "", "state": "", "city": ""}
            addr = loc.raw.get("address", {})
            country = addr.get("country", "")
            state = addr.get("state", "") or addr.get("region", "")
            city = (addr.get("city", "")
                    or addr.get("town", "")
                    or addr.get("village", "")
                    or addr.get("suburb", ""))
            return {"country": country, "state": state, "city": city}
        except Exception as e:
            logger.warning(f"Reverse geocode failed for ({lat}, {lon}): {e}")
            return {"country": "", "state": "", "city": ""}

    @staticmethod
    def write_exif_location(file_path: str, country: str, state: str,
                            city: str) -> bool:
        """Write IPTC/XMP location fields to image using exiftool."""
        import shutil
        exiftool_path = shutil.which("exiftool")
        if not exiftool_path:
            return False
        cmd = [exiftool_path, "-overwrite_original"]
        if country:
            cmd.append(f"-IPTC:Country-PrimaryLocationName={country}")
            cmd.append(f"-XMP-photoshop:Country={country}")
        if state:
            cmd.append(f"-IPTC:Province-State={state}")
            cmd.append(f"-XMP-photoshop:State={state}")
        if city:
            cmd.append(f"-IPTC:City={city}")
            cmd.append(f"-XMP-photoshop:City={city}")
        cmd.append(file_path)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"EXIF location write failed: {e}")
            return False

    @staticmethod
    def write_ownership_metadata(file_path: str, ownership_notice: str) -> Dict[str, any]:
        """Write ownership/copyright metadata fields using exiftool."""
        import shutil

        notice = (ownership_notice or "").strip()
        if not notice:
            return {"success": True, "message": "No ownership notice provided", "fields_written": 0}

        exiftool_path = shutil.which("exiftool")
        if not exiftool_path:
            return {"success": False, "message": "exiftool not found on system PATH", "fields_written": 0}

        cmd = [
            exiftool_path,
            "-overwrite_original",
            f"-EXIF:Copyright={notice}",
            f"-IPTC:CopyrightNotice={notice}",
            f"-XMP-dc:Rights={notice}",
            "-XMP-xmpRights:Marked=True",
            file_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if result.returncode == 0:
                return {"success": True, "message": result.stdout.strip(), "fields_written": 4}
            return {"success": False, "message": result.stderr.strip(), "fields_written": 0}
        except Exception as e:
            return {"success": False, "message": str(e), "fields_written": 0}

    @staticmethod
    def _resize_image_preserving_metadata(
        file_path: str, max_width: int = 1920, max_height: int = 1080
    ) -> Dict[str, any]:
        """Downsample oversized images in-place and copy metadata from original."""
        from PIL import Image
        import shutil

        suffix = Path(file_path).suffix.lower()
        format_map = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".tif": "TIFF",
            ".tiff": "TIFF",
            ".webp": "WEBP",
            ".bmp": "BMP",
            ".gif": "GIF",
        }

        original_copy = None
        try:
            with Image.open(file_path) as img:
                original_width, original_height = img.size
                image_format = img.format or format_map.get(suffix, "JPEG")

            if original_width <= max_width and original_height <= max_height:
                return {
                    "resized": False,
                    "original_width": original_width,
                    "original_height": original_height,
                    "new_width": original_width,
                    "new_height": original_height,
                    "metadata_preserved": True,
                }

            scale = min(max_width / float(original_width), max_height / float(original_height))
            new_width = max(1, int(original_width * scale))
            new_height = max(1, int(original_height * scale))

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                original_copy = tmp.name
            shutil.copy2(file_path, original_copy)

            with Image.open(file_path) as img:
                if image_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                save_kwargs: Dict[str, any] = {}
                if image_format == "JPEG":
                    save_kwargs["quality"] = 90
                    save_kwargs["optimize"] = True
                resized.save(file_path, format=image_format, **save_kwargs)

            metadata_preserved = False
            exiftool_path = shutil.which("exiftool")
            if exiftool_path:
                copy_result = subprocess.run(
                    [
                        exiftool_path,
                        "-overwrite_original",
                        "-TagsFromFile",
                        original_copy,
                        "-all:all",
                        "-unsafe",
                        file_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                metadata_preserved = copy_result.returncode == 0
                if not metadata_preserved:
                    logger.warning(
                        f"Metadata copy after resize failed for {Path(file_path).name}: {copy_result.stderr}"
                    )
            else:
                logger.warning("exiftool not found; metadata copy after resize skipped")

            return {
                "resized": True,
                "original_width": original_width,
                "original_height": original_height,
                "new_width": new_width,
                "new_height": new_height,
                "metadata_preserved": metadata_preserved,
            }
        except Exception as e:
            logger.warning(f"Resize failed for {Path(file_path).name}: {e}")
            if original_copy and Path(original_copy).exists():
                try:
                    shutil.copy2(original_copy, file_path)
                except Exception as restore_err:
                    logger.warning(f"Could not restore original image after resize failure: {restore_err}")
            return {"resized": False, "error": str(e), "metadata_preserved": False}
        finally:
            if original_copy and Path(original_copy).exists():
                try:
                    os.remove(original_copy)
                except Exception:
                    pass

    def resize_image_only(
        self, file_path: str, max_width: int = 1920, max_height: int = 1080
    ) -> Dict[str, any]:
        """Resize photo only (no keyword generation), preserving metadata when possible."""
        self._report(0.0, "Checking image dimensions...")
        resize_info = self._resize_image_preserving_metadata(
            file_path, max_width=max_width, max_height=max_height
        )
        self._report(1.0, "Done")
        return {
            "file_name": Path(file_path).name,
            "resize_info": resize_info,
        }

    @staticmethod
    def _filter_sensitive_keywords(
        keywords: List[str], blocked_keywords: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """Remove blocked/sensitive keywords while preserving other tags."""
        blocked_set = {k.strip().lower() for k in (blocked_keywords or []) if k and k.strip()}
        filtered: List[str] = []
        removed: List[str] = []

        for keyword in keywords:
            kw = (keyword or "").strip()
            if not kw:
                continue
            kw_lower = kw.lower()
            is_handle = kw_lower.startswith("@") or ("_" in kw_lower)
            if kw_lower in blocked_set or is_handle:
                removed.append(kw_lower)
                continue
            filtered.append(kw_lower)

        dedup_filtered = list(dict.fromkeys(filtered))
        dedup_removed = list(dict.fromkeys(removed))
        return dedup_filtered, dedup_removed

    def _llm_filter_sensitive_keywords(self, keywords: List[str]) -> List[str]:
        """Use a text LLM to remove personal/sensitive tags while keeping neutral content tags."""
        if not keywords:
            return []
        try:
            import ollama
            client = ollama.Client(timeout=30)

            text_model = None
            for model in self.TEXT_MODELS:
                try:
                    client.show(model)
                    text_model = model
                    break
                except Exception:
                    continue

            if not text_model:
                logger.warning("No text model available for keyword anonymization; using deterministic filtering only")
                return keywords

            payload = ", ".join(keywords)
            response = client.chat(
                model=text_model,
                messages=[{
                    "role": "user",
                    "content": (
                        "You are cleaning photo keywords for privacy. "
                        "Remove personal names, usernames/handles, nicknames, and relationship words "
                        "like friends/family/partner/mum/dad. "
                        "Keep non-person subject/location/object tags unchanged. "
                        "Return ONLY a comma-separated list of kept keywords in lowercase.\n\n"
                        f"Keywords: {payload}"
                    ),
                }],
                options={"temperature": 0.0, "num_predict": 200},
            )
            raw = (response.get("message", {}) or {}).get("content", "").strip()
            if not raw:
                return keywords
            kept = [k.strip().lower() for k in raw.split(",") if k.strip()]
            return list(dict.fromkeys(kept))
        except Exception as e:
            logger.warning(f"LLM keyword anonymization failed: {e}")
            return keywords

    def _hybrid_filter_sensitive_keywords(
        self, keywords: List[str], blocked_keywords: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """Hybrid privacy filter: LLM intent filter + deterministic blocked-list/handle filter."""
        llm_kept_keywords = self._llm_filter_sensitive_keywords(keywords)
        llm_removed = [k for k in keywords if k.lower() not in {x.lower() for x in llm_kept_keywords}]
        final_kept, blocked_removed = self._filter_sensitive_keywords(
            llm_kept_keywords, blocked_keywords=blocked_keywords
        )
        removed = list(dict.fromkeys([x.lower() for x in llm_removed] + blocked_removed))
        return final_kept, removed

    @staticmethod
    def clear_exif_keyword_tags_only(file_path: str) -> bool:
        """Remove only keyword tag fields, preserving captions/descriptions."""
        import shutil
        exiftool_path = shutil.which("exiftool")
        if not exiftool_path:
            return False
        try:
            result = subprocess.run(
                [exiftool_path, "-overwrite_original", "-XMP-dc:Subject=", "-IPTC:Keywords=", file_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Clear keyword tags failed: {e}")
            return False

    def anonymize_existing_photo_keywords(
        self, file_path: str, blocked_keywords: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """Anonymize existing keyword tags on a photo without running image description."""
        existing_keywords = self.read_exif_keywords(file_path)
        filtered_existing, removed_existing = self._hybrid_filter_sensitive_keywords(
            existing_keywords, blocked_keywords=blocked_keywords
        )
        if not removed_existing:
            return {
                "success": True,
                "existing_keywords": existing_keywords,
                "kept_keywords": filtered_existing,
                "removed_keywords": [],
                "message": "No sensitive keywords found to remove",
            }

        if not self.clear_exif_keyword_tags_only(file_path):
            return {
                "success": False,
                "existing_keywords": existing_keywords,
                "kept_keywords": existing_keywords,
                "removed_keywords": [],
                "message": "Failed to clear keyword fields before anonymization",
            }
        write_result = self.write_exif_keywords(file_path, filtered_existing, "")
        return {
            "success": bool(write_result.get("success")),
            "existing_keywords": existing_keywords,
            "kept_keywords": filtered_existing,
            "removed_keywords": removed_existing,
            "message": write_result.get("message", ""),
        }

    def keyword_image(self, file_path: str, city_radius_km: float = 5.0,
                      clear_keywords: bool = False,
                      clear_location: bool = False,
                      force_resize_max_1080: bool = False,
                      anonymize_keywords: bool = False,
                      blocked_keywords: Optional[List[str]] = None,
                      ownership_notice: str = "") -> Dict[str, any]:
        """Full pipeline: describe image, extract keywords, resolve GPS location, write to EXIF.

        Args:
            file_path: Path to image file.
            city_radius_km: Reserved for future proximity grouping.
            clear_keywords: If True, strip existing keywords/description before writing.
            clear_location: If True, strip existing location fields before writing.

        Returns dict with description, keywords, location, and write result.
        """
        self._report(0.0, "Reading image...")
        file_name = Path(file_path).name
        resize_info = {
            "resized": False,
            "metadata_preserved": True,
        }

        if force_resize_max_1080:
            self._report(0.01, "Checking image dimensions...")
            resize_info = self._resize_image_preserving_metadata(
                file_path, max_width=1920, max_height=1080
            )
            if resize_info.get("resized"):
                self._report(0.06, "Downsampled to gallery size")

        # Clear existing fields if requested
        if clear_keywords:
            self._report(0.02, "Clearing existing keywords...")
            self.clear_exif_keywords(file_path)
        if clear_location:
            self._report(0.04, "Clearing existing location fields...")
            self.clear_exif_location(file_path)

        # Read existing keywords so we can merge (after any clear)
        existing_keywords = self.read_exif_keywords(file_path)

        with open(file_path, "rb") as f:
            image_bytes = f.read()

        # GPS location lookup
        self._report(0.1, "Reading GPS data...")
        gps = self.read_gps(file_path)
        location = None
        has_gps = gps is not None
        if has_gps:
            self._report(0.15, "Reverse-geocoding location...")
            location = self.reverse_geocode(gps[0], gps[1])
            logger.info(f"GPS location for {file_name}: {location}")

        self._report(0.2, "Describing image with vision model...")
        description = self.describe_image(image_bytes)

        self._report(0.5, "Extracting keywords...")
        keywords = self.extract_keywords(description)
        removed_sensitive_keywords: List[str] = []

        # Add location tags to keywords
        if location:
            for field in ("city", "state", "country"):
                val = location.get(field, "")
                if val and val.lower() not in [k.lower() for k in keywords]:
                    keywords.append(val.lower())
        elif not has_gps:
            # Mark as no GPS in batch mode
            if "nogps" not in keywords:
                keywords.append("nogps")

        existing_keywords_for_merge = existing_keywords
        removed_existing_sensitive: List[str] = []
        if anonymize_keywords:
            # Sanitize generated keywords and existing metadata keywords.
            keywords, removed_generated_sensitive = self._hybrid_filter_sensitive_keywords(
                keywords, blocked_keywords=blocked_keywords
            )
            existing_keywords_for_merge, removed_existing_sensitive = self._hybrid_filter_sensitive_keywords(
                existing_keywords, blocked_keywords=blocked_keywords
            )
            removed_sensitive_keywords = list(dict.fromkeys(
                removed_generated_sensitive + removed_existing_sensitive
            ))

        # Deduplicate: only write keywords that aren't already in EXIF
        existing_lower = {k.lower() for k in existing_keywords_for_merge}
        new_keywords = [k for k in keywords if k.lower() not in existing_lower]

        self._report(0.8, "Writing keywords to EXIF...")
        if anonymize_keywords:
            # Replace existing keyword set with sanitized + generated keywords.
            combined_target_keywords = sorted(set(existing_keywords_for_merge + keywords))
            self.clear_exif_keyword_tags_only(file_path)
            write_result = self.write_exif_keywords(file_path, combined_target_keywords, description)
            new_keywords = [k for k in combined_target_keywords if k.lower() not in {x.lower() for x in existing_keywords}]
        else:
            write_result = self.write_exif_keywords(file_path, new_keywords, description)

        # Write location EXIF fields
        if location and any(location.values()):
            self.write_exif_location(
                file_path, location.get("country", ""),
                location.get("state", ""), location.get("city", ""),
            )

        ownership_result = None
        if ownership_notice.strip():
            ownership_result = self.write_ownership_metadata(file_path, ownership_notice.strip())

        # Combined set for display (existing + new)
        combined_keywords = (
            sorted(set(existing_keywords_for_merge + keywords))
            if anonymize_keywords
            else sorted(set(existing_keywords + keywords))
        )

        self._report(1.0, "Done")
        return {
            "file_name": file_name,
            "description": description,
            "keywords": combined_keywords,
            "new_keywords": new_keywords,
            "existing_keywords": existing_keywords,
            "removed_sensitive_keywords": removed_sensitive_keywords,
            "has_gps": has_gps,
            "location": location,
            "exif_result": write_result,
            "ownership_result": ownership_result,
            "resize_info": resize_info,
        }

    # ------------------------------------------------------------------
    # Docling fallback
    # ------------------------------------------------------------------

    def _try_docling(self, file_path: str) -> Optional[str]:
        """Try using Docling for higher-quality conversion. Returns None if unavailable."""
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(file_path)
            md = result.document.export_to_markdown()
            if md and md.strip():
                logger.info("Used Docling for PDF conversion")
                return md
        except Exception:
            pass
        return None
