# ## File: pages/7_Document_Extract.py
# Version: v5.8.0
# Date: 2026-01-29
# Purpose: Document extraction tools — Textifier (document to Markdown) and Anonymizer.

import streamlit as st
import sys
from pathlib import Path
import os
import shutil
import json
import re
import tempfile
import time
import zipfile
import io
import unicodedata
import subprocess
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core modules
from cortex_engine.anonymizer import DocumentAnonymizer, AnonymizationMapping, AnonymizationOptions
from cortex_engine.utils import get_logger, convert_windows_to_wsl_path
from cortex_engine.config_manager import ConfigManager
from cortex_engine.version_config import VERSION_STRING
from cortex_engine.journal_authority import classify_journal_authority
from cortex_engine.preface_classification import classify_credibility_tier
from cortex_engine.document_preface import add_document_preface

# Set up logging
logger = get_logger(__name__)

# Page configuration
st.set_page_config(page_title="Document & Photo Processing", layout="wide", page_icon="📝")

# Page metadata
PAGE_VERSION = VERSION_STRING
MAX_BATCH_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB


# ======================================================================
# Shared helpers
# ======================================================================


class _SessionUpload:
    """Persist uploaded file bytes across Streamlit reruns."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data or b""
        self.size = len(self._data)

    def getvalue(self) -> bytes:
        return self._data

def _get_knowledge_base_files(extensions: List[str]) -> List[Path]:
    """Return files from knowledge base directories matching given extensions."""
    config_manager = ConfigManager()
    config = config_manager.get_config()

    possible_dirs = []
    if config.get("ai_database_path"):
        base_path = Path(convert_windows_to_wsl_path(config["ai_database_path"]))
        possible_dirs.extend([
            base_path / "documents",
            base_path / "source_documents",
            base_path.parent / "documents",
            base_path.parent / "source_documents",
        ])
    possible_dirs.extend([
        project_root / "documents",
        project_root / "source_documents",
        project_root / "test_documents",
    ])

    files = []
    for dir_path in possible_dirs:
        if dir_path.exists():
            for fp in dir_path.glob("**/*"):
                if fp.is_file() and fp.suffix.lower() in extensions:
                    files.append(fp)
    return files


def _file_input_widget(key_prefix: str, allowed_types: List[str], label: str = "Choose a document:"):
    """Render upload / browse KB widget. Returns selected file path or None."""
    # Use a version counter so "Clear All Files" can reset the uploader widget
    if f"{key_prefix}_upload_version" not in st.session_state:
        st.session_state[f"{key_prefix}_upload_version"] = 0
    upload_version = st.session_state[f"{key_prefix}_upload_version"]

    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Browse Knowledge Base"],
        key=f"{key_prefix}_method",
    )

    selected_file = None

    if input_method == "Upload File":
        uploaded = st.file_uploader(label, type=allowed_types,
                                    key=f"{key_prefix}_upload_v{upload_version}",
                                    accept_multiple_files=(st.session_state.get(f"{key_prefix}_batch", False)))
        if uploaded:
            files = uploaded if isinstance(uploaded, list) else [uploaded]
            accepted_files = []
            accepted_bytes = 0
            for uf in files:
                size_bytes = int(getattr(uf, "size", 0) or 0)
                if size_bytes <= 0:
                    try:
                        size_bytes = len(uf.getvalue())
                    except Exception:
                        size_bytes = 0
                if accepted_bytes + size_bytes > MAX_BATCH_UPLOAD_BYTES:
                    break
                accepted_files.append(uf)
                accepted_bytes += size_bytes
            if not accepted_files:
                st.error("Upload exceeds 1GB total limit. Select fewer/smaller files.")
                return None
            if len(accepted_files) < len(files):
                st.warning(
                    f"Maximum 1GB total upload for this batch — only the first "
                    f"{len(accepted_files)} of {len(files)} files will be processed."
                )
            files = accepted_files
            temp_dir = Path(tempfile.gettempdir()) / f"cortex_{key_prefix}"
            temp_dir.mkdir(exist_ok=True, mode=0o755)
            paths = []
            for uf in files:
                dest = str(temp_dir / f"upload_{int(time.time())}_{uf.name}")
                with open(dest, "wb") as f:
                    f.write(uf.getvalue())
                os.chmod(dest, 0o644)
                paths.append(dest)
            if len(paths) == 1:
                selected_file = paths[0]
                st.success(f"Uploaded: {files[0].name}")
            else:
                selected_file = paths  # list for batch
                st.success(f"Uploaded {len(paths)} files")
    else:
        knowledge_files = _get_knowledge_base_files([f".{t}" for t in allowed_types])
        if knowledge_files:
            names = [f"{f.name} ({f.parent.name})" for f in knowledge_files]
            idx = st.selectbox("Select document:", range(len(names)),
                               format_func=lambda x: names[x], index=None,
                               placeholder="Choose a document...", key=f"{key_prefix}_kb")
            if idx is not None:
                selected_file = str(knowledge_files[idx])
                st.success(f"Selected: {knowledge_files[idx].name}")
        else:
            st.warning("No documents found in knowledge base directories")
            st.info("Try uploading a file instead")

    return selected_file


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def _user_visible_filename(file_path: str) -> str:
    """Strip internal upload prefixes from temp files for display/metadata."""
    name = Path(file_path).name
    m = re.match(r"^upload_\d+_(.+)$", name)
    return m.group(1) if m else name


def _user_visible_stem(file_path: str) -> str:
    return Path(_user_visible_filename(file_path)).stem


def _read_photo_metadata_preview(file_path: str) -> dict:
    """Read existing photo metadata fields for preview (keywords/description/location)."""
    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        return {"available": False, "reason": "exiftool not found on PATH"}
    try:
        result = subprocess.run(
            [
                exiftool_path,
                "-json",
                "-XMP-dc:Subject",
                "-IPTC:Keywords",
                "-XMP-dc:Description",
                "-IPTC:Caption-Abstract",
                "-EXIF:ImageDescription",
                "-XMP-photoshop:City",
                "-IPTC:City",
                "-XMP-photoshop:State",
                "-IPTC:Province-State",
                "-XMP-photoshop:Country",
                "-IPTC:Country-PrimaryLocationName",
                "-GPSLatitude",
                "-GPSLongitude",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"available": False, "reason": result.stderr.strip() or "exiftool read failed"}

        payload = json.loads(result.stdout)
        if not payload:
            return {"available": False, "reason": "No metadata found"}
        row = payload[0]

        keywords = []
        for field in ("Subject", "Keywords"):
            val = row.get(field, [])
            if isinstance(val, str):
                val = [val]
            for item in val:
                v = (item or "").strip()
                if v:
                    keywords.append(v)
        keywords = list(dict.fromkeys(keywords))

        description = (
            (row.get("Description") or "").strip()
            or (row.get("Caption-Abstract") or "").strip()
            or (row.get("ImageDescription") or "").strip()
        )
        city = (row.get("City") or "").strip()
        state = (row.get("State") or row.get("Province-State") or "").strip()
        country = (row.get("Country") or row.get("Country-PrimaryLocationName") or "").strip()

        gps_lat = row.get("GPSLatitude")
        gps_lon = row.get("GPSLongitude")
        gps = None
        if gps_lat not in (None, "") and gps_lon not in (None, ""):
            gps = f"{gps_lat}, {gps_lon}"

        return {
            "available": True,
            "keywords": keywords,
            "description": description,
            "city": city,
            "state": state,
            "country": country,
            "gps": gps,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def _write_photo_metadata_quick_edit(
    file_path: str,
    keywords: List[str],
    description: str,
    city: str,
    state: str,
    country: str,
) -> dict:
    """Apply quick metadata edits (replace keywords/description/location fields)."""
    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        return {"success": False, "message": "exiftool not found on PATH"}
    try:
        cmd = [
            exiftool_path,
            "-overwrite_original",
            # Clear existing keyword/caption/location fields so this acts as an explicit edit.
            "-XMP-dc:Subject=",
            "-IPTC:Keywords=",
            "-XMP-dc:Description=",
            "-IPTC:Caption-Abstract=",
            "-EXIF:ImageDescription=",
            "-IPTC:Country-PrimaryLocationName=",
            "-XMP-photoshop:Country=",
            "-IPTC:Province-State=",
            "-XMP-photoshop:State=",
            "-IPTC:City=",
            "-XMP-photoshop:City=",
        ]
        for kw in keywords:
            cmd.append(f"-XMP-dc:Subject+={kw}")
            cmd.append(f"-IPTC:Keywords+={kw}")
        desc = (description or "").strip()
        if desc:
            cmd.append(f"-XMP-dc:Description={desc}")
            cmd.append(f"-IPTC:Caption-Abstract={desc}")
            cmd.append(f"-EXIF:ImageDescription={desc}")
        if country.strip():
            cmd.append(f"-IPTC:Country-PrimaryLocationName={country.strip()}")
            cmd.append(f"-XMP-photoshop:Country={country.strip()}")
        if state.strip():
            cmd.append(f"-IPTC:Province-State={state.strip()}")
            cmd.append(f"-XMP-photoshop:State={state.strip()}")
        if city.strip():
            cmd.append(f"-IPTC:City={city.strip()}")
            cmd.append(f"-XMP-photoshop:City={city.strip()}")
        cmd.append(file_path)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return {"success": True, "message": result.stdout.strip()}
        return {"success": False, "message": result.stderr.strip() or "metadata write failed"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def _ascii_fold(text: str) -> str:
    """Fold unicode text to ASCII for robust search/parsing."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_ascii_text(text: str) -> str:
    """Normalize arbitrary text to ASCII-safe form."""
    cleaned = _clean_line(_ascii_fold(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
    return cleaned


def _normalize_person_name_ascii(name: str) -> str:
    """Normalize person names to ASCII-safe form while preserving punctuation."""
    normalized = _normalize_ascii_text(name)
    # Remove affiliation markers/superscripts commonly embedded in author lists.
    normalized = re.sub(r"\b\d+\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,;:-*")
    return normalized


def _first_nonempty_lines(text: str, limit: int = 30) -> List[str]:
    lines = [_clean_line(x) for x in (text or "").splitlines()]
    lines = [x for x in lines if x]
    return lines[:limit]


def _looks_like_person_name(candidate: str) -> bool:
    c = _normalize_ascii_text(_clean_line(re.sub(r"\d+", "", candidate)))
    if not c or len(c) < 5 or len(c) > 80:
        return False
    low = c.lower()
    disallow = [
        "university", "department", "school", "faculty", "institute", "hospital", "campus",
        "australia", "india", "srilanka", "melbourne", "brisbane", "gold coast", "corresponding author",
        "author:", "keywords", "abstract", "study", "comprehensive", "licensed", "creative commons",
        "lecturer", "postgraduate", "associate professor", "professor",
        "management", "countries", "country", "licensee", "republic", "sciences",
        "introduction", "change management",
    ]
    if any(x in low for x in disallow):
        return False
    if "@" in c or "http" in low:
        return False
    c = re.sub(r"\b(Dr|Prof)\.?\s+", "", c, flags=re.IGNORECASE)
    c = re.sub(r"\b(PhD|MD|MSc|BSc|BA|MA|MPhil|DPhil|MBA|BHlthSc|Mast\s+Nutr&Diet)\b\.?", "", c, flags=re.IGNORECASE)
    c = re.sub(r"\s+", " ", c).strip(" ,;:-")
    parts = c.split()
    if len(parts) < 2 or len(parts) > 5:
        return False
    for p in parts:
        if not re.match(r"^(?:[A-Z]\.?|[A-Z][A-Za-z'\-]+)$", p):
            return False
    return True


def _extract_names_from_author_block(raw: str) -> List[str]:
    """Extract person-name candidates from a likely author block."""
    text = _clean_line(raw)
    if not text:
        return []

    # Remove common non-author prefixes and obvious affiliation tails.
    text = re.sub(r"(?i)^authors?\s*[:\-]\s*", "", text)
    text = re.sub(r"(?i)^for referencing,\s*please use:\s*", "", text)
    text = re.sub(r"(?i)\b(university|department|faculty|institute|hospital)\b.*$", "", text)
    text = re.sub(r"[*†‡§]", " ", text)
    text = re.sub(r"\(\s*[^)]*\)", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")

    parts = [p.strip() for p in re.split(r",|;|\band\b", text, flags=re.IGNORECASE) if p.strip()]
    names: List[str] = []
    for part in parts:
        cleaned = _normalize_ascii_text(part).strip(" ,;:-")
        # Handle citation forms like "Surname, A.B." by flipping to "A.B. Surname".
        m = re.match(r"^([A-Z][A-Za-z'\-]+)\s*,\s*([A-Z](?:\.[A-Z])*\.?)$", cleaned)
        if m:
            cleaned = f"{m.group(2)} {m.group(1)}"
        if _looks_like_person_name(cleaned):
            names.append(cleaned)
    return list(dict.fromkeys(names))


def _photo_description_issue(description: str) -> Optional[str]:
    """Return a user-facing issue for placeholder image descriptions."""
    desc = (description or "").strip()
    if not desc.startswith("[Image:"):
        return None
    low = desc.lower()
    if "timed out" in low:
        return "Image description timed out. The vision model may be overloaded."
    if "unavailable" in low:
        return "Image description was skipped because the vision model was unavailable."
    if "error" in low:
        return "Image description failed due to a vision model error."
    if "logo/icon omitted" in low:
        return "Image looked like a logo or icon, so description was intentionally skipped."
    return desc


def _extract_available_at(md_content: str) -> str:
    text = md_content or ""
    doi_url = re.search(r"https?://(?:dx\.)?doi\.org/\S+", text, flags=re.IGNORECASE)
    if doi_url:
        return doi_url.group(0).rstrip(").,;]")

    doi = re.search(r"\bdoi\s*[:]\s*(10\.\d{4,9}/[^\s;,)\]]+)", text, flags=re.IGNORECASE)
    if doi:
        return f"https://doi.org/{doi.group(1)}"

    doi_bare = re.search(r"\b(10\.\d{4,9}/[^\s;,)\]]+)", text)
    if doi_bare:
        return f"https://doi.org/{doi_bare.group(1)}"

    url = re.search(r"https?://\S+", text, flags=re.IGNORECASE)
    if url:
        return url.group(0).rstrip(").,;]")

    return "Unknown"


def _sanitize_markdown_for_preface(md_content: str) -> str:
    """Remove known conversion-error boilerplate from markdown before LLM/keyword extraction."""
    def _is_logo_or_icon_caption(caption_text: str) -> bool:
        c = _clean_line(caption_text)
        if not c:
            return True
        low = c.lower()
        marker_hits = sum(
            1
            for marker in [
                "logo", "icon", "icons", "watermark", "emblem", "symbol",
                "badge", "seal", "silhouette", "letters", "initials",
            ]
            if marker in low
        )
        visual_hits = sum(
            1
            for marker in [
                "black and white", "circular", "circle", "strip", "background",
                "left icon", "right icon", "main colors", "main colours",
            ]
            if marker in low
        )
        return (marker_hits >= 1 and visual_hits >= 1) or marker_hits >= 2

    cleaned_lines = []
    error_markers = [
        "image could not be described",
        "vision model",
        "vlm processing failed",
        "image processing failed",
        "source_type': 'image_error'",
        "source_type: image_error",
    ]
    for raw_line in (md_content or "").splitlines():
        line = raw_line.strip()
        low = line.lower()
        if any(marker in low for marker in error_markers):
            continue
        # Drop inline image-caption blocks that are logo/icon noise.
        image_caption_match = re.match(r"^\s*>\s*\*\*\[Image[^\]]*\]\*\*:\s*(.+)$", line, flags=re.IGNORECASE)
        if image_caption_match:
            caption_body = image_caption_match.group(1).strip()
            if _is_logo_or_icon_caption(caption_body):
                continue
        # Drop markdown image embeds that mostly carry filenames/noise for metadata extraction.
        if line.startswith("![") and "](" in line:
            continue
        cleaned_lines.append(raw_line)
    return "\n".join(cleaned_lines)


def _extract_authors_from_markdown(md_content: str) -> List[str]:
    pre_abstract = re.split(r"(?im)^\s*abstract\b", md_content or "", maxsplit=1)[0][:12000]
    lines = _first_nonempty_lines(pre_abstract, limit=120)
    authors: List[str] = []

    # 1) Prefer explicit author/citation blocks first.
    explicit_patterns = [
        r"(?is)\bauthors?\b\s*[:\-]?\s*(.+?)(?=\n\s*(university|affiliations?|author contribution|doi|abstract|keywords?|for referencing|\Z))",
        r"(?is)for referencing,\s*please use:\s*(.+?)(?=\n|$)",
    ]
    for pat in explicit_patterns:
        for match in re.finditer(pat, pre_abstract, flags=re.IGNORECASE):
            authors.extend(_extract_names_from_author_block(match.group(1)))
    if authors:
        return list(dict.fromkeys(authors))[:12]

    # 2) Fall back to line-based heuristics.
    for line in lines:
        if len(line) > 260:
            continue
        if re.match(r"^\d+\s", line):
            continue
        # Skip likely title/topic lines unless they look like a name list delimiter line.
        if "," not in line and ";" not in line and " and " not in line.lower():
            continue
        cleaned = re.sub(r"([A-Za-z])\d+\b", r"\1", line)
        cleaned = re.sub(r"\band\b", ",", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(Dr|Prof)\.?\s+", "", cleaned, flags=re.IGNORECASE)
        parts = [p.strip() for p in re.split(r",|;", cleaned) if p.strip()]
        for part in parts:
            part = re.sub(r"\s+", " ", part).strip(" ,;:-")
            if _looks_like_person_name(part):
                authors.append(part)
        if len(authors) >= 12:
            break

    unique = list(dict.fromkeys(authors))
    return unique[:12]


def _guess_title_from_markdown(md_content: str, file_path: str) -> str:
    lines = _first_nonempty_lines(md_content, limit=120)
    skip_exact = {
        "viewpoint", "introduction", "abstract", "background", "keywords", "key words"
    }
    skip_contains = [
        "please cite this publication as",
        "this work is published under",
        "opinions expressed and arguments employed",
        "revised version",
        "corrigenda",
    ]
    # Academic fallback: title is often the line immediately before author names.
    for i in range(1, min(80, len(lines))):
        if _extract_authors_from_markdown(lines[i]):
            prev = _clean_line(lines[i - 1])
            if (
                len(prev) >= 20
                and not re.match(r"^page\s+\d+", prev, flags=re.IGNORECASE)
                and "licensed" not in prev.lower()
                and "creative commons" not in prev.lower()
                and not any(s in prev.lower() for s in skip_contains)
            ):
                return prev[:180]
    for i, line in enumerate(lines[:40]):
        low = line.lower().strip()
        if low in skip_exact and i + 1 < len(lines):
            nxt = lines[i + 1]
            if len(nxt) > 15:
                return nxt[:180]
        if re.match(r"^#+\s*page\s+\d+", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^page\s+\d+", low):
            continue
        if line.startswith("#"):
            title = _clean_line(line.lstrip("#"))
            if title and not re.match(r"^page\s+\d+", title, flags=re.IGNORECASE):
                return title[:180]
    for line in lines[:60]:
        low = line.lower()
        if re.match(r"^page\s+\d+", low):
            continue
        if len(line) < 15:
            continue
        if len(line) > 180:
            continue
        if any(x in low for x in skip_contains):
            continue
        if any(x in low for x in ["university", "hospital", "school of", "email:", "doi:", "journal", "www."]):
            continue
        if any(x in low for x in ["licensed", "creative commons", "corresponding author"]):
            continue
        if low in skip_exact:
            continue
        # Prefer sentence-case/title-like lines as title candidates.
        if len(re.findall(r"[A-Za-z]", line)) >= 12:
            return line[:180]
    return Path(file_path).stem


def _detect_source_type_hint(file_path: str, md_content: str) -> str:
    text = f"{_user_visible_filename(file_path)}\n{md_content[:20000]}".lower()

    academic_patterns = [
        r"\belsevier\b", r"\bspringer\b", r"\bwiley\b", r"\bieee\b",
        r"\bdoi\b", r"\bjournal\b", r"\bproceedings\b",
    ]
    consulting_patterns = [
        r"\bdeloitte\b", r"\bmckinsey\b", r"\bbain\b", r"\bbcg\b",
        r"\bkpmg\b", r"\bpwc\b", r"\baccenture\b", r"\bconsulting\b",
        r"\bernst\s*&\s*young\b", r"\bey\b",
    ]
    institutional_patterns = [
        r"\badha\b", r"\baidh\b", r"\baustralian digital health agency\b",
        r"\baustralasian institute of digital health\b",
        r"\bdepartment of health\b", r"\bministry\b", r"\bgovernment\b",
        r"\bwho\b", r"\boecd\b", r"\bworld bank\b", r"\bunited nations\b",
        r"\bnhs\b", r"\bcdc\b", r"\buniversity\b", r"\binstitute\b",
    ]
    ai_patterns = [
        r"\bperplexity\b", r"\bchatgpt\b", r"\bopenai\b", r"\bclaude\b", r"\bgemini\b",
        r"\bdeep research\b", r"\bdeep report\b", r"\bdeep report into\b",
        r"\bgenerated by ai\b", r"\bai-generated\b",
    ]

    def _has_any(patterns: List[str]) -> bool:
        return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

    has_academic = _has_any(academic_patterns)
    has_consulting = _has_any(consulting_patterns)
    has_institutional = _has_any(institutional_patterns)
    has_ai_markers = _has_any(ai_patterns)
    em_dash_count = text.count("—") + text.count("–")

    if has_academic:
        return "Academic"
    # Explicit institutional signals should not be overwritten by consulting hints.
    if has_institutional and not has_consulting:
        return "Other"
    if has_consulting:
        return "Consulting Company"
    # If not strongly human-sourced, bias toward AI when AI-style signals are present.
    if has_ai_markers or (em_dash_count >= 8 and not (has_academic or has_institutional or has_consulting)):
        return "AI Generated Report"
    return "Other"


def _infer_publisher_from_text(md_text: str, source_hint: str) -> str:
    text = (md_text or "").lower()
    publisher_patterns = [
        (r"\baustralian digital health agency\b|\badha\b", "Australian Digital Health Agency"),
        (r"\baustralasian institute of digital health\b|\baidh\b", "Australasian Institute of Digital Health"),
        (r"\bworld health organization\b|\bwho\b", "World Health Organization"),
        (r"\boecd\b", "OECD"),
        (r"\bworld bank\b", "World Bank"),
        (r"\bunited nations\b|\bun\b", "United Nations"),
        (r"\belsevier\b", "Elsevier"),
        (r"\bspringer\b", "Springer"),
        (r"\bwiley\b", "Wiley"),
        (r"\bieee\b", "IEEE"),
        (r"\bdeloitte\b", "Deloitte"),
        (r"\bmckinsey\b", "McKinsey"),
        (r"\bbain\b", "Bain"),
        (r"\bbcg\b", "BCG"),
        (r"\bkpmg\b", "KPMG"),
        (r"\bernst\s*&\s*young\b|\bey\b", "EY"),
        (r"\bpwc\b", "PwC"),
        (r"\baccenture\b", "Accenture"),
        (r"\bperplexity\b", "Perplexity"),
        (r"\bopenai\b", "OpenAI"),
        (r"\bclaude\b|\banthropic\b", "Anthropic"),
        (r"\bgemini\b|\bgoogle\b", "Google"),
    ]
    for pattern, label in publisher_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    if source_hint == "AI Generated Report":
        return "Unknown AI"
    return "Unknown"


def _extract_publishing_date(md_text: str, file_path: str) -> str:
    text = (md_text or "")[:12000]
    text_ascii = _ascii_fold(text)

    # Strong citation-line signal, e.g. "... (2024), ... DOI ..."
    citation_line = re.search(
        r"(?is)\bplease\s+cite\b[^\n]{0,300}?\((19|20)\d{2}\)",
        text_ascii,
        flags=re.IGNORECASE,
    )
    if citation_line:
        m = re.search(r"\((19|20)\d{2}\)", citation_line.group(0))
        if m:
            return m.group(0).strip("()")

    revised_pattern = re.search(
        r"(?i)\brevised\s+version\b[^A-Za-z0-9]{0,8}(?:,\s*)?([A-Za-z]{3,9}\s+20\d{2}|20\d{2})",
        text_ascii,
    )
    if revised_pattern:
        return _clean_line(revised_pattern.group(1))

    # Prefer explicitly labeled publication/update lines before generic year-only matches.
    label_pattern = (
        r"(?i)\b(?:published|publication date|date published|released|issued|last updated|updated)\b"
        r"[^A-Za-z0-9]{0,12}"
        r"(20\d{2}[-/][01]?\d[-/][0-3]?\d|[0-3]?\d\s+[A-Za-z]{3,9}\s+20\d{2}|[A-Za-z]{3,9}\s+[0-3]?\d,\s+20\d{2}|20\d{2})"
    )
    m = re.search(label_pattern, text_ascii)
    if m:
        return _clean_line(m.group(1))

    # Context-aware year extraction from citation/publication lines.
    scored_year = ""
    scored_value = -999
    for raw_line in text_ascii.splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue
        years = re.findall(r"\b((?:19|20)\d{2})\b", line)
        if not years:
            continue

        low = line.lower()
        score = 0
        if "please cite" in low or "cite this publication" in low:
            score += 8
        if "revised version" in low:
            score += 7
        if any(k in low for k in ["published", "publication", "issued", "release", "copyright"]):
            score += 6
        if any(k in low for k in ["isbn", "issn", "doi"]):
            score += 5
        if re.search(r"\((?:19|20)\d{2}\)", line):
            score += 4
        if low.startswith("> **[image"):
            score -= 5
        if "photo credits" in low:
            score -= 3

        candidate_year = years[0]
        if score > scored_value:
            scored_value = score
            scored_year = candidate_year

    if scored_year and scored_value >= 3:
        return scored_year

    generic_pattern = r"(20\d{2}[-/][01]?\d[-/][0-3]?\d|[0-3]?\d\s+[A-Za-z]{3,9}\s+20\d{2}|[A-Za-z]{3,9}\s+[0-3]?\d,\s+20\d{2})"
    m = re.search(generic_pattern, text_ascii)
    if m:
        return _clean_line(m.group(1))

    year_match = re.search(r"\b((?:19|20)\d{2})\b", text_ascii)
    if year_match:
        return year_match.group(1)

    # Last fallback: year in filename.
    filename = _user_visible_filename(file_path)
    m = re.search(r"\b(19|20)\d{2}\b", filename)
    if m:
        return m.group(0)
    return "Unknown"


def _extract_json_block(raw: str) -> Optional[dict]:
    if not raw:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _extract_preface_metadata_with_llm(file_path: str, md_content: str, source_hint: str) -> Optional[dict]:
    try:
        from cortex_engine.llm_interface import LLMInterface
        llm = LLMInterface(model="mistral:latest", temperature=0.1)
    except Exception as e:
        logger.warning(f"Could not initialize LLM for preface extraction: {e}")
        return None

    snippet = _sanitize_markdown_for_preface(md_content)[:18000]
    prompt = f"""
You extract publication metadata from markdown content.
Return STRICT JSON only with keys:
- title (string)
- source_type (one of: Academic, Consulting Company, AI Generated Report, Other)
- publisher (string)
- publishing_date (string)
- authors (array of strings)
- available_at (string)
- abstract (string)
- keywords (array of strings)
- credibility_tier_value (integer: 0..5)
- credibility_tier_key (one of: peer-reviewed, institutional, pre-print, editorial, commentary, unclassified)
- credibility_tier_label (one of: Peer-Reviewed, Institutional, Pre-Print, Editorial, Commentary, Unclassified)
- credibility (human-readable string, e.g. "Draft Institutional Report")

Rules:
- Use source hint: "{source_hint}" unless content strongly indicates a different source_type.
- If AI Generated Report and publisher cannot be identified, set publisher to "Unknown AI".
- If abstract is not explicitly present, generate a concise abstract from the document.
- If abstract exists, extract the full abstract paragraph(s), not just one line.
- For authors, include only person names and exclude affiliations/locations/institutions.
- For title, prefer the document title and never use abstract opening text or citation boilerplate (e.g. "Please cite this publication as").
- For available_at, prefer DOI URL if present, else canonical report URL, else "Unknown".
- For publishing_date, prioritize explicit year/date from citation/publishing lines, including patterns like "(2024)" and "Revised version, November 2024".
- Provide 5-12 useful keywords, with each keyword at most TWO WORDS.
- Credibility tiers:
  5 / peer-reviewed / Peer-Reviewed: NLM/PubMed, Nature, The Lancet, JAMA, BMJ
  4 / institutional / Institutional: WHO, UN/IPCC, OECD, World Bank, ABS, government depts, universities/institutes
  3 / pre-print / Pre-Print: arXiv, SSRN, bioRxiv, ResearchGate
  2 / editorial / Editorial: Scientific American, The Conversation, HBR
  1 / commentary / Commentary: blogs, newsletters, consulting reports, opinion
  0 / unclassified / Unclassified: not yet assessed
- If source_type is AI Generated Report, default to 0 / unclassified.
- Treat titles like "Deep report into..." and strongly AI-styled writing (frequent em-dashes) as AI-generated unless clear human institutional/academic markers are present.
- Ignore tiny/logo/icon-only figure captions when determining title, abstract, keywords, source_type, and date.
- If a field is unknown use "Unknown" (or [] for authors/keywords).

File name: {_user_visible_filename(file_path)}

Markdown content:
{snippet}
"""
    try:
        response = llm.generate(prompt, max_tokens=900)
        return _extract_json_block(response)
    except Exception as e:
        logger.warning(f"LLM preface extraction failed: {e}")
        return None


def _detect_document_stage(file_path: str, title: str, md_content: str) -> str:
    text = f"{_user_visible_filename(file_path)}\n{title}\n{md_content[:30000]}".lower()
    return "Draft" if re.search(r"\bdraft\b", text) else "Final"


def _check_url_availability(url: str, timeout: float = 8.0) -> Dict[str, str]:
    """Probe URL availability without downloading full content."""
    checked_at = datetime.now().strftime("%Y-%m-%d")
    if not url or url == "Unknown":
        return {
            "availability_status": "unknown",
            "availability_http_code": "",
            "availability_checked_at": checked_at,
            "availability_note": "No canonical source URL provided.",
        }

    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {
            "availability_status": "unknown",
            "availability_http_code": "",
            "availability_checked_at": checked_at,
            "availability_note": f"URL appears invalid: {url}",
        }

    headers = {
        "User-Agent": "CortexSuite/DocumentExtract (+source-integrity-check)",
        "Accept": "*/*",
    }
    methods = ("HEAD", "GET")
    last_http_error = None

    for method in methods:
        req_headers = dict(headers)
        if method == "GET":
            req_headers["Range"] = "bytes=0-0"
        req = Request(url, headers=req_headers, method=method)
        try:
            with urlopen(req, timeout=timeout) as resp:
                code = int(getattr(resp, "status", 200) or 200)
            if 200 <= code < 400:
                return {
                    "availability_status": "available",
                    "availability_http_code": str(code),
                    "availability_checked_at": checked_at,
                    "availability_note": f"Available as at {checked_at}.",
                }
            if code == 404:
                return {
                    "availability_status": "not_found",
                    "availability_http_code": "404",
                    "availability_checked_at": checked_at,
                    "availability_note": f"Previously available at: {url} but no longer available as at: {checked_at}.",
                }
            if code == 410:
                return {
                    "availability_status": "gone",
                    "availability_http_code": "410",
                    "availability_checked_at": checked_at,
                    "availability_note": f"Previously available at: {url} but has been removed (HTTP 410) as at: {checked_at}.",
                }
            if 400 <= code < 500:
                return {
                    "availability_status": "client_error",
                    "availability_http_code": str(code),
                    "availability_checked_at": checked_at,
                    "availability_note": f"Source URL returned HTTP {code} as at {checked_at}; verify whether the source moved, is access-restricted, or withdrawn.",
                }
            if code >= 500:
                return {
                    "availability_status": "server_error",
                    "availability_http_code": str(code),
                    "availability_checked_at": checked_at,
                    "availability_note": f"Source host returned HTTP {code} as at {checked_at}; availability could not be confirmed.",
                }
        except HTTPError as e:
            code = int(e.code)
            if code == 404:
                return {
                    "availability_status": "not_found",
                    "availability_http_code": "404",
                    "availability_checked_at": checked_at,
                    "availability_note": f"Previously available at: {url} but no longer available as at: {checked_at}.",
                }
            if code == 410:
                return {
                    "availability_status": "gone",
                    "availability_http_code": "410",
                    "availability_checked_at": checked_at,
                    "availability_note": f"Previously available at: {url} but has been removed (HTTP 410) as at: {checked_at}.",
                }
            if code == 405 and method == "HEAD":
                # Some hosts disallow HEAD; retry with ranged GET.
                continue
            last_http_error = e
        except URLError as e:
            return {
                "availability_status": "unreachable",
                "availability_http_code": "",
                "availability_checked_at": checked_at,
                "availability_note": f"Source URL could not be reached as at {checked_at}: {e.reason}",
            }
        except Exception as e:
            return {
                "availability_status": "unreachable",
                "availability_http_code": "",
                "availability_checked_at": checked_at,
                "availability_note": f"Source URL check failed as at {checked_at}: {e}",
            }

    if last_http_error is not None:
        code = int(last_http_error.code)
        return {
            "availability_status": "client_error" if 400 <= code < 500 else "server_error",
            "availability_http_code": str(code),
            "availability_checked_at": checked_at,
            "availability_note": f"Source URL returned HTTP {code} as at {checked_at}; verify whether the source moved, is access-restricted, or withdrawn.",
        }
    return {
        "availability_status": "unknown",
        "availability_http_code": "",
        "availability_checked_at": checked_at,
        "availability_note": f"Source URL availability is unknown as at {checked_at}.",
    }


def _classify_credibility_tier(
    file_path: str,
    source_type: str,
    publisher: str,
    available_at: str,
    md_content: str,
    availability_status: str = "unknown",
) -> Dict[str, str]:
    text = f"{_user_visible_filename(file_path)}\n{publisher}\n{available_at}\n{md_content[:50000]}".lower()
    tier_value, key, label = classify_credibility_tier(
        text=text,
        source_type=source_type,
        availability_status=availability_status,
    )
    stage = _detect_document_stage(file_path, "", md_content)
    if availability_status in {"not_found", "gone"}:
        credibility_text = f"{stage} {label} Report (Source Link Unavailable)"
    elif availability_status in {"client_error", "server_error", "unreachable"}:
        credibility_text = f"{stage} {label} Report (Source Link Unverified)"
    else:
        credibility_text = f"{stage} {label} Report"
    return {
        "credibility_tier_value": tier_value,
        "credibility_tier_key": key,
        "credibility_tier_label": label,
        "credibility": credibility_text,
    }


def _clean_keywords(keywords: List[str]) -> List[str]:
    banned = {
        "image", "vision", "model", "error", "could", "described", "failed",
        "processing", "source", "type", "unknown", "document",
    }
    cleaned: List[str] = []
    for keyword in keywords:
        k = re.sub(r"\s+", " ", str(keyword or "").strip().lower())
        if not k or len(k) < 3:
            continue
        if k in banned:
            continue
        if len(k) > 60:
            continue
        if len(k.split()) > 2:
            continue
        if "image" in k and "health" not in k:
            continue
        if "error" in k or "vision model" in k:
            continue
        cleaned.append(k)
    # preserve order while deduping
    return list(dict.fromkeys(cleaned))[:8]


def _fallback_preface_metadata(file_path: str, md_content: str, source_hint: str) -> dict:
    md_clean = _sanitize_markdown_for_preface(md_content)
    title = _guess_title_from_markdown(md_clean, file_path)
    lines = _first_nonempty_lines(md_clean, limit=120)
    text_lower = md_clean.lower()

    publisher = _infer_publisher_from_text(md_clean, source_hint)
    publishing_date = _extract_publishing_date(md_clean, file_path)
    available_at = _extract_available_at(md_clean)

    authors = _extract_authors_from_markdown(md_clean)

    abstract = ""
    abstract_match = re.search(
        r"(?is)\babstract\b\s*[:\-]?\s*(.+?)(?=\n\s*(keywords?|key words?|introduction|background|methods?|j\s+med|doi:|##\s*page|\Z))",
        md_clean,
    )
    if abstract_match:
        abstract = _clean_line(abstract_match.group(1))
    if not abstract:
        # Fallback summary from first meaningful lines.
        abstract = " ".join([l for l in lines if len(l) > 30][:4])[:900] or "Summary not available."

    keywords = []
    kw_match = re.search(
        r"(?is)\bkeywords?\b\s*[:\-]?\s*(.+?)(?=\n\s*(introduction|background|methods?|##\s*page|\Z))",
        md_content[:20000],
        flags=re.IGNORECASE,
    )
    if kw_match:
        kw_raw = _clean_line(kw_match.group(1))
        keywords = [k.strip().lower() for k in re.split(r",|;|\|", kw_raw) if k.strip()][:12]
    if not keywords:
        tokens = re.findall(r"\b[A-Za-z][A-Za-z\-]{3,}\b", md_clean[:8000])
        freq = {}
        stop = {
            "this", "that", "with", "from", "were", "have", "been", "into", "their", "about", "which",
            "document", "page", "pages", "report", "using", "used", "study", "journal", "research",
            "university", "australia", "gold", "coast"
        }
        for t in tokens:
            low = t.lower()
            if low in stop:
                continue
            freq[low] = freq.get(low, 0) + 1
        keywords = [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:10]]

    availability = _check_url_availability(available_at)
    credibility = _classify_credibility_tier(
        file_path=file_path,
        source_type=source_hint or "Other",
        publisher=publisher,
        available_at=available_at,
        availability_status=availability.get("availability_status", "unknown"),
        md_content=md_clean,
    )

    return {
        "title": title or _user_visible_stem(file_path),
        "source_type": source_hint or "Other",
        "publisher": publisher,
        "publishing_date": publishing_date,
        "authors": authors,
        "available_at": available_at,
        "availability_status": availability.get("availability_status", "unknown"),
        "availability_http_code": availability.get("availability_http_code", ""),
        "availability_checked_at": availability.get("availability_checked_at", ""),
        "availability_note": availability.get("availability_note", ""),
        "abstract": abstract,
        "keywords": _clean_keywords(keywords),
        **credibility,
    }


def _normalize_preface_metadata(file_path: str, source_hint: str, raw_meta: Optional[dict], fallback_meta: dict, md_content: str) -> dict:
    data = raw_meta or {}
    # Back-compat aliases from older systems.
    if "credibility_tier_value" not in data and "credibility_value" in data:
        data["credibility_tier_value"] = data.get("credibility_value")
    if "credibility_tier_key" not in data and "credibility_source" in data:
        data["credibility_tier_key"] = str(data.get("credibility_source", "")).strip().lower()
    title = _normalize_ascii_text(_clean_line(str(data.get("title", "")))) or _normalize_ascii_text(fallback_meta["title"])
    source_type = _clean_line(str(data.get("source_type", ""))) or source_hint or fallback_meta["source_type"]
    if source_type not in {"Academic", "Consulting Company", "AI Generated Report", "Other"}:
        source_type = source_hint if source_hint in {"Academic", "Consulting Company", "AI Generated Report", "Other"} else "Other"

    publisher = _normalize_ascii_text(_clean_line(str(data.get("publisher", "")))) or _normalize_ascii_text(fallback_meta["publisher"])
    if not publisher or publisher.lower() == "unknown":
        publisher = _infer_publisher_from_text(_sanitize_markdown_for_preface(md_content), source_type)
    if source_type == "AI Generated Report" and (not publisher or publisher.lower() == "unknown"):
        publisher = "Unknown AI"

    publishing_date = _clean_line(str(data.get("publishing_date", ""))) or fallback_meta["publishing_date"] or "Unknown"
    if not publishing_date or publishing_date == "Unknown":
        publishing_date = _extract_publishing_date(_sanitize_markdown_for_preface(md_content), file_path)
    available_at = _clean_line(str(data.get("available_at", ""))) or fallback_meta.get("available_at", "Unknown")
    if available_at != "Unknown":
        available_at = _extract_available_at(available_at)
    availability = _check_url_availability(available_at)

    # Final deterministic source-type pass using consolidated title/publisher/content.
    # This guards against LLM drift and false consulting classifications.
    consolidated_text = f"{title}\n{publisher}\n{_sanitize_markdown_for_preface(md_content)}"
    deterministic_source = _detect_source_type_hint(file_path, consolidated_text)
    if deterministic_source == "AI Generated Report":
        source_type = "AI Generated Report"
    elif source_type == "Consulting Company" and deterministic_source == "Other":
        # Prefer neutral "Other" when strong consulting markers are absent.
        source_type = "Other"

    authors_raw = data.get("authors", fallback_meta.get("authors", []))
    if isinstance(authors_raw, str):
        authors = [a.strip() for a in re.split(r",|;", authors_raw) if a.strip()]
    elif isinstance(authors_raw, list):
        authors = [str(a).strip() for a in authors_raw if str(a).strip()]
    else:
        authors = []
    if not authors:
        authors = fallback_meta.get("authors", [])
    authors = [_normalize_person_name_ascii(a) for a in authors if _looks_like_person_name(a)]
    authors = [a for a in authors if _looks_like_person_name(a)]
    authors = list(dict.fromkeys(authors))

    keywords_raw = data.get("keywords", fallback_meta.get("keywords", []))
    if isinstance(keywords_raw, str):
        keywords = [k.strip() for k in re.split(r",|;|\|", keywords_raw) if k.strip()]
    elif isinstance(keywords_raw, list):
        keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
    else:
        keywords = []
    if not keywords:
        keywords = fallback_meta.get("keywords", [])
    keywords = _clean_keywords([_normalize_ascii_text(k.lower()) for k in keywords if k])

    abstract = _clean_line(str(data.get("abstract", ""))) or fallback_meta["abstract"] or "Summary not available."

    cred_value_raw = data.get("credibility_tier_value", fallback_meta.get("credibility_tier_value", 0))
    try:
        cred_value = int(cred_value_raw)
    except Exception:
        cred_value = int(fallback_meta.get("credibility_tier_value", 0))
    if cred_value not in _CREDIBILITY_TIERS:
        cred_value = int(fallback_meta.get("credibility_tier_value", 0))
    # Deterministic policy alignment with credibility_spec.md.
    deterministic = _classify_credibility_tier(
        file_path=file_path,
        source_type=source_type,
        publisher=publisher,
        available_at=available_at,
        availability_status=availability.get("availability_status", "unknown"),
        md_content=_sanitize_markdown_for_preface(md_content),
    )
    det_value = int(deterministic.get("credibility_tier_value", 0))
    if source_type == "AI Generated Report" or det_value > 0:
        cred_value = det_value
    cred_key, cred_label = _CREDIBILITY_TIERS.get(cred_value, _CREDIBILITY_TIERS[0])
    stage = _detect_document_stage(file_path, title, _sanitize_markdown_for_preface(md_content))
    availability_status = availability.get("availability_status", "unknown")
    if availability_status in {"not_found", "gone"}:
        credibility_text = f"{stage} {cred_label} Report (Source Link Unavailable)"
    elif availability_status in {"client_error", "server_error", "unreachable"}:
        credibility_text = f"{stage} {cred_label} Report (Source Link Unverified)"
    else:
        credibility_text = f"{stage} {cred_label} Report"

    journal_authority = classify_journal_authority(
        title=title,
        text=_sanitize_markdown_for_preface(md_content),
    )

    source_integrity_flag = "ok"
    if availability_status in {"not_found", "gone"}:
        source_integrity_flag = "deprecated_or_removed"
    elif availability_status in {"client_error", "server_error", "unreachable", "unknown"}:
        source_integrity_flag = "unverified"

    return {
        "title": title or _user_visible_stem(file_path),
        "source_type": source_type,
        "publisher": publisher or "Unknown",
        "publishing_date": publishing_date,
        "authors": authors[:20],
        "available_at": available_at or "Unknown",
        "availability_status": availability_status,
        "availability_http_code": availability.get("availability_http_code", ""),
        "availability_checked_at": availability.get("availability_checked_at", ""),
        "availability_note": availability.get("availability_note", ""),
        "source_integrity_flag": source_integrity_flag,
        "keywords": keywords[:8],
        "abstract": abstract,
        "credibility_tier_value": cred_value,
        "credibility_tier_key": cred_key,
        "credibility_tier_label": cred_label,
        "credibility": credibility_text,
        **journal_authority,
    }


def _yaml_escape(value: str) -> str:
    v = str(value or "").replace("'", "''")
    return f"'{v}'"


def _build_preface(md_meta: dict) -> str:
    authors = md_meta.get("authors") or []
    keywords = md_meta.get("keywords") or []
    authors_yaml = "[" + ", ".join(_yaml_escape(a) for a in authors) + "]" if authors else "[]"
    keywords_yaml = "[" + ", ".join(_yaml_escape(k) for k in keywords) + "]" if keywords else "[]"
    lines = [
        "---",
        "preface_schema: '1.0'",
        f"title: {_yaml_escape(md_meta['title'])}",
        f"source_type: {_yaml_escape(md_meta['source_type'])}",
        f"publisher: {_yaml_escape(md_meta['publisher'])}",
        f"publishing_date: {_yaml_escape(md_meta['publishing_date'])}",
        f"authors: {authors_yaml}",
        f"available_at: {_yaml_escape(md_meta.get('available_at', 'Unknown'))}",
        f"availability_status: {_yaml_escape(md_meta.get('availability_status', 'unknown'))}",
        f"availability_http_code: {_yaml_escape(md_meta.get('availability_http_code', ''))}",
        f"availability_checked_at: {_yaml_escape(md_meta.get('availability_checked_at', ''))}",
        f"availability_note: {_yaml_escape(md_meta.get('availability_note', ''))}",
        f"source_integrity_flag: {_yaml_escape(md_meta.get('source_integrity_flag', 'unverified'))}",
        f"credibility_tier_value: {_yaml_escape(md_meta.get('credibility_tier_value', 0))}",
        f"credibility_tier_key: {_yaml_escape(md_meta.get('credibility_tier_key', 'unclassified'))}",
        f"credibility_tier_label: {_yaml_escape(md_meta.get('credibility_tier_label', 'Unclassified'))}",
        f"credibility: {_yaml_escape(md_meta.get('credibility', 'Unclassified Report'))}",
        f"journal_ranking_source: {_yaml_escape(md_meta.get('journal_ranking_source', 'scimagojr_2024'))}",
        f"journal_sourceid: {_yaml_escape(md_meta.get('journal_sourceid', ''))}",
        f"journal_title: {_yaml_escape(md_meta.get('journal_title', ''))}",
        f"journal_issn: {_yaml_escape(md_meta.get('journal_issn', ''))}",
        f"journal_sjr: {_yaml_escape(md_meta.get('journal_sjr', 0.0))}",
        f"journal_quartile: {_yaml_escape(md_meta.get('journal_quartile', ''))}",
        f"journal_rank_global: {_yaml_escape(md_meta.get('journal_rank_global', 0))}",
        f"journal_categories: {_yaml_escape(md_meta.get('journal_categories', ''))}",
        f"journal_areas: {_yaml_escape(md_meta.get('journal_areas', ''))}",
        f"journal_high_ranked: {_yaml_escape(md_meta.get('journal_high_ranked', False))}",
        f"journal_match_method: {_yaml_escape(md_meta.get('journal_match_method', 'none'))}",
        f"journal_match_confidence: {_yaml_escape(md_meta.get('journal_match_confidence', 0.0))}",
        f"keywords: {keywords_yaml}",
        f"abstract: {_yaml_escape(md_meta['abstract'])}",
        "---",
        "",
    ]
    return "\n".join(lines)


def _add_document_preface(file_path: str, md_content: str) -> str:
    # Shared engine routine used by both Textifier UI and URL-ingestor PDF->MD flow.
    return add_document_preface(file_path, md_content)


# ======================================================================
# Textifier tab
# ======================================================================

def _render_textifier_tab():
    """Render the Textifier tool UI."""
    st.markdown("Convert PDF, DOCX, PPTX, or image files (PNG/JPG) to rich Markdown with optional AI image descriptions.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Input")

        use_vision = st.toggle("Use Vision Model for images", value=True, key="txt_vision")
        pdf_mode_label = st.selectbox(
            "PDF processing mode",
            options=[
                "Hybrid (Recommended): Docling first, Qwen enhancement, fallback on timeout",
                "Docling only (best layout/tables)",
                "Qwen 30B cleanup (LLM-first, no Docling)",
            ],
            index=0,
            key="txt_pdf_mode",
            help="Hybrid runs Docling first, then enriches with vision/LLM, and falls back if timeout limits are hit.",
        )
        cleanup_provider = st.selectbox(
            "Cleanup LLM provider",
            options=["lmstudio", "ollama"],
            index=0,
            key="txt_cleanup_provider",
        )
        cleanup_model = st.text_input(
            "Cleanup model",
            value="qwen2.5:32b",
            key="txt_cleanup_model",
            help="Model name for markdown cleanup when using Qwen/Hybrid modes.",
        )
        st.caption("Timeout safeguards")
        docling_timeout_seconds = st.number_input(
            "Docling timeout (seconds)",
            min_value=30,
            max_value=1200,
            value=240,
            step=10,
            key="txt_docling_timeout_s",
            help="If Docling exceeds this time, Textifier falls back to legacy extraction.",
        )
        image_timeout_seconds = st.number_input(
            "Per-image vision timeout (seconds)",
            min_value=3,
            max_value=180,
            value=20,
            step=1,
            key="txt_image_timeout_s",
            help="Maximum time allowed for each image description.",
        )
        image_budget_seconds = st.number_input(
            "Total image-description budget (seconds)",
            min_value=10,
            max_value=1800,
            value=120,
            step=10,
            key="txt_image_budget_s",
            help="When budget is exceeded, remaining images are skipped with placeholders.",
        )
        batch_mode = st.toggle("Batch mode (multi-file)", value=False, key="txt_batch_toggle")
        st.session_state["textifier_batch"] = batch_mode
        batch_cooldown_seconds = st.slider(
            "Cooldown between files (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="txt_batch_cooldown_s",
            help="Adds a short pause between files to reduce system load during batch processing.",
        )

        selected = _file_input_widget("textifier", ["pdf", "docx", "pptx", "png", "jpg", "jpeg"])

        # Clear all uploaded files button
        if st.button("Clear All Files", key="txt_clear_all", use_container_width=True):
            # Bump upload widget version so Streamlit creates a fresh uploader
            ver = st.session_state.get("textifier_upload_version", 0)
            # Clear all textifier-related state except the version we're about to set
            for key in list(st.session_state.keys()):
                if key.startswith("textifier_"):
                    del st.session_state[key]
            if "textifier_results" in st.session_state:
                del st.session_state["textifier_results"]
            st.session_state["textifier_upload_version"] = ver + 1
            # Clean temp files
            temp_dir = Path(tempfile.gettempdir()) / "cortex_textifier"
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            st.rerun()

    with col2:
        st.header("Output")

        if selected:
            files_to_process = selected if isinstance(selected, list) else [selected]
            total_files = len(files_to_process)

            if st.button("Convert to Markdown", type="primary", use_container_width=True):
                from cortex_engine.textifier import DocumentTextifier

                results = {}
                progress = st.progress(0.0, "Starting conversion...")
                status_text = st.empty()

                for file_idx, fpath in enumerate(files_to_process):
                    fname = _user_visible_stem(fpath)
                    file_base = file_idx / total_files
                    file_span = 1.0 / total_files

                    def _on_progress(frac, msg, _base=file_base, _span=file_span, _name=_user_visible_filename(fpath)):
                        overall = min(_base + frac * _span, 1.0)
                        label = f"[{_name}] {msg}" if total_files > 1 else msg
                        progress.progress(overall, label)

                    if total_files > 1:
                        status_text.info(f"File {file_idx + 1}/{total_files}: {_user_visible_filename(fpath)}")

                    mode_map = {
                        "Hybrid (Recommended): Docling first, Qwen enhancement, fallback on timeout": "hybrid",
                        "Docling only (best layout/tables)": "docling",
                        "Qwen 30B cleanup (LLM-first, no Docling)": "qwen30b",
                    }
                    textifier_options = {
                        "use_vision": use_vision,
                        "pdf_strategy": mode_map.get(pdf_mode_label, "hybrid"),
                        "cleanup_provider": cleanup_provider,
                        "cleanup_model": cleanup_model,
                        "docling_timeout_seconds": float(docling_timeout_seconds),
                        "image_description_timeout_seconds": float(image_timeout_seconds),
                        "image_enrich_max_seconds": float(image_budget_seconds),
                    }
                    textifier = DocumentTextifier.from_options(textifier_options, on_progress=_on_progress)
                    try:
                        md_content = textifier.textify_file(fpath)
                    except Exception as e:
                        st.error(f"Failed to convert {_user_visible_filename(fpath)}: {e}")
                        logger.error(f"Textifier conversion error for {fpath}: {e}", exc_info=True)
                        continue

                    # Preface enrichment is best-effort. Never block markdown download on metadata issues.
                    try:
                        _on_progress(1.0, "Generating metadata preface...")
                        md_content = _add_document_preface(fpath, md_content)
                    except Exception as e:
                        st.warning(
                            f"Converted {_user_visible_filename(fpath)} but metadata preface failed; "
                            "download includes markdown without preface."
                        )
                        logger.warning(f"Textifier preface enrichment failed for {fpath}: {e}", exc_info=True)

                    results[f"{fname}_textified.md"] = md_content

                    if batch_cooldown_seconds > 0 and total_files > 1 and file_idx < total_files - 1:
                        progress.progress(
                            min((file_idx + 1) / total_files, 1.0),
                            f"Cooling down for {batch_cooldown_seconds:.1f}s before next file..."
                        )
                        time.sleep(batch_cooldown_seconds)

                progress.progress(1.0, "Done!")
                status_text.empty()

                if results:
                    st.session_state["textifier_results"] = results

        # Display results
        results = st.session_state.get("textifier_results")
        if results:
            st.divider()
            st.subheader("Results")

            if len(results) == 1:
                name, content = next(iter(results.items()))
                st.download_button("Download Markdown", content, file_name=name,
                                   mime="text/markdown", use_container_width=True)
                with st.expander("Preview", expanded=True):
                    st.markdown(content[:5000] + ("\n\n*... truncated ...*" if len(content) > 5000 else ""))
            else:
                # Zip download for batch
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, content in results.items():
                        zf.writestr(name, content)
                buf.seek(0)
                st.download_button("Download All (ZIP)", buf.getvalue(),
                                   file_name="textified_documents.zip",
                                   mime="application/zip", use_container_width=True)
                for name, content in results.items():
                    with st.expander(name):
                        st.markdown(content[:3000] + ("\n\n*... truncated ...*" if len(content) > 3000 else ""))
        elif selected:
            st.info("Click **Convert to Markdown** to process your document")
        else:
            st.info("Select a document from the left panel to get started")


# ======================================================================
# Anonymizer tab (original logic preserved)
# ======================================================================

def _render_anonymizer_tab():
    """Render the Anonymizer tool UI (original Document Anonymizer logic)."""
    st.markdown("Replace identifying information with generic placeholders for privacy protection.")

    if "anonymizer_results" not in st.session_state:
        st.session_state.anonymizer_results = {}
    if "current_anonymization" not in st.session_state:
        st.session_state.current_anonymization = None

    with st.expander("About Document Anonymizer", expanded=False):
        st.markdown("""
        **Protect sensitive information** by replacing identifying details with generic placeholders.

        **Features:**
        - **Smart Entity Detection**: Automatically finds people, companies, and locations
        - **Consistent Replacement**: Same entity always gets the same placeholder
        - **Multiple Formats**: PDF, Word, and text file support

        **Replacement Examples:**
        - People: John Smith -> Person A
        - Companies: Acme Corp -> Company 1
        - Contact Info: emails -> [EMAIL], phones -> [PHONE]
        """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Document Input")
        batch_mode = st.toggle("Batch mode", value=False, key="anonymizer_batch_toggle")
        st.session_state["anonymizer_batch"] = batch_mode
        selected_file = _file_input_widget("anonymizer", ["pdf", "docx", "txt"])

        has_files = selected_file is not None
        file_list = selected_file if isinstance(selected_file, list) else ([selected_file] if selected_file else [])

        if has_files:
            st.divider()
            st.subheader("Anonymization Settings")
            confidence_threshold = st.slider(
                "Entity Detection Confidence:",
                min_value=0.1, max_value=0.9, value=0.3, step=0.1,
                help="Lower values detect more entities (may include false positives)",
            )
            st.session_state.confidence_threshold = confidence_threshold
            st.caption("Granular redaction controls")
            redact_people = st.checkbox("Redact people", value=True, key="anon_opt_people")
            redact_organizations = st.checkbox("Redact organizations", value=True, key="anon_opt_orgs")
            redact_projects = st.checkbox("Redact projects", value=True, key="anon_opt_projects")
            redact_locations = st.checkbox("Redact locations", value=True, key="anon_opt_locations")
            redact_personal_pronouns = st.checkbox(
                "Redact personal pronouns (he/she/they/etc.)",
                value=False,
                key="anon_opt_pronouns",
            )
            redact_company_names = st.checkbox(
                "Redact custom company names",
                value=False,
                key="anon_opt_company_names",
            )
            preserve_source_formatting = st.checkbox(
                "Preserve source formatting",
                value=True,
                key="anon_opt_preserve_formatting",
                help="Remove <CR>/<CRLF> markers and reflow hard-wrapped lines while keeping lists/headings/table-like lines.",
            )
            custom_company_names_text = st.text_area(
                "Custom company names (comma-separated)",
                value="",
                key="anon_opt_company_names_text",
                help="Optional deterministic masking list for known organization names.",
                disabled=not redact_company_names,
            )
            anonymization_options = AnonymizationOptions(
                redact_people=redact_people,
                redact_organizations=redact_organizations,
                redact_projects=redact_projects,
                redact_locations=redact_locations,
                redact_personal_pronouns=redact_personal_pronouns,
                redact_company_names=redact_company_names,
                custom_company_names=[
                    name.strip() for name in custom_company_names_text.split(",") if name.strip()
                ],
                preserve_source_formatting=preserve_source_formatting,
            )
            st.session_state.anonymization_options = anonymization_options

    with col2:
        st.header("Anonymization Process")

        if has_files:
            if len(file_list) == 1:
                st.markdown(f"**File:** `{Path(file_list[0]).name}`")
            else:
                st.info(f"{len(file_list)} document(s) selected")

            if st.button("Start Anonymization", type="primary", use_container_width=True):
                progress_bar = st.progress(0, "Initializing anonymization...")

                # Shared mapping across batch so entities stay consistent
                anonymizer = DocumentAnonymizer()
                mapping = AnonymizationMapping()
                batch_results = []

                for idx, fpath in enumerate(file_list):
                    fname = Path(fpath).name
                    base_pct = idx / len(file_list)
                    try:
                        progress_bar.progress(
                            min(base_pct + 0.02, 1.0),
                            f"[{idx+1}/{len(file_list)}] Reading {fname}..."
                        )

                        result_path, result_mapping = anonymizer.anonymize_single_file(
                            input_path=fpath,
                            output_path=None,
                            mapping=mapping,
                            confidence_threshold=st.session_state.confidence_threshold,
                            options=st.session_state.get("anonymization_options"),
                        )
                        # Re-use the returned mapping for next file (consistent entities)
                        mapping = result_mapping

                        batch_results.append({
                            "original_file": fpath,
                            "anonymized_file": result_path,
                            "mapping": result_mapping,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })

                    except Exception as e:
                        st.error(f"**Failed:** {fname}: {e}")
                        logger.error(f"Anonymization error for {fpath}: {e}", exc_info=True)

                progress_bar.progress(1.0, "Anonymization complete!")

                if batch_results:
                    # For single file, keep backward compat
                    st.session_state.current_anonymization = batch_results[0] if len(batch_results) == 1 else None
                    st.session_state.anonymization_batch_results = batch_results

                    # Summary metrics (use the final mapping which has all entities)
                    final_mapping = batch_results[-1]["mapping"]
                    st.success(f"**Anonymized {len(batch_results)} document(s) successfully!**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("People", len([k for k, v in final_mapping.mappings.items() if v.startswith("Person")]))
                    with col_b:
                        st.metric("Companies", len([k for k, v in final_mapping.mappings.items() if v.startswith("Company")]))
                    with col_c:
                        st.metric("Projects", len([k for k, v in final_mapping.mappings.items() if v.startswith("Project")]))

        # --- Display results ---
        batch_results = st.session_state.get("anonymization_batch_results")
        single_result = st.session_state.get("current_anonymization")

        if batch_results and len(batch_results) > 1:
            # Batch results
            st.divider()
            st.subheader("Anonymization Results")

            # Zip download for all anonymized files
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in batch_results:
                    zf.write(r["anonymized_file"], Path(r["anonymized_file"]).name)
            buf.seek(0)
            st.download_button(
                f"Download All {len(batch_results)} Anonymized Documents",
                buf.getvalue(),
                file_name="anonymized_documents.zip",
                mime="application/zip",
                use_container_width=True,
            )

            # Mapping report (shared across batch)
            mapping_content = _generate_mapping_report(batch_results[-1]["mapping"])
            st.download_button(
                label="Download Mapping Reference",
                data=mapping_content,
                file_name=f"anonymization_mapping_{int(time.time())}.txt",
                mime="text/plain",
                help="Reference file showing original -> anonymized mappings (keep secure!)",
            )

            # Per-file expanders
            for r in batch_results:
                orig_name = Path(r["original_file"]).name
                anon_name = Path(r["anonymized_file"]).name
                with st.expander(f"{orig_name} -> {anon_name}", expanded=False):
                    try:
                        with open(r["anonymized_file"], "r", encoding="utf-8") as f:
                            content = f.read()
                        preview = content[:2000]
                        if len(content) > 2000:
                            preview += "\n\n... [Content truncated for preview] ..."
                        st.text_area("Preview:", preview, height=200, key=f"anon_preview_{orig_name}")
                        st.download_button(
                            f"Download {anon_name}",
                            content,
                            file_name=anon_name,
                            mime="text/plain",
                            key=f"anon_dl_{orig_name}",
                        )
                    except Exception as e:
                        st.error(f"Could not load: {e}")

            # Entity mappings table
            final_mapping = batch_results[-1]["mapping"]
            if final_mapping.mappings:
                with st.expander("Entity Mappings", expanded=False):
                    import pandas as pd
                    rows = [{"Original": orig, "Anonymized": anon, "Type": _get_entity_type(anon)}
                            for orig, anon in final_mapping.mappings.items()]
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        elif single_result or (batch_results and len(batch_results) == 1):
            result = single_result or batch_results[0]
            st.divider()
            st.subheader("Anonymization Results")

            col_orig, col_anon = st.columns(2)
            with col_orig:
                st.markdown("**Original File:**")
                st.code(Path(result["original_file"]).name)
            with col_anon:
                st.markdown("**Anonymized File:**")
                st.code(Path(result["anonymized_file"]).name)

            st.markdown("### Download Results")
            try:
                with open(result["anonymized_file"], "r", encoding="utf-8") as f:
                    anonymized_content = f.read()

                st.download_button(
                    label="Download Anonymized Document",
                    data=anonymized_content,
                    file_name=Path(result["anonymized_file"]).name,
                    mime="text/plain",
                    use_container_width=True,
                )

                mapping_content = _generate_mapping_report(result["mapping"])
                st.download_button(
                    label="Download Mapping Reference",
                    data=mapping_content,
                    file_name=f"anonymization_mapping_{int(time.time())}.txt",
                    mime="text/plain",
                    help="Reference file showing original -> anonymized mappings (keep secure!)",
                )

                with st.expander("Preview Anonymized Content", expanded=False):
                    preview = anonymized_content[:2000]
                    if len(anonymized_content) > 2000:
                        preview += "\n\n... [Content truncated for preview] ..."
                    st.text_area("Preview:", preview, height=300)

                if result["mapping"].mappings:
                    with st.expander("Entity Mappings", expanded=False):
                        import pandas as pd
                        rows = []
                        for original, anon in result["mapping"].mappings.items():
                            rows.append({"Original": original, "Anonymized": anon,
                                         "Type": _get_entity_type(anon)})
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Could not load results: {str(e)}")

        elif has_files:
            st.info("Click **Start Anonymization** to process your document(s)")
        else:
            st.info("Select a document from the left panel to get started")


def _generate_mapping_report(mapping: AnonymizationMapping) -> str:
    """Generate a formatted mapping report."""
    report = [
        "ANONYMIZATION MAPPING REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "WARNING: KEEP THIS FILE SECURE AND SEPARATE FROM ANONYMIZED DOCUMENTS",
        "",
    ]
    if not mapping.mappings:
        report.append("No entity mappings found.")
        return "\n".join(report)

    groups = {"Person": [], "Company": [], "Project": []}
    other = []
    for original, anon in mapping.mappings.items():
        placed = False
        for prefix in groups:
            if anon.startswith(prefix):
                groups[prefix].append((original, anon))
                placed = True
                break
        if not placed:
            other.append((original, anon))

    labels = {"Person": "PEOPLE", "Company": "COMPANIES", "Project": "PROJECTS"}
    for prefix, items in groups.items():
        if items:
            report.append(f"{labels[prefix]}:")
            for orig, anon in sorted(items):
                report.append(f"  {orig} -> {anon}")
            report.append("")
    if other:
        report.append("OTHER:")
        for orig, anon in sorted(other):
            report.append(f"  {orig} -> {anon}")

    return "\n".join(report)


def _get_entity_type(anonymized: str) -> str:
    for prefix, label in [("Person", "Person"), ("Company", "Company"),
                          ("Project", "Project"), ("Location", "Location")]:
        if anonymized.startswith(prefix):
            return label
    return "Other"


# ======================================================================
# PDF Image Extractor tab
# ======================================================================

def _render_pdf_image_extract_tab():
    """Render the PDF image extraction utility."""
    st.subheader("PDF Image Extractor")
    with st.expander("About PDF Image Extractor", expanded=False):
        st.write(
            "Extracts likely photographic image blocks from PDFs into JPG files. "
            "This is useful for scanned newsletters, reports, and old flatbed PDF scans where "
            "photos need to be pulled out for separate processing."
        )
        st.caption(
            "Filtering is heuristic-based: small graphics, header/footer art, and icons can be "
            "excluded using the size and edge controls below."
        )

    col1, col2 = st.columns([1, 1.2])

    with col1:
        batch_mode = st.toggle("Batch mode (multi-file)", value=False, key="pdfimg_batch_toggle")
        st.session_state["pdfimg_batch"] = batch_mode

        min_width_px = st.slider(
            "Minimum source image width (px)",
            min_value=100,
            max_value=3000,
            value=500,
            step=50,
            key="pdfimg_min_width_px",
            help="Ignore small embedded graphics narrower than this.",
        )
        min_height_px = st.slider(
            "Minimum source image height (px)",
            min_value=100,
            max_value=3000,
            value=500,
            step=50,
            key="pdfimg_min_height_px",
            help="Ignore small embedded graphics shorter than this.",
        )
        min_page_coverage_pct = st.slider(
            "Minimum on-page coverage (%)",
            min_value=0.0,
            max_value=40.0,
            value=2.0,
            step=0.5,
            key="pdfimg_min_coverage_pct",
            help="Filters out very small items even if their source image dimensions are large.",
        )
        ignore_edge_decorations = st.checkbox(
            "Ignore likely header/footer and edge decorations",
            value=True,
            key="pdfimg_ignore_edge",
            help="Skips small banner/icon-like images close to the page edges.",
        )
        edge_margin_pct = st.slider(
            "Edge margin exclusion band (%)",
            min_value=2.0,
            max_value=20.0,
            value=8.0,
            step=1.0,
            key="pdfimg_edge_margin_pct",
            disabled=not ignore_edge_decorations,
            help="The top, bottom, left, and right edge band used for decoration filtering.",
        )
        render_scale = st.slider(
            "Output render scale",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.25,
            key="pdfimg_render_scale",
            help="Higher values create larger JPG extractions but use more memory.",
        )
        split_full_page_scans = st.checkbox(
            "Split full-page scans into separate photo crops",
            value=False,
            key="pdfimg_split_scans",
            help="When a PDF page is one large scanned image, try to detect multiple photo regions inside it.",
        )
        split_min_crop_coverage_pct = st.slider(
            "Minimum split crop area (%)",
            min_value=1.0,
            max_value=25.0,
            value=6.0,
            step=0.5,
            key="pdfimg_split_min_crop_pct",
            disabled=not split_full_page_scans,
            help="Smaller values detect more candidate crops but can pick up layout noise.",
        )

        selected = _file_input_widget("pdfimg", ["pdf"], label="Choose PDF scan:")

        if st.button("Clear Extractor Files", key="pdfimg_clear_all", use_container_width=True):
            ver = st.session_state.get("pdfimg_upload_version", 0)
            for key in list(st.session_state.keys()):
                if key.startswith("pdfimg_"):
                    del st.session_state[key]
            st.session_state["pdfimg_upload_version"] = ver + 1
            st.rerun()

    with col2:
        st.header("Output")

        if selected:
            files_to_process = selected if isinstance(selected, list) else [selected]
            st.info(f"{len(files_to_process)} PDF file(s) ready for image extraction.")

            if st.button("Extract Images to ZIP", type="primary", use_container_width=True):
                from cortex_engine.textifier import DocumentTextifier

                extractor = DocumentTextifier(use_vision=False)
                all_images: List[Dict[str, object]] = []
                extraction_rows: List[Dict[str, object]] = []
                progress = st.progress(0.0, "Starting image scan...")

                for file_idx, fpath in enumerate(files_to_process):
                    visible_name = _user_visible_filename(fpath)
                    file_base = file_idx / len(files_to_process)
                    file_span = 1.0 / len(files_to_process)

                    def _on_progress(frac, msg, _base=file_base, _span=file_span, _name=visible_name):
                        overall = min(_base + frac * _span, 1.0)
                        label = f"[{_name}] {msg}" if len(files_to_process) > 1 else msg
                        progress.progress(overall, label)

                    extractor.on_progress = _on_progress
                    try:
                        result = extractor.extract_pdf_images(
                            fpath,
                            min_width_px=int(min_width_px),
                            min_height_px=int(min_height_px),
                            min_page_coverage_pct=float(min_page_coverage_pct),
                            ignore_edge_decorations=bool(ignore_edge_decorations),
                            edge_margin_pct=float(edge_margin_pct),
                            render_scale=float(render_scale),
                            split_full_page_scans=bool(split_full_page_scans),
                            split_min_crop_coverage_pct=float(split_min_crop_coverage_pct),
                        )
                    except Exception as e:
                        st.error(f"Failed to scan {_user_visible_filename(fpath)}: {e}")
                        logger.error(f"PDF image extraction failed for {fpath}: {e}", exc_info=True)
                        continue

                    extraction_rows.append(
                        {
                            "pdf": visible_name,
                            "pages": int(result.get("pages_scanned", 0) or 0),
                            "detected": int(result.get("detected_blocks", 0) or 0),
                            "extracted": len(result.get("images", []) or []),
                            "skipped_small": int(result.get("skipped_small", 0) or 0),
                            "skipped_edge": int(result.get("skipped_edge", 0) or 0),
                            "skipped_invalid": int(result.get("skipped_invalid", 0) or 0),
                            "split_generated": int(result.get("split_generated", 0) or 0),
                            "message": str(result.get("message", "") or ""),
                        }
                    )

                    for image in result.get("images", []) or []:
                        image_record = dict(image)
                        image_record["zip_path"] = f"{Path(visible_name).stem}/{image_record['file_name']}"
                        all_images.append(image_record)

                progress.progress(1.0, "Done!")

                st.session_state["pdfimg_results"] = extraction_rows
                if all_images:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        summary_lines = [
                            "Cortex PDF Image Extractor",
                            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            "",
                        ]
                        for row in extraction_rows:
                            summary_lines.append(
                                f"{row['pdf']}: extracted {row['extracted']} of {row['detected']} detected "
                                f"image block(s); skipped small={row['skipped_small']}, "
                                f"edge={row['skipped_edge']}, invalid={row['skipped_invalid']}."
                            )
                        summary_lines.append("")
                        summary_lines.append("Extracted files:")
                        for image in all_images:
                            zf.writestr(str(image["zip_path"]), image["bytes"])
                            summary_lines.append(
                                f"- {image['zip_path']} | page {image['page']} | "
                                f"source {image['intrinsic_width']}x{image['intrinsic_height']} px | "
                                f"coverage {image['coverage_pct']}% | bbox {image['bbox']}"
                            )
                        zf.writestr("extraction_summary.txt", "\n".join(summary_lines) + "\n")

                    zip_name = "pdf_extracted_images.zip" if len(files_to_process) > 1 else (
                        f"{Path(_user_visible_filename(files_to_process[0])).stem}_images.zip"
                    )
                    st.session_state["pdfimg_zip_bytes"] = buf.getvalue()
                    st.session_state["pdfimg_zip_name"] = zip_name
                else:
                    st.session_state.pop("pdfimg_zip_bytes", None)
                    st.session_state.pop("pdfimg_zip_name", None)

        rows = st.session_state.get("pdfimg_results")
        zip_bytes = st.session_state.get("pdfimg_zip_bytes")
        zip_name = st.session_state.get("pdfimg_zip_name", "pdf_extracted_images.zip")

        if rows:
            st.divider()
            st.subheader("Extraction Results")

            total_detected = sum(int(row.get("detected", 0) or 0) for row in rows)
            total_extracted = sum(int(row.get("extracted", 0) or 0) for row in rows)
            total_split = sum(int(row.get("split_generated", 0) or 0) for row in rows)
            metric_cols = st.columns(4)
            metric_cols[0].metric("PDFs Scanned", len(rows))
            metric_cols[1].metric("Image Blocks Found", total_detected)
            metric_cols[2].metric("JPGs Extracted", total_extracted)
            metric_cols[3].metric("Split Crops", total_split)

            for row in rows:
                st.write(
                    f"**{row['pdf']}**: extracted {row['extracted']} / {row['detected']} "
                    f"(small skipped: {row['skipped_small']}, edge skipped: {row['skipped_edge']}, "
                    f"invalid skipped: {row['skipped_invalid']}, split crops: {row['split_generated']})"
                )

            if zip_bytes:
                st.download_button(
                    "Download Extracted JPGs (ZIP)",
                    zip_bytes,
                    file_name=zip_name,
                    mime="application/zip",
                    use_container_width=True,
                )
            else:
                st.warning(
                    "No images met the current thresholds. Lower the size or coverage filters "
                    "if this PDF contains smaller photos."
                )


# ======================================================================
# Photo Processor tab
# ======================================================================

def _render_photo_keywords_tab():
    """Render the Photo Processor tool for batch resize and photo metadata workflows."""
    st.markdown(
        "Process photos in batch: resize for gallery use, generate AI keywords, "
        "clean sensitive tags, and write EXIF/XMP ownership metadata."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Input")

        st.session_state["photokw_batch"] = True  # always batch-capable

        # Upload version counter for clear button
        if "photokw_upload_version" not in st.session_state:
            st.session_state["photokw_upload_version"] = 0
        ver = st.session_state["photokw_upload_version"]

        uploaded_input = st.file_uploader(
            "Drop photos here:",
            type=["png", "jpg", "jpeg", "tiff", "webp", "gif", "bmp"],
            accept_multiple_files=True,
            key=f"photokw_upload_v{ver}",
        )
        upload_cache_key = "photokw_uploaded_cache"
        if uploaded_input:
            uploaded = [_SessionUpload(uf.name, uf.getvalue()) for uf in uploaded_input]
            st.session_state[upload_cache_key] = [
                {"name": uf.name, "data": uf.getvalue()}
                for uf in uploaded_input
            ]
        else:
            cached_uploads = st.session_state.get(upload_cache_key) or []
            uploaded = [
                _SessionUpload(str(item.get("name") or ""), item.get("data") or b"")
                for item in cached_uploads
                if isinstance(item, dict) and item.get("name")
            ]

        write_to_original = st.toggle(
            "Write to original files",
            value=False,
            key="photokw_write_original",
            help="When OFF, keywords are written to copies in a temp folder (originals untouched). "
                 "When ON, keywords are written directly to the uploaded files.",
        )

        city_radius = st.slider(
            "City location radius",
            min_value=1, max_value=50, value=5, step=1,
            key="photokw_city_radius",
            help="Radius (km) for city-level reverse geocoding of GPS coordinates. "
                 "Larger values may match broader city names for rural locations.",
        )

        clear_keywords = st.checkbox(
            "Clear existing keywords/tags first",
            value=False,
            key="photokw_clear_keywords",
            help="Remove all existing XMP Subject and IPTC Keywords before writing new ones.",
        )
        clear_location = st.checkbox(
            "Clear existing location fields first",
            value=False,
            key="photokw_clear_location",
            help="Remove existing Country, State, and City EXIF fields and rebuild from GPS/location hints.",
        )
        generate_description = st.checkbox(
            "Generate AI description + keywords",
            value=True,
            key="photokw_generate_description",
            help="Writes a fresh description and any new keywords. Turn this off to update only location/GPS metadata.",
        )
        populate_location = st.checkbox(
            "Fill location and GPS metadata",
            value=True,
            key="photokw_populate_location",
            help="Completes City/State/Country from GPS, or derives GPS from City/Country hints when GPS is missing.",
        )
        fallback_city = st.text_input(
            "Fallback city (optional)",
            value="",
            key="photokw_fallback_city",
            disabled=not populate_location,
            help="Used only when a photo has no GPS and no embedded location fields.",
        )
        fallback_country = st.text_input(
            "Fallback country (optional)",
            value="",
            key="photokw_fallback_country",
            disabled=not populate_location,
            help="If only a country is provided, Cortex will try that country's capital city.",
        )
        resize_profile = st.selectbox(
            "Resize profile",
            options=["Keep original dimensions", "Low (1920 x 1080)", "Medium (2560 x 1440)"],
            index=0,
            key="photokw_resize_profile",
            help="Maximum output dimensions. Photos already below the selected profile are not resized.",
        )
        no_resize_selected = resize_profile == "Keep original dimensions"
        convert_to_jpg = st.checkbox(
            "Convert resized output to JPG",
            value=False,
            key="photokw_convert_to_jpg",
            disabled=no_resize_selected,
            help="Only non-JPG sources are converted. Existing JPG files stay JPG and are only resized when needed.",
        )
        jpg_quality = st.slider(
            "JPG quality",
            min_value=60,
            max_value=100,
            value=90,
            step=1,
            key="photokw_jpg_quality",
            disabled=(not convert_to_jpg) or no_resize_selected,
            help="Only applies when JPG conversion is enabled.",
        )

        anonymize_keywords = st.checkbox(
            "Anonymize sensitive keywords",
            value=False,
            key="photokw_anonymize_keywords",
            help="Remove personal/sensitive tags from generated keywords using the blocked list below.",
        )
        blocked_keywords_text = st.text_input(
            "Blocked keywords (comma-separated)",
            value="friends,family,paul,paul_c,jacqui",
            key="photokw_blocked_keywords",
            help="These keywords are removed when anonymization is enabled.",
        )

        apply_ownership = st.checkbox(
            "Insert ownership info",
            value=True,
            key="photokw_apply_ownership",
            help="Write ownership/copyright metadata fields in EXIF/IPTC/XMP.",
        )
        ownership_notice = st.text_area(
            "Ownership notice",
            value="All rights (c) Longboardfella. Contact longboardfella.com for info on use of photos.",
            key="photokw_ownership_notice",
            height=90,
        )
        photokw_batch_cooldown_seconds = st.slider(
            "Cooldown between photos (seconds)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="photokw_batch_cooldown_s",
            help="Adds a short pause between photos to keep batch processing more responsive.",
        )

        if st.button("Clear All Photos", key="photokw_clear", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("photokw_") and key != "photokw_upload_version":
                    del st.session_state[key]
            st.session_state["photokw_upload_version"] = ver + 1
            if "photokw_results" in st.session_state:
                del st.session_state["photokw_results"]
            temp_dir = Path(tempfile.gettempdir()) / "cortex_photokw"
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            st.rerun()

    with col2:
        st.header("Results")

        if uploaded:
            accepted_uploads = []
            accepted_bytes = 0
            for uf in uploaded:
                size_bytes = int(getattr(uf, "size", 0) or 0)
                if size_bytes <= 0:
                    try:
                        size_bytes = len(uf.getvalue())
                    except Exception:
                        size_bytes = 0
                if accepted_bytes + size_bytes > MAX_BATCH_UPLOAD_BYTES:
                    break
                accepted_uploads.append(uf)
                accepted_bytes += size_bytes
            if not accepted_uploads:
                st.error("Selected photos exceed the 1GB total upload limit.")
                return
            if len(accepted_uploads) < len(uploaded):
                st.warning(
                    f"Maximum 1GB total upload per photo batch — only the first "
                    f"{len(accepted_uploads)} of {len(uploaded)} photos will be processed."
                )
            uploaded = accepted_uploads
            st.session_state[upload_cache_key] = [
                {"name": uf.name, "data": uf.getvalue()}
                for uf in uploaded
            ]
            total = len(uploaded)
            st.info(f"{total} photo(s) selected ({accepted_bytes / (1024 * 1024):.1f} MB total)")

            # Single-photo metadata preview for quick testing before processing.
            if total == 1:
                preview_photo = uploaded[0]
                preview_bytes = preview_photo.getvalue()
                preview_temp_dir = Path(tempfile.gettempdir()) / "cortex_photokw_preview"
                preview_temp_dir.mkdir(exist_ok=True, mode=0o755)

                # Keep a stable working copy path for this uploaded file across reruns,
                # so quick metadata edits are not lost.
                import hashlib
                preview_sig = f"{preview_photo.name}:{len(preview_bytes)}:{hashlib.md5(preview_bytes).hexdigest()}"
                existing_sig = st.session_state.get("photokw_single_upload_sig")
                existing_path = st.session_state.get("photokw_single_working_path")
                if preview_sig != existing_sig or not existing_path or not Path(existing_path).exists():
                    preview_path = preview_temp_dir / preview_photo.name
                    with open(preview_path, "wb") as f:
                        f.write(preview_bytes)
                    os.chmod(str(preview_path), 0o644)
                    st.session_state["photokw_single_upload_sig"] = preview_sig
                    st.session_state["photokw_single_working_path"] = str(preview_path)
                    st.session_state.pop("photokw_vlm_probe", None)
                else:
                    preview_path = Path(existing_path)

                probe_col, refresh_col = st.columns(2)
                run_probe = probe_col.button("Run VLM Diagnostic Probe", key="photokw_vlm_probe_btn", use_container_width=True)
                if refresh_col.button("Refresh Preview", key="photokw_preview_refresh", use_container_width=True):
                    st.rerun()

                if run_probe:
                    from cortex_engine.textifier import DocumentTextifier

                    probe_result = DocumentTextifier(use_vision=True).probe_image_vlm(preview_bytes, simple_prompt=True)
                    st.session_state["photokw_vlm_probe"] = probe_result

                preview_meta = _read_photo_metadata_preview(str(preview_path))
                probe_result = st.session_state.get("photokw_vlm_probe")

                with st.expander("Single Photo Metadata Preview", expanded=True):
                    st.image(preview_bytes, caption=preview_photo.name, width=420)
                    if probe_result:
                        st.markdown("**VLM Diagnostic Probe**")
                        st.json(probe_result)
                        st.caption(
                            "This shows the raw response shape from Ollama for the current photo using the simple prompt."
                        )
                        st.divider()
                    if preview_meta.get("available"):
                        description = preview_meta.get("description", "")
                        keywords = preview_meta.get("keywords", [])
                        city = preview_meta.get("city", "")
                        state = preview_meta.get("state", "")
                        country = preview_meta.get("country", "")
                        gps = preview_meta.get("gps")

                        if description:
                            st.markdown(f"**Description:** {description}")
                        else:
                            st.caption("No existing description found in metadata.")

                        if keywords:
                            st.markdown(f"**Keywords ({len(keywords)}):** {', '.join(keywords)}")
                        else:
                            st.caption("No existing keywords found in metadata.")

                        location_parts = [v for v in [city, state, country] if v]
                        if location_parts:
                            st.markdown(f"**Location fields:** {', '.join(location_parts)}")
                        else:
                            st.caption("No existing City/State/Country metadata found.")
                        if gps:
                            st.caption(f"GPS: {gps}")

                        st.divider()
                        st.markdown("**Quick Edit Metadata**")
                        edit_keywords = st.text_area(
                            "Edit keywords (comma-separated)",
                            value=", ".join(keywords),
                            key="photokw_edit_keywords",
                            height=90,
                        )
                        edit_description = st.text_area(
                            "Edit description",
                            value=description,
                            key="photokw_edit_description",
                            height=90,
                        )
                        ec1, ec2, ec3 = st.columns(3)
                        with ec1:
                            edit_city = st.text_input("City", value=city, key="photokw_edit_city")
                        with ec2:
                            edit_state = st.text_input("State", value=state, key="photokw_edit_state")
                        with ec3:
                            edit_country = st.text_input("Country", value=country, key="photokw_edit_country")

                        if st.button("Apply Quick Metadata Edits", key="photokw_apply_quick_edit", use_container_width=True):
                            edited_keywords = [k.strip() for k in edit_keywords.split(",") if k.strip()]
                            write_result = _write_photo_metadata_quick_edit(
                                str(preview_path),
                                keywords=edited_keywords,
                                description=edit_description,
                                city=edit_city,
                                state=edit_state,
                                country=edit_country,
                            )
                            if write_result.get("success"):
                                st.success("Metadata edits applied.")
                                st.rerun()
                            else:
                                st.error(f"Could not apply metadata edits: {write_result.get('message', 'Unknown error')}")
                    else:
                        st.info(f"Metadata preview unavailable: {preview_meta.get('reason', 'Unknown reason')}")

            resolution_map = {
                "Keep original dimensions": (None, None),
                "Low (1920 x 1080)": (1920, 1080),
                "Medium (2560 x 1440)": (2560, 1440),
            }
            max_width, max_height = resolution_map.get(resize_profile, (None, None))

            action_cols = st.columns(2)
            do_resize_only = action_cols[0].button("Resize Photos Only", use_container_width=True)
            do_keywords = action_cols[1].button("Process Selected Metadata", type="primary", use_container_width=True)

            if do_resize_only or do_keywords:
                if do_keywords and not any([generate_description, populate_location, apply_ownership]):
                    st.warning("Select at least one metadata action before processing.")
                    return

                from cortex_engine.textifier import DocumentTextifier

                # Save uploads to temp dir
                temp_dir = Path(tempfile.gettempdir()) / "cortex_photokw"
                temp_dir.mkdir(exist_ok=True, mode=0o755)
                file_paths = []
                if total == 1 and st.session_state.get("photokw_single_working_path"):
                    working_path = st.session_state.get("photokw_single_working_path")
                    if working_path and Path(working_path).exists():
                        dest = temp_dir / uploaded[0].name
                        shutil.copy2(working_path, dest)
                        os.chmod(str(dest), 0o644)
                        file_paths.append(str(dest))
                    else:
                        uf = uploaded[0]
                        dest = temp_dir / uf.name
                        with open(dest, "wb") as f:
                            f.write(uf.getvalue())
                        os.chmod(str(dest), 0o644)
                        file_paths.append(str(dest))
                else:
                    for uf in uploaded:
                        dest = temp_dir / uf.name
                        with open(dest, "wb") as f:
                            f.write(uf.getvalue())
                        os.chmod(str(dest), 0o644)
                        file_paths.append(str(dest))

                textifier = DocumentTextifier(use_vision=True)
                results = []
                mode = "resize_only" if do_resize_only else "keyword_metadata"
                progress = st.progress(0.0, "Starting resize..." if do_resize_only else "Starting metadata processing...")
                blocked_keywords = [k.strip().lower() for k in blocked_keywords_text.split(",") if k.strip()]

                for idx, fpath in enumerate(file_paths):
                    fname = Path(fpath).name

                    def _on_progress(frac, msg, _idx=idx, _total=total, _name=fname):
                        overall = min((_idx + frac) / _total, 1.0)
                        progress.progress(overall, f"[{_name}] {msg}")

                    textifier.on_progress = _on_progress
                    try:
                        if do_resize_only:
                            if max_width is None or max_height is None:
                                result = {
                                    "file_name": Path(fpath).name,
                                    "output_path": fpath,
                                    "resize_info": {
                                        "resized": False,
                                        "metadata_preserved": True,
                                        "skipped_resize": True,
                                    },
                                }
                            else:
                                result = textifier.resize_image_only(
                                    fpath,
                                    max_width=max_width,
                                    max_height=max_height,
                                    convert_to_jpg=convert_to_jpg,
                                    jpg_quality=jpg_quality,
                                )
                            output_path = str(result.get("output_path", fpath))
                            file_paths[idx] = output_path
                            if anonymize_keywords:
                                result["keyword_anonymize_result"] = textifier.anonymize_existing_photo_keywords(
                                    output_path, blocked_keywords=blocked_keywords
                                )
                            if apply_ownership and ownership_notice.strip():
                                result["ownership_result"] = textifier.write_ownership_metadata(
                                    output_path, ownership_notice.strip()
                                )
                        else:
                            result = textifier.keyword_image(
                                fpath, city_radius_km=city_radius,
                                clear_keywords=(clear_keywords if generate_description else False),
                                clear_location=(clear_location if populate_location else False),
                                generate_description=generate_description,
                                populate_location=populate_location,
                                anonymize_keywords=anonymize_keywords,
                                blocked_keywords=blocked_keywords,
                                fallback_city=fallback_city,
                                fallback_country=fallback_country,
                                ownership_notice=(ownership_notice.strip() if apply_ownership else ""),
                            )
                        results.append(result)
                    except Exception as e:
                        st.error(f"Failed: {fname}: {e}")
                        logger.error(f"Photo keyword error for {fpath}: {e}", exc_info=True)
                    if photokw_batch_cooldown_seconds > 0 and total > 1 and idx < total - 1:
                        progress.progress(
                            min((idx + 1) / total, 1.0),
                            f"Cooling down for {photokw_batch_cooldown_seconds:.1f}s before next photo..."
                        )
                        time.sleep(photokw_batch_cooldown_seconds)

                progress.progress(1.0, "Done!")

                # If writing to originals, user needs to copy back — but since
                # we're working on uploaded copies in temp, the writes already happened.
                # The user downloads the processed files.
                if results:
                    st.session_state["photokw_results"] = results
                    st.session_state["photokw_paths"] = file_paths
                    st.session_state["photokw_mode"] = mode

        # Display results
        results = st.session_state.get("photokw_results")
        file_paths = st.session_state.get("photokw_paths", [])
        photokw_mode = st.session_state.get("photokw_mode", "keyword_metadata")

        if results:
            st.divider()

            if photokw_mode == "resize_only":
                resized_count = sum(1 for r in results if r.get("resize_info", {}).get("resized"))
                total_removed = sum(len(r.get("keyword_anonymize_result", {}).get("removed_keywords", [])) for r in results)
                ownership_ok = sum(
                    1 for r in results
                    if not r.get("ownership_result") or r.get("ownership_result", {}).get("success")
                )
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Photos Processed", len(results))
                with mc2:
                    st.metric("Resized", f"{resized_count}/{len(results)}")
                with mc3:
                    st.metric("Sensitive Tags Removed", total_removed)
                with mc4:
                    st.metric("Ownership Written", f"{ownership_ok}/{len(results)}")
            else:
                # Summary metrics
                total_new = sum(len(r.get("new_keywords", [])) for r in results)
                total_existing = sum(len(r.get("existing_keywords", [])) for r in results)
                total_removed = sum(len(r.get("removed_sensitive_keywords", [])) for r in results)
                successful = sum(1 for r in results if r["exif_result"]["success"])
                location_written = sum(
                    1 for r in results if r.get("location_result", {}).get("location_written")
                )
                gps_written = sum(
                    1 for r in results if r.get("location_result", {}).get("gps_written")
                )
                ownership_ok = sum(
                    1 for r in results
                    if not r.get("ownership_result") or r.get("ownership_result", {}).get("success")
                )
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                with mc1:
                    st.metric("Photos Processed", len(results))
                with mc2:
                    st.metric("Existing Tags", total_existing)
                with mc3:
                    st.metric("New Tags Added", total_new)
                with mc4:
                    st.metric("Sensitive Tags Removed", total_removed)
                with mc5:
                    st.metric("Metadata Written", f"{successful}/{len(results)}")
                st.caption(
                    f"Location fields written: {location_written}/{len(results)} | "
                    f"GPS written: {gps_written}/{len(results)} | "
                    f"Ownership metadata written: {ownership_ok}/{len(results)}"
                )

            # Download — single file direct, multiple as zip
            if file_paths:
                if len(file_paths) == 1:
                    fpath = file_paths[0]
                    fname = Path(fpath).name
                    mime = "image/jpeg" if fname.lower().endswith((".jpg", ".jpeg")) else "image/png"
                    with open(fpath, "rb") as dl_f:
                        st.download_button(
                            f"Download {fname}",
                            dl_f.read(),
                            file_name=fname,
                            mime=mime,
                            use_container_width=True,
                        )
                else:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fpath in file_paths:
                            zf.write(fpath, Path(fpath).name)
                    buf.seek(0)
                    st.download_button(
                        f"Download All {len(file_paths)} Photos",
                        buf.getvalue(),
                        file_name="processed_photos.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

            # Per-image details
            if len(results) == 1:
                # Single photo — show inline preview (like Textifier)
                r = results[0]
                resize_info = r.get("resize_info", {})
                ownership_result = r.get("ownership_result")
                if photokw_mode == "resize_only":
                    if resize_info.get("skipped_resize"):
                        st.info(f"Left dimensions unchanged for {r['file_name']}")
                    elif resize_info.get("resized"):
                        st.success(
                            f"Resized {r['file_name']}: "
                            f"{resize_info.get('original_width')}x{resize_info.get('original_height')} -> "
                            f"{resize_info.get('new_width')}x{resize_info.get('new_height')}"
                        )
                    elif resize_info.get("converted_to_jpg"):
                        st.success(f"Converted to JPG: {r['file_name']}")
                    else:
                        st.info(f"No resize needed for {r['file_name']}")
                    if ownership_result:
                        if ownership_result.get("success"):
                            st.success("Ownership metadata written")
                        else:
                            st.warning(f"Ownership metadata write failed: {ownership_result.get('message', '')}")
                else:
                    exif = r["exif_result"]
                    desc_issue = _photo_description_issue(r.get("description", ""))
                    if exif.get("message") == "Keyword generation skipped":
                        st.info("Description/keyword generation skipped")
                    elif exif["success"]:
                        st.success(f"EXIF written: {exif['keywords_written']} keywords to {r['file_name']}")
                    else:
                        st.error(f"EXIF write failed: {exif['message']}")
                    if desc_issue:
                        st.warning(desc_issue)
                    if ownership_result:
                        if ownership_result.get("success"):
                            st.success("Ownership metadata written")
                        else:
                            st.warning(f"Ownership metadata write failed: {ownership_result.get('message', '')}")

                    # GPS / location feedback
                    location_result = r.get("location_result", {})
                    if not location_result.get("enabled"):
                        st.caption("Location/GPS processing was skipped.")
                    elif r.get("location") and any((r.get("location") or {}).values()):
                        loc = r["location"]
                        parts = [v for v in [loc.get("city"), loc.get("state"), loc.get("country")] if v]
                        if parts:
                            st.info(f"Location: **{', '.join(parts)}**")
                        if r.get("gps"):
                            st.caption(f"GPS: {r['gps'][0]:.5f}, {r['gps'][1]:.5f}")
                        if location_result.get("gps_written"):
                            st.success("GPS coordinates were derived and written.")
                    else:
                        st.warning(
                            f"No GPS or usable location hint was found for **{r['file_name']}**. "
                            "Add fallback City/Country to auto-fill empty photos."
                        )

                with st.expander("Preview", expanded=True):
                    # Show thumbnail of the photo
                    if file_paths and Path(file_paths[0]).exists():
                        st.image(file_paths[0], caption=r["file_name"], width=400)
                    if resize_info.get("resized"):
                        st.caption(
                            "Resized: "
                            f"{resize_info.get('original_width')}x{resize_info.get('original_height')} -> "
                            f"{resize_info.get('new_width')}x{resize_info.get('new_height')}"
                        )
                    if photokw_mode == "resize_only":
                        st.markdown(
                            f"**Metadata preserved after resize:** "
                            f"{'Yes' if resize_info.get('metadata_preserved') else 'Partial/Unknown'}"
                        )
                        if resize_info.get("skipped_resize"):
                            st.caption("Dimensions were left unchanged by request.")
                    else:
                        desc = r["description"] or "(no description generated)"
                        st.markdown(f"**Description:**\n\n{desc}")
                        desc_issue = _photo_description_issue(desc)
                        if desc_issue:
                            st.warning(desc_issue)
                        st.divider()
                        # Location fields
                        if r.get("location") and any(r["location"].values()):
                            loc = r["location"]
                            st.markdown(
                                f"**Location:** {loc.get('city', '')} · "
                                f"{loc.get('state', '')} · {loc.get('country', '')}"
                            )
                            if r.get("gps"):
                                st.caption(f"GPS: {r['gps'][0]:.5f}, {r['gps'][1]:.5f}")
                            st.divider()
                        existing = r.get("existing_keywords", [])
                        new_kw = r.get("new_keywords", [])
                        removed_kw = r.get("removed_sensitive_keywords", [])
                        if existing:
                            st.markdown(f"**Existing tags ({len(existing)}):** {', '.join(existing)}")
                        if new_kw:
                            st.markdown(f"**New tags added ({len(new_kw)}):** {', '.join(new_kw)}")
                        elif not existing:
                            st.warning("No keywords generated — the vision model may have failed to describe this image.")
                        if removed_kw:
                            st.caption(f"Removed sensitive tags: {', '.join(removed_kw)}")
                        st.markdown(f"**Combined keywords ({len(r['keywords'])}):**")
                        if r["keywords"]:
                            st.markdown(", ".join(r["keywords"]))
            else:
                # Batch mode
                if photokw_mode != "resize_only":
                    no_gps = [
                        r for r in results
                        if "nogps" in [kw.lower() for kw in r.get("keywords", [])]
                    ]
                    if no_gps:
                        st.warning(
                            f"**{len(no_gps)} photo(s) have no GPS data** — tagged with "
                            f"'nogps' for easy filtering: "
                            f"{', '.join(r['file_name'] for r in no_gps)}"
                        )
                for r in results:
                    resize_info = r.get("resize_info", {})
                    loc = r.get("location")
                    loc_label = ""
                    if loc and any(loc.values()):
                        parts = [v for v in [loc.get("city"), loc.get("state"), loc.get("country")] if v]
                        loc_label = f"  —  {', '.join(parts)}"
                    with st.expander(f"{r['file_name']}{loc_label}", expanded=False):
                        # Show thumbnail in batch mode too
                        idx = next((i for i, fp in enumerate(file_paths) if Path(fp).name == r["file_name"]), None)
                        if idx is not None and Path(file_paths[idx]).exists():
                            st.image(file_paths[idx], caption=r["file_name"], width=300)
                        if resize_info.get("resized"):
                            st.caption(
                                "Resized: "
                                f"{resize_info.get('original_width')}x{resize_info.get('original_height')} -> "
                                f"{resize_info.get('new_width')}x{resize_info.get('new_height')}"
                            )
                        if photokw_mode == "resize_only":
                            st.caption(
                                "Metadata preserved after resize: "
                                f"{'Yes' if resize_info.get('metadata_preserved') else 'Partial/Unknown'}"
                            )
                            if resize_info.get("converted_to_jpg"):
                                st.caption("Converted to JPG")
                            anon_result = r.get("keyword_anonymize_result")
                            if anon_result:
                                if anon_result.get("success"):
                                    removed = anon_result.get("removed_keywords", [])
                                    if removed:
                                        st.caption(f"Removed sensitive tags: {', '.join(removed)}")
                                    else:
                                        st.caption("No sensitive tags removed.")
                                else:
                                    st.warning(
                                        f"Keyword anonymization failed: {anon_result.get('message', 'Unknown error')}"
                                    )
                        else:
                            desc = r.get('description') or '(no description)'
                            st.markdown(f"**Description:** {desc}")
                            desc_issue = _photo_description_issue(desc)
                            if desc_issue:
                                st.warning(desc_issue)
                            if loc and any(loc.values()):
                                st.markdown(
                                    f"**Location:** {loc.get('city', '')} · "
                                    f"{loc.get('state', '')} · {loc.get('country', '')}"
                                )
                            elif not r.get("has_gps"):
                                st.caption("No GPS data — tagged 'nogps'")
                            existing = r.get("existing_keywords", [])
                            new_kw = r.get("new_keywords", [])
                            removed_kw = r.get("removed_sensitive_keywords", [])
                            if existing:
                                st.caption(f"Existing: {', '.join(existing)}")
                            if new_kw:
                                st.caption(f"Added: {', '.join(new_kw)}")
                            if removed_kw:
                                st.caption(f"Removed: {', '.join(removed_kw)}")
                            st.markdown(f"**Keywords ({len(r['keywords'])}):** {', '.join(r['keywords'])}")
                            exif = r["exif_result"]
                            if exif.get("message") == "Keyword generation skipped":
                                st.info("Description/keyword generation skipped")
                            elif exif["success"]:
                                st.success(f"EXIF written: {exif['keywords_written']} new keywords")
                            else:
                                st.error(f"EXIF write failed: {exif['message']}")

        elif uploaded:
            st.info("Choose an action: **Resize Photos Only** or **Process Selected Metadata**")
        else:
            st.info("Upload photos from the left panel to get started")


# ======================================================================
# Main
# ======================================================================

def main():
    st.title("Document or Photo Processing")
    st.caption(f"Version: {PAGE_VERSION} • Document conversion and privacy tools")

    tab_textifier, tab_pdfimg, tab_photo, tab_anonymizer = st.tabs(
        ["Textifier", "PDF Image Extractor", "Photo Processor", "Anonymizer"]
    )

    with tab_textifier:
        _render_textifier_tab()

    with tab_pdfimg:
        _render_pdf_image_extract_tab()

    with tab_photo:
        _render_photo_keywords_tab()

    with tab_anonymizer:
        _render_anonymizer_tab()


if __name__ == "__main__":
    main()

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
