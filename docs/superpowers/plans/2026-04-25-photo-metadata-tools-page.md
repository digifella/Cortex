# Photo & Metadata Tools Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `pages/20_Photo_Metadata_Tools.py` with a Photo Processor tab (moved from Document Extract) and a new LLM Metadata Sync tab that writes LLM-generated JPG keywords/descriptions into matching RAW source files via ExifTool.

**Architecture:** A new `cortex_engine/llm_metadata_sync/` package handles all business logic (index building, keyword merging, ExifTool subprocess calls). The Streamlit page imports from that package for the LMS tab, and copies the photo processor helpers verbatim from Document Extract. Document Extract drops to three tabs.

**Tech Stack:** Python 3.11, Streamlit, ExifTool (subprocess), pytest, PIL (already installed), pandas (already installed)

---

## File Map

**Create:**
- `cortex_engine/llm_metadata_sync/__init__.py`
- `cortex_engine/llm_metadata_sync/models.py`
- `cortex_engine/llm_metadata_sync/matcher.py`
- `cortex_engine/llm_metadata_sync/merger.py`
- `cortex_engine/llm_metadata_sync/exiftool_runner.py`
- `cortex_engine/llm_metadata_sync/sync.py`
- `pages/20_Photo_Metadata_Tools.py`
- `tests/unit/test_llm_metadata_sync/__init__.py`
- `tests/unit/test_llm_metadata_sync/test_matcher.py`
- `tests/unit/test_llm_metadata_sync/test_merger.py`
- `tests/unit/test_llm_metadata_sync/test_sync.py`

**Modify:**
- `pages/7_Document_Extract.py` — remove photo tab and photo-specific helpers
- `docker/pages/7_Document_Extract.py` — sync
- `docker/cortex_engine/llm_metadata_sync/` — sync new module

---

## Task 1: Module scaffold and models

**Files:**
- Create: `cortex_engine/llm_metadata_sync/__init__.py`
- Create: `cortex_engine/llm_metadata_sync/models.py`
- Create: `tests/unit/test_llm_metadata_sync/__init__.py`

- [ ] **Step 1: Create the package directory and empty files**

```bash
mkdir -p cortex_engine/llm_metadata_sync
touch cortex_engine/llm_metadata_sync/__init__.py
mkdir -p tests/unit/test_llm_metadata_sync
touch tests/unit/test_llm_metadata_sync/__init__.py
```

- [ ] **Step 2: Write `cortex_engine/llm_metadata_sync/models.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TargetType(Enum):
    SIDECAR = "sidecar"      # write to .xmp sidecar
    EMBEDDED = "embedded"    # write into TIF/PSD/DNG/PSB


class SidecarAction(Enum):
    CREATE = "create"        # XMP does not exist yet
    MERGE = "merge"          # XMP exists; clear then repopulate
    NONE = "none"            # embedded targets have no sidecar action


@dataclass
class SyncAction:
    jpg_path: Path
    target_path: Path        # XMP sidecar path OR embedded file path
    target_type: TargetType
    sidecar_action: SidecarAction
    raw_path: Path | None    # original RAW for sidecar mode (informational)


@dataclass
class SyncResult:
    action: SyncAction
    success: bool
    keywords_written: int
    description_written: bool
    error: str | None = None


@dataclass
class SyncReport:
    actions: list[SyncAction]
    results: list[SyncResult]
    orphaned_jpgs: list[Path]


@dataclass
class SyncConfig:
    raw_root: Path
    jpg_dir: Path
    filter_keywords: list[str] = field(default_factory=lambda: ["nogps"])
    keep_backups: bool = True
    dry_run: bool = False
    rating_suffix_range: tuple[int, int] = (1, 5)
    raw_extensions: tuple[str, ...] = (
        "RAF", "NEF", "CR2", "CR3", "ARW", "DNG",
        "RW2", "ORF", "PEF", "SRW", "IIQ", "3FR",
    )
    embed_extensions: tuple[str, ...] = ("tif", "tiff", "psd", "psb", "dng")
    deriv_patterns: tuple[str, ...] = (
        r"-Edit", r"-Edit-\d+",
        r"-Enhanced", r"-Enhanced-NR",
        r"-HDR", r"-HDR-\d+",
        r"-Pano", r"-Pano-\d+",
    )
```

- [ ] **Step 3: Commit**

```bash
git add cortex_engine/llm_metadata_sync/ tests/unit/test_llm_metadata_sync/__init__.py
git commit -m "feat: scaffold llm_metadata_sync package and models"
```

---

## Task 2: matcher.py (TDD)

**Files:**
- Create: `cortex_engine/llm_metadata_sync/matcher.py`
- Create: `tests/unit/test_llm_metadata_sync/test_matcher.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_llm_metadata_sync/test_matcher.py`:

```python
from pathlib import Path
import pytest
from cortex_engine.llm_metadata_sync.matcher import (
    strip_rating_suffix,
    build_raw_index,
    resolve_jpg,
)
from cortex_engine.llm_metadata_sync.models import SyncConfig, TargetType, SidecarAction


def _cfg(tmp_path: Path, **kwargs) -> SyncConfig:
    return SyncConfig(raw_root=tmp_path, jpg_dir=tmp_path, **kwargs)


# ── strip_rating_suffix ──────────────────────────────────────────────────────

def test_strip_rating_suffix_removes_trailing_number():
    assert strip_rating_suffix("2026-03-03-X-T5-5", (1, 5)) == "2026-03-03-X-T5"


def test_strip_rating_suffix_removes_rating_1():
    assert strip_rating_suffix("shot-1", (1, 5)) == "shot"


def test_strip_rating_suffix_leaves_out_of_range():
    # -6 is outside (1,5), so not stripped
    assert strip_rating_suffix("shot-6", (1, 5)) == "shot-6"


def test_strip_rating_suffix_no_suffix():
    assert strip_rating_suffix("shot", (1, 5)) == "shot"


def test_strip_rating_suffix_only_strips_trailing():
    # -3 is in the middle, not at end — no strip
    assert strip_rating_suffix("2026-03-10-3-final", (1, 5)) == "2026-03-10-3-final"


# ── build_raw_index ──────────────────────────────────────────────────────────

def test_index_raw_original_points_to_xmp(tmp_path):
    (tmp_path / "shot.RAF").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    assert "shot" in index
    assert tmp_path / "shot.xmp" in index["shot"]


def test_index_skips_acr(tmp_path):
    (tmp_path / "shot.acr").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    assert index == {}


def test_index_embedded_derivative_indexed_by_base_stem(tmp_path):
    (tmp_path / "shot-Edit.tif").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    assert "shot" in index
    assert tmp_path / "shot-Edit.tif" in index["shot"]


def test_index_multiple_matches_same_stem(tmp_path):
    (tmp_path / "shot.RAF").touch()
    (tmp_path / "shot-Edit.tif").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    assert len(index["shot"]) == 2


def test_index_is_case_folded(tmp_path):
    (tmp_path / "Shot.NEF").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    assert "shot" in index


def test_index_nested_directory(tmp_path):
    sub = tmp_path / "2026" / "March"
    sub.mkdir(parents=True)
    (sub / "deep.RAF").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    assert "deep" in index


def test_index_skips_non_raw_non_embed(tmp_path):
    (tmp_path / "shot.jpg").touch()
    (tmp_path / "shot.xmp").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    assert index == {}


# ── resolve_jpg ──────────────────────────────────────────────────────────────

def test_resolve_jpg_sidecar_create_when_xmp_absent(tmp_path):
    (tmp_path / "shot.RAF").touch()
    cfg = _cfg(tmp_path)
    index = build_raw_index(tmp_path, cfg)
    jpg = tmp_path / "shot.jpg"
    jpg.touch()
    actions = resolve_jpg(jpg, index, cfg)
    assert len(actions) == 1
    assert actions[0].target_type == TargetType.SIDECAR
    assert actions[0].sidecar_action == SidecarAction.CREATE


def test_resolve_jpg_sidecar_merge_when_xmp_exists(tmp_path):
    (tmp_path / "shot.RAF").touch()
    (tmp_path / "shot.xmp").touch()
    cfg = _cfg(tmp_path)
    index = build_raw_index(tmp_path, cfg)
    jpg = tmp_path / "shot.jpg"
    jpg.touch()
    actions = resolve_jpg(jpg, index, cfg)
    assert actions[0].sidecar_action == SidecarAction.MERGE


def test_resolve_jpg_embedded_derivative(tmp_path):
    (tmp_path / "shot-Edit.tif").touch()
    cfg = _cfg(tmp_path)
    index = build_raw_index(tmp_path, cfg)
    jpg = tmp_path / "shot.jpg"
    jpg.touch()
    actions = resolve_jpg(jpg, index, cfg)
    assert len(actions) == 1
    assert actions[0].target_type == TargetType.EMBEDDED
    assert actions[0].sidecar_action == SidecarAction.NONE


def test_resolve_jpg_strips_rating_suffix(tmp_path):
    (tmp_path / "shot.RAF").touch()
    cfg = _cfg(tmp_path)
    index = build_raw_index(tmp_path, cfg)
    jpg = tmp_path / "shot-5.jpg"
    jpg.touch()
    actions = resolve_jpg(jpg, index, cfg)
    assert len(actions) == 1


def test_resolve_jpg_orphan_returns_empty(tmp_path):
    cfg = _cfg(tmp_path)
    index = build_raw_index(tmp_path, cfg)
    jpg = tmp_path / "orphan.jpg"
    jpg.touch()
    assert resolve_jpg(jpg, index, cfg) == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/longboardfella/cortex_suite && source venv/bin/activate
pytest tests/unit/test_llm_metadata_sync/test_matcher.py -v 2>&1 | head -20
```

Expected: `ImportError` or `ModuleNotFoundError` — matcher.py doesn't exist yet.

- [ ] **Step 3: Write `cortex_engine/llm_metadata_sync/matcher.py`**

```python
from __future__ import annotations

import os
import re
from pathlib import Path

from .models import SidecarAction, SyncAction, SyncConfig, TargetType


def strip_rating_suffix(stem: str, suffix_range: tuple[int, int]) -> str:
    """Remove trailing -N rating suffix (e.g. '-5') if N is within suffix_range."""
    lo, hi = suffix_range
    for n in range(lo, hi + 1):
        suffix = f"-{n}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _deriv_regex(patterns: tuple[str, ...]) -> re.Pattern:
    alternation = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(f"({alternation})$", re.IGNORECASE)


def build_raw_index(raw_root: Path, config: SyncConfig) -> dict[str, list[Path]]:
    """Walk raw_root once and return {case-folded stem: [target paths]}.

    Target paths are:
    - .xmp sidecar path (may not exist yet) for raw originals
    - embedded file path for TIF/PSD/DNG/PSB derivatives
    ACR files are always skipped.
    """
    raw_exts = {e.lower() for e in config.raw_extensions}
    embed_exts = {e.lower() for e in config.embed_extensions}
    deriv_re = _deriv_regex(config.deriv_patterns)

    index: dict[str, list[Path]] = {}

    for dirpath, _dirs, filenames in os.walk(raw_root):
        dir_path = Path(dirpath)
        for filename in filenames:
            path = dir_path / filename
            ext = path.suffix.lstrip(".").lower()
            stem = path.stem

            if ext == "acr":
                continue

            if ext in raw_exts:
                key = stem.lower()
                sidecar = dir_path / f"{stem}.xmp"
                index.setdefault(key, []).append(sidecar)

            elif ext in embed_exts:
                m = deriv_re.search(stem)
                if m:
                    base_stem = stem[: m.start()]
                    key = base_stem.lower()
                    index.setdefault(key, []).append(path)

    return index


def resolve_jpg(
    jpg_path: Path, index: dict[str, list[Path]], config: SyncConfig
) -> list[SyncAction]:
    """Resolve a JPG to a list of SyncActions against the pre-built index."""
    stem = strip_rating_suffix(jpg_path.stem, config.rating_suffix_range)
    key = stem.lower()
    targets = index.get(key, [])

    embed_exts = {e.lower() for e in config.embed_extensions}
    actions: list[SyncAction] = []

    for target in targets:
        ext = target.suffix.lstrip(".").lower()

        if ext == "xmp":
            sidecar_action = SidecarAction.MERGE if target.exists() else SidecarAction.CREATE
            raw_path: Path | None = None
            for raw_ext in config.raw_extensions:
                candidate = target.parent / f"{target.stem}.{raw_ext}"
                if candidate.exists():
                    raw_path = candidate
                    break
            actions.append(
                SyncAction(
                    jpg_path=jpg_path,
                    target_path=target,
                    target_type=TargetType.SIDECAR,
                    sidecar_action=sidecar_action,
                    raw_path=raw_path,
                )
            )

        elif ext in embed_exts:
            actions.append(
                SyncAction(
                    jpg_path=jpg_path,
                    target_path=target,
                    target_type=TargetType.EMBEDDED,
                    sidecar_action=SidecarAction.NONE,
                    raw_path=None,
                )
            )

    return actions
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/unit/test_llm_metadata_sync/test_matcher.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add cortex_engine/llm_metadata_sync/matcher.py tests/unit/test_llm_metadata_sync/test_matcher.py
git commit -m "feat: add llm_metadata_sync matcher with tests"
```

---

## Task 3: merger.py (TDD)

**Files:**
- Create: `cortex_engine/llm_metadata_sync/merger.py`
- Create: `tests/unit/test_llm_metadata_sync/test_merger.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_llm_metadata_sync/test_merger.py`:

```python
from cortex_engine.llm_metadata_sync.merger import build_keyword_union


def test_existing_keywords_appear_first():
    result = build_keyword_union(["B", "A"], ["C"], filter_list=[])
    assert result == ["B", "A", "C"]


def test_new_keywords_appended_in_original_order():
    result = build_keyword_union([], ["Z", "A", "M"], filter_list=[])
    assert result == ["Z", "A", "M"]


def test_duplicates_deduplicated_first_seen_wins():
    result = build_keyword_union(["A", "B"], ["B", "C"], filter_list=[])
    assert result == ["A", "B", "C"]


def test_filter_removes_matching_keyword():
    result = build_keyword_union(["nogps", "Melbourne"], ["Beach"], filter_list=["nogps"])
    assert "nogps" not in result
    assert "Melbourne" in result
    assert "Beach" in result


def test_filter_is_case_insensitive_against_input():
    result = build_keyword_union(["NOGPS", "Moon"], [], filter_list=["nogps"])
    assert "NOGPS" not in result
    assert "Moon" in result


def test_filter_is_case_insensitive_from_new():
    result = build_keyword_union([], ["NoGPS", "Star"], filter_list=["nogps"])
    assert "NoGPS" not in result
    assert "Star" in result


def test_empty_inputs_return_empty():
    assert build_keyword_union([], [], filter_list=[]) == []


def test_empty_existing():
    assert build_keyword_union([], ["A", "B"], filter_list=[]) == ["A", "B"]


def test_empty_new():
    assert build_keyword_union(["A", "B"], [], filter_list=[]) == ["A", "B"]


def test_case_sensitive_dedup():
    # "Melbourne" and "melbourne" are treated as distinct keywords
    result = build_keyword_union(["Melbourne"], ["melbourne"], filter_list=[])
    assert "Melbourne" in result
    assert "melbourne" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_llm_metadata_sync/test_merger.py -v 2>&1 | head -10
```

Expected: `ImportError`.

- [ ] **Step 3: Write `cortex_engine/llm_metadata_sync/merger.py`**

```python
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def build_keyword_union(
    existing: list[str],
    new: list[str],
    filter_list: list[str],
) -> list[str]:
    """Return union of existing + new keywords.

    - Deduplicated, case-sensitive (first-seen order preserved)
    - existing keywords appear before new ones
    - Keywords matching any entry in filter_list (case-insensitive) are dropped
    """
    filter_set = {f.lower() for f in filter_list}
    seen: dict[str, None] = {}
    for kw in existing + new:
        if kw.lower() not in filter_set and kw not in seen:
            seen[kw] = None
    return list(seen.keys())


def read_jpg_metadata(jpg: Path) -> tuple[list[str], str]:
    """Read IPTC keywords and caption from a JPG via exiftool.

    Returns (keywords, description). Both empty if exiftool fails or file has none.
    """
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return [], ""
    result = subprocess.run(
        [exiftool, "-json", "-s", "-IPTC:Keywords", "-IPTC:Caption-Abstract", str(jpg)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return [], ""
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return [], ""
    if not payload:
        return [], ""
    row = payload[0]
    kws = row.get("Keywords", [])
    if isinstance(kws, str):
        kws = [kws]
    keywords = [k.strip() for k in kws if k.strip()]
    description = (row.get("Caption-Abstract") or "").strip()
    return keywords, description


def read_existing_keywords(target: Path) -> list[str]:
    """Read XMP-dc:Subject keywords from a target file via exiftool.

    Returns empty list if file doesn't exist or exiftool fails.
    """
    if not target.exists():
        return []
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return []
    result = subprocess.run(
        [exiftool, "-json", "-s", "-XMP-dc:Subject", str(target)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not payload:
        return []
    val = payload[0].get("Subject", [])
    if isinstance(val, str):
        val = [val]
    return [k.strip() for k in val if k.strip()]
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/unit/test_llm_metadata_sync/test_merger.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add cortex_engine/llm_metadata_sync/merger.py tests/unit/test_llm_metadata_sync/test_merger.py
git commit -m "feat: add llm_metadata_sync merger with tests"
```

---

## Task 4: exiftool_runner.py

**Files:**
- Create: `cortex_engine/llm_metadata_sync/exiftool_runner.py`

No unit tests for this module — ExifTool subprocess calls require a real binary. Covered by manual acceptance.

- [ ] **Step 1: Write `cortex_engine/llm_metadata_sync/exiftool_runner.py`**

```python
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import TargetType

_BENIGN_WARNINGS = ("IPTCDigest is not current",)


class ExifToolNotFoundError(RuntimeError):
    pass


def exiftool_path() -> str:
    """Return path to exiftool binary, raising ExifToolNotFoundError if absent."""
    path = shutil.which("exiftool")
    if not path:
        raise ExifToolNotFoundError(
            "exiftool not found on PATH. "
            "Install with: sudo apt install libimage-exiftool-perl"
        )
    return path


def is_available() -> bool:
    """Return True if exiftool is on PATH."""
    return shutil.which("exiftool") is not None


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    command: list[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def filtered_stderr(self) -> str:
        lines = [
            line
            for line in self.stderr.splitlines()
            if not any(w in line for w in _BENIGN_WARNINGS)
        ]
        return "\n".join(lines)


def _run(args: list[str]) -> RunResult:
    result = subprocess.run(args, capture_output=True, text=True)
    return RunResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        command=args,
    )


def _backup_flag(keep_backups: bool) -> str:
    return "-overwrite_original" if keep_backups else "-overwrite_original_in_place"


def clear_keyword_lists(
    target: Path, target_type: TargetType, keep_backups: bool
) -> RunResult:
    """Step 1 of two-step write: clear keyword lists on target.

    For SIDECAR: clears xmp-dc:subject only.
    For EMBEDDED: clears xmp-dc:subject AND iptc:Keywords.
    """
    et = exiftool_path()
    backup = _backup_flag(keep_backups)

    if target_type == TargetType.SIDECAR:
        args = [et, backup, "-xmp-dc:subject=", str(target)]
    else:
        args = [et, backup, "-xmp-dc:subject=", "-iptc:Keywords=", str(target)]

    return _run(args)


def write_metadata(
    jpg: Path,
    target: Path,
    target_type: TargetType,
    keywords: list[str],
    description: str,
    keep_backups: bool,
) -> RunResult:
    """Step 2 of two-step write: populate keywords and/or description.

    For SIDECAR: writes to xmp-dc namespace only.
    For EMBEDDED: writes to both xmp-dc and iptc namespaces (kept in sync).
    Description is copied from JPG's iptc:Caption-Abstract via -tagsfromfile.
    """
    et = exiftool_path()
    backup = _backup_flag(keep_backups)
    args = [et, backup]

    if target_type == TargetType.SIDECAR:
        for kw in keywords:
            args.append(f"-xmp-dc:subject+={kw}")
        if description:
            args += [
                "-tagsfromfile", str(jpg),
                "-xmp-dc:description<iptc:Caption-Abstract",
            ]
    else:  # EMBEDDED
        for kw in keywords:
            args.append(f"-xmp-dc:subject+={kw}")
            args.append(f"-iptc:Keywords+={kw}")
        if description:
            args += [
                "-tagsfromfile", str(jpg),
                "-xmp-dc:description<iptc:Caption-Abstract",
                "-iptc:Caption-Abstract<iptc:Caption-Abstract",
            ]

    args.append(str(target))
    return _run(args)
```

- [ ] **Step 2: Commit**

```bash
git add cortex_engine/llm_metadata_sync/exiftool_runner.py
git commit -m "feat: add llm_metadata_sync exiftool_runner"
```

---

## Task 5: sync.py (TDD)

**Files:**
- Create: `cortex_engine/llm_metadata_sync/sync.py`
- Create: `tests/unit/test_llm_metadata_sync/test_sync.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_llm_metadata_sync/test_sync.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cortex_engine.llm_metadata_sync.models import (
    SidecarAction,
    SyncAction,
    SyncConfig,
    TargetType,
)
from cortex_engine.llm_metadata_sync import sync


def _cfg(tmp_path: Path, **kwargs) -> SyncConfig:
    defaults = dict(
        raw_root=tmp_path / "raw",
        jpg_dir=tmp_path / "jpg",
        dry_run=False,
    )
    defaults.update(kwargs)
    return SyncConfig(**defaults)


def _make_files(tmp_path: Path, names: list[str]) -> None:
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "jpg").mkdir(exist_ok=True)
    for name in names:
        (tmp_path / "jpg" / f"{name}.jpg").touch()
        (tmp_path / "raw" / f"{name}.RAF").touch()


def _ok_result() -> MagicMock:
    r = MagicMock()
    r.ok = True
    r.filtered_stderr = ""
    return r


def _fail_result(msg: str = "permission denied") -> MagicMock:
    r = MagicMock()
    r.ok = False
    r.filtered_stderr = msg
    return r


# ── dry run ──────────────────────────────────────────────────────────────────

def test_dry_run_does_not_call_exiftool(tmp_path):
    _make_files(tmp_path, ["shot"])
    cfg = _cfg(tmp_path, dry_run=True)

    with patch("cortex_engine.llm_metadata_sync.sync.read_jpg_metadata", return_value=(["kw1"], "desc")), \
         patch("cortex_engine.llm_metadata_sync.sync.read_existing_keywords", return_value=[]), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.clear_keyword_lists") as mock_clear, \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.write_metadata") as mock_write:
        results = list(sync.run_sync(cfg))

    mock_clear.assert_not_called()
    mock_write.assert_not_called()
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].keywords_written == 1


# ── failure isolation ────────────────────────────────────────────────────────

def test_one_file_failure_does_not_abort_others(tmp_path):
    _make_files(tmp_path, ["a", "b"])
    # Create existing XMPs so the clear step is triggered
    (tmp_path / "raw" / "a.xmp").touch()
    (tmp_path / "raw" / "b.xmp").touch()
    cfg = _cfg(tmp_path)

    with patch("cortex_engine.llm_metadata_sync.sync.read_jpg_metadata", return_value=(["kw"], "desc")), \
         patch("cortex_engine.llm_metadata_sync.sync.read_existing_keywords", return_value=[]), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.clear_keyword_lists",
               side_effect=[_fail_result(), _ok_result()]), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.write_metadata",
               return_value=_ok_result()):
        results = list(sync.run_sync(cfg))

    assert len(results) == 2
    assert any(not r.success for r in results), "Expected one failure"
    assert any(r.success for r in results), "Expected one success"


# ── skip empty metadata ───────────────────────────────────────────────────────

def test_jpg_with_no_metadata_skips_exiftool(tmp_path):
    _make_files(tmp_path, ["empty"])
    cfg = _cfg(tmp_path)

    with patch("cortex_engine.llm_metadata_sync.sync.read_jpg_metadata", return_value=([], "")), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.clear_keyword_lists") as mock_clear, \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.write_metadata") as mock_write:
        results = list(sync.run_sync(cfg))

    mock_clear.assert_not_called()
    mock_write.assert_not_called()
    assert results[0].success is True
    assert results[0].keywords_written == 0


# ── orphan detection ─────────────────────────────────────────────────────────

def test_orphaned_jpgs_do_not_produce_actions(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "jpg").mkdir()
    (tmp_path / "jpg" / "orphan.jpg").touch()
    # No RAW file for orphan
    cfg = _cfg(tmp_path)

    with patch("cortex_engine.llm_metadata_sync.sync.read_jpg_metadata", return_value=(["kw"], "desc")):
        results = list(sync.run_sync(cfg))

    assert results == []


# ── two-step write called for existing sidecar ───────────────────────────────

def test_clear_called_before_write_for_existing_sidecar(tmp_path):
    _make_files(tmp_path, ["shot"])
    (tmp_path / "raw" / "shot.xmp").touch()  # existing sidecar triggers clear
    cfg = _cfg(tmp_path)

    mock_clear = MagicMock(return_value=_ok_result())
    mock_write = MagicMock(return_value=_ok_result())
    call_order = []
    mock_clear.side_effect = lambda *a, **kw: (call_order.append("clear"), _ok_result())[1]
    mock_write.side_effect = lambda *a, **kw: (call_order.append("write"), _ok_result())[1]

    with patch("cortex_engine.llm_metadata_sync.sync.read_jpg_metadata", return_value=(["kw"], "desc")), \
         patch("cortex_engine.llm_metadata_sync.sync.read_existing_keywords", return_value=[]), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.clear_keyword_lists", mock_clear), \
         patch("cortex_engine.llm_metadata_sync.exiftool_runner.write_metadata", mock_write):
        list(sync.run_sync(cfg))

    assert call_order == ["clear", "write"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_llm_metadata_sync/test_sync.py -v 2>&1 | head -10
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Write `cortex_engine/llm_metadata_sync/sync.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Generator

from . import exiftool_runner
from .matcher import build_raw_index, resolve_jpg
from .merger import build_keyword_union, read_existing_keywords, read_jpg_metadata
from .models import SyncAction, SyncConfig, SyncResult, TargetType


def run_sync(config: SyncConfig) -> Generator[SyncResult, None, None]:
    """Generator orchestrator. Yields one SyncResult per matched action.

    Builds the RAW index once, then iterates over all JPGs in jpg_dir.
    Orphaned JPGs (no RAW match) are silently skipped — caller detects
    them via the scan phase using build_raw_index + resolve_jpg directly.
    """
    index = build_raw_index(config.raw_root, config)

    jpgs = sorted(
        list(config.jpg_dir.glob("*.jpg")) + list(config.jpg_dir.glob("*.JPG"))
    )

    for jpg in jpgs:
        actions = resolve_jpg(jpg, index, config)
        for action in actions:
            yield _process_action(action, config)


def _process_action(action: SyncAction, config: SyncConfig) -> SyncResult:
    try:
        jpg_keywords, description = read_jpg_metadata(action.jpg_path)

        if not jpg_keywords and not description:
            return SyncResult(
                action=action,
                success=True,
                keywords_written=0,
                description_written=False,
                error=None,
            )

        existing_keywords = read_existing_keywords(action.target_path)
        merged_keywords = build_keyword_union(
            existing_keywords, jpg_keywords, config.filter_keywords
        )

        if config.dry_run:
            return SyncResult(
                action=action,
                success=True,
                keywords_written=len(merged_keywords),
                description_written=bool(description),
                error=None,
            )

        # Step 1: clear (only if target already exists)
        if action.target_path.exists():
            clear_result = exiftool_runner.clear_keyword_lists(
                action.target_path, action.target_type, config.keep_backups
            )
            if not clear_result.ok:
                return SyncResult(
                    action=action,
                    success=False,
                    keywords_written=0,
                    description_written=False,
                    error=clear_result.filtered_stderr or "exiftool clear failed",
                )

        # Step 2: write
        write_result = exiftool_runner.write_metadata(
            action.jpg_path,
            action.target_path,
            action.target_type,
            merged_keywords,
            description,
            config.keep_backups,
        )
        if not write_result.ok:
            return SyncResult(
                action=action,
                success=False,
                keywords_written=0,
                description_written=False,
                error=write_result.filtered_stderr or "exiftool write failed",
            )

        return SyncResult(
            action=action,
            success=True,
            keywords_written=len(merged_keywords),
            description_written=bool(description),
            error=None,
        )

    except Exception as exc:
        return SyncResult(
            action=action,
            success=False,
            keywords_written=0,
            description_written=False,
            error=str(exc),
        )
```

- [ ] **Step 4: Run all LMS tests — expect pass**

```bash
pytest tests/unit/test_llm_metadata_sync/ -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add cortex_engine/llm_metadata_sync/sync.py tests/unit/test_llm_metadata_sync/test_sync.py
git commit -m "feat: add llm_metadata_sync sync orchestrator with tests"
```

---

## Task 6: Create new page with Photo Processor tab

**Files:**
- Create: `pages/20_Photo_Metadata_Tools.py`

The photo tab and all its helpers are extracted verbatim from `pages/7_Document_Extract.py`. Copy the exact blocks listed below. Do NOT paraphrase — copy character-for-character.

- [ ] **Step 1: Read source blocks from Document Extract**

Open `pages/7_Document_Extract.py` and read these line ranges (they become the photo helpers in the new file):

| Block | Lines | Functions included |
|---|---|---|
| A | 3000–3275 | `_read_photo_metadata_preview`, `_write_photo_metadata_quick_edit`, `_halftone_strength_label`, `_zoom_crop_image`, `_resize_preview_image`, `_build_ab_window_image`, `_render_halftone_ab_compare` |
| B | 3363–3379 | `_photo_description_issue` |
| C | 5007–6084 | `_photokw_temp_dir`, `_photokw_manifest_path`, `_save_photokw_manifest`, `_load_photokw_manifest`, `_render_photo_keywords_tab` |

- [ ] **Step 2: Write `pages/20_Photo_Metadata_Tools.py` — scaffold + imports**

```python
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import streamlit as st
from PIL import Image, ImageOps

from cortex_engine.config_manager import ConfigManager
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger, resolve_db_root_path
from cortex_engine.version_config import VERSION_STRING

from cortex_engine.llm_metadata_sync import exiftool_runner
from cortex_engine.llm_metadata_sync.matcher import build_raw_index, resolve_jpg
from cortex_engine.llm_metadata_sync.models import SyncConfig, SyncReport
from cortex_engine.llm_metadata_sync.sync import run_sync

PAGE_VERSION = VERSION_STRING
MAX_BATCH_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB

st.set_page_config(
    page_title="Photo & Metadata Tools", layout="wide", page_icon="📷"
)

logger = get_logger(__name__)
```

- [ ] **Step 3: Append photo helper functions (Blocks A, B, C)**

After the imports block, paste Block A (lines 3000–3275 from Document Extract), then Block B (lines 3363–3379), then Block C (lines 5007–6084) exactly as they appear in the source file.

- [ ] **Step 4: Append the LMS placeholder tab and main()**

```python
# ── LLM Metadata Sync tab ────────────────────────────────────────────────────
# (implemented in Task 7)

def _render_lms_tab() -> None:
    st.info("LLM Metadata Sync — coming in next task.")


# ── Page footer ──────────────────────────────────────────────────────────────

def _render_page_footer() -> None:
    try:
        from cortex_engine.ui_components import render_version_footer
        render_version_footer()
    except Exception:
        pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Photo & Metadata Tools")
    st.caption(
        f"Version: {PAGE_VERSION} • Photo processing, batch keywords, and LLM metadata sync"
    )

    tab_photo, tab_lms = st.tabs(["Photo Processor", "LLM Metadata Sync"])

    with tab_photo:
        _render_photo_keywords_tab()

    with tab_lms:
        _render_lms_tab()

    _render_page_footer()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Start the app and verify Photo Processor tab works**

```bash
cd /home/longboardfella/cortex_suite && source venv/bin/activate
streamlit run pages/20_Photo_Metadata_Tools.py --server.port 8502 &
```

Open `http://localhost:8502` in a browser. Confirm the Photo Processor tab renders without errors. Check the LLM Metadata Sync tab shows the placeholder. Kill the server (`pkill -f "streamlit run pages/20"`).

- [ ] **Step 6: Commit**

```bash
git add pages/20_Photo_Metadata_Tools.py
git commit -m "feat: create Photo & Metadata Tools page with photo processor tab"
```

---

## Task 7: Implement LMS tab in new page

**Files:**
- Modify: `pages/20_Photo_Metadata_Tools.py` — replace placeholder `_render_lms_tab()`

- [ ] **Step 1: Replace the placeholder with the full LMS UI**

In `pages/20_Photo_Metadata_Tools.py`, find and replace the entire `_render_lms_tab` function with:

```python
def _lms_validate_path(p: str) -> Optional[str]:
    """Return error string if path is invalid, else None."""
    if not p.strip():
        return "Path is required."
    path = Path(p.strip())
    if not path.exists():
        return f"Directory does not exist: {path}"
    if not path.is_dir():
        return f"Not a directory: {path}"
    return None


def _render_lms_tab() -> None:
    st.markdown(
        "Propagate LLM-generated keywords and descriptions from a flat folder of "
        "tagged JPGs back into matching RAW source files and embedded derivatives "
        "(TIF/PSD/DNG) in your RAW library. Uses ExifTool to write XMP sidecars and "
        "embedded IPTC/XMP metadata."
    )

    st.info(
        "**Before running:** In Lightroom Classic, select the affected photos → "
        "**Metadata → Save Metadata to File** (Ctrl+S) to flush pending catalog-only "
        "edits to disk before this tool overwrites them."
    )

    # ExifTool check — bail early if not available
    if not exiftool_runner.is_available():
        st.error(
            "**ExifTool not found on PATH.** Install with:"
        )
        st.code("sudo apt install libimage-exiftool-perl", language="bash")
        return

    # Path inputs
    raw_root_str = st.text_input(
        "RAW root directory",
        placeholder="/mnt/f/Photos/RAW",
        key="lms_raw_root",
    )
    jpg_dir_str = st.text_input(
        "JPG source directory (flat, LLM-tagged JPGs)",
        placeholder="/mnt/f/Photos/Export/LLM-Tagged",
        key="lms_jpg_dir",
    )

    raw_err = _lms_validate_path(raw_root_str) if raw_root_str else None
    jpg_err = _lms_validate_path(jpg_dir_str) if jpg_dir_str else None
    if raw_err:
        st.error(raw_err)
    if jpg_err:
        st.error(jpg_err)
    paths_valid = bool(raw_root_str and jpg_dir_str and not raw_err and not jpg_err)

    # Advanced options
    with st.expander("Advanced options", expanded=False):
        filter_kw_str = st.text_area(
            "Filter keywords (comma-separated, case-insensitive)",
            value="nogps",
            key="lms_filter_kw",
        )
        keep_backups = st.toggle(
            "Keep ExifTool backups (_original files)", value=True, key="lms_keep_backups"
        )
        col1, col2 = st.columns(2)
        with col1:
            rating_lo = st.number_input(
                "Rating suffix min", value=1, min_value=1, max_value=9, key="lms_rating_lo"
            )
        with col2:
            rating_hi = st.number_input(
                "Rating suffix max", value=5, min_value=1, max_value=9, key="lms_rating_hi"
            )
        deriv_patterns_str = st.text_area(
            "Derivative suffix patterns (one per line, regex)",
            value="-Edit\n-Edit-\\d+\n-Enhanced\n-Enhanced-NR\n-HDR\n-HDR-\\d+\n-Pano\n-Pano-\\d+",
            key="lms_deriv_patterns",
        )

    def _build_config(dry_run: bool = False) -> SyncConfig:
        filter_kws = [k.strip() for k in filter_kw_str.split(",") if k.strip()]
        patterns = tuple(p.strip() for p in deriv_patterns_str.splitlines() if p.strip())
        return SyncConfig(
            raw_root=Path(raw_root_str.strip()),
            jpg_dir=Path(jpg_dir_str.strip()),
            filter_keywords=filter_kws,
            keep_backups=keep_backups,
            rating_suffix_range=(int(rating_lo), int(rating_hi)),
            deriv_patterns=patterns,
            dry_run=dry_run,
        )

    # Scan
    if st.button("Scan", disabled=not paths_valid, type="secondary"):
        cfg = _build_config()
        with st.spinner("Building index and scanning JPGs…"):
            index = build_raw_index(cfg.raw_root, cfg)
            jpgs = sorted(
                list(cfg.jpg_dir.glob("*.jpg")) + list(cfg.jpg_dir.glob("*.JPG"))
            )
            all_actions: list = []
            orphaned: list = []
            for jpg in jpgs:
                actions = resolve_jpg(jpg, index, cfg)
                if actions:
                    all_actions.extend(actions)
                else:
                    orphaned.append(jpg)
        report = SyncReport(actions=all_actions, results=[], orphaned_jpgs=orphaned)
        st.session_state["lms_report"] = report
        st.session_state["lms_config_snapshot"] = cfg
        st.session_state["lms_scan_clean"] = True

    report: Optional[SyncReport] = st.session_state.get("lms_report")

    # Warn if paths changed since last scan
    config_changed = False
    if report and paths_valid:
        snap: Optional[SyncConfig] = st.session_state.get("lms_config_snapshot")
        if snap and paths_valid:
            current_cfg = _build_config()
            if current_cfg.raw_root != snap.raw_root or current_cfg.jpg_dir != snap.jpg_dir:
                config_changed = True
                st.warning("⚠️ Paths changed since last scan — please re-scan before applying.")

    # Scan results display
    if report is not None:
        st.markdown("---")
        matched_jpgs = len({a.jpg_path for a in report.actions})
        total_jpgs = matched_jpgs + len(report.orphaned_jpgs)
        st.markdown(
            f"**Scan results:** {len(report.actions)} action(s) across "
            f"{matched_jpgs} matched JPG(s)"
        )

        if report.actions:
            import pandas as pd
            rows = [
                {
                    "JPG": a.jpg_path.name,
                    "Target": a.target_path.name,
                    "Type": a.target_type.value,
                    "Sidecar action": a.sidecar_action.value,
                }
                for a in report.actions
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.caption(
            f"Total JPGs: {total_jpgs} | Matched: {matched_jpgs} | "
            f"Orphaned: {len(report.orphaned_jpgs)}"
        )

        if report.orphaned_jpgs:
            with st.expander(f"Show orphaned JPGs ({len(report.orphaned_jpgs)})"):
                for p in report.orphaned_jpgs:
                    st.text(p.name)

        if not report.actions:
            st.info("No matches found. Check that paths are correct and filenames align.")

    # Dry/Live toggle — default Live run (index 0)
    run_mode = st.radio(
        "Run mode",
        ["Live run", "Dry run"],
        index=0,
        horizontal=True,
        key="lms_run_mode",
    )
    dry_run = run_mode == "Dry run"

    apply_disabled = not report or not report.actions or config_changed
    if st.button("Apply", disabled=apply_disabled, type="primary"):
        snap = st.session_state.get("lms_config_snapshot")
        if not snap:
            st.error("No scan found — please scan first.")
            st.stop()

        apply_cfg = _build_config(dry_run=dry_run)

        log_lines: list[str] = []
        success_count = error_count = 0
        t_start = time.time()
        total = len(report.actions)  # type: ignore[union-attr]

        with st.status("Applying…", expanded=True) as status_widget:
            progress_bar = st.progress(0)
            log_placeholder = st.empty()

            for idx, result in enumerate(run_sync(apply_cfg), start=1):
                progress_bar.progress(idx / total)
                if result.success:
                    success_count += 1
                    parts = [f"✓ {result.action.target_path.name}"]
                    if result.keywords_written:
                        parts.append(f"{result.keywords_written} kw")
                    if result.description_written:
                        parts.append("description")
                    line = " — ".join(parts)
                else:
                    error_count += 1
                    line = f"✗ {result.action.target_path.name} — {result.error}"
                log_lines.append(line)
                log_placeholder.text_area(
                    "Progress log",
                    "\n".join(log_lines[-50:]),
                    height=250,
                    key=f"lms_log_{idx}",
                )

            elapsed = time.time() - t_start
            label = f"Done — {success_count} succeeded, {error_count} failed ({elapsed:.1f}s)"
            state = "complete" if error_count == 0 else "error"
            status_widget.update(label=label, state=state)

        full_log = "\n".join(log_lines)
        st.download_button(
            "Download full log",
            full_log,
            file_name="lms_sync_log.txt",
            mime="text/plain",
        )

        if not dry_run:
            st.success(
                "**After running:** In Lightroom Classic, select the affected photos → "
                "**Metadata → Read Metadata from File**. This pulls the merged keywords "
                "and descriptions into the catalog. New keywords appear flat — drag them "
                "into your hierarchy if needed. LRC does not auto-detect external XMP "
                "changes; this step is required."
            )
```

- [ ] **Step 2: Verify the LMS tab renders**

```bash
streamlit run pages/20_Photo_Metadata_Tools.py --server.port 8502 &
```

Open the LMS tab. Verify:
- ExifTool banner (installed/not installed)
- Path inputs appear
- Advanced options expander works
- Scan button is disabled until both paths are entered
- Run mode radio defaults to "Live run"
- Apply button is disabled before scan

Kill: `pkill -f "streamlit run pages/20"`

- [ ] **Step 3: Commit**

```bash
git add pages/20_Photo_Metadata_Tools.py
git commit -m "feat: implement LLM Metadata Sync tab in Photo & Metadata Tools page"
```

---

## Task 8: Strip photo tab from Document Extract

**Files:**
- Modify: `pages/7_Document_Extract.py`

- [ ] **Step 1: Remove the photo tab from `main()`**

In `pages/7_Document_Extract.py`, find the `main()` function and replace:

```python
    tab_textifier, tab_pdfimg, tab_photo, tab_anonymizer = st.tabs(
        ["PDF Ingestor", "PDF Image Extractor", "Photo Processor", "Anonymizer"]
    )

    with tab_textifier:
        _render_textifier_tab()

    with tab_pdfimg:
        _render_pdf_image_extract_tab()

    with tab_photo:
        _render_photo_keywords_tab()

    with tab_anonymizer:
        _render_anonymizer_tab()
```

with:

```python
    tab_textifier, tab_pdfimg, tab_anonymizer = st.tabs(
        ["PDF Ingestor", "PDF Image Extractor", "Anonymizer"]
    )

    with tab_textifier:
        _render_textifier_tab()

    with tab_pdfimg:
        _render_pdf_image_extract_tab()

    with tab_anonymizer:
        _render_anonymizer_tab()
```

- [ ] **Step 2: Update the page title and caption**

Find in `main()`:

```python
    st.title("Document or Photo Processing")
    st.caption(f"Version: {PAGE_VERSION} • Document conversion, PDF extraction, photo processing, and privacy tools")
```

Replace with:

```python
    st.title("Document Processing")
    st.caption(f"Version: {PAGE_VERSION} • Document conversion, PDF extraction, and privacy tools")
```

- [ ] **Step 3: Remove photo helper function blocks**

Delete the following line ranges from `7_Document_Extract.py`. Delete each block in reverse order (highest line numbers first) so earlier line numbers stay accurate:

1. **Lines 5007–6084** — `_photokw_temp_dir`, `_photokw_manifest_path`, `_save_photokw_manifest`, `_load_photokw_manifest`, `_render_photo_keywords_tab` (the entire photo processor render function and its manifest helpers)

2. **Lines 3363–3379** — `_photo_description_issue`

3. **Lines 3000–3275** — `_read_photo_metadata_preview`, `_write_photo_metadata_quick_edit`, `_halftone_strength_label`, `_zoom_crop_image`, `_resize_preview_image`, `_build_ab_window_image`, `_render_halftone_ab_compare`

After deletion, verify no references to the removed functions remain:

```bash
grep -n "_render_photo_keywords_tab\|_read_photo_metadata_preview\|_write_photo_metadata_quick_edit\|_photo_description_issue\|_photokw_\|_halftone_strength_label\|_zoom_crop_image\|_resize_preview_image\|_build_ab_window_image\|_render_halftone_ab_compare" pages/7_Document_Extract.py
```

Expected: no output.

- [ ] **Step 4: Verify Document Extract still loads**

```bash
streamlit run pages/7_Document_Extract.py --server.port 8502 &
```

Open the page. Confirm three tabs: PDF Ingestor, PDF Image Extractor, Anonymizer. No photo tab. No errors in terminal. Kill: `pkill -f "streamlit run pages/7"`.

- [ ] **Step 5: Run existing tests to confirm no regressions**

```bash
pytest tests/unit/ -v --tb=short 2>&1 | tail -20
```

Expected: all existing tests pass (new LMS tests also pass).

- [ ] **Step 6: Commit**

```bash
git add pages/7_Document_Extract.py
git commit -m "refactor: remove photo processor tab from Document Extract (moved to page 20)"
```

---

## Task 9: Docker sync, full test run, and version bump

**Files:**
- Modify: `docker/pages/7_Document_Extract.py`
- Create: `docker/cortex_engine/llm_metadata_sync/` (copy of new module)
- Create: `docker/pages/20_Photo_Metadata_Tools.py`

- [ ] **Step 1: Sync files to docker directory**

```bash
cp pages/7_Document_Extract.py docker/pages/7_Document_Extract.py
cp pages/20_Photo_Metadata_Tools.py docker/pages/20_Photo_Metadata_Tools.py
cp -r cortex_engine/llm_metadata_sync docker/cortex_engine/llm_metadata_sync
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/unit/ -v --tb=short
```

Expected: all pass, including the new `test_llm_metadata_sync/` tests.

- [ ] **Step 3: Version bump and sync**

```bash
python scripts/version_manager.py --info
```

Update `cortex_engine/version_config.py` — increment the patch version and add to `VERSION_METADATA`:

```python
CORTEX_VERSION = "X.Y.Z+1"  # increment patch from current
VERSION_METADATA = {
    ...
    "new_features": [
        "Photo & Metadata Tools page (page 20) with Photo Processor and LLM Metadata Sync tabs",
        "LLM Metadata Sync: propagates JPG keywords/descriptions to RAW XMP sidecars and embedded TIF/PSD/DNG via ExifTool",
    ],
    ...
}
```

Then sync:

```bash
python scripts/version_manager.py --sync-all
python scripts/version_manager.py --update-changelog
python scripts/version_manager.py --check
```

Expected: `✓ Version consistency verified`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "release: add Photo & Metadata Tools page with LLM Metadata Sync

- New page 20_Photo_Metadata_Tools.py with two tabs:
  - Photo Processor: moved verbatim from Document Extract
  - LLM Metadata Sync: writes LLM-tagged JPG metadata to RAW XMP sidecars
    and embedded TIF/PSD/DNG files via ExifTool
- Document Extract reverts to PDF/text-only (3 tabs)
- New cortex_engine/llm_metadata_sync/ module: matcher, merger,
  exiftool_runner, sync with unit tests
- Docker distribution synced"
```

- [ ] **Step 5: Push**

```bash
git push origin main
```

---

## Manual Acceptance Checklist

Before closing this plan:

- [ ] Scan a directory of ~10 known JPGs against a known RAW tree — verify match count and target list are correct
- [ ] Dry run produces no file changes (verify modification times unchanged with `stat`)
- [ ] Live run on a copy of real data → open LRC → Read Metadata from File → confirm new keywords and description appear
- [ ] `.acr` files are untouched after a run
- [ ] Orphaned JPGs are listed in the UI and the run completes without error
- [ ] ExifTool `_original` backup files appear with backup mode on; absent with backup mode off
- [ ] Document Extract shows exactly three tabs and no Python errors
- [ ] Photo Processor tab in page 20 behaves identically to before the move
