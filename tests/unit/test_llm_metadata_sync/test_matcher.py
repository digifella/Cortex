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
    assert strip_rating_suffix("shot-6", (1, 5)) == "shot-6"


def test_strip_rating_suffix_no_suffix():
    assert strip_rating_suffix("shot", (1, 5)) == "shot"


def test_strip_rating_suffix_only_strips_trailing():
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


def test_index_double_edit_tif_indexed_by_base_stem(tmp_path):
    """2025-09-11 07-43-37-X-T5-Edit-Edit.tif must resolve to base key without any Edit suffix."""
    (tmp_path / "2025-09-11 07-43-37-X-T5-Edit-Edit.tif").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    key = "2025-09-11 07-43-37-x-t5"
    assert key in index, f"Expected key '{key}', got: {list(index.keys())}"
    assert tmp_path / "2025-09-11 07-43-37-X-T5-Edit-Edit.tif" in index[key]


def test_index_double_edit_tif_matches_rated_jpg(tmp_path):
    """Full round-trip: double-Edit TIF should match a rated JPG with the same base timestamp."""
    (tmp_path / "2025-09-11 07-43-37-X-T5-Edit-Edit.tif").touch()
    cfg = _cfg(tmp_path)
    index = build_raw_index(tmp_path, cfg)
    jpg = tmp_path / "2025-09-11 07-43-37-X-T5-5.jpg"
    jpg.touch()
    actions = resolve_jpg(jpg, index, cfg)
    assert len(actions) == 1
    assert actions[0].target_type == TargetType.EMBEDDED


def test_index_enhanced_nr_double_edit_tif_indexed_by_base_stem(tmp_path):
    """2025-01-08 15-58-04-X-T5-Enhanced-NR-Edit-Edit.tif must strip the full compound suffix."""
    (tmp_path / "2025-01-08 15-58-04-X-T5-Enhanced-NR-Edit-Edit.tif").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    key = "2025-01-08 15-58-04-x-t5"
    assert key in index, f"Expected key '{key}', got: {list(index.keys())}"
    assert tmp_path / "2025-01-08 15-58-04-X-T5-Enhanced-NR-Edit-Edit.tif" in index[key]


def test_index_enhanced_nr_double_edit_tif_matches_rated_jpg(tmp_path):
    """Full round-trip: Enhanced-NR-Edit-Edit TIF must match a rated JPG at the base timestamp."""
    (tmp_path / "2025-01-08 15-58-04-X-T5-Enhanced-NR-Edit-Edit.tif").touch()
    cfg = _cfg(tmp_path)
    index = build_raw_index(tmp_path, cfg)
    jpg = tmp_path / "2025-01-08 15-58-04-X-T5-5.jpg"
    jpg.touch()
    actions = resolve_jpg(jpg, index, cfg)
    assert len(actions) == 1
    assert actions[0].target_type == TargetType.EMBEDDED


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


def test_index_skips_xmp_standalone(tmp_path):
    """Bare .xmp files are not indexed (they're sidecar outputs, not sources)."""
    (tmp_path / "shot.xmp").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    assert index == {}


def test_index_jpg_indexed_as_jpg_replace_target(tmp_path):
    """Plain JPG in raw_root is indexed as a JPG_REPLACE target, not skipped."""
    (tmp_path / "shot.jpg").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    assert "shot" in index
    assert tmp_path / "shot.jpg" in index["shot"]


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


# ── JPG_REPLACE (mobile / catalog JPG sources) ───────────────────────────────

def test_index_catalog_jpg_indexed_as_jpg_replace(tmp_path):
    """Plain JPG in raw_root (no rating suffix) must appear in index as a JPG_REPLACE target."""
    (tmp_path / "2025-09-09 11-24-48-Pixel 9 Pro.jpg").touch()
    index = build_raw_index(tmp_path, _cfg(tmp_path))
    key = "2025-09-09 11-24-48-pixel 9 pro"
    assert key in index
    assert tmp_path / "2025-09-09 11-24-48-Pixel 9 Pro.jpg" in index[key]


def test_catalog_jpg_matches_rated_described_jpg(tmp_path):
    """Full round-trip: described rated JPG from jpg_dir must resolve to catalog JPG via JPG_REPLACE."""
    catalog_jpg = tmp_path / "2025-09-09 11-24-48-Pixel 9 Pro.jpg"
    catalog_jpg.touch()
    cfg = _cfg(tmp_path)
    index = build_raw_index(tmp_path, cfg)
    # described photo is the jpg_path (has rating suffix, is in jpg_dir)
    described = tmp_path / "2025-09-09 11-24-48-Pixel 9 Pro-5.jpg"
    described.touch()
    actions = resolve_jpg(described, index, cfg)
    assert len(actions) == 1
    assert actions[0].target_type == TargetType.JPG_REPLACE
    assert actions[0].target_path == catalog_jpg
    assert actions[0].jpg_path == described


def test_catalog_jpg_does_not_match_itself(tmp_path):
    """When jpg_dir == raw_root a catalog JPG must not create a self-replace action."""
    catalog_jpg = tmp_path / "2025-09-09 11-24-48-Pixel 9 Pro.jpg"
    catalog_jpg.touch()
    cfg = _cfg(tmp_path)
    index = build_raw_index(tmp_path, cfg)
    actions = resolve_jpg(catalog_jpg, index, cfg)
    assert actions == []
