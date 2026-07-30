"""Keywords living only in an XMP sidecar must survive enrichment.

Lightroom sometimes writes keywords to a `<stem>.xmp` beside a JPG/TIF rather
than into the file. exiftool does not follow sidecars, so a sidecar-only
keyword was invisible to `read_exif_keywords` — and because enrichment writes
back `existing + AI` keywords, an invisible keyword is a *deleted* keyword.
"""
import shutil
import subprocess

import pytest

from cortex_engine.textifier import DocumentTextifier

pytestmark = pytest.mark.skipif(
    shutil.which("exiftool") is None, reason="exiftool not installed"
)


def _make_jpg(path, keywords=()):
    """A real 8x8 JPG, optionally carrying embedded keywords."""
    from PIL import Image

    Image.new("RGB", (8, 8), (128, 128, 128)).save(path, "JPEG")
    if keywords:
        args = [f"-XMP-dc:Subject={k}" for k in keywords]
        args += [f"-IPTC:Keywords={k}" for k in keywords]
        subprocess.run(
            ["exiftool", "-overwrite_original", *args, str(path)],
            capture_output=True, check=True,
        )


def _make_sidecar(path, keywords):
    """A real .xmp sidecar carrying XMP-dc:Subject keywords."""
    args = [f"-XMP-dc:Subject={k}" for k in keywords]
    subprocess.run(
        ["exiftool", "-overwrite_original", *args, str(path)],
        capture_output=True, check=True,
    )


def test_sidecar_only_keyword_is_returned(tmp_path):
    """A keyword present only in the sidecar must not be lost."""
    jpg = tmp_path / "shot.jpg"
    _make_jpg(jpg, keywords=["embedded_tag"])
    _make_sidecar(tmp_path / "shot.xmp", ["sidecar_only_tag"])

    assert DocumentTextifier.read_exif_keywords(str(jpg)) == [
        "embedded_tag",
        "sidecar_only_tag",
    ]


def test_no_sidecar_returns_embedded_keywords_only(tmp_path):
    """Regression guard: the common no-sidecar case is unchanged."""
    jpg = tmp_path / "shot.jpg"
    _make_jpg(jpg, keywords=["embedded_tag"])

    assert DocumentTextifier.read_exif_keywords(str(jpg)) == ["embedded_tag"]


def test_sidecar_and_embedded_duplicates_are_deduped(tmp_path):
    """Same keyword in both places yields one entry, not two."""
    jpg = tmp_path / "shot.jpg"
    _make_jpg(jpg, keywords=["shared_tag"])
    _make_sidecar(tmp_path / "shot.xmp", ["Shared_Tag"])

    assert DocumentTextifier.read_exif_keywords(str(jpg)) == ["shared_tag"]


def test_sidecar_for_tif_master_is_read(tmp_path):
    """TIF masters are the other catalog format that gets stray sidecars."""
    from PIL import Image

    tif = tmp_path / "edit.tif"
    Image.new("RGB", (8, 8), (200, 200, 200)).save(tif, "TIFF")
    _make_sidecar(tmp_path / "edit.xmp", ["lightroom_tag"])

    assert DocumentTextifier.read_exif_keywords(str(tif)) == ["lightroom_tag"]


def test_sidecar_is_not_confused_with_a_different_stem(tmp_path):
    """Only the sidecar matching this file's stem is read."""
    jpg = tmp_path / "shot.jpg"
    _make_jpg(jpg, keywords=["embedded_tag"])
    _make_sidecar(tmp_path / "other.xmp", ["other_tag"])

    assert DocumentTextifier.read_exif_keywords(str(jpg)) == ["embedded_tag"]


def test_numeric_sidecar_keyword_does_not_wipe_all_keywords(tmp_path):
    """A year keyword comes back from exiftool as an int, not a str.

    Sidecars routinely carry year tags. Without coercion the whole read
    raises, is swallowed by the except, and returns [] — losing every
    keyword on the photo, not just the numeric one.
    """
    jpg = tmp_path / "shot.jpg"
    _make_jpg(jpg, keywords=["embedded_tag"])
    _make_sidecar(tmp_path / "shot.xmp", ["2025"])

    assert DocumentTextifier.read_exif_keywords(str(jpg)) == ["2025", "embedded_tag"]
