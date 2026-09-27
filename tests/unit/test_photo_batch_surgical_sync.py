"""Surgical `photo_batch.py sync --date/--files`: selection, staging, pixel
verification, the rating guard and the Lightroom reminders."""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "photo_batch.py"
_spec = importlib.util.spec_from_file_location("photo_batch", _PATH)
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)

needs_exiftool = pytest.mark.skipif(not shutil.which("exiftool"), reason="exiftool missing")


def _img(path, seed, fmt=None, size=(320, 240)):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (24, 32), dtype=np.uint8)  # coarse blocks survive JPEG
    Image.fromarray(base).resize(size, Image.NEAREST).convert("RGB").save(path, fmt)
    return path


def _exif(path, *args):
    subprocess.run(["exiftool", "-q", "-overwrite_original", *args, str(path)], check=True)


def _read(path, *tags):
    out = subprocess.run(["exiftool", "-s3", *tags, str(path)],
                         capture_output=True, text=True).stdout
    return out.strip()


# ── selection + staging ──────────────────────────────────────────────────────

def test_select_by_date_and_files_dedupes(tmp_path):
    for n in ("2021-04-13 08-26-30-Cam-3.jpg", "2021-04-13 09-00-00-Cam-3.jpg",
              "2021-04-14 10-00-00-Cam-3.jpg"):
        (tmp_path / n).write_bytes(b"x")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "2020-01-01 00-00-00-Cam-4.jpg").write_bytes(b"x")
    got = pb.select_exports(tmp_path, dates=["2021-04-13"],
                            files=["2021-04-13 08-26-30-Cam-3.jpg",
                                   str(other / "2020-01-01 00-00-00-Cam-4.jpg")])
    assert [p.name for p in got] == ["2021-04-13 08-26-30-Cam-3.jpg",
                                     "2021-04-13 09-00-00-Cam-3.jpg",
                                     "2020-01-01 00-00-00-Cam-4.jpg"]


def test_select_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pb.select_exports(tmp_path, files=["nope.jpg"])


def test_stage_refuses_name_collision(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "same.jpg").write_bytes(b"x")
    (b / "same.jpg").write_bytes(b"y")
    with pytest.raises(ValueError):
        pb.stage_exports([a / "same.jpg", b / "same.jpg"], tmp_path / "stage")


# ── pixel check ──────────────────────────────────────────────────────────────

def test_pixel_match_same_vs_different(tmp_path):
    same_a = pb._load_gray(_img(tmp_path / "a.jpg", 1))
    same_b = pb._load_gray(_img(tmp_path / "b.tif", 1, "TIFF", size=(640, 480)))
    other = pb._load_gray(_img(tmp_path / "c.jpg", 2))
    d, r = pb.pixel_match(same_a, same_b)
    assert d <= pb.PIXEL_MAX_DHASH and r >= pb.PIXEL_MIN_CORR
    d, r = pb.pixel_match(same_a, other)
    assert d > pb.PIXEL_MAX_DHASH or r < pb.PIXEL_MIN_CORR


# ── Lightroom reminders ──────────────────────────────────────────────────────

def test_lightroom_notes_follow_autowrite_setting(monkeypatch):
    monkeypatch.setattr(pb, "LRC_AUTO_WRITE_XMP", False)
    assert "Ctrl+S" in pb.lightroom_before_apply_note()
    assert "BEFORE --apply" in pb.lightroom_before_apply_note()
    monkeypatch.setattr(pb, "LRC_AUTO_WRITE_XMP", True)
    assert "Close Lightroom" in pb.lightroom_before_apply_note()


# ── end to end on real files ─────────────────────────────────────────────────

@pytest.fixture
def library(tmp_path):
    """exports/ with two same-day photos + raws/2021/2021-04 TIF masters."""
    exports = tmp_path / "exports"
    masters = tmp_path / "raws" / "2021" / "2021-04"
    exports.mkdir(), masters.mkdir(parents=True)
    names = ["2021-04-13 11-36-43-TestCam-3", "2021-04-13 11-37-01-TestCam-3"]
    for i, n in enumerate(names):
        _img(exports / f"{n}.jpg", 10 + i)
        _img(masters / f"{n}.tif", 10 + i, "TIFF")
        _exif(exports / f"{n}.jpg", "-XMP-xmp:Rating=3",
              f"-IPTC:Caption-Abstract=caption {i}", f"-XMP-dc:Description=caption {i}",
              "-XMP-dc:Subject=Dubbil Barril", "-IPTC:Keywords=Dubbil Barril")
        _exif(masters / f"{n}.tif", "-XMP-xmp:Rating=3", "-XMP-xmp:Label=Green")
    return exports, tmp_path / "raws" / "2021", masters, names


@needs_exiftool
def test_surgical_apply_touches_only_selected(library, capsys, monkeypatch):
    monkeypatch.setattr(pb, "LRC_AUTO_WRITE_XMP", False)
    exports, raw_root, masters, names = library
    res = pb.sync_photos(exports, raw_root, apply=True,
                         files=[f"{names[0]}.jpg"])
    out = capsys.readouterr().out
    assert res["succeeded"] == 1 and res["failed"] == 0 and res["drift"] == 0
    assert "Dubbil Barril" in _read(masters / f"{names[0]}.tif", "-XMP-dc:Subject")
    assert _read(masters / f"{names[1]}.tif", "-XMP-dc:Subject") == ""   # untouched
    assert _read(masters / f"{names[0]}.tif", "-XMP-xmp:Label") == "Green"
    assert "Pixel check: 1/1" in out
    assert "Read Metadata from Files" in out and "do NOT Save Metadata" in out
    assert "2021/2021-04: 1 photo(s)" in out


@needs_exiftool
def test_surgical_refuses_different_image(library, capsys):
    exports, raw_root, masters, names = library
    _img(masters / f"{names[0]}.tif", 99, "TIFF")        # master is another photo
    _exif(masters / f"{names[0]}.tif", "-XMP-xmp:Rating=3")
    res = pb.sync_photos(exports, raw_root, apply=True, dates=["2021-04-13"])
    assert res["blocked"] and not res["applied"]
    assert "DIFFERENT IMAGE?" in capsys.readouterr().out
    assert _read(masters / f"{names[1]}.tif", "-XMP-dc:Subject") == ""   # nothing written


@needs_exiftool
def test_surgical_refuses_rating_change_unless_allowed(library, capsys):
    exports, raw_root, masters, names = library
    _exif(masters / f"{names[0]}.tif", "-XMP-xmp:Rating=5")
    res = pb.sync_photos(exports, raw_root, apply=True, files=[f"{names[0]}.jpg"])
    assert res["blocked"] and "rating 5 -> 3" in capsys.readouterr().out
    res = pb.sync_photos(exports, raw_root, apply=True, files=[f"{names[0]}.jpg"],
                         allow_rating_change=True)
    assert res["succeeded"] == 1 and res["drift"] == 0
    assert _read(masters / f"{names[0]}.tif", "-XMP-xmp:Rating") == "3"


@needs_exiftool
def test_dry_run_writes_nothing_and_says_ctrl_s_first(library, capsys, monkeypatch):
    monkeypatch.setattr(pb, "LRC_AUTO_WRITE_XMP", False)
    exports, raw_root, masters, names = library
    res = pb.sync_photos(exports, raw_root, apply=False, dates=["2021-04-13"])
    assert not res["applied"] and not res["blocked"]
    assert "Ctrl+S" in capsys.readouterr().out
    assert _read(masters / f"{names[0]}.tif", "-XMP-dc:Subject") == ""
