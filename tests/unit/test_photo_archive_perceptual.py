# tests/unit/test_photo_archive_perceptual.py
import csv
import pytest
from PIL import Image

from scripts.photo_archive import db, perceptual


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def _img(path, colour, size=(240, 180)):
    Image.new("RGB", size, colour).save(path)
    return str(path)


def _add(conn, path, dt="2019-03-04T10:00:00", ext=".jpg", kind="image", state="exif_read"):
    return db.upsert_file(conn, path=str(path), top_folder="t", rel_dir="",
                          filename=str(path).split("/")[-1], ext=ext, size=1,
                          mtime=1.0, kind=kind, exif_dt=dt, state=state)


def test_perceptual_hash_is_stable_for_same_image(tmp_path):
    a = _img(tmp_path / "a.jpg", (10, 120, 200))
    b = _img(tmp_path / "b.jpg", (10, 120, 200))
    assert perceptual.perceptual_hash(a) == perceptual.perceptual_hash(b)


def test_perceptual_hash_survives_a_resize(tmp_path):
    # The whole point of tier 3: a rendition at a different size is the SAME
    # photograph, and byte hashing cannot see that.
    big = Image.new("RGB", (400, 300), (0, 0, 0))
    for x in range(400):
        for y in range(0, 300, 7):
            big.putpixel((x, y), (x % 256, y % 256, 90))
    big.save(tmp_path / "big.jpg", quality=95)
    big.resize((200, 150)).save(tmp_path / "small.jpg", quality=95)
    assert perceptual.perceptual_hash(str(tmp_path / "big.jpg")) == \
           perceptual.perceptual_hash(str(tmp_path / "small.jpg"))


def test_perceptual_hash_differs_for_different_images(tmp_path):
    a = _img(tmp_path / "a.jpg", (255, 0, 0))
    b = Image.new("RGB", (240, 180))
    for x in range(240):
        for y in range(180):
            b.putpixel((x, y), ((x * 7) % 256, (y * 13) % 256, 40))
    b.save(tmp_path / "b.jpg", quality=95)
    assert perceptual.perceptual_hash(a) != perceptual.perceptual_hash(str(tmp_path / "b.jpg"))


def test_only_candidates_sharing_a_timestamp_are_hashed(conn, tmp_path):
    _add(conn, _img(tmp_path / "solo.jpg", (1, 2, 3)), dt="2001-01-01T00:00:00")
    _add(conn, _img(tmp_path / "p1.jpg", (9, 9, 9)))
    _add(conn, _img(tmp_path / "p2.jpg", (9, 9, 9)))
    stats = perceptual.hash_candidates(conn)
    assert stats["hashed"] == 2
    solo = conn.execute("SELECT percept_hash FROM files WHERE filename='solo.jpg'").fetchone()[0]
    assert solo is None


def test_raw_and_video_are_never_decoded(conn, tmp_path):
    _add(conn, tmp_path / "a.raf", ext=".raf", kind="raw")
    _add(conn, tmp_path / "b.raf", ext=".raf", kind="raw")
    _add(conn, tmp_path / "c.mov", ext=".mov", kind="video")
    _add(conn, tmp_path / "d.mov", ext=".mov", kind="video")
    assert perceptual.hash_candidates(conn)["hashed"] == 0


def test_quarantined_files_are_skipped(conn, tmp_path):
    _add(conn, _img(tmp_path / "q1.jpg", (5, 5, 5)), state="quarantined")
    _add(conn, _img(tmp_path / "q2.jpg", (5, 5, 5)), state="quarantined")
    assert perceptual.hash_candidates(conn)["hashed"] == 0


def test_unreadable_file_counts_as_error_and_continues(conn, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    _add(conn, str(bad))
    _add(conn, _img(tmp_path / "ok.jpg", (7, 7, 7)))
    stats = perceptual.hash_candidates(conn)
    assert stats["errors"] == 1 and stats["hashed"] == 1


def test_plan_groups_only_within_the_same_timestamp(conn, tmp_path):
    _add(conn, _img(tmp_path / "x1.jpg", (30, 60, 90)), dt="2019-03-04T10:00:00")
    _add(conn, _img(tmp_path / "x2.jpg", (30, 60, 90)), dt="2019-03-04T10:00:00")
    _add(conn, _img(tmp_path / "y1.jpg", (30, 60, 90)), dt="2020-01-01T00:00:00")
    _add(conn, _img(tmp_path / "y2.jpg", (30, 60, 90)), dt="2020-01-01T00:00:00")
    perceptual.hash_candidates(conn)
    out = tmp_path / "t3.csv"
    stats = perceptual.plan_tier3(conn, str(out))
    # identical pixels, but two DIFFERENT capture instants -> two groups, never one
    assert stats["groups"] == 2
    rows = list(csv.DictReader(out.open()))
    assert len([r for r in rows if r["role"] == "keep"]) == 2


def test_plan_is_report_only_and_never_marks_state(conn, tmp_path):
    _add(conn, _img(tmp_path / "a.jpg", (4, 4, 4)))
    _add(conn, _img(tmp_path / "b.jpg", (4, 4, 4)))
    perceptual.hash_candidates(conn)
    perceptual.plan_tier3(conn, str(tmp_path / "t3.csv"))
    states = {r[0] for r in conn.execute("SELECT state FROM files")}
    assert states == {"exif_read"}, "tier 3 is reviewable only - it must not act"


def test_min_size_filter_skips_small_files(conn, tmp_path):
    small = _img(tmp_path / "s1.jpg", (2, 2, 2), size=(40, 30))
    _img(tmp_path / "s2.jpg", (2, 2, 2), size=(40, 30))
    import os
    for p in ("s1.jpg", "s2.jpg"):
        db.upsert_file(conn, path=str(tmp_path / p), top_folder="t", rel_dir="",
                       filename=p, ext=".jpg", size=os.path.getsize(tmp_path / p),
                       mtime=1.0, kind="image", exif_dt="2019-03-04T10:00:00")
    assert perceptual.hash_candidates(conn, min_size=10_000_000)["hashed"] == 0
    assert perceptual.hash_candidates(conn, min_size=0)["hashed"] == 2


def test_group_flags_separate_edits_and_formats(conn, tmp_path):
    def add(name, ext):
        p = tmp_path / name
        _img(p, (44, 88, 122))
        db.upsert_file(conn, path=str(p), top_folder="t", rel_dir="",
                       filename=name, ext=ext, size=100, mtime=1.0,
                       kind="image", exif_dt="2019-03-04T10:00:00")
    add("shot.jpg", ".jpg")
    add("shot-Edit.jpg", ".jpg")
    perceptual.hash_candidates(conn, min_size=0)
    out = tmp_path / "t3.csv"
    perceptual.plan_tier3(conn, str(out))
    flags = {r["flag"] for r in csv.DictReader(out.open())}
    assert flags == {"edit_pair"}, "an original/-Edit pair must be flagged, not proposed"


def test_same_format_no_edit_is_flagged_reviewable(conn, tmp_path):
    for name in ("one.jpg", "two.jpg"):
        p = tmp_path / name
        _img(p, (17, 17, 90))
        db.upsert_file(conn, path=str(p), top_folder="t", rel_dir="",
                       filename=name, ext=".jpg", size=100, mtime=1.0,
                       kind="image", exif_dt="2019-03-04T10:00:00")
    perceptual.hash_candidates(conn, min_size=0)
    out = tmp_path / "t3.csv"
    perceptual.plan_tier3(conn, str(out))
    assert {r["flag"] for r in csv.DictReader(out.open())} == {"candidate"}
