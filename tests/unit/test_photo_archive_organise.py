# tests/unit/test_photo_archive_organise.py
import csv
import os
import pytest
from scripts.photo_archive import db, organise


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def test_target_name_matches_existing_export_convention():
    assert organise.target_name("2019-01-01T15:41:12", "X-M1", ".jpg") == \
        "2019-01-01 15-41-12-X-M1.jpg"


def test_target_name_drops_trailing_hyphen_without_model():
    # Never produce the dangling "1994-01-01 21-03-33--4.jpg" form.
    assert organise.target_name("1994-01-01T21:03:33", None, ".jpg") == \
        "1994-01-01 21-03-33.jpg"
    assert organise.target_name("1994-01-01T21:03:33", "", ".jpg") == \
        "1994-01-01 21-03-33.jpg"


def test_target_name_preserves_internal_spaces_in_model():
    assert organise.target_name("1994-03-01T14:55:53", "HP pstc5200", ".jpg") == \
        "1994-03-01 14-55-53-HP pstc5200.jpg"


def test_target_name_lowercases_extension():
    assert organise.target_name("2019-01-01T15:41:12", "X-M1", ".JPG").endswith(".jpg")


def test_target_dir_is_year_then_year_month():
    got = organise.target_dir("P:", "2019-03-04T10:00:00")
    assert got == os.path.join("P:", "2019", "2019-03")


def test_undated_rows_route_to_undated_tree(conn, tmp_path):
    db.upsert_file(conn, path=r"P:\family_Randoms\sub\x.jpg",
                   top_folder="family_Randoms", rel_dir="sub", filename="x.jpg",
                   ext=".jpg", size=1, mtime=1.0, kind="image", exif_dt=None)
    out = tmp_path / "plan.csv"
    organise.plan_organise(conn, "P:", str(out))
    row = next(r for r in csv.DictReader(out.open()))
    assert row["dst"] == os.path.join("P:", "_UNDATED", "family_Randoms",
                                      "sub", "x.jpg")


def test_dated_rows_route_to_year_month(conn, tmp_path):
    db.upsert_file(conn, path=r"P:\family_Randoms\x.jpg",
                   top_folder="family_Randoms", rel_dir="", filename="x.jpg",
                   ext=".jpg", size=1, mtime=1.0, kind="image",
                   exif_dt="2019-03-04T10:00:00", camera_model="X-T5")
    out = tmp_path / "plan.csv"
    organise.plan_organise(conn, "P:", str(out))
    row = next(r for r in csv.DictReader(out.open()))
    assert row["dst"] == os.path.join("P:", "2019", "2019-03",
                                      "2019-03-04 10-00-00-X-T5.jpg")


def test_sidecar_takes_parent_stem_and_destination(conn, tmp_path):
    parent = db.upsert_file(conn, path=r"P:\family_Randoms\x.raf",
                            top_folder="family_Randoms", rel_dir="",
                            filename="x.raf", ext=".raf", size=1, mtime=1.0,
                            kind="raw", exif_dt="2019-03-04T10:00:00",
                            camera_model="X-T5")
    db.upsert_file(conn, path=r"P:\family_Randoms\x.xmp",
                   top_folder="family_Randoms", rel_dir="", filename="x.xmp",
                   ext=".xmp", size=1, mtime=1.0, kind="sidecar",
                   sidecar_of=parent)
    out = tmp_path / "plan.csv"
    organise.plan_organise(conn, "P:", str(out))
    rows = {r["src"]: r["dst"] for r in csv.DictReader(out.open())}
    assert rows[r"P:\family_Randoms\x.xmp"] == os.path.join(
        "P:", "2019", "2019-03", "2019-03-04 10-00-00-X-T5.xmp")


def test_quarantined_rows_are_not_organised(conn, tmp_path):
    fid = db.upsert_file(conn, path=r"P:\_DUPES\family_Randoms\x.jpg",
                         top_folder="family_Randoms", rel_dir="",
                         filename="x.jpg", ext=".jpg", size=1, mtime=1.0,
                         kind="image", exif_dt="2019-03-04T10:00:00")
    db.set_state(conn, fid, "quarantined")
    out = tmp_path / "plan.csv"
    stats = organise.plan_organise(conn, "P:", str(out))
    assert stats["planned"] == 0
