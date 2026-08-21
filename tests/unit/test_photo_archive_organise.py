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


def test_colliding_timestamps_get_distinct_destinations(conn, tmp_path):
    for n in ("a", "b", "c"):
        db.upsert_file(conn, path=rf"P:\family_Randoms\{n}.jpg",
                       top_folder="family_Randoms", rel_dir="", filename=f"{n}.jpg",
                       ext=".jpg", size=1, mtime=1.0, kind="image",
                       exif_dt="1994-01-01T21:03:33", camera_model="HP pstc5200")
    out = tmp_path / "plan.csv"
    stats = organise.plan_organise(conn, "P:", str(out))
    dsts = [r["dst"] for r in csv.DictReader(out.open())]
    assert len(dsts) == 3
    assert len(set(dsts)) == 3, "plan must not list the same destination twice"
    assert stats["collisions"] == 2


def test_collision_suffixes_are_deterministic(conn, tmp_path):
    for n in ("a", "b"):
        db.upsert_file(conn, path=rf"P:\family_Randoms\{n}.jpg",
                       top_folder="family_Randoms", rel_dir="", filename=f"{n}.jpg",
                       ext=".jpg", size=1, mtime=1.0, kind="image",
                       exif_dt="1994-01-01T21:03:33", camera_model="X")
    first = tmp_path / "1.csv"
    second = tmp_path / "2.csv"
    organise.plan_organise(conn, "P:", str(first))
    organise.plan_organise(conn, "P:", str(second))
    assert first.read_text() == second.read_text()


def test_sidecar_inherits_parent_collision_suffix(conn, tmp_path):
    # Two RAWs share a timestamp; each has its own sidecar. Each .xmp must
    # follow ITS OWN parent, suffix included - never the other one's.
    for n in ("a", "b"):
        pid = db.upsert_file(conn, path=rf"P:\family_Randoms\{n}.raf",
                             top_folder="family_Randoms", rel_dir="",
                             filename=f"{n}.raf", ext=".raf", size=1, mtime=1.0,
                             kind="raw", exif_dt="2019-03-04T10:00:00",
                             camera_model="X-T5")
        db.upsert_file(conn, path=rf"P:\family_Randoms\{n}.xmp",
                       top_folder="family_Randoms", rel_dir="", filename=f"{n}.xmp",
                       ext=".xmp", size=1, mtime=1.0, kind="sidecar", sidecar_of=pid)
    out = tmp_path / "plan.csv"
    organise.plan_organise(conn, "P:", str(out))
    plan = {r["src"]: r["dst"] for r in csv.DictReader(out.open())}
    for n in ("a", "b"):
        raw = plan[rf"P:\family_Randoms\{n}.raf"]
        side = plan[rf"P:\family_Randoms\{n}.xmp"]
        assert os.path.splitext(raw)[0] == os.path.splitext(side)[0]
    assert len(set(plan.values())) == 4


def test_unique_returns_collision_flag():
    taken = set()
    assert organise._unique("P:/a/x.jpg", taken) == ("P:/a/x.jpg", False)
    assert organise._unique("P:/a/x.jpg", taken) == ("P:/a/x-2.jpg", True)
    assert organise._unique("P:/a/x.jpg", taken) == ("P:/a/x-3.jpg", True)


def test_non_photo_files_are_left_alone(conn, tmp_path):
    # Google Drive leaves 164-byte .gdrive stubs beside real photos. This tool
    # organises photographs; relocating arbitrary files is scope it should not
    # take, and sweeping them into _UNDATED would be surprising.
    db.upsert_file(conn, path=r"P:\a\x.jpg.gdrive", top_folder="a", rel_dir="",
                   filename="x.jpg.gdrive", ext=".gdrive", size=164, mtime=1.0,
                   kind="other")
    db.upsert_file(conn, path=r"P:\a\real.jpg", top_folder="a", rel_dir="",
                   filename="real.jpg", ext=".jpg", size=1, mtime=1.0,
                   kind="image", exif_dt="2019-03-04T10:00:00")
    out = tmp_path / "plan.csv"
    organise.plan_organise(conn, "P:", str(out))
    srcs = [r["src"] for r in csv.DictReader(out.open())]
    assert r"P:\a\real.jpg" in srcs
    assert r"P:\a\x.jpg.gdrive" not in srcs
