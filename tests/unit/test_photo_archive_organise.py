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
    # All three are contested, so all three are counted and all three carry a
    # distinguisher. Letting the first arrival keep the clean name would
    # discard that one file's information while preserving the others'.
    assert stats["collisions"] == 3
    assert all("-" in d.rsplit("-", 1)[-1] or d.endswith(("a.jpg", "b.jpg", "c.jpg"))
               for d in dsts)


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


def test_distinguisher_strips_redundant_date_and_model():
    # "1992-09-03 14-43-35_WA_5640 x 3760_Film Scanner.jpg" already repeats the
    # date and model that the new name carries; only "_WA_5640 x 3760" is new.
    d = organise._distinguisher("1992-09-03 14-43-35_WA_5640 x 3760_Film Scanner.jpg",
                                "Film Scanner")
    assert "1992" not in d
    assert "Film Scanner" not in d
    assert "WA" in d and "5640" in d


def test_distinguisher_sanitises_and_caps_length():
    d = organise._distinguisher('a<b>c:d"e/f\\g|h?i*j.jpg', None)
    assert not any(ch in d for ch in '<>:"/\\|?*')
    long = organise._distinguisher("x" * 200 + ".jpg", None)
    assert len(long) <= 60


def test_collision_preserves_original_information(conn, tmp_path):
    for stem in ("1992-09-03 14-43-35_WA_5640 x 3760_Film Scanner",
                 "1992-09-03 14-43-35_Holiday_3760 x 5640_Film Scanner"):
        db.upsert_file(conn, path=rf"P:\family_Randoms\{stem}.jpg",
                       top_folder="family_Randoms", rel_dir="",
                       filename=f"{stem}.jpg", ext=".jpg", size=1, mtime=1.0,
                       kind="image", exif_dt="1992-09-03T14:43:35",
                       camera_model="Film Scanner")
    out = tmp_path / "plan.csv"
    organise.plan_organise(conn, "P:", str(out))
    dsts = [r["dst"] for r in csv.DictReader(out.open())]
    assert len(set(dsts)) == 2
    joined = " ".join(dsts)
    # BOTH must keep their distinguishing text - neither is privileged.
    assert "WA" in joined and "Holiday" in joined
    assert not any(d.endswith("-2.jpg") for d in dsts)


def test_identical_distinguishers_still_fall_back_to_counter(conn, tmp_path):
    # Same original stem in two folders - the distinguisher cannot separate
    # them, so the numeric suffix must still guarantee uniqueness.
    for folder in ("family_Randoms", "Google Drive Photos"):
        db.upsert_file(conn, path=rf"P:\{folder}\beach.jpg", top_folder=folder,
                       rel_dir="", filename="beach.jpg", ext=".jpg", size=1,
                       mtime=1.0, kind="image", exif_dt="2019-03-04T10:00:00",
                       camera_model="X-T5")
    out = tmp_path / "plan.csv"
    organise.plan_organise(conn, "P:", str(out))
    dsts = [r["dst"] for r in csv.DictReader(out.open())]
    assert len(set(dsts)) == 2
