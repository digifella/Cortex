import csv
import pytest
from scripts.photo_archive import db, dupes


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def _add(conn, path, top="family_Randoms", rel="", size=100, sha="abc",
         mtime=100.0, dt=None, model=None, ext=".jpg", kind="image"):
    return db.upsert_file(conn, path=path, top_folder=top, rel_dir=rel,
                          filename=path.split("\\")[-1], ext=ext, size=size,
                          mtime=mtime, kind=kind, sha256=sha,
                          exif_dt=dt, camera_model=model)


def _rows(conn):
    return conn.execute("SELECT * FROM files ORDER BY id").fetchall()


def test_keeper_avoids_dupes_named_folder(conn):
    _add(conn, r"P:\0 and 1 star photos originals and dupes\a.jpg",
         top="0 and 1 star photos originals and dupes")
    _add(conn, r"P:\family_Randoms\a.jpg")
    keeper = dupes.choose_keeper(_rows(conn))
    assert keeper["top_folder"] == "family_Randoms"


def test_keeper_avoids_backup_named_folder(conn):
    _add(conn, r"P:\Backup Consolidated Photos\a.jpg",
         top="Backup Consolidated Photos")
    _add(conn, r"P:\family_Randoms\a.jpg")
    assert dupes.choose_keeper(_rows(conn))["top_folder"] == "family_Randoms"


def test_keeper_prefers_richer_exif(conn):
    _add(conn, r"P:\family_Randoms\a.jpg", dt=None)
    _add(conn, r"P:\family_Randoms\b.jpg", dt="2019-01-01T10:00:00")
    assert dupes.choose_keeper(_rows(conn))["exif_dt"] is not None


def test_keeper_prefers_camera_model_when_dates_equal(conn):
    _add(conn, r"P:\family_Randoms\a.jpg", dt="2019-01-01T10:00:00", model=None)
    _add(conn, r"P:\family_Randoms\b.jpg", dt="2019-01-01T10:00:00", model="X-T5")
    assert dupes.choose_keeper(_rows(conn))["camera_model"] == "X-T5"


def test_keeper_prefers_shallower_path(conn):
    _add(conn, r"P:\family_Randoms\deep\deeper\a.jpg", rel="deep\\deeper")
    _add(conn, r"P:\family_Randoms\b.jpg", rel="")
    assert dupes.choose_keeper(_rows(conn))["rel_dir"] == ""


def test_keeper_prefers_older_mtime(conn):
    _add(conn, r"P:\family_Randoms\a.jpg", mtime=500.0)
    _add(conn, r"P:\family_Randoms\b.jpg", mtime=100.0)
    assert dupes.choose_keeper(_rows(conn))["mtime"] == 100.0


def test_keeper_is_deterministic_on_full_tie(conn):
    _add(conn, r"P:\family_Randoms\b.jpg")
    _add(conn, r"P:\family_Randoms\a.jpg")
    first = dupes.choose_keeper(_rows(conn))["path"]
    second = dupes.choose_keeper(list(reversed(_rows(conn))))["path"]
    assert first == second == r"P:\family_Randoms\a.jpg"


def test_raw_and_jpg_never_grouped(conn):
    _add(conn, r"P:\family_Randoms\a.jpg", sha="same", ext=".jpg", kind="image")
    _add(conn, r"P:\family_Randoms\a.raf", sha="same", ext=".raf", kind="raw")
    assert list(dupes.tier1_groups(conn)) == []


def test_tier1_group_requires_two_members(conn):
    _add(conn, r"P:\family_Randoms\a.jpg", sha="solo")
    assert list(dupes.tier1_groups(conn)) == []


def test_plan_tier1_writes_csv_with_keeper_flag(conn, tmp_path):
    _add(conn, r"P:\Backup Consolidated Photos\a.jpg",
         top="Backup Consolidated Photos")
    _add(conn, r"P:\family_Randoms\a.jpg")
    out = tmp_path / "plan.csv"
    stats = dupes.plan_tier1(conn, str(out))
    rows = list(csv.DictReader(out.open()))
    assert stats["groups"] == 1
    assert stats["to_quarantine"] == 1
    keepers = [r for r in rows if r["role"] == "keep"]
    assert len(keepers) == 1
    assert keepers[0]["top_folder"] == "family_Randoms"


def test_sidecars_are_never_grouped_as_duplicates(conn):
    _add(conn, r"P:\family_Randoms\a.xmp", sha="s", ext=".xmp", kind="sidecar")
    _add(conn, r"P:\family_Randoms\b.xmp", sha="s", ext=".xmp", kind="sidecar")
    assert list(dupes.tier1_groups(conn)) == []


def test_keeper_prefers_clean_name_over_copy_suffix(conn):
    # Google Drive names re-uploads "IMG_0176 (1).JPG". The clean name is the
    # original; the (N) is the copy. mtime cannot decide this - a bulk download
    # stamps them all within minutes in arbitrary order.
    _add(conn, r"P:\Google Drive Photos\IMG_0176 (1).JPG", mtime=1.0)
    _add(conn, r"P:\Google Drive Photos\IMG_0176.JPG", mtime=999.0)
    keeper = dupes.choose_keeper(_rows(conn))
    assert keeper["path"].endswith("IMG_0176.JPG")


def test_keeper_prefers_clean_name_over_highest_copy_number(conn):
    for n, mt in ((" (1)", 3.0), (" (2)", 2.0), (" (3)", 1.0), ("", 9.0)):
        _add(conn, rf"P:\Google Drive Photos\BH_2010_278{n}.jpg", mtime=mt)
    keeper = dupes.choose_keeper(_rows(conn))
    assert keeper["path"].endswith("BH_2010_278.jpg")


def test_keeper_prefers_clean_name_over_windows_copy_suffix(conn):
    _add(conn, r"P:\family_Randoms\photo - Copy.jpg", mtime=1.0)
    _add(conn, r"P:\family_Randoms\photo.jpg", mtime=999.0)
    assert dupes.choose_keeper(_rows(conn))["path"].endswith("photo.jpg")


def test_copy_suffix_does_not_outrank_folder_demotion(conn):
    # A clean name inside a Backup folder still loses to a (1) outside it.
    _add(conn, r"P:\Backup Consolidated Photos\x.jpg", top="Backup Consolidated Photos")
    _add(conn, r"P:\family_Randoms\x (1).jpg")
    assert dupes.choose_keeper(_rows(conn))["top_folder"] == "family_Randoms"


def test_is_copy_suffixed_detects_forms():
    assert dupes._is_copy("IMG_0176 (1).JPG") is True
    assert dupes._is_copy("BH_2010_278 (12).jpg") is True
    assert dupes._is_copy("photo - Copy.jpg") is True
    assert dupes._is_copy("photo - Copy (2).jpg") is True
    assert dupes._is_copy("IMG_0176.JPG") is False
    # A legitimate name that merely contains brackets must NOT be flagged.
    assert dupes._is_copy("Trip (Italy) 2019.jpg") is False


def test_keeper_prefers_clean_name_over_macos_space_number(conn):
    # macOS names duplicates "_N4A5939 2.jpg". Seen 10,809 times on P:.
    _add(conn, r"P:\family_Randoms\_N4A5939 3.jpg", mtime=1.0)
    _add(conn, r"P:\family_Randoms\_N4A5939 2.jpg", mtime=2.0)
    _add(conn, r"P:\family_Randoms\_N4A5939.jpg", mtime=999.0)
    assert dupes.choose_keeper(_rows(conn))["path"].endswith("_N4A5939.jpg")


def test_three_digit_catalog_names_are_not_treated_as_copies():
    # "Family 158.jpg" is a real catalog name on P:, not a copy of "Family".
    # Restricting to 1-2 digits keeps macOS copies without eating these.
    assert dupes._is_copy("Family 158.jpg") is False
    assert dupes._is_copy("Crowfam 2011.jpg") is False
    assert dupes._is_copy("_N4A5939 2.jpg") is True
    assert dupes._is_copy("_N4A5939 12.jpg") is True


def test_rejects_folder_loses_to_backup_folder(conn):
    # Both are "demoted", but they are not equally bad. A survivor must never
    # be left in the rejects folder when an organised copy exists - Paul may
    # delete that folder wholesale, and it would take the survivors with it.
    _add(conn, r"P:\0 and 1 star photos originals and dupes\1 star\x.tif",
         top="0 and 1 star photos originals and dupes", rel="1 star")
    _add(conn, r"P:\Backup Consolidated Photos\2025\Argentina\x.tif",
         top="Backup Consolidated Photos", rel=r"2025\Argentina")
    keeper = dupes.choose_keeper(_rows(conn))
    assert keeper["top_folder"] == "Backup Consolidated Photos"


def test_rejects_rank_is_worse_than_backup_rank(conn):
    _add(conn, r"P:\0 and 1 star photos originals and dupes\a.jpg",
         top="0 and 1 star photos originals and dupes")
    _add(conn, r"P:\Backup Consolidated Photos\b.jpg",
         top="Backup Consolidated Photos")
    _add(conn, r"P:\family_Randoms\c.jpg", top="family_Randoms")
    ranks = {r["top_folder"]: dupes._demoted(r) for r in _rows(conn)}
    assert ranks["family_Randoms"] == 0
    assert ranks["Backup Consolidated Photos"] == 1
    assert ranks["0 and 1 star photos originals and dupes"] == 2


def test_undated_shallow_reject_still_loses_to_deep_organised(conn):
    # The exact live case: shallow path in the rejects folder was winning.
    _add(conn, r"P:\0 and 1 star photos originals and dupes\1 star\XT5A0756-Edit.tif",
         top="0 and 1 star photos originals and dupes", rel="1 star")
    _add(conn, r"P:\Backup Consolidated Photos\2025\Imported\Argentina\Buenos Aires\XT5A0756-Edit.tif",
         top="Backup Consolidated Photos", rel=r"2025\Imported\Argentina\Buenos Aires")
    assert dupes.choose_keeper(_rows(conn))["top_folder"] == "Backup Consolidated Photos"
