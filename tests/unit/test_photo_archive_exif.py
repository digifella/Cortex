# tests/unit/test_photo_archive_exif.py
from scripts.photo_archive import exif


def test_prefers_subsec_over_datetimeoriginal():
    dt, src = exif.pick_datetime({
        "SubSecDateTimeOriginal": "2019:01:01 15:41:12.45",
        "DateTimeOriginal": "2019:01:01 15:41:12",
        "CreateDate": "2001:01:01 00:00:00",
    })
    assert dt == "2019-01-01T15:41:12"
    assert src == "SubSecDateTimeOriginal"


def test_falls_back_to_createdate():
    dt, src = exif.pick_datetime({"CreateDate": "2005:07:04 08:09:10"})
    assert dt == "2005-07-04T08:09:10"
    assert src == "CreateDate"


def test_zero_date_is_rejected():
    # exiftool returns this for files with a null date field; it is not a date.
    dt, src = exif.pick_datetime({"DateTimeOriginal": "0000:00:00 00:00:00"})
    assert dt is None and src is None


def test_missing_tags_give_none():
    assert exif.pick_datetime({}) == (None, None)


def test_malformed_date_is_rejected():
    assert exif.pick_datetime({"DateTimeOriginal": "not a date"}) == (None, None)


def test_model_sanitised_for_filesystem():
    assert exif.sanitise_model("X-T5") == "X-T5"
    assert exif.sanitise_model("HP pstc5200") == "HP pstc5200"
    assert exif.sanitise_model("Canon/EOS:60D") == "Canon-EOS-60D"
    assert exif.sanitise_model("  X100V  ") == "X100V"


def test_empty_model_returns_empty_string():
    assert exif.sanitise_model("") == ""
    assert exif.sanitise_model(None) == ""


import pytest
from scripts.photo_archive import db


class _FakeReader:
    """Stands in for ExifReader so tests never spawn exiftool."""
    def __init__(self, mapping):
        self.mapping = mapping
        self.closed = False
        self.batches = []

    def read_many(self, paths):
        self.batches.append(list(paths))
        return {p: self.mapping.get(p, {}) for p in paths}

    def close(self):
        self.closed = True


@pytest.fixture
def idx():
    c = db.connect(":memory:")
    db.init_schema(c)
    return c


def _add(conn, path, kind="image", ext=".jpg"):
    return db.upsert_file(conn, path=path, top_folder="t", rel_dir="",
                          filename=path.split("\\")[-1], ext=ext, size=1,
                          mtime=1.0, kind=kind)


def test_norm_matches_exiftool_forward_slash_reporting():
    # exiftool echoes SourceFile with forward slashes; our paths use backslashes.
    assert exif._norm(r"P:\a\b.jpg") == exif._norm("P:/a/b.jpg")
    assert exif._norm("\\\\?\\P:\\a\\b.jpg") == exif._norm("P:/a/b.jpg")


def test_read_exif_writes_date_and_model(idx):
    _add(idx, r"P:\a\b.jpg")
    reader = _FakeReader({r"P:\a\b.jpg": {"DateTimeOriginal": "2019:03:04 10:11:12",
                                          "Model": "X-T5"}})
    stats = exif.read_exif_into_index(idx, reader=reader)
    row = idx.execute("SELECT exif_dt, exif_dt_source, camera_model, state "
                      "FROM files").fetchone()
    assert row["exif_dt"] == "2019-03-04T10:11:12"
    assert row["exif_dt_source"] == "DateTimeOriginal"
    assert row["camera_model"] == "X-T5"
    assert row["state"] == "exif_read"
    assert stats["dated"] == 1


def test_undated_row_is_still_marked_read(idx):
    # Otherwise every run re-reads them forever and the stage never finishes.
    _add(idx, r"P:\a\b.jpg")
    stats = exif.read_exif_into_index(idx, reader=_FakeReader({}))
    row = idx.execute("SELECT exif_dt, state FROM files").fetchone()
    assert row["exif_dt"] is None
    assert row["state"] == "exif_read"
    assert stats["undated"] == 1


def test_second_run_reads_nothing(idx):
    _add(idx, r"P:\a\b.jpg")
    exif.read_exif_into_index(idx, reader=_FakeReader({}))
    stats = exif.read_exif_into_index(idx, reader=_FakeReader({}))
    assert stats["read"] == 0


def test_sidecars_and_other_files_are_not_read(idx):
    _add(idx, r"P:\a\b.xmp", kind="sidecar", ext=".xmp")
    _add(idx, r"P:\a\b.txt", kind="other", ext=".txt")
    stats = exif.read_exif_into_index(idx, reader=_FakeReader({}))
    assert stats["read"] == 0


def test_exiftool_args_strip_long_path_prefix():
    """exiftool 12.85 REJECTS \\\\?\\ paths - it reads '?' as a wildcard and
    returns "Wildcards don't work in the directory specification".

    walk.py stores every Windows path with that prefix, so passing stored paths
    straight through makes EVERY exif read fail silently: read_many returns {},
    every photo is marked undated, and the whole archive routes to _UNDATED
    with no error. Verified against real exiftool on Windows 2026-08-21.
    """
    args = exif._exiftool_args(["\\\\?\\P:\\a\\b.jpg", "P:\\c\\d.jpg"])
    assert not any(a.startswith("\\\\?\\") for a in args)
    assert args[-2:] == ["P:\\a\\b.jpg", "P:\\c\\d.jpg"]


def test_exiftool_args_include_all_date_tags():
    args = exif._exiftool_args(["x.jpg"])
    for tag in ("-DateTimeOriginal", "-SubSecDateTimeOriginal", "-CreateDate",
                "-Model", "-json", "-fast2"):
        assert tag in args
