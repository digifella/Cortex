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
