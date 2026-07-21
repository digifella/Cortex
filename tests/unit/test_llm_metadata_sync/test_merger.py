from cortex_engine.llm_metadata_sync import merger
from cortex_engine.llm_metadata_sync.merger import build_keyword_union, read_existing_keywords


def test_read_existing_keywords_coerces_numeric_keywords_to_str(monkeypatch, tmp_path):
    """exiftool can return a purely-numeric keyword (e.g. a year like 2005) as a
    JSON int rather than a string; must not crash on .strip()."""
    target = tmp_path / "2005-08-26 17-56-19--4.tif"
    target.touch()
    monkeypatch.setattr(merger.shutil, "which", lambda name: "/usr/bin/exiftool")

    class FakeResult:
        returncode = 0
        stdout = '[{"Subject": [2005, "London", "family"]}]'

    monkeypatch.setattr(merger.subprocess, "run", lambda *a, **k: FakeResult())
    assert read_existing_keywords(target) == ["2005", "London", "family"]


def test_existing_keywords_appear_first():
    result = build_keyword_union(["B", "A"], ["C"], filter_list=[])
    assert result == ["B", "A", "C"]


def test_new_keywords_appended_in_original_order():
    result = build_keyword_union([], ["Z", "A", "M"], filter_list=[])
    assert result == ["Z", "A", "M"]


def test_duplicates_deduplicated_first_seen_wins():
    result = build_keyword_union(["A", "B"], ["B", "C"], filter_list=[])
    assert result == ["A", "B", "C"]


def test_filter_removes_matching_keyword():
    result = build_keyword_union(["nogps", "Melbourne"], ["Beach"], filter_list=["nogps"])
    assert "nogps" not in result
    assert "Melbourne" in result
    assert "Beach" in result


def test_filter_is_case_insensitive_against_existing():
    result = build_keyword_union(["NOGPS", "Moon"], [], filter_list=["nogps"])
    assert "NOGPS" not in result
    assert "Moon" in result


def test_filter_is_case_insensitive_from_new():
    result = build_keyword_union([], ["NoGPS", "Star"], filter_list=["nogps"])
    assert "NoGPS" not in result
    assert "Star" in result


def test_empty_inputs_return_empty():
    assert build_keyword_union([], [], filter_list=[]) == []


def test_empty_existing():
    assert build_keyword_union([], ["A", "B"], filter_list=[]) == ["A", "B"]


def test_empty_new():
    assert build_keyword_union(["A", "B"], [], filter_list=[]) == ["A", "B"]


def test_case_sensitive_dedup():
    # "Melbourne" and "melbourne" are distinct keywords
    result = build_keyword_union(["Melbourne"], ["melbourne"], filter_list=[])
    assert "Melbourne" in result
    assert "melbourne" in result
