"""Tests for photo_dedup — the enrichment/dedup pass over poorly-scanned folders.

These pin the three faults found on the 1993-1998 Pre-Dig run (2026-08-17), each
of which wrote bad data to Lightroom masters or exports.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import photo_dedup  # noqa: E402


def write_state(directory: Path, state: dict, links: dict = None, groups: list = None):
    (directory / photo_dedup.CHECKPOINT).write_text(json.dumps(state))
    if links is not None:
        (directory / photo_dedup.LINKS).write_text(json.dumps(links))
    if groups is not None:
        (directory / photo_dedup.GROUPS).write_text(json.dumps(groups))


class Args:
    def __init__(self, export_dir, apply=False):
        self.export_dir = export_dir
        self.apply = apply


class TestDuplicateMarking:
    """An enrichment-only run has no `group` pass, so nothing is a keeper.

    Deducing "duplicate" from "no keeper claims this" flagged every catalog
    master — 343 of them on the Pre-Dig run — and the user's workflow is to
    filter on `Duplicate` and move those photos out of the catalog.
    """

    def test_enrichment_only_run_marks_nothing(self, tmp_path, capsys):
        (tmp_path / "a.jpg").write_bytes(b"x")
        write_state(
            tmp_path,
            {"a.jpg": {"phash": "0" * 16, "caption": "a photo", "keywords": ["beach"]}},
            links={"a.jpg": {"catalog": "cat.tif", "distance": 0,
                             "paths": [str(tmp_path / "cat.tif")]}},
            # no groups file at all — the enrichment-only shape
        )
        photo_dedup.cmd_apply(Args(tmp_path))
        out = capsys.readouterr().out
        assert "(0 to mark Duplicate)" in out
        assert "DRY RUN" in out

    def test_known_duplicate_is_still_marked(self, tmp_path, capsys):
        for n in ("keep.jpg", "dup.jpg"):
            (tmp_path / n).write_bytes(b"x")
        write_state(
            tmp_path,
            {"keep.jpg": {"phash": "0" * 16, "caption": "kept"},
             "dup.jpg": {"phash": "0" * 16, "caption": "dupe"}},
            links={"keep.jpg": {"catalog": "k.tif", "distance": 0,
                                "paths": [str(tmp_path / "k.tif")]},
                   "dup.jpg": {"catalog": "d.tif", "distance": 0,
                               "paths": [str(tmp_path / "d.tif")]}},
            groups=[{"keeper": "keep.jpg", "duplicates": ["dup.jpg"]}],
        )
        photo_dedup.cmd_apply(Args(tmp_path))
        assert "(1 to mark Duplicate)" in capsys.readouterr().out


class TestPlaceResolution:
    def test_gps_beats_haiku_landmark_guess(self):
        rec = {"city": "Hoi An", "country": "Vietnam",
               "geo": {"city": "Melbourne", "state": "Victoria", "country": "Australia"}}
        assert photo_dedup.place_of(rec) == {
            "city": "Melbourne", "state": "Victoria", "country": "Australia"}

    def test_falls_back_to_landmark_when_no_gps_fix(self):
        rec = {"city": "Hoi An", "country": "Vietnam",
               "geo": {"city": "", "state": "", "country": ""}}
        place = photo_dedup.place_of(rec)
        assert place["city"] == "Hoi An" and place["country"] == "Vietnam"

    def test_no_gps_is_tagged_nogps(self):
        assert photo_dedup.place_keywords({"geo": None}) == ["nogps"]

    def test_geo_stage_not_run_yields_no_tags(self):
        assert photo_dedup.place_keywords({"city": "Rome"}) == []

    def test_place_becomes_lowercase_keywords(self):
        rec = {"geo": {"city": "Gold Coast", "state": "Queensland", "country": "Australia"}}
        assert photo_dedup.place_keywords(rec) == ["gold coast", "queensland", "australia"]


class TestJunkDescriptions:
    """Scanner/upscaler boilerplate is long enough to pass the "already
    captioned" gate, so it survived as a caption and reached the masters."""

    @pytest.mark.parametrize("text", [
        "File written by Adobe Photoshop\xa8 5.2",
        "Upscaled with Gigapixel v8.4.4. 1600x1270 => 3200x2540 (2x)",
        "Scanned with LiDE 210",
    ])
    def test_boilerplate_is_not_a_caption(self, text):
        assert photo_dedup.JUNK_DESC.match(text)

    def test_real_caption_is_kept(self):
        assert not photo_dedup.JUNK_DESC.match(
            "Paul and Jacqui stand together on a stone bridge railing.")


class TestCityChain:
    """textifier.reverse_geocode stops at city/town/village/suburb, which left
    103 of 400 rural photos with no place at all."""

    def test_city_district_is_used_when_no_city(self, monkeypatch):
        class Loc:
            raw = {"address": {"city_district": "Wilsons Promontory",
                               "state": "Victoria", "country": "Australia"}}

        monkeypatch.setattr(photo_dedup, "geocode", photo_dedup.geocode)
        import geopy.geocoders

        class FakeNominatim:
            def __init__(self, **kw):
                pass

            def reverse(self, *a, **kw):
                return Loc()

        monkeypatch.setattr(geopy.geocoders, "Nominatim", FakeNominatim)
        assert photo_dedup.geocode(-39.03, 146.32) == {
            "country": "Australia", "state": "Victoria", "city": "Wilsons Promontory"}
