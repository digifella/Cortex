"""Tests for the VLM keyword hint stop-list.

The hint injects existing EXIF keywords into the prompt as "Known subjects".
Catalog bookkeeping tags describe the *record*, not the picture, and actively
mislead the model: `icon` / `building_icon` / `artwork` are what made both
Haiku and the local VLM answer "[Image: logo/icon omitted]" for 16 ordinary
photographs (Sydney at dawn, NYC, Venice, Crete).
"""
from cortex_engine.textifier import DocumentTextifier as DT


class TestKeywordHintStopList:
    def test_icon_is_excluded(self):
        hint = DT._build_keyword_hint(["sydney", "icon", "dawn"])
        assert "icon" not in hint
        assert "sydney" in hint and "dawn" in hint

    def test_suffixed_icon_tag_is_excluded(self):
        hint = DT._build_keyword_hint(["building_icon", "harbour"])
        assert "building_icon" not in hint
        assert "harbour" in hint

    def test_artwork_is_excluded(self):
        assert "artwork" not in DT._build_keyword_hint(["venice", "artwork"])

    def test_workflow_tags_are_excluded(self):
        hint = DT._build_keyword_hint(["adobe_cloud_sync", "exclude_temp", "crete"])
        assert "adobe_cloud_sync" not in hint and "exclude_temp" not in hint
        assert "crete" in hint

    def test_case_is_ignored(self):
        assert "Icon" not in DT._build_keyword_hint(["Icon", "milan"])

    def test_known_subjects_omitted_when_all_keywords_are_noise(self):
        hint = DT._build_keyword_hint(["icon", "artwork"])
        assert "Known subjects" not in hint

    def test_location_still_emitted_when_all_keywords_are_noise(self):
        hint = DT._build_keyword_hint(["icon"], location={"city": "Milan", "country": "Italy"})
        assert "Milan" in hint and "Italy" in hint
        assert "Known subjects" not in hint

    def test_noise_does_not_consume_the_twenty_tag_budget(self):
        # 5 noise tags then 20 real ones: all 20 real tags must survive.
        kws = ["icon", "artwork", "exclude_temp", "adobe_cloud_sync", "building_icon"]
        kws += [f"subject{i}" for i in range(20)]
        hint = DT._build_keyword_hint(kws)
        for i in range(20):
            assert f"subject{i}" in hint

    def test_real_subjects_are_untouched(self):
        hint = DT._build_keyword_hint(["stonehenge", "jacqui_c", "greece"])
        assert "stonehenge" in hint and "jacqui_c" in hint and "greece" in hint

    def test_empty_input_returns_empty_string(self):
        assert DT._build_keyword_hint([]) == ""
