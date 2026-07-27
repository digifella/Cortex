"""Tests for offline reverse geocoding used by photo location enrichment."""
import pytest

from cortex_engine import offline_geocoder


class TestResolveModes:
    def test_online_mode_uses_online_fn(self):
        called = {}

        def fake_online(lat, lon):
            called["hit"] = (lat, lon)
            return {"city": "Brisbane", "state": "Queensland", "country": "Australia"}

        loc, source = offline_geocoder.resolve(-27.5, 153.0, mode="online",
                                               online_fn=fake_online)
        assert source == "online"
        assert loc["city"] == "Brisbane"
        assert called["hit"] == (-27.5, 153.0)

    def test_online_mode_without_fn_returns_empty(self):
        loc, source = offline_geocoder.resolve(-27.5, 153.0, mode="online", online_fn=None)
        assert source == "online"
        assert loc == {"country": "", "state": "", "city": ""}

    def test_offline_mode_never_calls_online(self):
        def boom(lat, lon):
            raise AssertionError("online lookup must not run in offline mode")

        loc, source = offline_geocoder.resolve(-27.5, 153.0, mode="offline", online_fn=boom)
        assert source == "offline"

    def test_auto_prefers_online_when_it_returns_a_result(self):
        def fake_online(lat, lon):
            return {"city": "Brisbane", "state": "Queensland", "country": "Australia"}

        loc, source = offline_geocoder.resolve(-27.5, 153.0, mode="auto",
                                               online_fn=fake_online)
        assert source == "online"
        assert loc["city"] == "Brisbane"

    def test_auto_falls_back_when_online_raises(self):
        def broken_online(lat, lon):
            raise ConnectionError("no network")

        loc, source = offline_geocoder.resolve(-27.5, 153.0, mode="auto",
                                               online_fn=broken_online)
        assert source in {"offline", "none"}

    def test_auto_falls_back_when_online_returns_empty(self):
        def empty_online(lat, lon):
            return {"country": "", "state": "", "city": ""}

        loc, source = offline_geocoder.resolve(-27.5, 153.0, mode="auto",
                                               online_fn=empty_online)
        assert source in {"offline", "none"}


class TestDescribeMode:
    def test_online_mode_never_warns(self):
        # Online mode does not need the offline dataset.
        assert offline_geocoder.describe_mode("online") is None


@pytest.mark.skipif(not offline_geocoder.is_available(),
                    reason="reverse_geocoder/pycountry not installed")
class TestOfflineLookup:
    """Integration tests against the bundled GeoNames dataset."""

    def test_resolves_australian_coordinates(self):
        loc = offline_geocoder.reverse_geocode_offline(-27.4846, 152.9948)
        assert loc["country"] == "Australia"
        assert loc["state"] == "Queensland"
        assert loc["city"]  # a nearby suburb — exact name is dataset-dependent

    def test_returns_expected_shape(self):
        loc = offline_geocoder.reverse_geocode_offline(-33.8688, 151.2093)
        assert set(loc) == {"country", "state", "city"}

    def test_invalid_coordinates_degrade_gracefully(self):
        loc = offline_geocoder.reverse_geocode_offline("not-a-number", None)
        assert loc == {"country": "", "state": "", "city": ""}
