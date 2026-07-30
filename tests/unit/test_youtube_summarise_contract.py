"""Contract for youtube_summarise job input.

The handler at worker/handlers/youtube_summarise.py imports
validate_youtube_summarise_input at module scope, so a missing or broken
validator takes the whole worker.handlers package down at import time — not
just this job.
"""
import pytest

from cortex_engine import handoff_contract as hc


def test_youtube_summarise_is_a_supported_job_type():
    assert "youtube_summarise" in hc.SUPPORTED_JOB_TYPES


class TestUrls:
    def test_urls_are_required(self):
        with pytest.raises(ValueError, match="urls"):
            hc.validate_youtube_summarise_input({})

    def test_blank_urls_are_stripped_and_rejected_when_nothing_remains(self):
        with pytest.raises(ValueError, match="urls"):
            hc.validate_youtube_summarise_input({"urls": ["  ", ""]})

    def test_urls_are_whitespace_normalised(self):
        out = hc.validate_youtube_summarise_input(
            {"urls": ["  https://youtu.be/abc  ", ""]})
        assert out["urls"] == ["https://youtu.be/abc"]


class TestApiChoice:
    def test_defaults_to_gemini_flash(self):
        out = hc.validate_youtube_summarise_input({"urls": ["u"]})
        assert out["api_choice"] == "gemini-flash"

    def test_unknown_api_is_rejected(self):
        with pytest.raises(ValueError, match="api_choice"):
            hc.validate_youtube_summarise_input({"urls": ["u"], "api_choice": "gpt-9"})


class TestOutputModes:
    def test_defaults_to_summary(self):
        out = hc.validate_youtube_summarise_input({"urls": ["u"]})
        assert out["output_modes"] == ["summary"]

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="output_modes"):
            hc.validate_youtube_summarise_input(
                {"urls": ["u"], "output_modes": ["haiku"]})

    def test_duplicates_are_collapsed_preserving_order(self):
        out = hc.validate_youtube_summarise_input(
            {"urls": ["u"], "output_modes": ["timestamps", "summary", "timestamps"]})
        assert out["output_modes"] == ["timestamps", "summary"]


class TestClipWindow:
    def test_end_must_be_after_start(self):
        with pytest.raises(ValueError, match="end_time_seconds"):
            hc.validate_youtube_summarise_input({
                "urls": ["u"],
                "youtube_options": {"start_time_seconds": 90, "end_time_seconds": 30},
            })

    def test_overlap_must_be_smaller_than_chunk(self):
        with pytest.raises(ValueError, match="chunk_overlap_seconds"):
            hc.validate_youtube_summarise_input({
                "urls": ["u"],
                "youtube_options": {"chunk_duration_seconds": 60,
                                    "chunk_overlap_seconds": 60},
            })

    def test_negative_offsets_are_rejected(self):
        with pytest.raises(ValueError, match="start_time_seconds"):
            hc.validate_youtube_summarise_input({
                "urls": ["u"],
                "youtube_options": {"start_time_seconds": -1},
            })

    def test_options_default_to_zero(self):
        out = hc.validate_youtube_summarise_input({"urls": ["u"]})
        assert out["youtube_options"] == {
            "start_time_seconds": 0,
            "end_time_seconds": 0,
            "chunk_duration_seconds": 0,
            "chunk_overlap_seconds": 0,
        }
