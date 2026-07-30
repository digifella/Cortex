"""Tests for placeholder detection and caption provenance.

A placeholder is a failure marker, not a caption. Treating one as success is how
26 photos ended up with "[Image: vision model returned empty description]" as
their permanent caption.
"""
import subprocess
from unittest.mock import patch

from cortex_engine.textifier import DocumentTextifier as DT


class TestPlaceholderDetection:
    def test_empty_description_placeholder(self):
        assert DT.is_placeholder_description("[Image: vision model returned empty description]")

    def test_unavailable_placeholder(self):
        assert DT.is_placeholder_description("[Image: could not be described — vision model unavailable]")

    def test_logo_placeholder_counts_as_retryable(self):
        # A stronger model often describes a photo the small one dismissed.
        assert DT.is_placeholder_description("[Image: logo/icon omitted]")

    def test_real_caption_is_not_a_placeholder(self):
        assert not DT.is_placeholder_description("A surfer walks the dark sand at dawn.")

    def test_caption_mentioning_image_is_not_a_placeholder(self):
        assert not DT.is_placeholder_description("An image of a surfer hangs on the wall.")

    def test_empty_and_none(self):
        assert not DT.is_placeholder_description("")
        assert not DT.is_placeholder_description(None)

    def test_leading_whitespace_tolerated(self):
        assert DT.is_placeholder_description("  [Image: logo/icon omitted]")


class TestCaptionProvenance:
    def test_writes_both_iptc_and_xmp_fields(self, tmp_path):
        target = tmp_path / "photo.jpg"
        target.write_bytes(b"stub")
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            result = DT.write_caption_provenance(str(target), "gemma4:e2b-it-qat")
        assert result["success"]
        args = run.call_args[0][0]
        assert any("IPTC:Writer-Editor=" in a for a in args)
        assert any("XMP-photoshop:CaptionWriter=" in a for a in args)

    def test_value_names_model_and_version(self, tmp_path):
        target = tmp_path / "photo.jpg"
        target.write_bytes(b"stub")
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "", "")
            DT.write_caption_provenance(str(target), "qwen3-vl:8b")
        written = [a for a in run.call_args[0][0] if a.startswith("-IPTC:Writer-Editor=")][0]
        assert "qwen3-vl:8b" in written
        assert "Cortex" in written

    def test_empty_model_is_a_noop(self, tmp_path):
        result = DT.write_caption_provenance(str(tmp_path / "x.jpg"), "")
        assert result["success"] and result["fields_written"] == 0

    def test_exiftool_failure_is_reported(self, tmp_path):
        target = tmp_path / "photo.jpg"
        target.write_bytes(b"stub")
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, "", "boom")
            result = DT.write_caption_provenance(str(target), "llava:7b")
        assert not result["success"]
        assert "boom" in result["message"]
