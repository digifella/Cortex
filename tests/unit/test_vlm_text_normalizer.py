"""Tests for _normalize_vlm_text — stripping model reasoning from descriptions.

Local vision models narrate their analysis before answering. The pipeline
scrubs that post-hoc, because prompting alone does not reliably stop it.
These cases come from real qwen3-vl:8b and llava:7b output.
"""
from cortex_engine.textifier import DocumentTextifier as DT


class TestReasoningHandoff:
    def test_strips_so_colon_handoff(self):
        raw = ("The main thing is a dark cocktail in a glass with salt rim, ice, "
               "and a jalapeño slice. So: A dark cocktail in a salt-rimmed glass "
               "holds ice cubes and a jalapeño slice.")
        out = DT._normalize_vlm_text(raw)
        assert out.startswith("A dark cocktail in a salt-rimmed glass")
        assert "The main thing is" not in out

    def test_strips_main_thing_without_handoff(self):
        raw = "The main thing is a surfer at dawn. A lone surfer walks along dark sand."
        assert DT._normalize_vlm_text(raw) == "A lone surfer walks along dark sand."

    def test_strips_what_stands_out(self):
        raw = "What stands out is the colour. A red door frames the entrance."
        assert DT._normalize_vlm_text(raw) == "A red door frames the entrance."


class TestFalsePositiveGuards:
    """The strip rules must not eat ordinary description."""

    def test_mid_sentence_so_survives(self):
        raw = "Two women stand smiling, so, they appear happy together."
        assert DT._normalize_vlm_text(raw) == raw

    def test_descriptive_foreground_survives(self):
        raw = "a group of people dining in a restaurant. There is a man in the foreground."
        assert DT._normalize_vlm_text(raw) == raw

    def test_clean_prose_untouched(self):
        raw = "A dark cocktail sits garnished with cucumber on a bar top."
        assert DT._normalize_vlm_text(raw) == raw

    def test_sentence_containing_also_survives(self):
        raw = "A surfer rides a wave. Also visible is a distant headland."
        assert DT._normalize_vlm_text(raw) == raw


class TestExistingBehaviourStillWorks:
    def test_strips_think_tags(self):
        raw = "<think>hmm let me look</think>A beach at dawn."
        assert DT._normalize_vlm_text(raw) == "A beach at dawn."

    def test_strips_leading_filler(self):
        assert DT._normalize_vlm_text("Okay, A beach at dawn.") == "A beach at dawn."

    def test_strips_self_directive(self):
        raw = "Focus on the subject. A lone surfer walks the sand."
        assert DT._normalize_vlm_text(raw) == "A lone surfer walks the sand."

    def test_strips_description_label(self):
        assert DT._normalize_vlm_text("Description: A beach at dawn.") == "A beach at dawn."

    def test_empty_input(self):
        assert DT._normalize_vlm_text("") == ""
