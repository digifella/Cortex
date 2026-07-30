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


class TestNumPredictIsModelAware:
    """Reasoning models need headroom to reach an answer; instruct models need a cap.

    A single global value regressed one class or the other: at 140 qwen3-vl spent
    the whole budget thinking and returned nothing; at 512 llava overran the
    35-word style limit (36 words avg vs 25).
    """

    def test_reasoning_model_gets_headroom(self):
        assert DT._num_predict_for("qwen3-vl:8b") == 640
        assert DT._num_predict_for("qwen3-vl:4b") == 640

    def test_instruct_models_get_a_tight_cap(self):
        for model in ("llava:7b", "minicpm-v:8b", "qwen2.5vl:7b", "gemma3:4b"):
            assert DT._num_predict_for(model) == 160, model

    def test_gemma4_is_a_reasoning_model_despite_the_it_naming(self):
        # "-it-qat" reads as instruction-tuned, but Gemma 4 thinks before
        # answering: at 160 tokens it produced 635 chars of reasoning and no
        # caption. gemma3 and earlier are plain instruct models.
        assert DT._num_predict_for("gemma4:e2b-it-qat") == 640
        assert DT._num_predict_for("gemma4:e4b-it-qat") == 640
        assert DT._num_predict_for("gemma3:4b") == 160

    def test_unknown_model_gets_middle_default(self):
        assert DT._num_predict_for("something-new:3b") == 200

    def test_matching_is_case_insensitive(self):
        assert DT._num_predict_for("QWEN3-VL:8B") == 640

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("CORTEX_VLM_NUM_PREDICT", "999")
        assert DT._num_predict_for("llava:7b") == 999

    def test_non_numeric_override_ignored(self, monkeypatch):
        monkeypatch.setenv("CORTEX_VLM_NUM_PREDICT", "lots")
        assert DT._num_predict_for("llava:7b") == 160

    def test_empty_model_name(self):
        assert DT._num_predict_for("") == 200
