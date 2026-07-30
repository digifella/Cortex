"""Tests for VRAM-adaptive vision model selection.

The same install runs on an 8GB laptop and a 48GB workstation. Selection must
adapt: too large a model silently spills to CPU (10-100x slower) or crashes the
Ollama runner outright.
"""
import pytest

from cortex_engine import vision_model_selector as sel

LAPTOP_MODELS = ["gemma4:e2b-it-qat", "qwen3-vl:8b", "qwen3-vl:4b", "llava:7b"]


class TestSelectionAdaptsToVram:
    def test_workstation_picks_highest_quality(self):
        model, _ = sel.select_vision_model(available=LAPTOP_MODELS, free_mb=48000)
        assert model == "qwen3-vl:8b"  # best measured content accuracy

    def test_workstation_prefers_32b_when_installed(self):
        model, _ = sel.select_vision_model(
            available=LAPTOP_MODELS + ["qwen3-vl:32b"], free_mb=48000)
        assert model == "qwen3-vl:32b"

    def test_laptop_with_lightroom_open_picks_small_model(self):
        model, _ = sel.select_vision_model(available=LAPTOP_MODELS, free_mb=4450)
        assert model == "gemma4:e2b-it-qat"

    def test_laptop_free_still_rejects_model_that_needs_headroom(self):
        # qwen3-vl:8b is 7400MB; with 1024MB headroom it does not fit 7959MB.
        # This is the configuration that crashed the runner in real use.
        model, _ = sel.select_vision_model(available=LAPTOP_MODELS, free_mb=7959)
        assert model != "qwen3-vl:8b"

    def test_headroom_is_configurable(self):
        model, _ = sel.select_vision_model(
            available=LAPTOP_MODELS, free_mb=7959, headroom_mb=0)
        assert model == "qwen3-vl:8b"


class TestDegradation:
    def test_no_models_installed(self):
        model, reason = sel.select_vision_model(available=[], free_mb=8000)
        assert model is None
        assert "no ollama models" in reason.lower()

    def test_unknown_models_only(self):
        model, reason = sel.select_vision_model(available=["mistral:latest"], free_mb=8000)
        assert model is None
        assert "known vision model" in reason

    def test_nothing_fits_falls_back_to_smallest_with_warning(self):
        model, reason = sel.select_vision_model(available=LAPTOP_MODELS, free_mb=512)
        assert model == "gemma4:e2b-it-qat"
        assert "CPU spill" in reason
        assert "Lightroom" in reason  # actionable: tells the user what to close

    def test_no_gpu_picks_smallest(self):
        model, reason = sel.select_vision_model(available=LAPTOP_MODELS, free_mb=None)
        # free_mb=None triggers a live probe; on a CPU-only host that yields None
        assert model in {p.name for p in sel.VISION_MODEL_PROFILES}
        assert reason


class TestProfiles:
    def test_reasoning_models_get_large_budgets(self):
        for name in ("qwen3-vl:8b", "qwen3-vl:4b", "gemma4:e2b-it-qat"):
            assert sel.num_predict_for(name) == 640, name

    def test_instruct_models_get_small_budgets(self):
        assert sel.num_predict_for("llava:7b") == 160

    def test_unknown_model_uses_default(self):
        assert sel.num_predict_for("nope:1b", default=123) == 123

    def test_profiles_are_quality_ranked_consistently(self):
        by_name = sel.PROFILES_BY_NAME
        # qwen3-vl:8b read the jalapeno garnish correctly; llava called it a pickle.
        assert by_name["qwen3-vl:8b"].quality > by_name["llava:7b"].quality
        # gemma4 e2b beat llava on clean rate (5/6 vs 2/6) at a third the VRAM.
        assert by_name["gemma4:e2b-it-qat"].quality > by_name["llava:7b"].quality

    def test_vram_figures_are_resident_not_download_size(self):
        # `ollama list` reports 6.1GB for qwen3-vl:8b; it is 7.4GB resident.
        assert sel.PROFILES_BY_NAME["qwen3-vl:8b"].vram_mb == 7400


class TestProbes:
    def test_free_vram_returns_int_or_none(self):
        v = sel.free_vram_mb()
        assert v is None or isinstance(v, int)

    def test_installed_models_returns_list(self):
        assert isinstance(sel.installed_models(), list)

    def test_describe_selection_is_a_string(self):
        assert isinstance(sel.describe_selection(), str)


class TestMeasured32bFootprint:
    """qwen3-vl:32b is resident at 24GB, not the 22GB originally profiled.

    Measured on the RTX 8000 via `ollama ps` during inference (ctx 32768):
    `qwen3-vl:32b  24 GB  100% GPU`. At the old 22000 figure the selector would
    green-light the model with ~23GB free, leaving it sharing a 46GB card with
    LM Studio's 35b — 45.2GB of 46GB used and only ~424MB spare, which is where
    the Ollama runner crashes rather than degrades.
    """

    def test_32b_is_rejected_when_only_the_old_estimate_would_fit(self):
        # 23.5GB free clears the old 22000+1024 bar but not the measured one.
        model, _ = sel.select_vision_model(
            available=["qwen3-vl:32b", "qwen3-vl:8b"], free_mb=23500)
        assert model == "qwen3-vl:8b"

    def test_32b_is_chosen_when_it_genuinely_fits(self):
        model, _ = sel.select_vision_model(
            available=["qwen3-vl:32b", "qwen3-vl:8b"], free_mb=26000)
        assert model == "qwen3-vl:32b"
