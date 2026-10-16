"""LM Studio vision provider in DocumentTextifier.

The provider exists to reuse a VLM that is ALREADY loaded in LM Studio, instead
of making Ollama evict it to load a smaller model into leftover VRAM. It must
degrade to "" (so the caller falls through to Ollama) whenever LM Studio is
unreachable, has no vision model loaded, or errors.
"""

import io
import pytest

from cortex_engine import lmstudio_vision
from cortex_engine.textifier import DocumentTextifier

LOADED_VLM = {"id": "qwen/qwen3.6-35b-a3b", "type": "vlm", "state": "loaded"}
IDLE_VLM = {"id": "google/gemma-4-31b", "type": "vlm", "state": "not-loaded"}
LOADED_LLM = {"id": "qwen3-coder-30b", "type": "llm", "state": "loaded"}
V1_QWEN = {"key": "qwen/qwen3.6-35b-a3b", "type": "llm",
           "capabilities": {"vision": True},
           "loaded_instances": [{"id": "qwen/qwen3.6-35b-a3b"}]}


@pytest.fixture
def textifier(monkeypatch):
    # auto_select_vision shells out to `ollama list` and nvidia-smi; off for tests.
    monkeypatch.delenv("CORTEX_LMSTUDIO_VISION_MODEL", raising=False)
    monkeypatch.setenv("CORTEX_LMSTUDIO_BASE_URL", "http://testhost:1234/v1")
    return DocumentTextifier(use_vision=True, auto_select_vision=False)


def _mock_models(monkeypatch, *, v1=(), v0=(), recorder=None):
    def fake(path):
        if recorder is not None:
            recorder.append(path)
        return {"models": list(v1)} if path.endswith("v1/models") else {"data": list(v0)}

    monkeypatch.setattr(lmstudio_vision, "_models", fake)


def test_picks_the_loaded_vision_model(textifier, monkeypatch):
    _mock_models(monkeypatch, v1=[V1_QWEN])
    assert textifier._lmstudio_loaded_vlm() == "qwen/qwen3.6-35b-a3b"


def test_ignores_vision_models_that_are_not_loaded(textifier, monkeypatch):
    # Loading one on demand would recreate the eviction fight this avoids.
    _mock_models(monkeypatch, v0=[IDLE_VLM])
    assert textifier._lmstudio_loaded_vlm() is None


def test_ignores_loaded_text_only_models(textifier, monkeypatch):
    _mock_models(monkeypatch, v0=[LOADED_LLM])
    assert textifier._lmstudio_loaded_vlm() is None


def test_returns_none_when_unreachable(textifier, monkeypatch):
    def boom(path):
        raise OSError("connection refused")

    monkeypatch.setattr(lmstudio_vision, "_models", boom)
    assert textifier._lmstudio_loaded_vlm() is None


def test_old_v0_api_still_works(textifier, monkeypatch):
    _mock_models(monkeypatch, v0=[LOADED_VLM])
    assert textifier._lmstudio_loaded_vlm() == "qwen/qwen3.6-35b-a3b"


def test_result_is_cached_across_images(textifier, monkeypatch):
    calls = []
    _mock_models(monkeypatch, v1=[V1_QWEN], recorder=calls)
    textifier._lmstudio_loaded_vlm()
    textifier._lmstudio_loaded_vlm()
    textifier._lmstudio_loaded_vlm()
    assert len(calls) == 1  # a document with 50 images must probe once


def test_negative_result_is_also_cached(textifier, monkeypatch):
    calls = []
    _mock_models(monkeypatch, v0=[IDLE_VLM], recorder=calls)
    assert textifier._lmstudio_loaded_vlm() is None
    assert textifier._lmstudio_loaded_vlm() is None
    assert len(calls) == 2  # each native API is checked only once per instance


def test_probes_native_v1_model_capabilities(textifier, monkeypatch):
    calls = []
    _mock_models(monkeypatch, v1=[V1_QWEN], recorder=calls)
    textifier._lmstudio_loaded_vlm()
    assert calls == ["/api/v1/models"]


def test_describe_returns_empty_when_no_model_loaded(textifier, monkeypatch):
    # The fall-through guarantee: no model -> "" so the caller tries Ollama.
    _mock_models(monkeypatch, v0=[IDLE_VLM])
    assert textifier._describe_with_lmstudio("ZmFrZQ==") == ""


def _sample_image():
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (24, 24), "blue").save(out, format="JPEG")
    return out.getvalue()


def test_qwen_result_skips_haiku(textifier, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(textifier, "_describe_with_lmstudio", lambda *a, **kw: "A blue square.")

    def fail_haiku(*a, **kw):
        raise AssertionError("Haiku must not be called after a Qwen description")

    monkeypatch.setattr(textifier, "_describe_with_claude", fail_haiku)
    assert textifier.describe_image(_sample_image()) == "A blue square."


def test_haiku_follows_unavailable_qwen(textifier, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []

    def no_qwen(*a, **kw):
        calls.append("qwen")
        return ""

    def haiku(*a, **kw):
        calls.append("haiku")
        return "A blue square."

    monkeypatch.setattr(textifier, "_describe_with_lmstudio", no_qwen)
    monkeypatch.setattr(textifier, "_describe_with_claude", haiku)
    assert textifier.describe_image(_sample_image()) == "A blue square."
    assert calls == ["qwen", "haiku"]
