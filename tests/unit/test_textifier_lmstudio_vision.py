"""LM Studio vision provider in DocumentTextifier.

The provider exists to reuse a VLM that is ALREADY loaded in LM Studio, instead
of making Ollama evict it to load a smaller model into leftover VRAM. It must
degrade to "" (so the caller falls through to Ollama) whenever LM Studio is
unreachable, has no vision model loaded, or errors.
"""

import io
import json
import urllib.request

import pytest

from cortex_engine.textifier import DocumentTextifier


def _payload(*entries):
    body = json.dumps({"data": list(entries)}).encode("utf-8")
    return lambda *a, **k: io.BytesIO(body)


LOADED_VLM = {"id": "qwen/qwen3.6-35b-a3b", "type": "vlm", "state": "loaded"}
IDLE_VLM = {"id": "google/gemma-4-31b", "type": "vlm", "state": "not-loaded"}
LOADED_LLM = {"id": "qwen3-coder-30b", "type": "llm", "state": "loaded"}


@pytest.fixture
def textifier(monkeypatch):
    # auto_select_vision shells out to `ollama list` and nvidia-smi; off for tests.
    monkeypatch.delenv("CORTEX_LMSTUDIO_VISION_MODEL", raising=False)
    monkeypatch.setenv("CORTEX_LMSTUDIO_BASE_URL", "http://testhost:1234/v1")
    return DocumentTextifier(use_vision=True, auto_select_vision=False)


class _Ctx:
    """urlopen returns a context manager; mimic just enough of it."""

    def __init__(self, fh):
        self._fh = fh

    def __enter__(self):
        return self._fh

    def __exit__(self, *exc):
        return False


def _mock_urlopen(monkeypatch, entries, recorder=None):
    def fake(url, timeout=None):
        if recorder is not None:
            recorder.append(url)
        return _Ctx(io.BytesIO(json.dumps({"data": list(entries)}).encode("utf-8")))

    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_picks_the_loaded_vision_model(textifier, monkeypatch):
    _mock_urlopen(monkeypatch, [IDLE_VLM, LOADED_VLM, LOADED_LLM])
    assert textifier._lmstudio_loaded_vlm() == "qwen/qwen3.6-35b-a3b"


def test_ignores_vision_models_that_are_not_loaded(textifier, monkeypatch):
    # Loading one on demand would recreate the eviction fight this avoids.
    _mock_urlopen(monkeypatch, [IDLE_VLM])
    assert textifier._lmstudio_loaded_vlm() is None


def test_ignores_loaded_text_only_models(textifier, monkeypatch):
    _mock_urlopen(monkeypatch, [LOADED_LLM])
    assert textifier._lmstudio_loaded_vlm() is None


def test_returns_none_when_unreachable(textifier, monkeypatch):
    def boom(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert textifier._lmstudio_loaded_vlm() is None


def test_env_override_wins_without_probing(textifier, monkeypatch):
    calls = []
    _mock_urlopen(monkeypatch, [LOADED_VLM], recorder=calls)
    monkeypatch.setenv("CORTEX_LMSTUDIO_VISION_MODEL", "my/pinned-vlm")
    assert textifier._lmstudio_loaded_vlm() == "my/pinned-vlm"
    assert calls == []  # override must not hit the network


def test_result_is_cached_across_images(textifier, monkeypatch):
    calls = []
    _mock_urlopen(monkeypatch, [LOADED_VLM], recorder=calls)
    textifier._lmstudio_loaded_vlm()
    textifier._lmstudio_loaded_vlm()
    textifier._lmstudio_loaded_vlm()
    assert len(calls) == 1  # a document with 50 images must probe once


def test_negative_result_is_also_cached(textifier, monkeypatch):
    calls = []
    _mock_urlopen(monkeypatch, [IDLE_VLM], recorder=calls)
    assert textifier._lmstudio_loaded_vlm() is None
    assert textifier._lmstudio_loaded_vlm() is None
    assert len(calls) == 1  # a miss must not re-probe per image


def test_probes_the_native_api_not_v1(textifier, monkeypatch):
    calls = []
    _mock_urlopen(monkeypatch, [LOADED_VLM], recorder=calls)
    textifier._lmstudio_loaded_vlm()
    # /v1 has no type/state fields; the loaded-model info is on the native API.
    assert calls == ["http://testhost:1234/api/v0/models"]


def test_describe_returns_empty_when_no_model_loaded(textifier, monkeypatch):
    # The fall-through guarantee: no model -> "" so the caller tries Ollama.
    _mock_urlopen(monkeypatch, [IDLE_VLM])
    assert textifier._describe_with_lmstudio("ZmFrZQ==") == ""
