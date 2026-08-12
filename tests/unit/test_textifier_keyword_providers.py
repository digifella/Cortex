"""Keyword-extraction provider chain in DocumentTextifier.

extract_keywords must prefer the model already resident in LM Studio, fall back
to the Ollama TEXT_MODELS list when LM Studio is unreachable or has nothing
loaded, and only then drop to the naive splitter. The silent drop to the naive
splitter is the dangerous case — its output gets synced into raw masters — so
the ordering is pinned here.
"""

import pytest

from cortex_engine.textifier import DocumentTextifier


DESCRIPTION = (
    "A luxury Cartier watch with a brown leather strap sits centered against a "
    "neutral gray background, displaying a cream dial with black Arabic numerals."
)


@pytest.fixture
def textifier(monkeypatch):
    # auto_select_vision shells out to `ollama list` and nvidia-smi; off for tests.
    monkeypatch.delenv("CORTEX_LMSTUDIO_VISION_MODEL", raising=False)
    return DocumentTextifier(use_vision=True, auto_select_vision=False)


def test_lmstudio_is_preferred_and_ollama_is_not_called(textifier, monkeypatch):
    monkeypatch.setattr(
        DocumentTextifier, "_keywords_from_lmstudio",
        lambda self, prompt: "cartier, watch, leather, strap",
    )

    def boom(self, prompt):
        raise AssertionError("Ollama must not be consulted when LM Studio answers")

    monkeypatch.setattr(DocumentTextifier, "_keywords_from_ollama", boom)

    assert textifier.extract_keywords(DESCRIPTION) == [
        "cartier", "watch", "leather", "strap",
    ]


def test_falls_back_to_ollama_when_lmstudio_unavailable(textifier, monkeypatch):
    monkeypatch.setattr(
        DocumentTextifier, "_keywords_from_lmstudio", lambda self, prompt: None
    )
    monkeypatch.setattr(
        DocumentTextifier, "_keywords_from_ollama",
        lambda self, prompt: "beach, wetsuit, surf",
    )

    assert textifier.extract_keywords(DESCRIPTION) == ["beach", "wetsuit", "surf"]


def test_falls_back_to_naive_splitter_when_both_providers_fail(textifier, monkeypatch):
    monkeypatch.setattr(
        DocumentTextifier, "_keywords_from_lmstudio", lambda self, prompt: None
    )
    monkeypatch.setattr(
        DocumentTextifier, "_keywords_from_ollama", lambda self, prompt: None
    )

    keywords = textifier.extract_keywords(DESCRIPTION)

    # The naive splitter keeps filler the LLM path would drop — that difference
    # is exactly why a silent fallback is worth noticing.
    assert "cartier" in keywords
    assert "sits" in keywords


def test_empty_and_placeholder_descriptions_extract_nothing(textifier, monkeypatch):
    def boom(self, prompt):
        raise AssertionError("no provider should be called for an empty description")

    monkeypatch.setattr(DocumentTextifier, "_keywords_from_lmstudio", boom)
    monkeypatch.setattr(DocumentTextifier, "_keywords_from_ollama", boom)

    assert textifier.extract_keywords("") == []
    assert textifier.extract_keywords("[Image: logo/icon omitted]") == []


def test_anchor_keywords_reach_the_prompt(textifier):
    prompt = textifier._keyword_prompt(DESCRIPTION, ["Melbourne", "jewellery"])

    assert "Melbourne, jewellery" in prompt
    assert DESCRIPTION in prompt
    # Anchors are ground truth, not suggestions — the instruction must survive.
    assert "verbatim" in prompt


def test_jargon_and_oversized_tags_are_filtered(textifier, monkeypatch):
    monkeypatch.setattr(
        DocumentTextifier, "_keywords_from_lmstudio",
        lambda self, prompt: "cartier, bokeh, mood, watch, " + ("x" * 60),
    )

    assert textifier.extract_keywords(DESCRIPTION) == ["cartier", "watch"]
