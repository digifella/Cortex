import io
import os

import pytest
from PIL import Image

from cortex_engine.haiku_figure_caption import caption_figure_with_haiku


def _tiny_red_square_png() -> bytes:
    img = Image.new("RGB", (200, 200), color=(220, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.skipif(
    not os.path.exists(os.path.expanduser("~/cortex_suite/.env")),
    reason="requires cortex_suite/.env with ANTHROPIC_API_KEY",
)
def test_captions_a_real_image():
    caption = caption_figure_with_haiku(_tiny_red_square_png())
    assert isinstance(caption, str)
    assert len(caption) > 0


def test_returns_empty_string_on_missing_key(monkeypatch):
    # _load_api_key() falls back to reading cortex_suite/.env directly, which
    # has a real key on this machine, so just deleting the env var doesn't
    # simulate "no key available" -- mock the loader itself instead.
    import cortex_engine.haiku_figure_caption as mod
    monkeypatch.setattr(mod, "_load_api_key", lambda: "")
    caption = caption_figure_with_haiku(_tiny_red_square_png())
    assert caption == ""
