import os
import tempfile

from cortex_engine.image_caption_cache import ImageCaptionCache


def test_cache_miss_then_hit(tmp_path):
    db_path = str(tmp_path / "cache.db")
    cache = ImageCaptionCache(db_path=db_path)

    image_bytes = b"fake-png-bytes-for-testing"
    assert cache.get(image_bytes) is None

    cache.put(image_bytes, "A bar chart showing quarterly revenue.")
    assert cache.get(image_bytes) == "A bar chart showing quarterly revenue."


def test_different_bytes_do_not_collide(tmp_path):
    db_path = str(tmp_path / "cache.db")
    cache = ImageCaptionCache(db_path=db_path)

    cache.put(b"image-one", "First caption")
    cache.put(b"image-two", "Second caption")

    assert cache.get(b"image-one") == "First caption"
    assert cache.get(b"image-two") == "Second caption"


def test_cache_persists_across_instances(tmp_path):
    db_path = str(tmp_path / "cache.db")
    ImageCaptionCache(db_path=db_path).put(b"persist-me", "Persisted caption")

    reopened = ImageCaptionCache(db_path=db_path)
    assert reopened.get(b"persist-me") == "Persisted caption"
