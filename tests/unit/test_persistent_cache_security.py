"""
Unit Tests for Persistent Cache SQL Identifier Validation
"""

import pytest

from cortex_engine.utils.persistent_cache import PersistentCache


def test_rejects_invalid_table_name(temp_dir):
    cache_db = temp_dir / "cache.db"

    with pytest.raises(ValueError):
        PersistentCache(str(cache_db), table_name="cache; DROP TABLE users;")


def test_accepts_safe_table_name(temp_dir):
    cache_db = temp_dir / "cache.db"
    cache = PersistentCache(str(cache_db), table_name="query_cache_v2")

    assert cache.table_name == "query_cache_v2"

