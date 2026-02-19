"""
Unit Tests for Collection Manager Integrity Helpers
"""

import pytest

from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.exceptions import CollectionError


class _FakeVectorCollection:
    def __init__(self, metadatas=None, should_fail=False):
        self._metadatas = metadatas or []
        self._should_fail = should_fail

    def get(self, where=None, include=None):
        if self._should_fail:
            raise RuntimeError("vector backend failure")
        return {"metadatas": self._metadatas}


class TestPruneMissingDocReferences:
    def test_prunes_missing_doc_ids_from_collection(self):
        manager = WorkingCollectionManager.__new__(WorkingCollectionManager)
        manager.collections = {
            "default": {"name": "default", "doc_ids": []},
            "alpha": {"name": "alpha", "doc_ids": ["doc-1", "doc-2", "doc-3"]},
        }

        def _remove(name, doc_ids_to_remove):
            remaining = [
                doc_id for doc_id in manager.collections[name]["doc_ids"]
                if doc_id not in doc_ids_to_remove
            ]
            manager.collections[name]["doc_ids"] = remaining

        manager.remove_from_collection = _remove

        vector_collection = _FakeVectorCollection(
            metadatas=[
                {"doc_id": "doc-1"},
                {"doc_id": "doc-3"},
            ]
        )
        result = manager.prune_missing_doc_references("alpha", vector_collection)

        assert result["collection"] == "alpha"
        assert result["total_before"] == 3
        assert result["existing_count"] == 2
        assert result["removed_count"] == 1
        assert result["removed_doc_ids"] == ["doc-2"]
        assert manager.collections["alpha"]["doc_ids"] == ["doc-1", "doc-3"]

    def test_returns_empty_result_for_unknown_collection(self):
        manager = WorkingCollectionManager.__new__(WorkingCollectionManager)
        manager.collections = {"default": {"name": "default", "doc_ids": []}}
        vector_collection = _FakeVectorCollection(metadatas=[])

        result = manager.prune_missing_doc_references("missing", vector_collection)

        assert result["collection"] == "missing"
        assert result["total_before"] == 0
        assert result["existing_count"] == 0
        assert result["removed_count"] == 0
        assert result["removed_doc_ids"] == []

    def test_raises_collection_error_when_backend_fails(self):
        manager = WorkingCollectionManager.__new__(WorkingCollectionManager)
        manager.collections = {"alpha": {"name": "alpha", "doc_ids": ["doc-1"]}}
        vector_collection = _FakeVectorCollection(should_fail=True)

        with pytest.raises(CollectionError):
            manager.prune_missing_doc_references("alpha", vector_collection)

