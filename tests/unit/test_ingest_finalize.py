"""
Unit Tests for Finalize Ingestion Edge Cases
"""

import json

from cortex_engine import ingest_cortex


class TestFinalizeIngestion:
    def test_returns_cleanly_when_staging_file_missing(self, temp_dir):
        ingest_cortex.finalize_ingestion(str(temp_dir))
        assert not (temp_dir / "staging_ingestion.json").exists()

    def test_empty_staging_removes_file_and_clears_batch(self, temp_dir, monkeypatch):
        staging_file = temp_dir / "staging_ingestion.json"
        staging_file.write_text(
            json.dumps(
                {
                    "documents": [],
                    "target_collection": "default",
                    "version": "2.0",
                }
            ),
            encoding="utf-8",
        )

        cleared = {"called": False}

        class _FakeBatchState:
            def __init__(self, db_path):
                self.db_path = db_path

            def clear_batch(self):
                cleared["called"] = True

        class _FakeGraphManager:
            def __init__(self, graph_file_path):
                self.graph_file_path = graph_file_path

        monkeypatch.setattr(ingest_cortex, "BatchState", _FakeBatchState)
        monkeypatch.setattr(ingest_cortex, "EnhancedGraphManager", _FakeGraphManager)
        monkeypatch.setattr(ingest_cortex, "_ensure_directory_cross_platform", lambda _path: None)

        ingest_cortex.finalize_ingestion(str(temp_dir))

        assert cleared["called"] is True
        assert not staging_file.exists()

