"""
Unit Tests for Ingestion Recovery Logic
"""

import json

from cortex_engine.ingestion_recovery import IngestionRecoveryManager


class TestFindOrphanedDocuments:
    """Test orphaned-document detection behavior."""

    def test_marks_missing_file_paths_as_orphaned(self):
        manager = IngestionRecoveryManager.__new__(IngestionRecoveryManager)
        ingested_files = {
            "/docs/kept.pdf": {"doc_id": "doc-1", "status": "ingested"},
            "/docs/missing.pdf": {"doc_id": "doc-2", "status": "ingested"},
        }
        chroma_data = {"file_paths": {"/docs/kept.pdf"}, "doc_ids": {"doc-1"}}

        orphaned = manager._find_orphaned_documents(ingested_files, chroma_data)

        assert len(orphaned) == 1
        assert orphaned[0]["file_path"] == "/docs/missing.pdf"
        assert orphaned[0]["file_hash"] == "doc-2"
        assert orphaned[0]["file_name"] == "missing.pdf"

    def test_skips_user_excluded_entries(self):
        manager = IngestionRecoveryManager.__new__(IngestionRecoveryManager)
        ingested_files = {
            "/docs/excluded-by-status.pdf": {"doc_id": "doc-x", "status": "excluded"},
            "/docs/excluded-by-doc-id.pdf": {"doc_id": "user_excluded_123", "status": "ingested"},
            "/docs/old-format-excluded.pdf": "user_excluded_legacy",
            "/docs/normal.pdf": {"doc_id": "doc-9", "status": "ingested"},
        }
        chroma_data = {"file_paths": set(), "doc_ids": set()}

        orphaned = manager._find_orphaned_documents(ingested_files, chroma_data)
        orphaned_paths = {item["file_path"] for item in orphaned}

        assert orphaned_paths == {"/docs/normal.pdf"}


class TestCleanupOrphanedLogEntries:
    """Test orphaned log cleanup behavior."""

    def test_removes_orphaned_entries_from_log_file(self, temp_dir):
        manager = IngestionRecoveryManager.__new__(IngestionRecoveryManager)
        manager.ingested_log_path = str(temp_dir / "ingested_files.log")

        original_log = {
            "/docs/ok.pdf": {"doc_id": "doc-ok"},
            "/docs/orphaned.pdf": {"doc_id": "doc-missing"},
        }
        with open(manager.ingested_log_path, "w", encoding="utf-8") as f:
            json.dump(original_log, f)

        manager.analyze_ingestion_state = lambda: {
            "orphaned_documents": [
                {"file_path": "/docs/orphaned.pdf", "file_name": "orphaned.pdf"}
            ]
        }

        result = manager.cleanup_orphaned_log_entries()

        assert result["status"] == "success"
        assert result["entries_removed"] == 1
        assert result["original_count"] == 2
        assert result["new_count"] == 1
        assert result["removed_files"] == ["orphaned.pdf"]

        with open(manager.ingested_log_path, "r", encoding="utf-8") as f:
            cleaned = json.load(f)
        assert list(cleaned.keys()) == ["/docs/ok.pdf"]

    def test_no_orphaned_entries_returns_success_without_changes(self, temp_dir):
        manager = IngestionRecoveryManager.__new__(IngestionRecoveryManager)
        manager.ingested_log_path = str(temp_dir / "ingested_files.log")

        with open(manager.ingested_log_path, "w", encoding="utf-8") as f:
            json.dump({"/docs/ok.pdf": {"doc_id": "doc-ok"}}, f)

        manager.analyze_ingestion_state = lambda: {"orphaned_documents": []}

        result = manager.cleanup_orphaned_log_entries()

        assert result["status"] == "success"
        assert result["entries_removed"] == 0
        assert "No orphaned log entries found" in result["message"]
