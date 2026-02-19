"""
Unit Tests for Batch Manager Cleanup Helpers
"""

from pathlib import Path

from cortex_engine.batch_manager import clear_all_ingestion_state


class TestClearAllIngestionState:
    def test_clears_batch_and_staging_files(self, temp_dir):
        batch_file = temp_dir / "batch_state.json"
        staging_file = temp_dir / "staging_ingestion.json"
        batch_file.write_text("{}", encoding="utf-8")
        staging_file.write_text("{}", encoding="utf-8")

        result = clear_all_ingestion_state(str(temp_dir))

        assert result["batch_state_cleared"] is True
        assert result["staging_cleared"] is True
        assert result["errors"] == []
        assert not batch_file.exists()
        assert not staging_file.exists()

    def test_missing_files_treated_as_cleared(self, temp_dir):
        result = clear_all_ingestion_state(str(temp_dir))

        assert result["batch_state_cleared"] is True
        assert result["staging_cleared"] is True
        assert result["errors"] == []

    def test_collects_error_when_unlink_fails(self, temp_dir, monkeypatch):
        batch_file = temp_dir / "batch_state.json"
        batch_file.write_text("{}", encoding="utf-8")
        staging_file = temp_dir / "staging_ingestion.json"
        staging_file.write_text("{}", encoding="utf-8")

        original_unlink = Path.unlink

        def _failing_unlink(self, *args, **kwargs):
            if self.name == "batch_state.json":
                raise PermissionError("denied")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr("cortex_engine.batch_manager._is_wsl_env", lambda: False)
        monkeypatch.setattr(Path, "unlink", _failing_unlink)

        result = clear_all_ingestion_state(str(temp_dir))

        assert result["batch_state_cleared"] is False
        assert result["staging_cleared"] is True
        assert len(result["errors"]) == 1
        assert "Failed to clear batch state" in result["errors"][0]
        assert batch_file.exists()
        assert not staging_file.exists()

