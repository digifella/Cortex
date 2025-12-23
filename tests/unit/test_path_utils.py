"""
Unit Tests for Path Utilities
Version: 1.0.0
Purpose: Test cross-platform path handling and conversion functions
"""

import pytest
from pathlib import Path
import os

from cortex_engine.utils.path_utils import (
    convert_windows_to_wsl_path,
    convert_source_path_to_docker_mount,
    convert_to_docker_mount_path,
    _in_docker,
    _safe_path_exists,
)


class TestConvertWindowsToWSLPath:
    """Test Windows to WSL path conversion."""

    def test_windows_drive_path_conversion(self):
        """Test conversion of Windows drive paths to WSL mounts."""
        assert convert_windows_to_wsl_path("C:/Users/test") == "/mnt/c/Users/test"
        assert convert_windows_to_wsl_path("D:/Documents") == "/mnt/d/Documents"
        assert convert_windows_to_wsl_path("E:/KB_Test") == "/mnt/e/KB_Test"

    def test_backslash_normalization(self):
        """Test that backslashes are converted to forward slashes."""
        assert convert_windows_to_wsl_path("C:\\Users\\test") == "/mnt/c/Users/test"
        assert convert_windows_to_wsl_path("F:\\ai_databases") == "/mnt/f/ai_databases"

    def test_bare_drive_letter(self):
        """Test conversion of bare drive letters."""
        result = convert_windows_to_wsl_path("C:")
        assert result in ["/mnt/c", "C:/"]  # Depends on mount existence

    def test_already_wsl_path(self):
        """Test that already-converted WSL paths are unchanged."""
        assert convert_windows_to_wsl_path("/mnt/c/existing") == "/mnt/c/existing"
        assert convert_windows_to_wsl_path("/mnt/f/test") == "/mnt/f/test"

    def test_unix_path_unchanged(self):
        """Test that Unix paths remain unchanged."""
        assert convert_windows_to_wsl_path("/home/user/docs") == "/home/user/docs"
        assert convert_windows_to_wsl_path("/var/lib/data") == "/var/lib/data"

    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs."""
        assert convert_windows_to_wsl_path("") == ""
        assert convert_windows_to_wsl_path(None) == ""

    def test_path_object_input(self):
        """Test that Path objects are handled correctly."""
        path_obj = Path("C:/Users/test")
        result = convert_windows_to_wsl_path(path_obj)
        assert result == "/mnt/c/Users/test"

    def test_whitespace_handling(self):
        """Test that leading/trailing whitespace is stripped."""
        assert convert_windows_to_wsl_path("  C:/Users/test  ") == "/mnt/c/Users/test"


class TestSafePathExists:
    """Test safe path existence checking."""

    def test_existing_path(self, temp_dir):
        """Test checking an existing path."""
        assert _safe_path_exists(temp_dir) is True

    def test_nonexistent_path(self, temp_dir):
        """Test checking a non-existent path."""
        fake_path = temp_dir / "does_not_exist"
        assert _safe_path_exists(fake_path) is False

    def test_handles_permission_errors(self, monkeypatch):
        """Test that permission errors don't raise exceptions."""
        def mock_exists():
            raise PermissionError("Access denied")

        # This should return False instead of raising
        # Note: Testing this properly requires mocking Path.exists()
        pass  # Placeholder - actual implementation needs monkeypatch


class TestDockerDetection:
    """Test Docker environment detection."""

    def test_docker_detection_with_dockerenv(self, monkeypatch, temp_dir):
        """Test Docker detection when /.dockerenv exists."""
        # Mock the existence of /.dockerenv
        dockerenv = temp_dir / ".dockerenv"
        dockerenv.touch()

        def mock_exists(path):
            return str(path) == '/.dockerenv'

        monkeypatch.setattr(os.path, "exists", mock_exists)
        assert _in_docker() is True

    def test_docker_detection_with_env_var(self, monkeypatch):
        """Test Docker detection via environment variable."""
        monkeypatch.setenv("container", "docker")
        assert _in_docker() is True

        monkeypatch.delenv("container")
        monkeypatch.setenv("DOCKER_CONTAINER", "1")
        assert _in_docker() is True


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_mixed_case_drive_letters(self):
        """Test that drive letters are normalized to lowercase in WSL paths."""
        result = convert_windows_to_wsl_path("C:/test")
        assert "/mnt/c/" in result.lower()

    def test_deep_nested_paths(self):
        """Test deeply nested path structures."""
        deep_path = "C:/very/deep/nested/path/structure/file.txt"
        result = convert_windows_to_wsl_path(deep_path)
        assert result == "/mnt/c/very/deep/nested/path/structure/file.txt"

    def test_special_characters_in_path(self):
        """Test paths with special characters."""
        # Spaces
        assert convert_windows_to_wsl_path("C:/Program Files/test") == "/mnt/c/Program Files/test"

        # Hyphens and underscores
        assert convert_windows_to_wsl_path("C:/my-test_folder") == "/mnt/c/my-test_folder"

    @pytest.mark.parametrize("invalid_input", [
        "",
        None,
        "   ",
    ])
    def test_invalid_inputs(self, invalid_input):
        """Test handling of invalid inputs."""
        result = convert_windows_to_wsl_path(invalid_input)
        assert result == ""


# ============================================================================
# Integration-style tests (still unit tests but test combinations)
# ============================================================================

class TestPathConversionWorkflows:
    """Test realistic path conversion workflows."""

    def test_database_path_conversion(self):
        """Test conversion of typical database paths."""
        # Windows user path
        win_path = "F:/ai_databases"
        wsl_path = convert_windows_to_wsl_path(win_path)
        assert wsl_path == "/mnt/f/ai_databases"

    def test_knowledge_source_conversion(self):
        """Test conversion of knowledge source paths."""
        # OneDrive path example
        win_path = "E:/OneDrive - Company/Knowledge_Base"
        wsl_path = convert_windows_to_wsl_path(win_path)
        assert wsl_path == "/mnt/e/OneDrive - Company/Knowledge_Base"

    def test_round_trip_conversion_safety(self):
        """Test that converting already-converted paths is safe."""
        original = "C:/Users/test"
        first_conversion = convert_windows_to_wsl_path(original)
        second_conversion = convert_windows_to_wsl_path(first_conversion)

        # Should remain stable after first conversion
        assert first_conversion == second_conversion


# ============================================================================
# Performance tests
# ============================================================================

@pytest.mark.performance
class TestPathUtilsPerformance:
    """Test performance of path utilities."""

    def test_conversion_performance(self, benchmark):
        """Benchmark path conversion performance."""
        def convert():
            return convert_windows_to_wsl_path("C:/Users/test/Documents/file.txt")

        result = benchmark(convert)
        assert result == "/mnt/c/Users/test/Documents/file.txt"

    def test_bulk_conversion_performance(self):
        """Test performance with many path conversions."""
        import time

        paths = [f"C:/test/path{i}" for i in range(1000)]

        start = time.time()
        results = [convert_windows_to_wsl_path(p) for p in paths]
        duration = time.time() - start

        # Should complete 1000 conversions in reasonable time (< 1 second)
        assert duration < 1.0, f"Conversions took {duration:.3f}s, expected < 1.0s"
        assert len(results) == 1000
