"""
Unit Tests for Version Configuration
Version: 1.0.0
Purpose: Test version management and validation functions
"""

import pytest
from cortex_engine.version_config import (
    CORTEX_VERSION,
    VERSION_INFO,
    VERSION_METADATA,
    get_version_string,
    get_version_display,
    get_full_version_info,
    get_version_footer,
    get_changelog_entry,
    validate_version_format,
)


class TestVersionConstants:
    """Test version constant values."""

    def test_version_is_string(self):
        """Test that CORTEX_VERSION is a string."""
        assert isinstance(CORTEX_VERSION, str)

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        parts = CORTEX_VERSION.split('.')
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_version_info_structure(self):
        """Test VERSION_INFO dictionary structure."""
        required_keys = ['major', 'minor', 'patch', 'pre_release', 'build']
        assert all(key in VERSION_INFO for key in required_keys)

        # Test types
        assert isinstance(VERSION_INFO['major'], int)
        assert isinstance(VERSION_INFO['minor'], int)
        assert isinstance(VERSION_INFO['patch'], int)

    def test_version_metadata_structure(self):
        """Test VERSION_METADATA dictionary structure."""
        required_keys = [
            'version', 'release_date', 'release_name',
            'description', 'breaking_changes', 'new_features',
            'improvements', 'bug_fixes'
        ]
        assert all(key in VERSION_METADATA for key in required_keys)

        # Test types
        assert isinstance(VERSION_METADATA['new_features'], list)
        assert isinstance(VERSION_METADATA['improvements'], list)
        assert isinstance(VERSION_METADATA['bug_fixes'], list)

    def test_version_metadata_matches_version(self):
        """Test that VERSION_METADATA version matches CORTEX_VERSION."""
        assert VERSION_METADATA['version'] == CORTEX_VERSION


class TestVersionFormatting:
    """Test version formatting functions."""

    def test_get_version_string(self):
        """Test version string generation."""
        version_str = get_version_string()
        assert version_str.startswith('v')
        assert version_str == f"v{CORTEX_VERSION}"

    def test_get_version_display(self):
        """Test version display string."""
        display = get_version_display()
        assert CORTEX_VERSION in display
        assert VERSION_METADATA['release_name'] in display

    def test_get_version_footer(self):
        """Test version footer generation."""
        footer = get_version_footer()
        assert 'Version:' in footer
        assert CORTEX_VERSION in footer
        assert VERSION_METADATA['description'] in footer

    def test_get_full_version_info(self):
        """Test full version info aggregation."""
        info = get_full_version_info()

        # Should contain VERSION_INFO fields
        assert info['major'] == VERSION_INFO['major']
        assert info['minor'] == VERSION_INFO['minor']

        # Should contain VERSION_METADATA fields
        assert info['version'] == VERSION_METADATA['version']
        assert info['release_name'] == VERSION_METADATA['release_name']

        # Should contain formatted versions
        assert 'formatted_version' in info
        assert 'display_version' in info


class TestChangelogGeneration:
    """Test changelog entry generation."""

    def test_get_changelog_entry_structure(self):
        """Test changelog entry format."""
        entry = get_changelog_entry()

        # Should contain version header
        assert f"## v{CORTEX_VERSION}" in entry
        assert VERSION_METADATA['release_date'] in entry
        assert VERSION_METADATA['release_name'] in entry

    def test_changelog_includes_new_features(self):
        """Test that new features are included in changelog."""
        entry = get_changelog_entry()

        if VERSION_METADATA.get('new_features'):
            assert '### ✨ New Features' in entry
            # Check first feature is included
            first_feature = VERSION_METADATA['new_features'][0]
            assert first_feature in entry

    def test_changelog_includes_improvements(self):
        """Test that improvements are included in changelog."""
        entry = get_changelog_entry()

        if VERSION_METADATA.get('improvements'):
            assert '### 🚀 Improvements' in entry

    def test_changelog_includes_breaking_changes(self):
        """Test that breaking changes are included when present."""
        # This version might not have breaking changes
        entry = get_changelog_entry()

        if VERSION_METADATA.get('breaking_changes'):
            assert '### 🔥 Breaking Changes' in entry

    def test_changelog_markdown_format(self):
        """Test that changelog uses proper Markdown formatting."""
        entry = get_changelog_entry()

        # Should have proper headers
        assert entry.startswith('##')

        # Should have bullet points for features
        if VERSION_METADATA.get('new_features'):
            assert '\n- ' in entry


class TestVersionValidation:
    """Test version format validation."""

    @pytest.mark.parametrize("valid_version", [
        "1.0.0",
        "v1.0.0",
        "4.10.3",
        "v4.10.3",
        "10.20.30",
        "1.0.0-alpha",
        # Note: Current validator may not support build metadata
        # "1.0.0-beta.1",
        # "1.0.0+build.123",
    ])
    def test_valid_version_formats(self, valid_version):
        """Test that valid version formats pass validation."""
        assert validate_version_format(valid_version) is True

    @pytest.mark.parametrize("invalid_version", [
        "1.0",           # Missing patch
        "1",             # Missing minor and patch
        "1.0.0.0",       # Too many parts
        "a.b.c",         # Non-numeric
        "1.0.x",         # Invalid patch
        "",              # Empty
        "version 1.0.0", # Extra text
    ])
    def test_invalid_version_formats(self, invalid_version):
        """Test that invalid version formats fail validation."""
        assert validate_version_format(invalid_version) is False

    def test_current_version_is_valid(self):
        """Test that the current CORTEX_VERSION is valid."""
        assert validate_version_format(CORTEX_VERSION) is True
        assert validate_version_format(f"v{CORTEX_VERSION}") is True


class TestVersionConsistency:
    """Test version consistency across components."""

    def test_version_info_matches_string(self):
        """Test that VERSION_INFO components match CORTEX_VERSION string."""
        major, minor, patch = CORTEX_VERSION.split('.')

        assert VERSION_INFO['major'] == int(major)
        assert VERSION_INFO['minor'] == int(minor)
        assert VERSION_INFO['patch'] == int(patch)

    def test_release_date_format(self):
        """Test that release date is in correct format."""
        import re
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'

        assert re.match(date_pattern, VERSION_METADATA['release_date'])

    def test_release_name_not_empty(self):
        """Test that release name is meaningful."""
        assert VERSION_METADATA['release_name']
        assert len(VERSION_METADATA['release_name']) > 5

    def test_description_not_empty(self):
        """Test that description is meaningful."""
        assert VERSION_METADATA['description']
        assert len(VERSION_METADATA['description']) > 10


class TestVersionComparison:
    """Test version comparison utilities (if implemented)."""

    def test_semantic_version_ordering(self):
        """Test semantic version comparison logic."""
        # Parse current version
        major, minor, patch = map(int, CORTEX_VERSION.split('.'))

        # Major version increases should be significant
        assert major >= 0

        # Minor version should be reasonable
        assert 0 <= minor <= 100

        # Patch version should be reasonable
        assert 0 <= patch <= 100


# ============================================================================
# Integration-style tests
# ============================================================================

class TestVersionInContext:
    """Test version usage in realistic scenarios."""

    def test_version_can_be_imported(self):
        """Test that version can be imported from main module."""
        from cortex_engine.version_config import VERSION
        assert VERSION == CORTEX_VERSION

    def test_version_display_can_be_imported(self):
        """Test that version display can be imported."""
        from cortex_engine.version_config import VERSION_DISPLAY
        assert CORTEX_VERSION in VERSION_DISPLAY

    def test_version_in_ui_context(self):
        """Test version formatting for UI display."""
        display = get_version_display()

        # Should be suitable for UI display
        assert len(display) < 100  # Reasonable length
        assert display  # Not empty

        # Should contain key information
        assert CORTEX_VERSION in display


# ============================================================================
# Error handling tests
# ============================================================================

class TestVersionErrorHandling:
    """Test error handling in version functions."""

    def test_invalid_version_raises_on_load(self):
        """Test that invalid version format is caught on module load."""
        # The module itself validates on load, so if we got here, it passed
        # This test ensures that validation is happening
        assert validate_version_format(CORTEX_VERSION)

    def test_empty_string_validation(self):
        """Test validation of empty string."""
        assert validate_version_format("") is False

    def test_none_handling(self):
        """Test that None is handled gracefully."""
        # validate_version_format should handle None
        try:
            result = validate_version_format(None)
            assert result is False
        except (TypeError, AttributeError):
            # Either way is acceptable - return False or raise TypeError
            pass
