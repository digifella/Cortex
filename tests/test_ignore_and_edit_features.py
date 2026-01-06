"""
Unit Tests for Ignore and Quick Edit Features
Version: 1.0.0
Date: 2026-01-06

Tests for:
- Two-level ignore (session vs permanent)
- Quick edit entity profile values
- LLM-based context classification
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from cortex_engine.workspace_manager import WorkspaceManager
from cortex_engine.workspace_model import WorkspaceState, MentionBinding
from cortex_engine.entity_profile_manager import EntityProfileManager
from cortex_engine.entity_profile_schema import EntityProfile, ProfileMetadata, CompanyInfo, ContactInfo, Address, EntityType
from cortex_engine.markup_engine import MarkupEngine
from cortex_engine.llm_interface import LLMInterface


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    workspace_dir = Path(tempfile.mkdtemp())
    profile_dir = Path(tempfile.mkdtemp())

    yield workspace_dir, profile_dir

    # Cleanup
    shutil.rmtree(workspace_dir)
    shutil.rmtree(profile_dir)


@pytest.fixture
def test_entity_profile(temp_dirs):
    """Create a test entity profile."""
    _, profile_dir = temp_dirs

    manager = EntityProfileManager(profile_dir)

    profile = manager.create_entity_profile(
        entity_id="test_entity",
        entity_name="Test Company",
        entity_type=EntityType.CONSULTING_FIRM,
        legal_name="Test Company Pty Ltd",
        abn="12345678901",
        phone="+61 2 1234 5678",
        email="test@company.com.au",
        registered_office_street="123 Test St",
        registered_office_city="Sydney",
        registered_office_state="NSW",
        registered_office_postcode="2000"
    )

    return manager, profile


@pytest.fixture
def test_workspace(temp_dirs, test_entity_profile):
    """Create a test workspace."""
    workspace_dir, _ = temp_dirs
    entity_manager, profile = test_entity_profile

    manager = WorkspaceManager(workspace_dir)

    workspace = manager.create_workspace(
        workspace_id="test_workspace",
        workspace_name="Test Workspace",
        tender_name="Test Tender",
        tender_reference="TEST-001"
    )

    # Bind entity
    workspace = manager.bind_entity(
        workspace_id="test_workspace",
        entity_id="test_entity",
        entity_name="Test Company"
    )

    return manager, workspace, entity_manager


class TestIgnoreFunctionality:
    """Test ignore features (session vs permanent)."""

    def test_ignore_permanent_marks_in_database(self, test_workspace):
        """Test that permanent ignore marks mention in database."""
        manager, workspace, _ = test_workspace

        # Add a test mention
        mention = MentionBinding(
            mention_text="@email",
            mention_type="simple",
            field_path="contact.email",
            location="Line 10"
        )

        workspace = manager.add_mention_bindings("test_workspace", [mention])

        # Ignore permanently
        workspace = manager.update_mention_binding(
            workspace_id="test_workspace",
            mention_text="@email",
            ignored=True
        )

        # Verify it's marked as ignored
        assert workspace.mentions[0].ignored is True
        assert workspace.metadata.ignored_mentions == 1

        # Verify it persists after reload
        reloaded = manager.get_workspace("test_workspace")
        assert reloaded.mentions[0].ignored is True

    def test_ignored_mentions_excluded_from_pending(self, test_workspace):
        """Test that ignored mentions don't appear in pending list."""
        manager, workspace, _ = test_workspace

        # Add mentions
        mentions = [
            MentionBinding(
                mention_text="@email",
                mention_type="simple",
                field_path="contact.email",
                location="Line 10"
            ),
            MentionBinding(
                mention_text="@phone",
                mention_type="simple",
                field_path="contact.phone",
                location="Line 20"
            )
        ]

        workspace = manager.add_mention_bindings("test_workspace", mentions)

        # Ignore one
        workspace = manager.update_mention_binding(
            workspace_id="test_workspace",
            mention_text="@email",
            ignored=True
        )

        # Get pending mentions (should exclude ignored)
        pending = [m for m in workspace.mentions if not m.approved and not m.rejected and not m.ignored]

        assert len(pending) == 1
        assert pending[0].mention_text == "@phone"

    def test_ignore_counts_as_reviewed(self, test_workspace):
        """Test that ignored mentions count toward review progress."""
        manager, workspace, _ = test_workspace

        # Add mentions
        mentions = [
            MentionBinding(
                mention_text="@email",
                mention_type="simple",
                field_path="contact.email",
                location="Line 10"
            ),
            MentionBinding(
                mention_text="@phone",
                mention_type="simple",
                field_path="contact.phone",
                location="Line 20"
            )
        ]

        workspace = manager.add_mention_bindings("test_workspace", mentions)

        # Ignore one
        workspace = manager.update_mention_binding(
            workspace_id="test_workspace",
            mention_text="@email",
            ignored=True
        )

        # Count reviewed (approved + rejected + ignored)
        total = len(workspace.mentions)
        reviewed = sum(1 for m in workspace.mentions if m.approved or m.rejected or m.ignored)

        assert total == 2
        assert reviewed == 1  # One ignored


class TestQuickEditFeature:
    """Test quick edit functionality."""

    def test_quick_edit_updates_simple_field(self, test_workspace):
        """Test that quick edit updates simple contact fields."""
        manager, workspace, entity_manager = test_workspace

        # Load profile
        profile = entity_manager.get_entity_profile("test_entity")

        # Update email
        old_email = profile.contact.email
        new_email = "newemail@company.com.au"

        profile.contact.email = new_email
        entity_manager._save_profile(profile)

        # Reload and verify
        updated_profile = entity_manager.get_entity_profile("test_entity")
        assert updated_profile.contact.email == new_email
        assert updated_profile.contact.email != old_email

    def test_quick_edit_updates_custom_field(self, test_workspace):
        """Test that quick edit can update custom fields."""
        manager, workspace, entity_manager = test_workspace

        # Load profile
        profile = entity_manager.get_entity_profile("test_entity")

        # Add custom field
        profile.add_custom_field(
            field_name="test_custom_field",
            field_value="original_value",
            description="Test field"
        )
        entity_manager._save_profile(profile)

        # Update it
        profile = entity_manager.get_entity_profile("test_entity")
        custom_field = profile.get_custom_field("test_custom_field")
        assert custom_field is not None
        assert custom_field.field_value == "original_value"

        # Update
        profile.add_custom_field(
            field_name="test_custom_field",
            field_value="updated_value"
        )
        entity_manager._save_profile(profile)

        # Verify
        updated_profile = entity_manager.get_entity_profile("test_entity")
        updated_field = updated_profile.get_custom_field("test_custom_field")
        assert updated_field.field_value == "updated_value"


class TestLLMClassification:
    """Test LLM-based context classification."""

    def test_llm_classification_detects_informational(self, test_entity_profile):
        """Test that LLM can classify informational fields."""
        entity_manager, profile = test_entity_profile

        # Create mock LLM
        class MockLLM(LLMInterface):
            def generate(self, prompt, temperature=0.7):
                if "contracts@digitalhealth.gov.au" in prompt:
                    return "INFORMATIONAL"
                return "REQUEST"

        llm = MockLLM()
        engine = MarkupEngine(entity_manager, llm)

        # Test informational context
        context_lines = [
            "RFT Contact Details",
            "Email: contracts@digitalhealth.gov.au",
            "Phone: 02 1234 5678"
        ]

        is_request = engine._is_request_field(
            line="Email: contracts@digitalhealth.gov.au",
            context_lines=context_lines,
            field_type="email"
        )

        assert is_request is False

    def test_llm_classification_detects_request(self, test_entity_profile):
        """Test that LLM can classify request fields."""
        entity_manager, profile = test_entity_profile

        # Create mock LLM
        class MockLLM(LLMInterface):
            def generate(self, prompt, temperature=0.7):
                if "Please provide your" in prompt:
                    return "REQUEST"
                return "INFORMATIONAL"

        llm = MockLLM()
        engine = MarkupEngine(entity_manager, llm)

        # Test request context
        context_lines = [
            "Tenderer Details",
            "Please provide your contact email:",
            "Email: _______________"
        ]

        is_request = engine._is_request_field(
            line="Email:",
            context_lines=context_lines,
            field_type="email"
        )

        assert is_request is True

    def test_llm_classification_defaults_to_true_on_error(self, test_entity_profile):
        """Test that LLM classification defaults to suggesting on error."""
        entity_manager, profile = test_entity_profile

        # Create mock LLM that raises exception
        class FailingLLM(LLMInterface):
            def generate(self, prompt, temperature=0.7):
                raise Exception("LLM service unavailable")

        llm = FailingLLM()
        engine = MarkupEngine(entity_manager, llm)

        # Should default to True (suggest) when LLM fails
        is_request = engine._is_request_field(
            line="Email:",
            context_lines=["Test context"],
            field_type="email"
        )

        assert is_request is True  # Conservative default


class TestIntegration:
    """Integration tests combining features."""

    def test_multiple_ignore_and_approve(self, test_workspace):
        """Test mixing ignored, approved, and pending mentions."""
        manager, workspace, entity_manager = test_workspace

        # Add multiple mentions
        mentions = [
            MentionBinding(
                mention_text="@email",
                mention_type="simple",
                field_path="contact.email",
                location="Line 10",
                suggested_by_llm=False
            ),
            MentionBinding(
                mention_text="@phone",
                mention_type="simple",
                field_path="contact.phone",
                location="Line 20",
                suggested_by_llm=False
            ),
            MentionBinding(
                mention_text="@abn",
                mention_type="simple",
                field_path="company.abn",
                location="Line 30",
                suggested_by_llm=False
            )
        ]

        workspace = manager.add_mention_bindings("test_workspace", mentions)

        # Ignore one
        workspace = manager.update_mention_binding(
            workspace_id="test_workspace",
            mention_text="@email",
            ignored=True
        )

        # Approve one
        workspace = manager.update_mention_binding(
            workspace_id="test_workspace",
            mention_text="@phone",
            approved=True
        )

        # Check counts
        assert workspace.metadata.total_mentions == 3
        assert workspace.metadata.ignored_mentions == 1
        assert workspace.metadata.approved_mentions == 1

        # Pending should be 1 (only @abn)
        pending = [m for m in workspace.mentions if not m.approved and not m.rejected and not m.ignored]
        assert len(pending) == 1
        assert pending[0].mention_text == "@abn"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
