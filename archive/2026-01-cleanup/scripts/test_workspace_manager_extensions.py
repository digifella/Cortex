"""
Test script for WorkspaceManager field mapping extensions.

Tests:
1. Save field mappings
2. Load field mappings
3. Update field mapping
4. Verify persistence
"""

import sys
from pathlib import Path
import tempfile
import shutil
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cortex_engine.workspace_manager import WorkspaceManager
from cortex_engine.workspace_schema import FieldMapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_workspace_manager_extensions():
    """Test WorkspaceManager field mapping methods."""

    # Create temporary database directory
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir)

        print("=" * 80)
        print("WORKSPACE MANAGER EXTENSIONS TEST")
        print("=" * 80)

        # Initialize WorkspaceManager
        manager = WorkspaceManager(db_path)

        # Create a test workspace
        print("\n1. Creating test workspace...")
        workspace = manager.create_workspace(
            tender_id="TEST-001",
            tender_filename="test_tender.docx",
            entity_id="test_entity",
            entity_name="Test Company"
        )
        workspace_id = workspace.workspace_id
        print(f"✅ Created workspace: {workspace_id}")

        # Create sample field mappings
        print("\n2. Creating sample field mappings...")
        field_mappings = [
            FieldMapping(
                field_id="field_1",
                field_location="Table 1, Row 1",
                field_description="Company ABN",
                field_type="abn",
                matched_data="12345678901",
                data_source="organization.abn",
                confidence=1.0,
                user_approved=False
            ),
            FieldMapping(
                field_id="field_2",
                field_location="Table 1, Row 2",
                field_description="Email address",
                field_type="email",
                matched_data="test@example.com",
                data_source="organization.email",
                confidence=1.0,
                user_approved=False
            ),
            FieldMapping(
                field_id="field_3",
                field_location="Table 1, Row 3",
                field_description="Years of experience",
                field_type="years_experience",
                matched_data="Over 10 years of experience in construction",
                data_source="semantic_extraction",
                confidence=0.85,
                user_approved=False
            ),
        ]

        # Save field mappings
        print("\n3. Saving field mappings...")
        success = manager.save_field_mappings(workspace_id, field_mappings)
        if success:
            print(f"✅ Saved {len(field_mappings)} field mappings")
        else:
            print("❌ Failed to save field mappings")
            return False

        # Load field mappings back
        print("\n4. Loading field mappings...")
        loaded_mappings = manager.get_field_mappings(workspace_id)
        if loaded_mappings:
            print(f"✅ Loaded {len(loaded_mappings)} field mappings")
            for mapping in loaded_mappings:
                print(f"   - {mapping.field_id}: {mapping.field_description} = {mapping.matched_data}")
        else:
            print("❌ Failed to load field mappings")
            return False

        # Verify loaded mappings match saved ones
        print("\n5. Verifying loaded data matches saved data...")
        assert len(loaded_mappings) == len(field_mappings), "Count mismatch!"
        for saved, loaded in zip(field_mappings, loaded_mappings):
            assert saved.field_id == loaded.field_id, f"Field ID mismatch: {saved.field_id} != {loaded.field_id}"
            assert saved.matched_data == loaded.matched_data, f"Data mismatch for {saved.field_id}"
            assert saved.user_approved == loaded.user_approved, f"Approval status mismatch for {saved.field_id}"
        print("✅ All data matches!")

        # Update a field mapping
        print("\n6. Updating field mapping (field_2)...")
        update_success = manager.update_field_mapping(
            workspace_id,
            "field_2",
            {
                "user_approved": True,
                "user_override": "updated@example.com"
            }
        )
        if update_success:
            print("✅ Successfully updated field_2")
        else:
            print("❌ Failed to update field mapping")
            return False

        # Load again and verify update persisted
        print("\n7. Verifying update persisted...")
        updated_mappings = manager.get_field_mappings(workspace_id)
        field_2 = next((m for m in updated_mappings if m.field_id == "field_2"), None)

        if field_2:
            print(f"   Field 2 data:")
            print(f"   - user_approved: {field_2.user_approved}")
            print(f"   - user_override: {field_2.user_override}")

            assert field_2.user_approved == True, "user_approved not updated!"
            assert field_2.user_override == "updated@example.com", "user_override not updated!"
            print("✅ Update persisted correctly!")
        else:
            print("❌ Could not find field_2 after update")
            return False

        # Test updating non-existent field
        print("\n8. Testing error handling (non-existent field)...")
        bad_update = manager.update_field_mapping(
            workspace_id,
            "field_999",
            {"user_approved": True}
        )
        if not bad_update:
            print("✅ Correctly rejected update for non-existent field")
        else:
            print("❌ Should have failed for non-existent field")
            return False

        # Check workspace metadata was updated
        print("\n9. Verifying workspace metadata...")
        updated_workspace = manager.get_workspace(workspace_id)
        print(f"   - Field count: {updated_workspace.field_count}")
        print(f"   - Matched count: {updated_workspace.matched_field_count}")
        print(f"   - Last modified: {updated_workspace.last_modified}")
        assert updated_workspace.field_count == 3, "Field count not updated!"
        assert updated_workspace.matched_field_count == 3, "Matched count not updated!"
        print("✅ Workspace metadata correctly updated!")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        return True


if __name__ == "__main__":
    try:
        success = test_workspace_manager_extensions()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        sys.exit(1)
