"""
End-to-End Workflow Test
Version: 1.0.0
Date: 2026-01-05

Purpose: Test complete proposal workflow from workspace creation to export.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cortex_engine.workspace_manager import WorkspaceManager
from cortex_engine.workspace_model import WorkspaceState
from cortex_engine.entity_profile_manager import EntityProfileManager
from cortex_engine.document_processor import DocumentProcessor
from cortex_engine.markup_engine import MarkupEngine
from cortex_engine.content_generator import ContentGenerator
from cortex_engine.field_substitution_engine import FieldSubstitutionEngine
from cortex_engine.mention_parser import MentionParser
from cortex_engine.llm_interface import LLMInterface
from cortex_engine.utils import get_logger, convert_windows_to_wsl_path

logger = get_logger(__name__)

def test_complete_workflow():
    """Test the complete proposal workflow."""

    print("\n" + "="*80)
    print("🧪 PROPOSAL WORKFLOW - END-TO-END TEST")
    print("="*80 + "\n")

    # Load configuration
    print("📦 Loading configuration...")
    config_path = Path(__file__).parent / "cortex_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    db_path = convert_windows_to_wsl_path(config.get('ai_database_path', '/mnt/f/ai_databases'))
    print(f"   Database path: {db_path}")

    # Initialize managers
    print("📦 Initializing managers...")
    workspaces_path = Path(db_path) / "workspaces"
    workspace_manager = WorkspaceManager(workspaces_path)
    entity_manager = EntityProfileManager(Path(db_path))
    llm = LLMInterface(model="mistral-small3.2")

    # Check for existing entity profiles
    entities = entity_manager.list_entity_profiles()
    if not entities:
        print("❌ No entity profiles found!")
        print("💡 Please create an entity profile first using Entity Profile Manager page")
        return False

    entity_id = entities[0].entity_id
    entity_name = entities[0].entity_name
    print(f"✅ Using entity: {entity_name} ({entity_id})")

    # Step 1: Create workspace
    print("\n📝 STEP 1: Creating workspace...")
    try:
        workspace_id = f"test_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        workspace = workspace_manager.create_workspace(
            workspace_id=workspace_id,
            workspace_name="End-to-End Test Workspace",
            tender_name="Test Tender RFT-2026-001",
            tender_reference="RFT-2026-001",
            created_by="Test Script"
        )
        print(f"✅ Workspace created: {workspace_id}")
        print(f"   State: {workspace.metadata.state}")
    except Exception as e:
        print(f"❌ Failed to create workspace: {e}")
        return False

    # Step 2: Create test document
    print("\n📄 STEP 2: Creating test document...")
    try:
        test_doc_content = """
TEST TENDER DOCUMENT
RFT-2026-001: Professional Services

SECTION 1: COMPANY DETAILS

Legal Entity Name:
ABN:
ACN:
Registered Office Address:

Contact Details:
Phone:
Email:
Website:

SECTION 2: INSURANCE

Insurance Coverage:
Public Liability:
Policy Number:

SECTION 3: EXECUTIVE SUMMARY

Executive Summary:
Please provide a brief overview of your company.

SECTION 4: TEAM MEMBERS

Please provide CVs for all proposed team members.

Project Manager: @cv[paul_smith]

SECTION 5: RELEVANT EXPERIENCE

Please provide examples of relevant projects.

@project_summary[dha_health_services]

SECTION 6: REFERENCES

Please provide at least 2 professional references.

@reference[john_smith_dha]
"""

        workspace_path = workspace_manager.base_path / workspace_id
        doc_path = workspace_path / "documents" / "test_tender.txt"

        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(test_doc_content)

        print(f"✅ Test document created: {doc_path.name}")
        print(f"   Length: {len(test_doc_content)} characters")
    except Exception as e:
        print(f"❌ Failed to create test document: {e}")
        return False

    # Step 3: Bind entity to workspace
    print("\n🔗 STEP 3: Binding entity to workspace...")
    try:
        workspace = workspace_manager.bind_entity(
            workspace_id=workspace_id,
            entity_id=entity_id,
            entity_name=entity_name
        )
        print(f"✅ Entity bound: {entity_name}")
        print(f"   State: {workspace.metadata.state}")
    except Exception as e:
        print(f"❌ Failed to bind entity: {e}")
        return False

    # Step 4: Run markup analysis
    print("\n🔍 STEP 4: Running markup analysis...")
    try:
        markup_engine = MarkupEngine(entity_manager, llm)
        suggested_mentions = markup_engine.analyze_document(
            test_doc_content,
            entity_id
        )

        print(f"✅ Markup analysis complete")
        print(f"   Found {len(suggested_mentions)} suggested mentions")

        # Add mentions to workspace
        workspace = workspace_manager.add_mention_bindings(
            workspace_id=workspace_id,
            mentions=suggested_mentions
        )

        # Transition to MARKUP_SUGGESTED
        workspace = workspace_manager.update_workspace_state(
            workspace_id=workspace_id,
            new_state=WorkspaceState.MARKUP_SUGGESTED
        )
        print(f"   State: {workspace.metadata.state}")

    except Exception as e:
        print(f"❌ Failed markup analysis: {e}")
        return False

    # Step 5: Review and approve mentions
    print("\n✅ STEP 5: Reviewing and approving mentions...")
    try:
        approved_count = 0
        for mention in workspace.mentions:
            # Auto-approve all for testing
            workspace = workspace_manager.update_mention_binding(
                workspace_id=workspace_id,
                mention_text=mention.mention_text,
                approved=True
            )
            approved_count += 1

        print(f"✅ Approved {approved_count} mentions")

        # Transition through review states
        workspace = workspace_manager.update_workspace_state(
            workspace_id=workspace_id,
            new_state=WorkspaceState.MARKUP_REVIEWED
        )
        workspace = workspace_manager.update_workspace_state(
            workspace_id=workspace_id,
            new_state=WorkspaceState.ENTITY_BOUND
        )
        print(f"   State: {workspace.metadata.state}")

    except Exception as e:
        print(f"❌ Failed to approve mentions: {e}")
        return False

    # Step 6: Generate LLM content
    print("\n🤖 STEP 6: Generating LLM content...")
    try:
        content_generator = ContentGenerator(entity_manager, llm)
        llm_mentions = workspace.get_llm_mentions()

        print(f"   Found {len(llm_mentions)} mentions requiring LLM generation")

        for mention in llm_mentions:
            print(f"\n   Generating: {mention.mention_text}")

            # Determine generation type
            if "@cv[" in mention.mention_text:
                generation_type = "cv"
            elif "@project_summary[" in mention.mention_text:
                generation_type = "project_summary"
            elif "@reference[" in mention.mention_text:
                generation_type = "reference"
            else:
                continue

            try:
                generated_content = content_generator.generate_content(
                    mention,
                    entity_id,
                    generation_type
                )

                # Update mention with generated content
                workspace = workspace_manager.update_mention_binding(
                    workspace_id=workspace_id,
                    mention_text=mention.mention_text,
                    resolved_value=generated_content
                )

                print(f"   ✅ Generated {len(generated_content)} characters")

            except Exception as e:
                print(f"   ⚠️ Generation failed: {e}")
                continue

        print(f"\n✅ LLM content generation complete")

        # Transition through generation states
        workspace = workspace_manager.update_workspace_state(
            workspace_id=workspace_id,
            new_state=WorkspaceState.CONTENT_GENERATED
        )
        workspace = workspace_manager.update_workspace_state(
            workspace_id=workspace_id,
            new_state=WorkspaceState.DRAFT_READY
        )
        print(f"   State: {workspace.metadata.state}")

    except Exception as e:
        print(f"❌ Failed LLM generation: {e}")
        return False

    # Step 7: Export final document
    print("\n📤 STEP 7: Exporting final document...")
    try:
        # Build replacements
        parser = MentionParser()
        engine = FieldSubstitutionEngine(entity_manager)
        mentions_in_doc = parser.parse_all(test_doc_content)

        replacements = {}
        for parsed_mention in mentions_in_doc:
            # Try workspace mentions first
            workspace_mention = next(
                (m for m in workspace.mentions if m.mention_text == parsed_mention.raw_text),
                None
            )

            if workspace_mention and workspace_mention.resolved_value:
                replacements[parsed_mention.raw_text] = workspace_mention.resolved_value
            else:
                # Resolve from entity profile
                result = engine.resolve(parsed_mention, entity_id)
                if result.success:
                    replacements[parsed_mention.raw_text] = result.value
                else:
                    replacements[parsed_mention.raw_text] = "[NOT RESOLVED]"

        print(f"   Built {len(replacements)} replacements")

        # Apply replacements
        final_content = test_doc_content
        for mention, value in replacements.items():
            if value is not None:
                final_content = final_content.replace(mention, str(value))
            else:
                logger.warning(f"Skipping replacement for {mention}: value is None")

        # Save export
        export_path = workspace_path / "exports" / f"test_tender_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(final_content)

        print(f"✅ Document exported: {export_path.name}")
        print(f"   Final length: {len(final_content)} characters")

        # Transition through final states
        workspace = workspace_manager.update_workspace_state(
            workspace_id=workspace_id,
            new_state=WorkspaceState.IN_REVIEW
        )
        workspace = workspace_manager.update_workspace_state(
            workspace_id=workspace_id,
            new_state=WorkspaceState.APPROVED
        )
        workspace = workspace_manager.update_workspace_state(
            workspace_id=workspace_id,
            new_state=WorkspaceState.EXPORTED
        )
        print(f"   State: {workspace.metadata.state}")

    except Exception as e:
        print(f"❌ Failed export: {e}")
        return False

    # Final summary
    print("\n" + "="*80)
    print("✅ WORKFLOW TEST COMPLETE")
    print("="*80)
    print(f"\nWorkspace ID: {workspace_id}")
    print(f"Final State: {workspace.metadata.state}")
    print(f"Total Mentions: {len(workspace.mentions)}")
    print(f"Approved: {sum(1 for m in workspace.mentions if m.approved)}")
    print(f"Generated Content: {sum(1 for m in workspace.mentions if m.resolved_value)}")
    print(f"Export Path: {export_path}")

    print("\n📊 Workspace Progress:")
    progress = workspace.get_progress_percentage()
    print(f"   {progress:.1f}% complete")

    print("\n✅ All workflow steps completed successfully!")
    return True


if __name__ == "__main__":
    try:
        success = test_complete_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
