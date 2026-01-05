#!/usr/bin/env python3
"""
Test Mention System
Demonstrates parsing and resolving @mentions with the Longboardfella entity.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cortex_engine.mention_parser import MentionParser
from cortex_engine.field_substitution_engine import FieldSubstitutionEngine
from cortex_engine.entity_profile_manager import EntityProfileManager
from cortex_engine.config_manager import ConfigManager
from cortex_engine.utils import convert_windows_to_wsl_path

def main():
    print("=" * 80)
    print("MENTION SYSTEM TEST")
    print("=" * 80)

    # Initialize components
    config = ConfigManager().get_config()
    db_path = convert_windows_to_wsl_path(config.get('ai_database_path'))

    manager = EntityProfileManager(Path(db_path))
    parser = MentionParser()
    engine = FieldSubstitutionEngine(manager)

    entity_id = "longboardfella_consulting"

    print(f"\n✅ Components initialized")
    print(f"   Entity: {entity_id}")

    # Test cases
    test_mentions = [
        # Simple fields
        "@companyname",
        "@abn",
        "@acn",
        "@email",
        "@phone",
        "@website",
        "@registered_office",

        # Structured access
        "@insurance.public_liability.coverage",
        "@insurance.public_liability.policy_number",
        "@team.paul_smith.role",
        "@team.paul_smith.full_name",

        # Content generation (will require LLM)
        "@cv[paul_smith]",
        "@project_summary[health_transformation_2023]",
        "@reference[sarah_johnson]",

        # Narrative
        "@narrative[company_overview]",
    ]

    print("\n" + "=" * 80)
    print("TESTING MENTIONS")
    print("=" * 80)

    for mention_text in test_mentions:
        print(f"\n📝 Testing: {mention_text}")
        print("-" * 80)

        # Parse
        parsed = parser.parse(mention_text)
        print(f"   Type: {parsed.mention_type.value}")
        print(f"   Path: {parsed.field_path}")
        if parsed.parameters:
            print(f"   Parameters: {parsed.parameters}")

        # Resolve
        result = engine.resolve(parsed, entity_id)

        if result.success:
            if result.requires_llm:
                print(f"   ⏳ Requires LLM generation")
                print(f"   Context: {result.generation_context}")
            else:
                print(f"   ✅ Resolved: {result.value[:100]}..." if len(result.value) > 100 else f"   ✅ Resolved: {result.value}")
        else:
            print(f"   ❌ Error: {result.error_message}")

    # Test parsing all mentions in a document
    print("\n\n" + "=" * 80)
    print("TESTING DOCUMENT PARSING")
    print("=" * 80)

    sample_document = """
    TENDER RESPONSE - EXECUTIVE SUMMARY

    Company: @companyname
    ABN: @abn
    Email: @email

    @narrative[company_overview]

    We are pleased to submit this proposal. Our team, led by @team.paul_smith.full_name,
    has extensive experience in digital transformation.

    Recent Project: @project_summary[health_transformation_2023]

    Insurance Coverage: @insurance.public_liability.coverage

    Reference: @reference[sarah_johnson]
    """

    print("\n📄 Sample Document:")
    print(sample_document)

    print("\n🔍 Detected Mentions:")
    print("-" * 80)

    mentions = parser.parse_all(sample_document)
    print(f"   Found {len(mentions)} mentions")

    for i, mention in enumerate(mentions, 1):
        print(f"\n   {i}. {mention.raw_text}")
        print(f"      Type: {mention.mention_type.value}")
        print(f"      Path: {mention.field_path}")

    # Resolve all mentions
    print("\n\n📊 Resolution Summary:")
    print("-" * 80)

    results = engine.resolve_all(mentions, entity_id)

    successful = sum(1 for r in results.values() if r.success)
    requires_llm = sum(1 for r in results.values() if r.requires_llm)
    errors = sum(1 for r in results.values() if not r.success)

    print(f"   Total Mentions: {len(results)}")
    print(f"   ✅ Successfully Resolved: {successful}")
    print(f"   ⏳ Requires LLM: {requires_llm}")
    print(f"   ❌ Errors: {errors}")

    # Show replacements
    print("\n\n🔄 Document with Resolved Values:")
    print("=" * 80)

    replaced_doc = sample_document

    for mention_text, result in results.items():
        if result.success and not result.requires_llm:
            # Truncate long values for display
            value = result.value if len(result.value) <= 100 else result.value[:97] + "..."
            replaced_doc = replaced_doc.replace(mention_text, value)
        elif result.requires_llm:
            replaced_doc = replaced_doc.replace(mention_text, f"[LLM_GENERATE: {mention_text}]")

    print(replaced_doc)

    print("\n" + "=" * 80)
    print("✅ MENTION SYSTEM TEST COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
