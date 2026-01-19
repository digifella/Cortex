#!/usr/bin/env python3
"""
Test Tender Fill
Demonstrates filling a tender document with @mentions.
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

# Sample tender document with @mentions
SAMPLE_TENDER = """
================================================================================
REQUEST FOR TENDER - RFT12345
Department of Digital Services
CONSULTING SERVICES - DIGITAL TRANSFORMATION
================================================================================

SECTION 1: COMPANY DETAILS
================================================================================

1.1 Legal Entity Name
@companyname

1.2 Australian Business Number (ABN)
@abn

1.3 Australian Company Number (ACN)
@acn

1.4 Registered Office Address
@registered_office

1.5 Contact Details
Phone: @phone
Email: @email
Website: @website


SECTION 2: EXECUTIVE SUMMARY
================================================================================

@narrative[company_overview]


SECTION 3: COMPANY PROFILE
================================================================================

3.1 Insurance Coverage

We maintain comprehensive insurance coverage appropriate for consulting services:

Public Liability Insurance:
- Coverage Amount: @insurance.public_liability.coverage
- Policy Number: @insurance.public_liability.policy_number
- Insurer: @insurance.public_liability.insurer


SECTION 4: TEAM COMPOSITION
================================================================================

4.1 Proposed Team Lead

Name: @team.paul_smith.full_name
Role: @team.paul_smith.role
Email: @team.paul_smith.email
Phone: @team.paul_smith.phone

Brief Biography:
@team.paul_smith.bio.brief

Detailed CV:
[Note: This would typically require LLM generation]
@cv[paul_smith]


SECTION 5: RELEVANT EXPERIENCE
================================================================================

5.1 Recent Similar Project

@project_summary[health_transformation_2023]

[Note: This would typically require LLM generation with project details]


SECTION 6: REFERENCES
================================================================================

6.1 Client Reference

@reference[sarah_johnson]

[Note: This would typically require LLM generation formatting the reference nicely]


SECTION 7: CERTIFICATIONS AND CAPABILITIES
================================================================================

We maintain the following certifications relevant to this engagement:

@capability.iso_9001_2015.capability_name
- Certification Body: @capability.iso_9001_2015.certification_body
- Certificate Number: @capability.iso_9001_2015.certification_number
- Status: @capability.iso_9001_2015.status

================================================================================
END OF TENDER RESPONSE
================================================================================
"""


def main():
    print("=" * 80)
    print("TENDER FILL TEST")
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

    # Show original document
    print("\n\n" + "=" * 80)
    print("ORIGINAL TENDER DOCUMENT (WITH @MENTIONS)")
    print("=" * 80)
    print(SAMPLE_TENDER)

    # Parse all mentions
    print("\n\n" + "=" * 80)
    print("PARSING MENTIONS")
    print("=" * 80)

    mentions = parser.parse_all(SAMPLE_TENDER)
    print(f"\n✅ Found {len(mentions)} mentions")

    for i, mention in enumerate(mentions, 1):
        print(f"\n{i:2d}. {mention.raw_text}")
        print(f"    Type: {mention.mention_type.value}")
        print(f"    Path: {mention.field_path}")

    # Resolve all mentions
    print("\n\n" + "=" * 80)
    print("RESOLVING MENTIONS")
    print("=" * 80)

    results = engine.resolve_all(mentions, entity_id)

    successful = sum(1 for r in results.values() if r.success)
    requires_llm = sum(1 for r in results.values() if r.requires_llm)
    errors = sum(1 for r in results.values() if not r.success)

    print(f"\n📊 Resolution Summary:")
    print(f"   Total Mentions: {len(results)}")
    print(f"   ✅ Successfully Resolved: {successful}")
    print(f"   ⏳ Requires LLM: {requires_llm}")
    print(f"   ❌ Errors: {errors}")

    if errors > 0:
        print("\n⚠️  Errors:")
        for mention_text, result in results.items():
            if not result.success:
                print(f"   - {mention_text}: {result.error_message}")

    # Generate filled document
    print("\n\n" + "=" * 80)
    print("FILLED TENDER DOCUMENT")
    print("=" * 80)

    filled_document = SAMPLE_TENDER

    for mention_text, result in results.items():
        if result.success:
            if result.requires_llm:
                # Mark for LLM generation
                replacement = f"\n[🤖 LLM GENERATION REQUIRED]\n[Mention: {mention_text}]\n[Context: {result.generation_context}]\n"
            else:
                # Use resolved value
                replacement = result.value

            filled_document = filled_document.replace(mention_text, replacement)

    print(filled_document)

    # Summary
    print("\n\n" + "=" * 80)
    print("✅ TENDER FILL TEST COMPLETE!")
    print("=" * 80)

    print(f"""
📊 Summary:
   - Original mentions: {len(mentions)}
   - Resolved immediately: {successful}
   - Requiring LLM generation: {requires_llm}
   - Errors: {errors}

💡 Next Steps:
   1. Mentions marked [🤖 LLM GENERATION REQUIRED] would be processed by:
      - Loading relevant data from entity profile
      - Sending to LLM with context
      - Formatting output appropriately

   2. For full workflow, implement Phase 2:
      - Workspace creation and management
      - Document upload and markup suggestion
      - Human review interface
      - LLM content generation
      - Final document export
""")


if __name__ == "__main__":
    main()
