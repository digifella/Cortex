"""
Test script to populate workspace with fake company data for Phase 2 testing.

This bypasses the extraction process and directly creates structured data
so we can test the field matching UI.
"""

import sys
from pathlib import Path
from datetime import datetime, date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cortex_engine.workspace_manager import WorkspaceManager
from cortex_engine.tender_schema import (
    OrganizationProfile,
    Insurance,
    InsuranceType,
    Qualification,
    QualificationType,
    WorkExperience,
    ProjectExperience,
    Reference,
    Capability,
    StructuredKnowledge
)
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger
from cortex_engine.config_manager import ConfigManager

logger = get_logger(__name__)


def create_fake_company_data():
    """Create realistic fake company data for Longboardfella Consulting."""

    # Organization profile
    organization = OrganizationProfile(
        legal_name="Longboardfella Consulting Pty Ltd",
        trading_names=["Longboardfella", "LBF Consulting"],
        abn="12345678901",
        acn="123456789",
        address={
            "street": "123 Beach Road",
            "city": "Sydney",
            "state": "NSW",
            "postcode": "2000",
            "country": "Australia"
        },
        postal_address=None,
        phone="+61 2 1234 5678",
        email="info@longboardfella.com.au",
        website="https://www.longboardfella.com.au",
        last_updated=datetime.now(),
        source_documents=["test_data"]
    )

    # Insurance policies
    insurances = [
        Insurance(
            insurance_type=InsuranceType.PUBLIC_LIABILITY,
            insurer="Insurance Australia Group",
            policy_number="PL-2024-123456",
            coverage_amount=20000000.0,
            effective_date=date(2024, 1, 1),
            expiry_date=date(2025, 12, 31),
            coverage_description="Public and products liability coverage",
            source_documents=["test_data"]
        ),
        Insurance(
            insurance_type=InsuranceType.PROFESSIONAL_INDEMNITY,
            insurer="QBE Insurance",
            policy_number="PI-2024-789012",
            coverage_amount=10000000.0,
            effective_date=date(2024, 1, 1),
            expiry_date=date(2025, 12, 31),
            coverage_description="Professional indemnity for consulting services",
            source_documents=["test_data"]
        ),
        Insurance(
            insurance_type=InsuranceType.WORKERS_COMPENSATION,
            insurer="Allianz",
            policy_number="WC-2024-345678",
            coverage_amount=5000000.0,
            effective_date=date(2024, 7, 1),
            expiry_date=date(2025, 6, 30),
            coverage_description="Workers compensation for all employees",
            source_documents=["test_data"]
        )
    ]

    # Team qualifications
    qualifications = [
        Qualification(
            person_name="Paul Smith",
            qualification_name="Master of Business Administration",
            qualification_type=QualificationType.DEGREE,
            institution="University of Technology Sydney",
            date_obtained=date(2015, 6, 1),
            description="Business Strategy specialization",
            source_documents=["test_data"]
        ),
        Qualification(
            person_name="Paul Smith",
            qualification_name="Certified Management Consultant",
            qualification_type=QualificationType.CERTIFICATION,
            institution="Institute of Management Consultants",
            date_obtained=date(2018, 3, 15),
            description="Management Consulting certification",
            source_documents=["test_data"]
        )
    ]

    # Work experience
    work_experiences = [
        WorkExperience(
            person_name="Paul Smith",
            role="Senior Consultant",
            organization="Longboardfella Consulting",
            start_date=date(2015, 1, 1),
            end_date=None,  # Current
            responsibilities=["Leading consulting engagements", "Strategic advisory", "Project management"],
            source_documents=["test_data"]
        ),
        WorkExperience(
            person_name="Paul Smith",
            role="Business Analyst",
            organization="Deloitte Consulting",
            start_date=date(2010, 6, 1),
            end_date=date(2014, 12, 31),
            responsibilities=["Business process analysis", "Requirements gathering", "Stakeholder management"],
            source_documents=["test_data"]
        )
    ]

    # Project experience
    projects = [
        ProjectExperience(
            project_name="Department of Health Digital Transformation",
            client="Australian Department of Health",
            start_date=date(2023, 1, 1),
            end_date=date(2024, 6, 30),
            description="Led digital transformation initiative to modernize health data systems and improve patient outcomes",
            role="Lead Consultant",
            value=850000.0,
            deliverables=["Strategic roadmap", "Implementation plan", "Change management framework"],
            outcomes=["30% improvement in data processing", "Reduced reporting time by 50%"],
            technologies=["Cloud platforms", "Data analytics", "API integration"],
            team_size=8,
            source_documents=["test_data"]
        ),
        ProjectExperience(
            project_name="State Government Procurement Reform",
            client="NSW Department of Finance",
            start_date=date(2022, 3, 1),
            end_date=date(2023, 2, 28),
            description="Redesigned procurement processes and implemented new digital procurement platform",
            role="Strategy Advisor",
            value=620000.0,
            deliverables=["Process redesign", "Platform selection", "Training materials"],
            outcomes=["40% reduction in procurement cycle time", "Improved supplier engagement"],
            technologies=["E-procurement systems", "Workflow automation"],
            team_size=5,
            source_documents=["test_data"]
        )
    ]

    # Client references
    references = [
        Reference(
            contact_name="Dr. Sarah Johnson",
            contact_title="Director of Digital Strategy",
            organization="Australian Department of Health",
            phone="+61 2 6289 1234",
            email="sarah.johnson@health.gov.au",
            relationship="Client - Project Sponsor",
            project_context="Digital Transformation project 2023-2024",
            source_documents=["test_data"]
        ),
        Reference(
            contact_name="Michael Chen",
            contact_title="Chief Procurement Officer",
            organization="NSW Department of Finance",
            phone="+61 2 9228 5678",
            email="michael.chen@finance.nsw.gov.au",
            relationship="Client - Engagement Manager",
            project_context="Procurement Reform project 2022-2023",
            source_documents=["test_data"]
        ),
        Reference(
            contact_name="Emma Williams",
            contact_title="Program Director",
            organization="Department of Veterans' Affairs",
            phone="+61 3 9281 9012",
            email="emma.williams@dva.gov.au",
            relationship="Former Client",
            project_context="Veterans services improvement program 2021",
            source_documents=["test_data"]
        )
    ]

    # Organizational capabilities
    capabilities = [
        Capability(
            capability_name="ISO 9001:2015 Quality Management",
            description="Certified quality management system for consulting services",
            certification_body="SAI Global",
            certification_number="QMS-2024-AUS-12345",
            date_obtained=date(2020, 3, 15),
            expiry_date=date(2026, 3, 14),
            scope="Management consulting and advisory services",
            source_documents=["test_data"]
        )
    ]

    # Create StructuredKnowledge object
    structured_data = StructuredKnowledge(
        organization=organization,
        insurances=insurances,
        team_qualifications=qualifications,
        team_work_experience=work_experiences,
        projects=projects,
        references=references,
        capabilities=capabilities,
        extraction_date=datetime.now(),
        kb_version="test_v1.0",
        total_documents_processed=4
    )

    return structured_data


def populate_workspace_with_test_data(workspace_id: str):
    """Populate a workspace with test data."""

    print("=" * 80)
    print("WORKSPACE TEST DATA POPULATION")
    print("=" * 80)

    # Get workspace manager
    config_manager = ConfigManager()
    config = config_manager.get_config()
    db_path = config.get('ai_database_path')
    wsl_db_path = convert_windows_to_wsl_path(db_path)

    workspace_manager = WorkspaceManager(Path(wsl_db_path))

    # Check workspace exists
    workspace = workspace_manager.get_workspace(workspace_id)
    if not workspace:
        print(f"❌ Workspace {workspace_id} not found!")
        print(f"\nAvailable workspaces:")
        for ws in workspace_manager.list_workspaces():
            print(f"  - {ws.workspace_id}")
        return False

    print(f"\n✅ Found workspace: {workspace.workspace_name}")
    print(f"   ID: {workspace.workspace_id}")
    print(f"   Status: {workspace.status}")

    # Create test data
    print(f"\n📊 Creating test company data...")
    structured_data = create_fake_company_data()

    # Show summary
    stats = structured_data.summary_stats
    print(f"\n✅ Created test data:")
    print(f"   Organization: {structured_data.organization.legal_name}")
    print(f"   ABN: {structured_data.organization.abn}")
    print(f"   Insurances: {stats['insurances']}")
    print(f"   Qualifications: {stats['qualifications']}")
    print(f"   Work Experiences: {stats['work_experiences']}")
    print(f"   Projects: {stats['projects']}")
    print(f"   References: {stats['references']}")
    print(f"   Capabilities: {stats['capabilities']}")

    # Populate workspace
    print(f"\n📦 Populating workspace...")

    # Import populate method
    from cortex_engine.tender_data_extractor import TenderDataExtractor
    from cortex_engine.adaptive_model_manager import AdaptiveModelManager
    import asyncio

    # Create temporary extractor just for the populate method
    model_manager = AdaptiveModelManager()
    extractor = TenderDataExtractor(
        vector_index=None,  # Not needed for populate
        knowledge_graph=None,
        model_manager=model_manager,
        db_path=Path(wsl_db_path)
    )

    success = asyncio.run(
        extractor.populate_workspace_with_extraction(
            workspace_manager=workspace_manager,
            workspace_id=workspace_id,
            structured_data=structured_data
        )
    )

    if success:
        # Update workspace status
        from cortex_engine.workspace_schema import WorkspaceStatus
        workspace_manager.update_workspace_status(workspace_id, WorkspaceStatus.IN_PROGRESS)

        # Refresh workspace
        updated_workspace = workspace_manager.get_workspace(workspace_id)

        print(f"\n✅ Workspace populated successfully!")
        print(f"   Documents in workspace: {updated_workspace.document_count}")
        print(f"   Status: {updated_workspace.status}")

        print(f"\n" + "=" * 80)
        print(f"✨ SUCCESS! Workspace ready for Phase 2 field matching testing")
        print(f"=" * 80)
        print(f"\nNext steps:")
        print(f"1. Go to Proposal Workflow")
        print(f"2. Select this workspace")
        print(f"3. Go to Step 4: Match Fields")
        print(f"4. Test the field matching UI with this data")

        return True
    else:
        print(f"\n❌ Failed to populate workspace")
        return False


if __name__ == "__main__":
    # Default to the current workspace
    workspace_id = "workspace_RFT12493_noentity_2026-01-05"

    # Allow override via command line
    if len(sys.argv) > 1:
        workspace_id = sys.argv[1]

    try:
        success = populate_workspace_with_test_data(workspace_id)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)
