#!/usr/bin/env python3
"""
Create Test Entity Profile
Quick script to create Longboardfella Consulting entity profile for testing.
"""

import sys
from pathlib import Path
from datetime import date

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cortex_engine.entity_profile_manager import EntityProfileManager
from cortex_engine.entity_profile_schema import (
    EntityType,
    Address,
    TeamMember,
    Bio,
    Qualification,
    Experience,
    Project,
    Timeline,
    Financials,
    ProjectTeam,
    TeamRole,
    ProjectDescription,
    Deliverable,
    Outcome,
    Reference,
    ReferenceRelationship,
    ReferenceAvailability,
    ReferenceContext,
    RelationshipType,
    Insurance,
    Coverage,
    InsuranceDates,
    InsuranceScope,
    InsuranceType,
    RenewalStatus,
    Capability,
    CapabilityDescription,
    CapabilityDates,
    CapabilityScope,
    CapabilityType
)
from cortex_engine.config_manager import ConfigManager
from cortex_engine.utils import convert_windows_to_wsl_path

def main():
    print("=" * 80)
    print("Creating Longboardfella Consulting Entity Profile")
    print("=" * 80)

    # Initialize manager
    config = ConfigManager().get_config()
    db_path = convert_windows_to_wsl_path(config.get('ai_database_path'))
    manager = EntityProfileManager(Path(db_path))

    # Create entity profile
    print("\n1️⃣ Creating entity profile...")

    address = Address(
        street="123 Beach Road",
        city="Sydney",
        state="NSW",
        postcode="2000",
        country="Australia"
    )

    profile = manager.create_entity_profile(
        entity_id="longboardfella_consulting",
        entity_name="Longboardfella Consulting Pty Ltd",
        entity_type=EntityType.CONSULTING_FIRM,
        legal_name="Longboardfella Consulting Pty Ltd",
        abn="12345678901",
        acn="123456789",
        address=address,
        phone="+61 2 1234 5678",
        email="info@longboardfella.com.au",
        website="https://www.longboardfella.com.au",
        trading_names=["Longboardfella", "LBF Consulting"]
    )

    print(f"✅ Created: {profile.metadata.entity_name}")
    print(f"   ABN: {profile.format_abn()}")
    print(f"   ACN: {profile.format_acn()}")

    # Add team member
    print("\n2️⃣ Adding team member...")

    paul_bio = Bio(
        brief="Paul Smith is a senior consultant with over 14 years of experience in strategic advisory and digital transformation for government agencies.",
        full="""Paul Smith is a senior consultant specializing in digital transformation and process improvement for government agencies. With an MBA from UTS and over 14 years of experience, Paul has led numerous high-value engagements across health, finance, and veterans' affairs sectors.

Paul's expertise includes strategic planning, change management, and stakeholder engagement for complex government IT initiatives. He has a proven track record of delivering measurable outcomes, including a recent $850K digital health transformation that achieved 30% improvement in data processing efficiency."""
    )

    paul = TeamMember(
        person_id="paul_smith",
        full_name="Paul Smith",
        preferred_name="Paul",
        role="Senior Consultant",
        email="paul.smith@longboardfella.com.au",
        phone="+61 400 123 456",
        qualifications=[
            Qualification(
                name="Master of Business Administration",
                institution="University of Technology Sydney",
                year=2015,
                specialization="Business Strategy"
            ),
            Qualification(
                name="Certified Management Consultant",
                institution="Institute of Management Consultants",
                year=2018
            )
        ],
        experience=[
            Experience(
                role="Senior Consultant",
                organization="Longboardfella Consulting",
                start_date=date(2015, 1, 1),
                end_date=None,
                location="Sydney, Australia",
                responsibilities=[
                    "Leading consulting engagements for government agencies",
                    "Strategic advisory on digital transformation",
                    "Project management for complex initiatives"
                ],
                achievements=[
                    "Led $850K digital health transformation project",
                    "Delivered 30% improvement in client data processing"
                ]
            ),
            Experience(
                role="Business Analyst",
                organization="Deloitte Consulting",
                start_date=date(2010, 6, 1),
                end_date=date(2014, 12, 31),
                location="Sydney, Australia",
                responsibilities=[
                    "Business process analysis",
                    "Requirements gathering",
                    "Stakeholder management"
                ]
            )
        ],
        bio=paul_bio
    )

    manager.add_team_member("longboardfella_consulting", paul)
    print(f"✅ Added: {paul.full_name} - {paul.role}")

    # Add projects
    print("\n3️⃣ Adding projects...")

    health_project = Project(
        project_id="health_transformation_2023",
        project_name="Department of Health Digital Transformation",
        client="Australian Department of Health",
        sector="government",
        project_type="digital_transformation",
        timeline=Timeline(
            start_date=date(2023, 1, 1),
            end_date=date(2024, 6, 30),
            duration_months=18
        ),
        financials=Financials(
            contract_value=850000.00,
            currency="AUD",
            payment_structure="milestone_based"
        ),
        team=ProjectTeam(
            size=8,
            our_staff=["paul_smith"],
            roles=[
                TeamRole(role="Lead Consultant", person="paul_smith")
            ]
        ),
        description=ProjectDescription(
            brief="Led digital transformation initiative to modernize health data systems and improve patient outcomes.",
            full="""Longboardfella Consulting led a comprehensive digital transformation initiative for the Australian Department of Health, modernizing legacy data systems and implementing cloud-based analytics platforms. The 18-month engagement involved strategic planning, system architecture, change management, and stakeholder engagement across multiple departmental divisions."""
        ),
        deliverables=[
            Deliverable(name="Strategic Roadmap", description="5-year digital transformation strategy"),
            Deliverable(name="Implementation Plan", description="Detailed 18-month delivery plan"),
            Deliverable(name="Change Management Framework", description="Stakeholder engagement program")
        ],
        outcomes=[
            Outcome(
                metric="Data processing efficiency",
                improvement="30% improvement",
                measurement="Processing time reduction"
            ),
            Outcome(
                metric="Reporting time",
                improvement="50% reduction",
                measurement="Time to generate monthly reports"
            )
        ],
        technologies=["Cloud platforms (AWS)", "Data analytics", "API integration"]
    )

    manager.add_project("longboardfella_consulting", health_project)
    print(f"✅ Added: {health_project.project_name} (${health_project.financials.contract_value:,.0f})")

    # Add references
    print("\n4️⃣ Adding references...")

    sarah_ref = Reference(
        reference_id="sarah_johnson",
        contact_name="Dr. Sarah Johnson",
        title="Director of Digital Strategy",
        organization="Australian Department of Health",
        email="sarah.johnson@health.gov.au",
        phone="+61 2 6289 1234",
        relationship=ReferenceRelationship(
            type=RelationshipType.CLIENT,
            role="Project Sponsor",
            projects=["health_transformation_2023"]
        ),
        availability=ReferenceAvailability(
            available=True,
            preferred_contact="email"
        ),
        context=ReferenceContext(
            working_relationship="Sarah was the Project Sponsor for our 2023-2024 digital health transformation engagement.",
            can_speak_to=[
                "Strategic planning capabilities",
                "Stakeholder management skills",
                "Delivery quality and timeliness"
            ]
        ),
        quote="Longboardfella Consulting delivered exceptional results on our digital transformation initiative. Their strategic approach and stakeholder engagement skills were instrumental in achieving a 30% improvement in our data processing efficiency."
    )

    manager.add_reference("longboardfella_consulting", sarah_ref)
    print(f"✅ Added: {sarah_ref.contact_name} ({sarah_ref.organization})")

    # Add insurance
    print("\n5️⃣ Adding insurance policies...")

    public_liability = Insurance(
        policy_id="public_liability",
        policy_type=InsuranceType.PUBLIC_LIABILITY,
        insurer="Insurance Australia Group",
        policy_number="PL-2024-123456",
        coverage=Coverage(
            amount=20000000.00,
            currency="AUD",
            formatted="$20,000,000",
            description="Public and products liability coverage"
        ),
        dates=InsuranceDates(
            effective_date=date(2024, 1, 1),
            expiry_date=date(2025, 12, 31),
            renewal_status=RenewalStatus.CURRENT
        ),
        scope=InsuranceScope(
            geographic="Australia and New Zealand",
            activities=[
                "Management consulting services",
                "Strategic advisory",
                "Digital transformation consulting"
            ],
            exclusions=["Software development", "Direct employment services"]
        )
    )

    manager.add_insurance("longboardfella_consulting", public_liability)
    print(f"✅ Added: {public_liability.policy_type.value} - {public_liability.coverage.formatted}")

    # Add capability
    print("\n6️⃣ Adding capabilities...")

    iso_cert = Capability(
        capability_id="iso_9001_2015",
        capability_name="ISO 9001:2015 Quality Management",
        capability_type=CapabilityType.CERTIFICATION,
        description=CapabilityDescription(
            brief="Certified quality management system for consulting services",
            full="Longboardfella Consulting maintains ISO 9001:2015 certification for quality management in consulting and advisory services."
        ),
        certification_body="SAI Global",
        certification_number="QMS-2024-AUS-12345",
        dates=CapabilityDates(
            obtained=date(2020, 3, 15),
            expiry=date(2026, 3, 14)
        ),
        scope=CapabilityScope(
            services=[
                "Management consulting",
                "Strategic advisory",
                "Digital transformation consulting"
            ],
            locations=["Sydney, Australia"],
            standards=["ISO 9001:2015"]
        )
    )

    manager.add_capability("longboardfella_consulting", iso_cert)
    print(f"✅ Added: {iso_cert.capability_name}")

    # Update narrative
    print("\n7️⃣ Adding narrative content...")

    narrative = """# Longboardfella Consulting Pty Ltd

## Company Overview

Longboardfella Consulting is a leading strategic advisory firm specializing in digital transformation and process improvement for government agencies. With over a decade of experience in the Australian public sector, we help organizations modernize their systems, optimize processes, and deliver measurable outcomes.

Our team combines deep technical expertise with practical government experience, enabling us to navigate complex stakeholder landscapes and deliver sustainable change. We pride ourselves on our collaborative approach, working closely with clients to build internal capability while achieving project objectives.

## Core Capabilities

### Digital Transformation
We help government agencies modernize legacy systems and embrace digital technologies. Our approach combines strategic planning, technical architecture, change management, and stakeholder engagement to deliver sustainable transformation.

**Key Services:**
- Digital strategy development
- Cloud migration and implementation
- Data analytics and business intelligence
- API integration and system modernization
- Agile delivery and DevOps practices

### Strategic Advisory
Our consultants provide expert guidance on complex policy, process, and technology challenges. We work at the intersection of strategy, operations, and technology to help agencies achieve their missions more effectively.

**Key Services:**
- Strategic planning and roadmaps
- Business case development
- Process optimization and redesign
- Organizational change management
- Governance framework design

### Stakeholder Engagement
We excel at managing complex stakeholder landscapes typical of government projects. Our proven methodologies ensure all voices are heard while maintaining project momentum and achieving consensus.

**Key Services:**
- Stakeholder analysis and mapping
- Engagement strategy and planning
- Workshop facilitation
- Change impact assessment
- Communication planning and execution

## Competitive Advantages

1. **Government Expertise:** Over 90% of our work is with Australian government agencies. We understand the unique challenges, constraints, and opportunities in the public sector.

2. **Proven Track Record:** We have successfully delivered over 50 major engagements, with an average client satisfaction score of 4.8/5.0.

3. **Outcome Focus:** Our projects deliver measurable results. Recent achievements include 30% efficiency improvements, 50% reduction in processing times, and 89% user satisfaction rates.

4. **Quality Assurance:** ISO 9001:2015 certified quality management system ensures consistent delivery excellence across all engagements.

5. **Knowledge Transfer:** We don't just do the work—we build client capability through structured knowledge transfer and coaching.

## Quality Assurance

Longboardfella Consulting maintains ISO 9001:2015 certification, demonstrating our commitment to quality management excellence. Our quality management system covers:

- Project delivery methodology
- Risk management and mitigation
- Quality assurance and review processes
- Continuous improvement practices
- Client satisfaction measurement

We conduct regular internal audits and maintain comprehensive project documentation to ensure consistent delivery quality and continuous improvement.

## Sustainability Commitment

We are committed to sustainable business practices and minimizing our environmental impact:

- Carbon-neutral operations through offset programs
- Paperless project delivery where possible
- Sustainable procurement practices
- Support for Indigenous business participation
- Community engagement and pro bono work for not-for-profit organizations
"""

    manager.update_narrative("longboardfella_consulting", narrative)
    print("✅ Added narrative content (6 sections)")

    # Summary
    print("\n" + "=" * 80)
    print("✨ ENTITY PROFILE CREATED SUCCESSFULLY!")
    print("=" * 80)

    profile = manager.get_entity_profile("longboardfella_consulting")
    print(f"\n📊 Summary:")
    print(f"   Entity: {profile.metadata.entity_name}")
    print(f"   ABN: {profile.format_abn()}")
    print(f"   Team Members: {len(manager.list_team_members('longboardfella_consulting'))}")
    print(f"   Projects: {len(manager.list_projects('longboardfella_consulting'))}")
    print(f"   References: {len(manager.list_references('longboardfella_consulting'))}")
    print(f"   Insurance: {len(manager.list_insurance('longboardfella_consulting'))}")
    print(f"   Capabilities: {len(manager.list_capabilities('longboardfella_consulting'))}")

    print(f"\n📁 Files created at:")
    print(f"   {manager.profiles_dir / 'longboardfella_consulting'}")

    print("\n✅ Ready to test mention system!")

if __name__ == "__main__":
    main()
