"""
Entity Profile Manager - UI
Version: 1.0.0
Date: 2026-01-05

Purpose: Streamlit UI for creating and managing entity profiles for mention-based proposals.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import date
import yaml

# Set page config
st.set_page_config(
    page_title="Entity Profile Manager",
    page_icon="🏢",
    layout="wide"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import theme and components
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.ui_components import render_version_footer

# Apply theme
apply_theme()

# Import entity profile system
from cortex_engine.entity_profile_manager import EntityProfileManager
from cortex_engine.entity_profile_schema import (
    EntityType,
    EntityStatus,
    TeamMember,
    Qualification,
    Experience,
    Certification,
    Bio,
    Project,
    Timeline,
    Financials,
    ProjectTeam,
    ProjectDescription,
    Deliverable,
    Outcome,
    TeamRole,
    Reference,
    ReferenceRelationship,
    ReferenceAvailability,
    ReferenceContext,
    ReferenceConfidentiality,
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
    CapabilityType,
    Address
)
from cortex_engine.config_manager import ConfigManager
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger

logger = get_logger(__name__)

# ============================================
# INITIALIZATION
# ============================================

def initialize_manager():
    """Initialize entity profile manager."""
    if 'entity_manager' not in st.session_state:
        config = ConfigManager().get_config()
        db_path = convert_windows_to_wsl_path(config.get('ai_database_path'))
        st.session_state.entity_manager = EntityProfileManager(Path(db_path))

    return st.session_state.entity_manager

# ============================================
# MAIN UI
# ============================================

st.title("🏢 Entity Profile Manager")
st.markdown("Manage entity profiles for mention-based proposal system")
st.markdown("---")

manager = initialize_manager()

# ============================================
# SIDEBAR: ENTITY SELECTION
# ============================================

with st.sidebar:
    st.header("📂 Entity Profiles")

    # List entities
    profiles = manager.list_entity_profiles()

    if profiles:
        entity_names = {p.entity_name: p.entity_id for p in profiles}
        selected_name = st.selectbox(
            "Select Entity",
            options=["-- Create New --"] + list(entity_names.keys())
        )

        if selected_name != "-- Create New --":
            selected_entity_id = entity_names[selected_name]
        else:
            selected_entity_id = None
    else:
        st.info("No entity profiles yet. Create your first one below!")
        selected_entity_id = None

    st.markdown("---")

    # Quick stats for selected entity
    if selected_entity_id:
        profile = manager.get_entity_profile(selected_entity_id)

        st.metric("Team Members", len(manager.list_team_members(selected_entity_id)))
        st.metric("Projects", len(manager.list_projects(selected_entity_id)))
        st.metric("References", len(manager.list_references(selected_entity_id)))
        st.metric("Insurance Policies", len(manager.list_insurance(selected_entity_id)))
        st.metric("Capabilities", len(manager.list_capabilities(selected_entity_id)))

# ============================================
# MAIN CONTENT AREA
# ============================================

if selected_entity_id is None:
    # CREATE NEW ENTITY
    section_header("➕", "Create New Entity Profile", "Set up a new organization profile")

    with st.form("create_entity_form"):
        col1, col2 = st.columns(2)

        with col1:
            entity_id = st.text_input(
                "Entity ID*",
                help="URL-safe identifier (lowercase, underscores). E.g., 'longboardfella_consulting'"
            )
            entity_name = st.text_input(
                "Entity Name*",
                help="Display name. E.g., 'Longboardfella Consulting Pty Ltd'"
            )
            entity_type = st.selectbox(
                "Entity Type*",
                options=[t.value for t in EntityType]
            )

        with col2:
            legal_name = st.text_input(
                "Legal Name*",
                help="Official registered business name"
            )
            abn = st.text_input("ABN (11 digits)", max_chars=11)
            acn = st.text_input("ACN (9 digits)", max_chars=9)

        st.subheader("Contact Information")

        col1, col2 = st.columns(2)

        with col1:
            street = st.text_input("Street Address*")
            city = st.text_input("City*")
            state = st.text_input("State*")

        with col2:
            postcode = st.text_input("Postcode*")
            country = st.text_input("Country", value="Australia")

        phone = st.text_input("Phone*")
        email = st.text_input("Email*")
        website = st.text_input("Website")

        submit = st.form_submit_button("✅ Create Entity Profile", type="primary")

        if submit:
            try:
                # Validate required fields
                if not all([entity_id, entity_name, legal_name, street, city, state, postcode, phone, email]):
                    st.error("Please fill in all required fields (*)")
                else:
                    # Create address
                    address = Address(
                        street=street,
                        city=city,
                        state=state,
                        postcode=postcode,
                        country=country
                    )

                    # Create entity profile
                    profile = manager.create_entity_profile(
                        entity_id=entity_id,
                        entity_name=entity_name,
                        entity_type=EntityType(entity_type),
                        legal_name=legal_name,
                        abn=abn if abn else None,
                        acn=acn if acn else None,
                        address=address,
                        phone=phone,
                        email=email,
                        website=website if website else None
                    )

                    st.success(f"✅ Created entity profile: {entity_name}")
                    st.info("👉 Select the entity from the sidebar to manage team, projects, and more.")
                    st.rerun()

            except Exception as e:
                st.error(f"Error creating entity: {e}")
                logger.error(f"Failed to create entity: {e}", exc_info=True)

else:
    # MANAGE EXISTING ENTITY
    profile = manager.get_entity_profile(selected_entity_id)

    # Tabs for different sections
    tab_profile, tab_team, tab_projects, tab_references, tab_insurance, tab_capabilities, tab_narrative = st.tabs([
        "📋 Profile",
        "👥 Team",
        "🏗️ Projects",
        "📞 References",
        "🛡️ Insurance",
        "⭐ Capabilities",
        "📝 Narrative"
    ])

    # ========================================
    # TAB: PROFILE
    # ========================================

    with tab_profile:
        section_header("📋", "Entity Profile", f"Managing: {profile.metadata.entity_name}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Entity Type", profile.metadata.entity_type.value.replace('_', ' ').title())
            st.metric("Status", profile.metadata.status.value.title())

        with col2:
            st.metric("Created", profile.metadata.created_date.strftime("%Y-%m-%d"))
            st.metric("Last Updated", profile.metadata.last_updated.strftime("%Y-%m-%d"))

        with col3:
            st.metric("Version", profile.metadata.version)
            st.metric("Tags", len(profile.metadata.tags))

        st.markdown("---")

        # Company Information
        with st.expander("🏢 Company Information", expanded=True):
            st.write(f"**Legal Name:** {profile.company.legal_name}")
            st.write(f"**Trading Names:** {', '.join(profile.company.trading_names) if profile.company.trading_names else 'None'}")
            st.write(f"**ABN:** {profile.format_abn()}")
            st.write(f"**ACN:** {profile.format_acn()}")

        # Contact Information
        with st.expander("📧 Contact Information", expanded=True):
            st.write(f"**Address:**")
            st.write(profile.contact.registered_office.formatted(single_line=False))
            st.write(f"**Phone:** {profile.contact.phone}")
            st.write(f"**Email:** {profile.contact.email}")
            if profile.contact.website:
                st.write(f"**Website:** {profile.contact.website}")

        # Edit Profile Button
        if st.button("✏️ Edit Profile"):
            st.info("Profile editing coming soon! For now, edit the YAML file directly.")

    # ========================================
    # TAB: TEAM
    # ========================================

    with tab_team:
        section_header("👥", "Team Members", f"{len(manager.list_team_members(selected_entity_id))} members")

        # List team members
        team_members = manager.list_team_members(selected_entity_id)

        if team_members:
            for member in team_members:
                with st.expander(f"👤 {member.full_name} - {member.role}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Person ID:** {member.person_id}")
                        st.write(f"**Role:** {member.role}")
                        if member.email:
                            st.write(f"**Email:** {member.email}")
                        if member.phone:
                            st.write(f"**Phone:** {member.phone}")

                    with col2:
                        st.write(f"**Qualifications:** {len(member.qualifications)}")
                        st.write(f"**Experience:** {len(member.experience)}")
                        st.write(f"**Certifications:** {len(member.certifications)}")

                    if member.bio:
                        st.write("**Bio:**")
                        st.write(member.bio.brief)

                    if st.button(f"🗑️ Remove {member.full_name}", key=f"remove_team_{member.person_id}"):
                        manager.remove_team_member(selected_entity_id, member.person_id)
                        st.success(f"Removed {member.full_name}")
                        st.rerun()
        else:
            st.info("No team members yet. Add your first team member!")

        st.markdown("---")

        # Add Team Member
        with st.expander("➕ Add Team Member"):
            st.write("*For detailed team member profiles, edit YAML files directly for now.*")

            with st.form("add_team_member"):
                person_id = st.text_input("Person ID*", help="e.g., paul_smith")
                full_name = st.text_input("Full Name*")
                role = st.text_input("Role*")
                email = st.text_input("Email")
                phone = st.text_input("Phone")

                brief_bio = st.text_area("Brief Bio (2-3 sentences)")

                submit_team = st.form_submit_button("Add Team Member")

                if submit_team:
                    try:
                        bio = Bio(brief=brief_bio, full=brief_bio) if brief_bio else None

                        team_member = TeamMember(
                            person_id=person_id,
                            full_name=full_name,
                            role=role,
                            email=email if email else None,
                            phone=phone if phone else None,
                            bio=bio
                        )

                        manager.add_team_member(selected_entity_id, team_member)
                        st.success(f"✅ Added {full_name}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error adding team member: {e}")

    # ========================================
    # TAB: PROJECTS
    # ========================================

    with tab_projects:
        section_header("🏗️", "Projects", f"{len(manager.list_projects(selected_entity_id))} projects")

        # List projects
        projects = manager.list_projects(selected_entity_id)

        if projects:
            for project in projects:
                with st.expander(f"🏗️ {project.project_name} ({project.client})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Client:** {project.client}")
                        st.write(f"**Period:** {project.timeline.start_date.strftime('%Y-%m-%d')} to {project.timeline.end_date.strftime('%Y-%m-%d') if project.timeline.end_date else 'Ongoing'}")
                        st.write(f"**Duration:** {project.timeline.duration_months} months")

                    with col2:
                        st.write(f"**Value:** ${project.financials.contract_value:,.0f} {project.financials.currency}")
                        st.write(f"**Team Size:** {project.team.size}")

                    st.write("**Description:**")
                    st.write(project.description.brief)

                    if project.deliverables:
                        st.write("**Deliverables:**")
                        for d in project.deliverables:
                            st.write(f"- {d.name}")

                    if project.outcomes:
                        st.write("**Outcomes:**")
                        for o in project.outcomes:
                            st.write(f"- {o.metric}: {o.improvement}")

                    if st.button(f"🗑️ Remove Project", key=f"remove_project_{project.project_id}"):
                        manager.remove_project(selected_entity_id, project.project_id)
                        st.success(f"Removed {project.project_name}")
                        st.rerun()
        else:
            st.info("No projects yet. Add your first project!")

        st.markdown("---")
        st.info("💡 **Tip:** For detailed project profiles with deliverables and outcomes, edit YAML files directly.")

    # ========================================
    # TAB: REFERENCES
    # ========================================

    with tab_references:
        section_header("📞", "References", f"{len(manager.list_references(selected_entity_id))} references")

        # List references
        references = manager.list_references(selected_entity_id)

        if references:
            for ref in references:
                with st.expander(f"📞 {ref.contact_name} ({ref.organization})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Title:** {ref.title}")
                        st.write(f"**Organization:** {ref.organization}")
                        st.write(f"**Relationship:** {ref.relationship.type.value.title()} - {ref.relationship.role}")

                    with col2:
                        if ref.email:
                            st.write(f"**Email:** {ref.email}")
                        if ref.phone:
                            st.write(f"**Phone:** {ref.phone}")
                        st.write(f"**Available:** {'Yes' if ref.availability.available else 'No'}")

                    if ref.context:
                        st.write("**Working Relationship:**")
                        st.write(ref.context.working_relationship)

                    if ref.quote:
                        st.info(f"💬 *\"{ref.quote}\"*")

                    if st.button(f"🗑️ Remove Reference", key=f"remove_ref_{ref.reference_id}"):
                        manager.remove_reference(selected_entity_id, ref.reference_id)
                        st.success(f"Removed {ref.contact_name}")
                        st.rerun()
        else:
            st.info("No references yet. Add your first reference!")

        st.markdown("---")
        st.info("💡 **Tip:** For detailed reference profiles, edit YAML files directly.")

    # ========================================
    # TAB: INSURANCE
    # ========================================

    with tab_insurance:
        section_header("🛡️", "Insurance Policies", f"{len(manager.list_insurance(selected_entity_id))} policies")

        # List insurance
        insurance_policies = manager.list_insurance(selected_entity_id)

        if insurance_policies:
            for policy in insurance_policies:
                # Check if expired
                is_expired = policy.dates.is_expired
                days_until_expiry = policy.dates.days_until_expiry

                status_emoji = "🔴" if is_expired else "🟡" if days_until_expiry < 60 else "🟢"

                with st.expander(f"{status_emoji} {policy.policy_type.value.replace('_', ' ').title()} - {policy.coverage.formatted}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Insurer:** {policy.insurer}")
                        st.write(f"**Policy Number:** {policy.policy_number}")
                        st.write(f"**Coverage:** {policy.coverage.formatted}")

                    with col2:
                        st.write(f"**Effective:** {policy.dates.effective_date.strftime('%Y-%m-%d')}")
                        st.write(f"**Expiry:** {policy.dates.expiry_date.strftime('%Y-%m-%d')}")

                        if is_expired:
                            st.error(f"**Status:** EXPIRED")
                        elif days_until_expiry < 60:
                            st.warning(f"**Status:** Expires in {days_until_expiry} days")
                        else:
                            st.success(f"**Status:** Current ({days_until_expiry} days remaining)")

                    st.write(f"**Description:** {policy.coverage.description}")

                    if st.button(f"🗑️ Remove Policy", key=f"remove_insurance_{policy.policy_id}"):
                        manager.remove_insurance(selected_entity_id, policy.policy_id)
                        st.success(f"Removed policy")
                        st.rerun()
        else:
            st.info("No insurance policies yet. Add your first policy!")

        st.markdown("---")
        st.info("💡 **Tip:** For detailed insurance profiles, edit YAML files directly.")

    # ========================================
    # TAB: CAPABILITIES
    # ========================================

    with tab_capabilities:
        section_header("⭐", "Capabilities & Certifications", f"{len(manager.list_capabilities(selected_entity_id))} capabilities")

        # List capabilities
        capabilities = manager.list_capabilities(selected_entity_id)

        if capabilities:
            for cap in capabilities:
                is_expired = cap.dates.is_expired
                status_emoji = "🔴" if is_expired else "🟢"

                with st.expander(f"{status_emoji} {cap.capability_name}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Type:** {cap.capability_type.value.title()}")
                        if cap.certification_body:
                            st.write(f"**Certification Body:** {cap.certification_body}")
                        if cap.certification_number:
                            st.write(f"**Certification #:** {cap.certification_number}")

                    with col2:
                        st.write(f"**Obtained:** {cap.dates.obtained.strftime('%Y-%m-%d')}")
                        if cap.dates.expiry:
                            st.write(f"**Expiry:** {cap.dates.expiry.strftime('%Y-%m-%d')}")
                            if is_expired:
                                st.error("**Status:** EXPIRED")
                            else:
                                st.success("**Status:** Current")
                        else:
                            st.info("**Status:** No expiry")

                    st.write(f"**Description:** {cap.description.brief}")

                    if st.button(f"🗑️ Remove Capability", key=f"remove_cap_{cap.capability_id}"):
                        manager.remove_capability(selected_entity_id, cap.capability_id)
                        st.success(f"Removed capability")
                        st.rerun()
        else:
            st.info("No capabilities yet. Add your first capability!")

        st.markdown("---")
        st.info("💡 **Tip:** For detailed capability profiles, edit YAML files directly.")

    # ========================================
    # TAB: NARRATIVE
    # ========================================

    with tab_narrative:
        section_header("📝", "Narrative Content", "Long-form content for proposals")

        # Get current narrative
        narrative = manager.get_narrative(selected_entity_id)

        # Editable text area
        new_narrative = st.text_area(
            "Narrative Content (Markdown)",
            value=narrative,
            height=500,
            help="Edit your narrative content in Markdown format. Use ## headings for sections."
        )

        col1, col2 = st.columns([1, 4])

        with col1:
            if st.button("💾 Save Narrative", type="primary"):
                manager.update_narrative(selected_entity_id, new_narrative)
                st.success("✅ Narrative saved!")
                st.rerun()

        with col2:
            if st.button("🔄 Reset to Original"):
                st.rerun()

        st.markdown("---")

        st.info("""
        **💡 Tip:** Use this format for narrative sections:

        ```markdown
        ## Company Overview
        Your company overview text here...

        ## Core Capabilities
        Your capabilities description...

        ## Competitive Advantages
        Your competitive advantages...
        ```

        These sections can be referenced in proposals using @narrative[section_name]
        """)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
render_version_footer("Entity Profile Manager", "v1.0.0", "2026-01-05")
