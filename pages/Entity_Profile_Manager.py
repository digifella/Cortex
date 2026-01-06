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
        # Helper text
        st.info("""
        💡 **Quick Guide:**
        - **Entity ID**: URL-safe identifier (e.g., `longboardfella_consulting`) - lowercase, underscores only
        - **Entity Name**: Display name (e.g., `Longboardfella Consulting Pty Ltd`) - can have spaces, capitals
        """)

        col1, col2 = st.columns(2)

        with col1:
            entity_name = st.text_input(
                "Entity Name*",
                placeholder="Longboardfella Consulting Pty Ltd",
                help="Display name - can include spaces, capitals, special characters"
            )

            # Auto-suggest Entity ID based on Entity Name
            suggested_id = ""
            if entity_name:
                # Convert to URL-safe ID: lowercase, replace spaces with underscores, remove special chars
                suggested_id = entity_name.lower()
                suggested_id = suggested_id.replace(' ', '_')
                suggested_id = ''.join(c for c in suggested_id if c.isalnum() or c in ['_', '-'])

            entity_id = st.text_input(
                "Entity ID*",
                value=suggested_id,
                placeholder="longboardfella_consulting",
                help="Auto-generated from Entity Name. Must be lowercase, underscores/hyphens only."
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
    tab_profile, tab_team, tab_projects, tab_references, tab_insurance, tab_capabilities, tab_custom, tab_narrative = st.tabs([
        "📋 Profile",
        "👥 Team",
        "🏗️ Projects",
        "📞 References",
        "🛡️ Insurance",
        "⭐ Capabilities",
        "🔧 Custom Fields",
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

        # Edit Profile
        st.markdown("---")

        if st.button("✏️ Edit Profile Information"):
            st.session_state['editing_profile'] = not st.session_state.get('editing_profile', False)

        if st.session_state.get('editing_profile', False):
            with st.form("edit_profile_form"):
                st.subheader("Edit Profile")

                col1, col2 = st.columns(2)

                with col1:
                    edit_legal_name = st.text_input("Legal Name*", value=profile.company.legal_name)
                    edit_trading_names = st.text_input(
                        "Trading Names (comma-separated)",
                        value=", ".join(profile.company.trading_names) if profile.company.trading_names else ""
                    )
                    edit_abn = st.text_input("ABN", value=profile.company.abn or "", max_chars=11)
                    edit_acn = st.text_input("ACN", value=profile.company.acn or "", max_chars=9)

                with col2:
                    edit_phone = st.text_input("Phone*", value=profile.contact.phone)
                    edit_email = st.text_input("Email*", value=profile.contact.email)
                    edit_website = st.text_input("Website", value=profile.contact.website or "")

                st.subheader("Registered Office Address")

                col1, col2 = st.columns(2)

                with col1:
                    edit_street = st.text_input("Street*", value=profile.contact.registered_office.street)
                    edit_city = st.text_input("City*", value=profile.contact.registered_office.city)
                    edit_state = st.text_input("State*", value=profile.contact.registered_office.state)

                with col2:
                    edit_postcode = st.text_input("Postcode*", value=profile.contact.registered_office.postcode)
                    edit_country = st.text_input("Country*", value=profile.contact.registered_office.country)

                col1, col2 = st.columns(2)

                with col1:
                    save = st.form_submit_button("💾 Save Changes", type="primary")
                with col2:
                    cancel = st.form_submit_button("❌ Cancel")

                if cancel:
                    st.session_state['editing_profile'] = False
                    st.rerun()

                if save:
                    try:
                        # Update profile
                        profile.company.legal_name = edit_legal_name
                        profile.company.trading_names = [t.strip() for t in edit_trading_names.split(",")] if edit_trading_names else []
                        profile.company.abn = edit_abn if edit_abn else None
                        profile.company.acn = edit_acn if edit_acn else None

                        profile.contact.phone = edit_phone
                        profile.contact.email = edit_email
                        profile.contact.website = edit_website if edit_website else None

                        profile.contact.registered_office.street = edit_street
                        profile.contact.registered_office.city = edit_city
                        profile.contact.registered_office.state = edit_state
                        profile.contact.registered_office.postcode = edit_postcode
                        profile.contact.registered_office.country = edit_country

                        # Save to file
                        manager.update_entity_metadata(selected_entity_id, profile)

                        st.success("✅ Profile updated successfully!")
                        st.session_state['editing_profile'] = False
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error updating profile: {e}")
                        logger.error(f"Failed to update profile: {e}", exc_info=True)

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

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(f"✏️ Edit", key=f"edit_team_{member.person_id}"):
                            st.session_state[f'editing_team_{member.person_id}'] = True
                            st.rerun()

                    with col2:
                        if st.button(f"🗑️ Remove", key=f"remove_team_{member.person_id}"):
                            manager.remove_team_member(selected_entity_id, member.person_id)
                            st.success(f"Removed {member.full_name}")
                            st.rerun()

                # Edit form for this team member
                if st.session_state.get(f'editing_team_{member.person_id}', False):
                    with st.form(f"edit_team_form_{member.person_id}"):
                        st.subheader(f"Edit {member.full_name}")

                        col1, col2 = st.columns(2)

                        with col1:
                            edit_full_name = st.text_input("Full Name*", value=member.full_name)
                            edit_preferred_name = st.text_input("Preferred Name", value=member.preferred_name or "")
                            edit_role = st.text_input("Role*", value=member.role)

                        with col2:
                            edit_email = st.text_input("Email", value=member.email or "")
                            edit_phone = st.text_input("Phone", value=member.phone or "")

                        edit_brief_bio = st.text_area(
                            "Brief Bio",
                            value=member.bio.brief if member.bio else "",
                            help="2-3 sentence summary"
                        )

                        edit_full_bio = st.text_area(
                            "Full Bio",
                            value=member.bio.full if member.bio else "",
                            height=150,
                            help="Detailed biography for proposals"
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            save_team = st.form_submit_button("💾 Save Changes", type="primary")
                        with col2:
                            cancel_team = st.form_submit_button("❌ Cancel")

                        if cancel_team:
                            st.session_state[f'editing_team_{member.person_id}'] = False
                            st.rerun()

                        if save_team:
                            try:
                                # Update team member
                                member.full_name = edit_full_name
                                member.preferred_name = edit_preferred_name if edit_preferred_name else None
                                member.role = edit_role
                                member.email = edit_email if edit_email else None
                                member.phone = edit_phone if edit_phone else None

                                if edit_brief_bio or edit_full_bio:
                                    member.bio = Bio(
                                        brief=edit_brief_bio if edit_brief_bio else edit_full_bio[:200],
                                        full=edit_full_bio if edit_full_bio else edit_brief_bio
                                    )

                                # Save to file
                                manager.update_team_member(selected_entity_id, member)

                                st.success(f"✅ Updated {member.full_name}")
                                st.session_state[f'editing_team_{member.person_id}'] = False
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error updating team member: {e}")
                                logger.error(f"Failed to update team member: {e}", exc_info=True)
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

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(f"✏️ Edit", key=f"edit_project_{project.project_id}"):
                            st.session_state[f'editing_project_{project.project_id}'] = True
                            st.rerun()

                    with col2:
                        if st.button(f"🗑️ Remove", key=f"remove_project_{project.project_id}"):
                            manager.remove_project(selected_entity_id, project.project_id)
                            st.success(f"Removed {project.project_name}")
                            st.rerun()

                # Edit form for this project
                if st.session_state.get(f'editing_project_{project.project_id}', False):
                    with st.form(f"edit_project_form_{project.project_id}"):
                        st.subheader(f"Edit {project.project_name}")

                        col1, col2 = st.columns(2)

                        with col1:
                            edit_project_name = st.text_input("Project Name*", value=project.project_name)
                            edit_client = st.text_input("Client*", value=project.client)
                            edit_sector = st.text_input("Sector", value=project.sector or "")

                        with col2:
                            edit_start_date = st.date_input("Start Date*", value=project.timeline.start_date)
                            edit_end_date = st.date_input("End Date", value=project.timeline.end_date if project.timeline.end_date else None)
                            edit_contract_value = st.number_input(
                                "Contract Value*",
                                min_value=0.0,
                                value=float(project.financials.contract_value),
                                step=1000.0
                            )

                        edit_brief_desc = st.text_area(
                            "Brief Description*",
                            value=project.description.brief,
                            help="2-3 sentence summary"
                        )

                        edit_full_desc = st.text_area(
                            "Full Description",
                            value=project.description.full if project.description.full else "",
                            height=150,
                            help="Detailed project description"
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            save_project = st.form_submit_button("💾 Save Changes", type="primary")
                        with col2:
                            cancel_project = st.form_submit_button("❌ Cancel")

                        if cancel_project:
                            st.session_state[f'editing_project_{project.project_id}'] = False
                            st.rerun()

                        if save_project:
                            try:
                                # Update project
                                project.project_name = edit_project_name
                                project.client = edit_client
                                project.sector = edit_sector if edit_sector else None

                                # Calculate duration
                                if edit_end_date:
                                    duration_months = ((edit_end_date.year - edit_start_date.year) * 12 +
                                                     (edit_end_date.month - edit_start_date.month))
                                else:
                                    duration_months = None

                                project.timeline = Timeline(
                                    start_date=edit_start_date,
                                    end_date=edit_end_date,
                                    duration_months=duration_months
                                )

                                project.financials = Financials(
                                    contract_value=edit_contract_value,
                                    currency=project.financials.currency,
                                    payment_structure=project.financials.payment_structure
                                )

                                project.description = ProjectDescription(
                                    brief=edit_brief_desc,
                                    full=edit_full_desc if edit_full_desc else edit_brief_desc
                                )

                                # Save to file
                                manager.update_project(selected_entity_id, project)

                                st.success(f"✅ Updated {project.project_name}")
                                st.session_state[f'editing_project_{project.project_id}'] = False
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error updating project: {e}")
                                logger.error(f"Failed to update project: {e}", exc_info=True)
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

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(f"✏️ Edit", key=f"edit_ref_{ref.reference_id}"):
                            st.session_state[f'editing_ref_{ref.reference_id}'] = True
                            st.rerun()

                    with col2:
                        if st.button(f"🗑️ Remove", key=f"remove_ref_{ref.reference_id}"):
                            manager.remove_reference(selected_entity_id, ref.reference_id)
                            st.success(f"Removed {ref.contact_name}")
                            st.rerun()

                # Edit form for this reference
                if st.session_state.get(f'editing_ref_{ref.reference_id}', False):
                    with st.form(f"edit_ref_form_{ref.reference_id}"):
                        st.subheader(f"Edit {ref.contact_name}")

                        col1, col2 = st.columns(2)

                        with col1:
                            edit_contact_name = st.text_input("Contact Name*", value=ref.contact_name)
                            edit_title = st.text_input("Title*", value=ref.title)
                            edit_organization = st.text_input("Organization*", value=ref.organization)

                        with col2:
                            edit_email = st.text_input("Email", value=ref.email or "")
                            edit_phone = st.text_input("Phone", value=ref.phone or "")
                            edit_available = st.checkbox("Available for contact", value=ref.availability.available)

                        edit_relationship_role = st.text_input(
                            "Relationship Role*",
                            value=ref.relationship.role,
                            help="e.g., 'Project Sponsor', 'Client Manager'"
                        )

                        edit_working_relationship = st.text_area(
                            "Working Relationship",
                            value=ref.context.working_relationship if ref.context else "",
                            help="Describe how you worked together"
                        )

                        edit_quote = st.text_area(
                            "Quote/Testimonial",
                            value=ref.quote or "",
                            help="Optional testimonial from this reference"
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            save_ref = st.form_submit_button("💾 Save Changes", type="primary")
                        with col2:
                            cancel_ref = st.form_submit_button("❌ Cancel")

                        if cancel_ref:
                            st.session_state[f'editing_ref_{ref.reference_id}'] = False
                            st.rerun()

                        if save_ref:
                            try:
                                # Update reference
                                ref.contact_name = edit_contact_name
                                ref.title = edit_title
                                ref.organization = edit_organization
                                ref.email = edit_email if edit_email else None
                                ref.phone = edit_phone if edit_phone else None
                                ref.quote = edit_quote if edit_quote else None

                                ref.relationship.role = edit_relationship_role

                                ref.availability = ReferenceAvailability(
                                    available=edit_available,
                                    preferred_contact=ref.availability.preferred_contact
                                )

                                if edit_working_relationship:
                                    ref.context = ReferenceContext(
                                        working_relationship=edit_working_relationship,
                                        can_speak_to=ref.context.can_speak_to if ref.context else []
                                    )

                                # Save to file
                                manager.update_reference(selected_entity_id, ref)

                                st.success(f"✅ Updated {ref.contact_name}")
                                st.session_state[f'editing_ref_{ref.reference_id}'] = False
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error updating reference: {e}")
                                logger.error(f"Failed to update reference: {e}", exc_info=True)
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

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(f"✏️ Edit", key=f"edit_insurance_{policy.policy_id}"):
                            st.session_state[f'editing_insurance_{policy.policy_id}'] = True
                            st.rerun()

                    with col2:
                        if st.button(f"🗑️ Remove", key=f"remove_insurance_{policy.policy_id}"):
                            manager.remove_insurance(selected_entity_id, policy.policy_id)
                            st.success(f"Removed policy")
                            st.rerun()

                # Edit form for this insurance policy
                if st.session_state.get(f'editing_insurance_{policy.policy_id}', False):
                    with st.form(f"edit_insurance_form_{policy.policy_id}"):
                        st.subheader(f"Edit {policy.policy_type.value.replace('_', ' ').title()}")

                        col1, col2 = st.columns(2)

                        with col1:
                            edit_insurer = st.text_input("Insurer*", value=policy.insurer)
                            edit_policy_number = st.text_input("Policy Number*", value=policy.policy_number)
                            edit_coverage_amount = st.number_input(
                                "Coverage Amount*",
                                min_value=0.0,
                                value=float(policy.coverage.amount),
                                step=100000.0
                            )

                        with col2:
                            edit_effective_date = st.date_input("Effective Date*", value=policy.dates.effective_date)
                            edit_expiry_date = st.date_input("Expiry Date*", value=policy.dates.expiry_date)

                        edit_description = st.text_area(
                            "Coverage Description",
                            value=policy.coverage.description,
                            help="Describe what this policy covers"
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            save_insurance = st.form_submit_button("💾 Save Changes", type="primary")
                        with col2:
                            cancel_insurance = st.form_submit_button("❌ Cancel")

                        if cancel_insurance:
                            st.session_state[f'editing_insurance_{policy.policy_id}'] = False
                            st.rerun()

                        if save_insurance:
                            try:
                                # Update insurance policy
                                policy.insurer = edit_insurer
                                policy.policy_number = edit_policy_number

                                policy.coverage = Coverage(
                                    amount=edit_coverage_amount,
                                    currency=policy.coverage.currency,
                                    formatted=f"${edit_coverage_amount:,.0f}",
                                    description=edit_description
                                )

                                policy.dates = InsuranceDates(
                                    effective_date=edit_effective_date,
                                    expiry_date=edit_expiry_date,
                                    renewal_status=policy.dates.renewal_status
                                )

                                # Save to file
                                manager.update_insurance(selected_entity_id, policy)

                                st.success(f"✅ Updated insurance policy")
                                st.session_state[f'editing_insurance_{policy.policy_id}'] = False
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error updating insurance: {e}")
                                logger.error(f"Failed to update insurance: {e}", exc_info=True)
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

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button(f"✏️ Edit", key=f"edit_cap_{cap.capability_id}"):
                            st.session_state[f'editing_cap_{cap.capability_id}'] = True
                            st.rerun()

                    with col2:
                        if st.button(f"🗑️ Remove", key=f"remove_cap_{cap.capability_id}"):
                            manager.remove_capability(selected_entity_id, cap.capability_id)
                            st.success(f"Removed capability")
                            st.rerun()

                # Edit form for this capability
                if st.session_state.get(f'editing_cap_{cap.capability_id}', False):
                    with st.form(f"edit_cap_form_{cap.capability_id}"):
                        st.subheader(f"Edit {cap.capability_name}")

                        col1, col2 = st.columns(2)

                        with col1:
                            edit_capability_name = st.text_input("Capability Name*", value=cap.capability_name)
                            edit_certification_body = st.text_input(
                                "Certification Body",
                                value=cap.certification_body or ""
                            )
                            edit_certification_number = st.text_input(
                                "Certification Number",
                                value=cap.certification_number or ""
                            )

                        with col2:
                            edit_obtained_date = st.date_input("Date Obtained*", value=cap.dates.obtained)
                            edit_expiry_date = st.date_input(
                                "Expiry Date",
                                value=cap.dates.expiry if cap.dates.expiry else None
                            )

                        edit_brief_desc = st.text_area(
                            "Brief Description*",
                            value=cap.description.brief,
                            help="Short description of this capability"
                        )

                        edit_full_desc = st.text_area(
                            "Full Description",
                            value=cap.description.full if cap.description.full else "",
                            height=100,
                            help="Detailed description"
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            save_cap = st.form_submit_button("💾 Save Changes", type="primary")
                        with col2:
                            cancel_cap = st.form_submit_button("❌ Cancel")

                        if cancel_cap:
                            st.session_state[f'editing_cap_{cap.capability_id}'] = False
                            st.rerun()

                        if save_cap:
                            try:
                                # Update capability
                                cap.capability_name = edit_capability_name
                                cap.certification_body = edit_certification_body if edit_certification_body else None
                                cap.certification_number = edit_certification_number if edit_certification_number else None

                                cap.description = CapabilityDescription(
                                    brief=edit_brief_desc,
                                    full=edit_full_desc if edit_full_desc else edit_brief_desc
                                )

                                cap.dates = CapabilityDates(
                                    obtained=edit_obtained_date,
                                    expiry=edit_expiry_date if edit_expiry_date else None
                                )

                                # Save to file
                                manager.update_capability(selected_entity_id, cap)

                                st.success(f"✅ Updated capability")
                                st.session_state[f'editing_cap_{cap.capability_id}'] = False
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error updating capability: {e}")
                                logger.error(f"Failed to update capability: {e}", exc_info=True)
        else:
            st.info("No capabilities yet. Add your first capability!")

        st.markdown("---")
        st.info("💡 **Tip:** For detailed capability profiles, edit YAML files directly.")

    # ========================================
    # TAB: CUSTOM FIELDS
    # ========================================

    with tab_custom:
        section_header("🔧", "Custom Fields", "Tender-specific custom data fields")

        st.info("💡 Custom fields are created during proposal workflows when you need tender-specific data that doesn't fit standard profile fields. They're saved here for reuse across proposals.")

        # Display existing custom fields
        if profile.custom_fields:
            st.subheader(f"📋 {len(profile.custom_fields)} Custom Fields")

            for idx, field in enumerate(profile.custom_fields):
                with st.expander(f"🔧 {field.field_name}", expanded=False):
                    st.markdown(f"**Field Name:** `@{field.field_name}`")
                    st.markdown(f"**Field Path:** `custom_fields.{field.field_name}`")

                    if field.description:
                        st.caption("Description")
                        st.info(field.description)

                    st.caption("Current Value")
                    st.code(field.field_value, language=None)

                    st.caption("Metadata")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"Created: {field.created_date}")
                    with col2:
                        if field.last_used:
                            st.text(f"Last Used: {field.last_used}")
                        else:
                            st.text("Last Used: Never")

                    st.markdown("---")

                    # Edit form
                    with st.form(key=f"edit_custom_field_{idx}"):
                        st.caption("✏️ EDIT FIELD")

                        new_value = st.text_area(
                            "Field Value",
                            value=field.field_value,
                            height=100
                        )

                        new_description = st.text_input(
                            "Description",
                            value=field.description or ""
                        )

                        col_save, col_delete = st.columns(2)

                        with col_save:
                            if st.form_submit_button("💾 Save Changes", use_container_width=True):
                                field.field_value = new_value
                                field.description = new_description if new_description else None
                                manager._save_profile(profile)
                                st.success(f"✅ Updated custom field: {field.field_name}")
                                st.rerun()

                        with col_delete:
                            if st.form_submit_button("🗑️ Delete", use_container_width=True):
                                profile.custom_fields.remove(field)
                                manager._save_profile(profile)
                                st.success(f"✅ Deleted custom field: {field.field_name}")
                                st.rerun()

        else:
            st.info("📭 No custom fields yet. Create them during proposal workflows using the 'Replace with Custom Field' feature in the Review tab.")

        # Add new custom field manually
        st.markdown("---")
        st.subheader("➕ Add New Custom Field")

        with st.form(key="add_custom_field_manual"):
            st.caption("Create a custom field manually")

            new_field_name = st.text_input(
                "Field Name",
                help="Lowercase letters, numbers, and underscores only (e.g., specified_personnel_1_email)"
            )

            new_field_value = st.text_area(
                "Field Value",
                height=100
            )

            new_field_description = st.text_input(
                "Description (optional)",
                help="What this field represents"
            )

            if st.form_submit_button("✨ Create Custom Field", type="primary"):
                if not new_field_name or not new_field_value:
                    st.error("Field name and value are required")
                else:
                    try:
                        profile.add_custom_field(
                            field_name=new_field_name,
                            field_value=new_field_value,
                            description=new_field_description if new_field_description else None
                        )
                        manager._save_profile(profile)
                        st.success(f"✅ Created custom field: @{new_field_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating custom field: {str(e)}")

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

render_version_footer()
