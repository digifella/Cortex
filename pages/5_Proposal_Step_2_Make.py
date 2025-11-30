# ## File: pages/5_Proposal_Step_2_Make.py
# Version: v4.10.3
# Date: 2025-08-31
# Purpose: A central hub for creating, loading, and managing proposals.
#          - FEATURE (v2.2.0): Added a confirmation step before deleting a
#            proposal to prevent accidental data loss.

import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.proposal_manager import ProposalManager
from cortex_engine.help_system import help_system

st.set_page_config(layout="wide", page_title="Cortex Proposal Management")

# Initialize the manager
prop_mgr = ProposalManager()

# --- Initialize Session State for this page ---
if 'confirming_delete_proposal_id' not in st.session_state:
    st.session_state.confirming_delete_proposal_id = None

st.title("🗂️ 5. Proposal Step 2 Make")
st.caption("Create a new proposal or load an existing one to continue working.")

# Quick access to Proposal Copilot
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Open Proposal Copilot", use_container_width=True, help="Access the Proposal Copilot for AI-assisted writing and editing"):
        st.switch_page("pages/_Proposal_Copilot.py")

st.divider()

# Add help system
help_system.show_help_menu()

# Show help modal if requested
if st.session_state.get("show_help_modal", False):
    help_topic = st.session_state.get("help_topic", "overview")
    help_system.show_help_modal(help_topic)

# Show contextual help for this page
help_system.show_contextual_help("proposals")

# --- Section: Create New Proposal ---
with st.expander("🚀 Create a New Proposal", expanded=True):
    with st.form("new_proposal_form"):
        new_proposal_name = st.text_input("Proposal Name", 
                                          help="📝 Give your proposal a descriptive name (e.g., 'Company X - AI Consulting Project'). This will help you identify it later.")
        submitted = st.form_submit_button("Create and Start")

        if submitted:
            if new_proposal_name:
                with st.spinner("Creating new proposal..."):
                    proposal_id = prop_mgr.create_proposal(new_proposal_name)

                    # Set the current proposal ID in session state for other pages
                    st.session_state['current_proposal_id'] = proposal_id

                    # Clear any potentially lingering state from other proposals
                    keys_to_clear = ['parsed_instructions', 'doc_template', 'section_content', 'generated_doc_bytes', 'original_filename', 'last_uploaded_file']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]

                    # Switch to the co-pilot page
                    st.switch_page("pages/_Proposal_Copilot.py")
            else:
                st.error("Please enter a name for the proposal.")

st.divider()

# --- Section: Load Existing Proposals ---
st.header("📂 Existing Proposals")

proposals_list = prop_mgr.list_proposals()

if not proposals_list:
    st.info("No existing proposals found. Create one above to get started.")
else:
    # Use st.columns to create a header
    c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1, 1])
    c1.write("**Proposal Name**")
    c2.write("**Last Modified**")
    c3.write("**Status**")

    st.markdown("---")

    for p in proposals_list:
        with st.container(border=True): # Use a container to group each proposal row and its confirmation
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
            with col1:
                st.markdown(f"**{p['name']}**")
                st.caption(f"ID: {p['id']}")
            with col2:
                st.write(p['last_modified'].strftime('%Y-%m-%d %H:%M'))
            with col3:
                # Use a selectbox to allow status changes
                new_status = st.selectbox(
                    "Status",
                    ["Drafting", "Review", "Approved", "Archived"],
                    index=["Drafting", "Review", "Approved", "Archived"].index(p['status']),
                    key=f"status_{p['id']}",
                    label_visibility="collapsed"
                )
                if new_status != p['status']:
                    prop_mgr.update_proposal_status(p['id'], new_status)
                    st.toast(f"Updated '{p['name']}' status to {new_status}")
                    st.rerun()
            with col4:
                if st.button("Load", key=f"load_{p['id']}", use_container_width=True):
                     with st.spinner(f"Loading '{p['name']}'..."):
                        st.session_state['current_proposal_id'] = p['id']
                        st.switch_page("pages/_Proposal_Copilot.py")
            with col5:
                if st.button("❌", key=f"del_{p['id']}", use_container_width=True):
                    # Set the proposal to be confirmed for deletion
                    st.session_state.confirming_delete_proposal_id = p['id']
                    st.rerun()

            # --- Deletion Confirmation UI ---
            if st.session_state.confirming_delete_proposal_id == p['id']:
                st.warning(f"**Are you sure you want to permanently delete the proposal '{p['name']}'?** This action cannot be undone.")
                confirm_cols = st.columns(2)
                with confirm_cols[0]:
                    if st.button("YES, DELETE PERMANENTLY", key=f"confirm_del_{p['id']}", use_container_width=True, type="primary"):
                        prop_mgr.delete_proposal(p['id'])
                        st.session_state.confirming_delete_proposal_id = None
                        st.rerun()
                with confirm_cols[1]:
                    if st.button("Cancel", key=f"cancel_del_{p['id']}", use_container_width=True):
                        st.session_state.confirming_delete_proposal_id = None
                        st.rerun()

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
