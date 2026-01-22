"""
Mention Review UI Component for Streamlit
Version: 2.1.0

Provides context-rich mention review with auto-scrolling document view.
Features:
- Auto-scroll to current mention location
- Focused "page view" of document section
- Field explanations for each mention type
- Scrollable context magnifier
"""

import streamlit as st
from typing import List, Dict
import re


def get_field_explanation(field_path: str, mention_type: str) -> Dict[str, str]:
    """
    Provide human-readable explanation of what a field mention represents.

    Returns dict with 'title', 'description', and 'example' keys.
    """
    # Common field explanations
    field_explanations = {
        'companyname': {
            'title': 'Company Name',
            'description': 'The registered business name from your entity profile',
            'example': 'e.g., "Longboardfella Consulting Pty Ltd"'
        },
        'abn': {
            'title': 'Australian Business Number',
            'description': 'Your ABN from entity profile > business information',
            'example': 'e.g., "12 345 678 901"'
        },
        'acn': {
            'title': 'Australian Company Number',
            'description': 'Your ACN from entity profile > business information',
            'example': 'e.g., "123 456 789"'
        },
        'email': {
            'title': 'Contact Email',
            'description': 'Primary contact email from entity profile',
            'example': 'e.g., "contact@company.com.au"'
        },
        'phone': {
            'title': 'Contact Phone',
            'description': 'Primary contact phone number from entity profile',
            'example': 'e.g., "+61 2 1234 5678"'
        },
        'registered_office': {
            'title': 'Registered Office Address',
            'description': 'The registered office address from entity profile > contact section',
            'example': 'e.g., "Level 5, 123 Business St, Sydney NSW 2000"'
        },
        'wgea': {
            'title': 'WGEA Registration',
            'description': 'Workplace Gender Equality Agency registration status or office details',
            'example': 'e.g., "Registered" or specific office location'
        },
    }

    # Try to find exact match first
    field_key = field_path.split('.')[-1].lower()
    if field_key in field_explanations:
        return field_explanations[field_key]

    # Check for partial matches
    for key, explanation in field_explanations.items():
        if key in field_path.lower():
            return explanation

    # Generic fallback
    return {
        'title': field_path.replace('.', ' > ').replace('_', ' ').title(),
        'description': f'Data from entity profile field: {field_path}',
        'example': 'Will be pulled from your entity profile'
    }


def inject_review_css():
    """Inject minimal custom CSS for mention highlighting only."""
    st.markdown("""
    <style>
        .mention-active {
            background: #0d9488;
            color: white;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-weight: 600;
        }

        .mention-approved {
            background: #d1fae5;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            border-bottom: 2px solid #059669;
            font-weight: 600;
        }

        .mention-rejected {
            background: #fee2e2;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            opacity: 0.5;
            text-decoration: line-through;
        }

        .mention-pending {
            background: #fed7aa;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            border-bottom: 2px solid #ea580c;
            font-weight: 600;
        }

        .document-viewer {
            max-height: 70vh;
            overflow-y: auto;
            padding: 1.5rem;
            background: #fafafa;
            border-radius: 8px;
            border: 1px solid #e5e5e5;
            line-height: 1.8;
            font-family: monospace;
            font-size: 0.9em;
        }

        .line-number {
            color: #999;
            margin-right: 1em;
            user-select: none;
        }
    </style>
    """, unsafe_allow_html=True)


def render_mention_review(
    workspace,
    workspace_manager,
    entity_profile_manager,
    document_text: str,
    mentions: List
):
    """
    Render mention review interface with auto-scrolling document view.

    Args:
        workspace: Current workspace object
        workspace_manager: Workspace manager instance
        entity_profile_manager: Entity profile manager instance
        document_text: Full document text
        mentions: List of MentionBinding objects
    """
    inject_review_css()

    # Initialize session state
    if 'review_mention_index' not in st.session_state:
        st.session_state.review_mention_index = 0

    # Get pending mentions (exclude approved, rejected, and ignored)
    pending = [m for m in mentions if not m.approved and not m.rejected and not m.ignored]

    if not pending:
        st.success("✅ All mentions reviewed!")
        return

    # Ensure valid index
    current_idx = min(st.session_state.review_mention_index, len(pending) - 1)
    current_mention = pending[current_idx]

    # Calculate progress
    total = len(mentions)
    reviewed = sum(1 for m in mentions if m.approved or m.rejected or m.ignored)
    progress_pct = (reviewed / total) if total > 0 else 0

    # Split document into lines
    lines = document_text.split('\n')

    # Extract line number from location (e.g., "Line 1705" -> 1705)
    target_line_num = 0
    location_match = re.search(r'Line\s+(\d+)', current_mention.location)
    if location_match:
        target_line_num = int(location_match.group(1)) - 1  # Convert to 0-indexed

    # Get field explanation
    field_info = get_field_explanation(current_mention.field_path, current_mention.mention_type)

    # Create layout
    col_doc, col_review = st.columns([2, 1], gap="large")

    with col_doc:
        st.subheader("📄 Document Page View")
        st.caption(f"Showing area around {current_mention.location}")

        # Show focused "page view" - 25 lines before and after target
        page_start = max(0, target_line_num - 25)
        page_end = min(len(lines), target_line_num + 26)
        page_lines = lines[page_start:page_end]

        # Build highlighted page view with line numbers
        highlighted_lines = []
        for i, line in enumerate(page_lines):
            actual_line_num = page_start + i + 1  # 1-indexed for display
            line_text = line.replace('<', '&lt;').replace('>', '&gt;')  # Escape HTML

            # Check if this line contains any mentions
            is_target_line = (page_start + i) == target_line_num

            if is_target_line:
                # Highlight the entire target line
                highlighted_lines.append(
                    f'<span class="line-number">{actual_line_num:4d}</span>'
                    f'<span class="mention-active">{line_text}</span>'
                )
            else:
                highlighted_lines.append(
                    f'<span class="line-number">{actual_line_num:4d}</span>{line_text}'
                )

        page_html = '<br>'.join(highlighted_lines)

        # Display focused page view
        st.markdown(
            f'<div class="document-viewer">{page_html}</div>',
            unsafe_allow_html=True
        )

    with col_review:
        st.subheader("🔍 Review Mention")

        # Progress
        st.progress(progress_pct, text=f"{reviewed} of {total} reviewed")

        st.markdown("---")

        # Current mention details
        mention_type_color = {
            'simple': '🔷',
            'structured': '🔶',
            'narrative': '📝',
            'generated': '✨'
        }.get(current_mention.mention_type, '📌')

        st.markdown(f"**{mention_type_color} Type:** {current_mention.mention_type.title()}")

        # Mention text (larger, more prominent)
        st.markdown(f"### `{current_mention.mention_text}`")

        # Field explanation - what this mention represents
        with st.expander("ℹ️ What is this field?", expanded=True):
            st.markdown(f"**{field_info['title']}**")
            st.caption(field_info['description'])
            st.info(field_info['example'])

        st.markdown("---")

        # Location details
        st.caption("📍 LOCATION")
        st.info(current_mention.location)

        st.caption("🔗 FIELD PATH")
        st.info(f"`{current_mention.field_path}`")

        # Context "magnifying glass" - scrollable detail view
        st.caption("🔍 CONTEXT MAGNIFIER (surrounding text)")

        # Extract 7 lines of context (3 before, target, 3 after)
        context_start = max(0, target_line_num - 3)
        context_end = min(len(lines), target_line_num + 4)
        context_lines = lines[context_start:context_end]

        # Highlight target line in context
        context_with_highlight = []
        for i, line in enumerate(context_lines):
            actual_line_num = context_start + i + 1
            if (context_start + i) == target_line_num:
                context_with_highlight.append(f">>> {actual_line_num:4d}: {line}")
            else:
                context_with_highlight.append(f"    {actual_line_num:4d}: {line}")

        context_text = '\n'.join(context_with_highlight) if context_with_highlight else "No context available"

        # Display in scrollable text area
        st.text_area(
            "Context detail",
            value=context_text,
            height=180,
            disabled=True,
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Action buttons - Primary actions
        col_approve, col_reject = st.columns(2)

        with col_approve:
            if st.button(
                "✅ Approve",
                key=f"approve_{current_idx}",
                use_container_width=True,
                type="primary"
            ):
                workspace_manager.update_mention_binding(
                    workspace.metadata.workspace_id,
                    current_mention.mention_text,
                    approved=True
                )
                if current_idx < len(pending) - 1:
                    st.session_state.review_mention_index += 1
                st.rerun()

        with col_reject:
            if st.button(
                "❌ Reject",
                key=f"reject_{current_idx}",
                use_container_width=True
            ):
                workspace_manager.update_mention_binding(
                    workspace.metadata.workspace_id,
                    current_mention.mention_text,
                    rejected=True
                )
                if current_idx < len(pending) - 1:
                    st.session_state.review_mention_index += 1
                st.rerun()

        # Secondary actions - Ignore options
        st.markdown("---")

        with st.expander("🔕 Ignore Options", expanded=False):
            st.caption("Choose how to ignore this suggestion")

            col_temp, col_perm = st.columns(2)

            with col_temp:
                if st.button(
                    "Skip for Now",
                    key=f"ignore_temp_{current_idx}",
                    use_container_width=True,
                    help="Skip during this review session only"
                ):
                    # Just move to next without marking in database
                    if current_idx < len(pending) - 1:
                        st.session_state.review_mention_index += 1
                    st.rerun()

            with col_perm:
                if st.button(
                    "Ignore Forever",
                    key=f"ignore_perm_{current_idx}",
                    use_container_width=True,
                    help="Never suggest this again for this document"
                ):
                    workspace_manager.update_mention_binding(
                        workspace.metadata.workspace_id,
                        current_mention.mention_text,
                        ignored=True
                    )
                    if current_idx < len(pending) - 1:
                        st.session_state.review_mention_index += 1
                    st.rerun()

        # Quick Edit - Update entity profile value
        st.markdown("---")

        with st.expander("✏️ Quick Edit Value", expanded=False):
            st.caption("Update the value in your entity profile")

            # Get current value from entity profile
            try:
                from cortex_engine.field_substitution_engine import FieldSubstitutionEngine
                from cortex_engine.mention_parser import MentionParser

                parser = MentionParser()
                parsed = parser.parse(current_mention.mention_text)

                engine = FieldSubstitutionEngine(entity_profile_manager)
                result = engine.resolve(parsed, workspace.metadata.entity_id)

                current_value = result.value if (result and result.success) else ""

            except Exception as e:
                current_value = ""
                st.warning(f"Could not load current value: {e}")

            with st.form(key=f"quick_edit_{current_idx}"):
                new_value = st.text_area(
                    "New Value",
                    value=current_value,
                    height=100,
                    help="Update this value in your entity profile"
                )

                if st.form_submit_button("💾 Save & Approve", type="primary"):
                    try:
                        # Load entity profile
                        profile = entity_profile_manager.get_entity_profile(workspace.metadata.entity_id)

                        # Update the field value based on field path
                        # For simple fields like contact.registered_office
                        field_parts = current_mention.field_path.split('.')

                        if len(field_parts) == 2 and field_parts[0] == 'contact':
                            field_name = field_parts[1]
                            if field_name == 'registered_office' and hasattr(profile.contact.registered_office, 'street'):
                                # It's an Address object - parse the single-line input
                                # Simple parsing: assume format like "Street, City STATE POSTCODE, Country"
                                from cortex_engine.entity_profile_schema import Address

                                # Just update street with the full address for now
                                # User can refine in Entity Profile Manager if needed
                                # Set country to empty to avoid duplication in formatted output
                                profile.contact.registered_office = Address(
                                    street=new_value,
                                    city="",
                                    state="",
                                    postcode="",
                                    country=""
                                )
                                entity_profile_manager._save_profile(profile)

                                # Approve the mention
                                workspace_manager.update_mention_binding(
                                    workspace.metadata.workspace_id,
                                    current_mention.mention_text,
                                    approved=True,
                                    resolved_value=new_value
                                )

                                st.success(f"✅ Updated {current_mention.field_path} and approved!")
                                st.info("💡 Full address stored in street field. Refine in Entity Profile Manager if needed.")
                                if current_idx < len(pending) - 1:
                                    st.session_state.review_mention_index += 1
                                st.rerun()
                            elif hasattr(profile.contact, field_name):
                                setattr(profile.contact, field_name, new_value)
                                entity_profile_manager._save_profile(profile)

                                # Approve the mention
                                workspace_manager.update_mention_binding(
                                    workspace.metadata.workspace_id,
                                    current_mention.mention_text,
                                    approved=True,
                                    resolved_value=new_value
                                )

                                st.success(f"✅ Updated {current_mention.field_path} and approved!")
                                if current_idx < len(pending) - 1:
                                    st.session_state.review_mention_index += 1
                                st.rerun()
                        elif field_parts[0] == 'custom_fields':
                            # Handle custom fields
                            custom_field_name = field_parts[1] if len(field_parts) > 1 else field_parts[0]
                            profile.add_custom_field(
                                field_name=custom_field_name,
                                field_value=new_value
                            )
                            entity_profile_manager._save_profile(profile)

                            workspace_manager.update_mention_binding(
                                workspace.metadata.workspace_id,
                                current_mention.mention_text,
                                approved=True,
                                resolved_value=new_value
                            )

                            st.success(f"✅ Updated custom field and approved!")
                            if current_idx < len(pending) - 1:
                                st.session_state.review_mention_index += 1
                            st.rerun()
                        else:
                            st.info("💡 For this field type, use 'Replace with Custom Field' or Entity Profile Manager")

                    except Exception as e:
                        st.error(f"Error updating value: {e}")

        # Replace with custom field option
        st.markdown("---")

        # Initialize session state for custom field form
        if 'show_custom_field_form' not in st.session_state:
            st.session_state.show_custom_field_form = False

        if st.button(
            "🔄 Replace with Custom Field",
            key=f"custom_{current_idx}",
            use_container_width=True,
            help="Create a new custom field to replace this suggestion"
        ):
            st.session_state.show_custom_field_form = not st.session_state.show_custom_field_form

        # Show custom field form if toggled
        if st.session_state.show_custom_field_form:
            with st.form(key=f"custom_field_form_{current_idx}"):
                st.caption("CREATE CUSTOM FIELD")

                # Suggest a field name based on context
                suggested_name = f"custom_{current_mention.field_path.replace('.', '_')}"

                custom_field_name = st.text_input(
                    "Field Name",
                    value=suggested_name,
                    help="Lowercase letters, numbers, and underscores only (e.g., specified_personnel_1_email)"
                )

                custom_field_value = st.text_area(
                    "Field Value",
                    height=100,
                    help="The actual data for this field (e.g., paul.cooper@company.com.au)"
                )

                custom_field_description = st.text_input(
                    "Description (optional)",
                    help="What this field represents (e.g., 'Paul Cooper - Lead Consultant email')"
                )

                col_submit, col_cancel = st.columns(2)

                with col_submit:
                    submitted = st.form_submit_button(
                        "✨ Create & Replace",
                        use_container_width=True,
                        type="primary"
                    )

                with col_cancel:
                    cancelled = st.form_submit_button(
                        "Cancel",
                        use_container_width=True
                    )

                if submitted:
                    if not custom_field_name or not custom_field_value:
                        st.error("Field name and value are required")
                    else:
                        try:
                            # Replace mention with custom field
                            workspace_manager.replace_mention_with_custom_field(
                                workspace_id=workspace.metadata.workspace_id,
                                mention_text=current_mention.mention_text,
                                custom_field_name=custom_field_name,
                                custom_field_value=custom_field_value,
                                custom_field_description=custom_field_description if custom_field_description else None,
                                entity_profile_manager=entity_profile_manager
                            )

                            st.success(f"✅ Created custom field '@{custom_field_name}' and saved to entity profile!")
                            st.session_state.show_custom_field_form = False

                            # Move to next mention
                            if current_idx < len(pending) - 1:
                                st.session_state.review_mention_index += 1

                            st.rerun()

                        except Exception as e:
                            st.error(f"Error creating custom field: {str(e)}")

                if cancelled:
                    st.session_state.show_custom_field_form = False
                    st.rerun()

        # Navigation
        st.markdown("---")
        col_prev, col_next = st.columns(2)

        with col_prev:
            if st.button(
                "⬅️ Previous",
                key="nav_prev",
                disabled=(current_idx == 0),
                use_container_width=True
            ):
                st.session_state.review_mention_index -= 1
                st.rerun()

        with col_next:
            if st.button(
                "➡️ Next",
                key="nav_next",
                disabled=(current_idx >= len(pending) - 1),
                use_container_width=True
            ):
                st.session_state.review_mention_index += 1
                st.rerun()

        # Jump to specific mention
        st.markdown("---")
        st.caption("JUMP TO")

        for i, mention in enumerate(pending[:10]):  # Show first 10
            status_emoji = "👉" if mention == current_mention else "⏸️"
            truncated_text = mention.mention_text[:25] + ("..." if len(mention.mention_text) > 25 else "")

            if st.button(
                f"{status_emoji} {truncated_text}",
                key=f"jump_{i}",
                use_container_width=True
            ):
                st.session_state.review_mention_index = i
                st.rerun()

        if len(pending) > 10:
            st.caption(f"...and {len(pending) - 10} more")
