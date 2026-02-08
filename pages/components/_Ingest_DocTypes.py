"""Shared document type management UI helpers for Knowledge Ingest pages."""

from __future__ import annotations

from datetime import datetime

import streamlit as st


def render_document_type_management(*, get_document_type_manager) -> None:
    """Render document type/category mapping management surface."""
    st.header("📋 Document Type Management")
    st.markdown("Configure document categories, types, and keyword mappings for intelligent document classification.")

    doc_type_manager = get_document_type_manager()

    tab1, tab2, tab3, tab4 = st.tabs(["📂 Categories", "🏷️ Type Mappings", "📊 Overview", "⚙️ Settings"])

    with tab1:
        st.subheader("Document Categories")
        st.markdown("Organize document types into logical categories for better organization.")

        categories = doc_type_manager.get_categories()

        for category_name, category_data in categories.items():
            with st.expander(f"📁 {category_name} ({len(category_data['types'])} types)", expanded=False):
                st.markdown(f"**Description:** {category_data['description']}")
                st.markdown(f"**Document Types:** {', '.join(category_data['types'])}")

                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    new_type = st.text_input(f"Add new type to {category_name}:", key=f"new_type_{category_name}")
                    if st.button(f"➕ Add Type", key=f"add_type_{category_name}"):
                        if new_type and doc_type_manager.add_type_to_category(category_name, new_type):
                            st.success(f"Added '{new_type}' to {category_name}")
                            st.session_state.show_maintenance = True
                            st.rerun()
                        if new_type:
                            st.warning(f"'{new_type}' already exists in {category_name}")

                with col2:
                    if category_data["types"]:
                        type_to_remove = st.selectbox(
                            f"Remove type from {category_name}:",
                            [""] + category_data["types"],
                            key=f"remove_type_{category_name}",
                        )
                        if type_to_remove and st.button(f"🗑️ Remove", key=f"remove_btn_{category_name}"):
                            if doc_type_manager.remove_type_from_category(category_name, type_to_remove):
                                st.success(f"Removed '{type_to_remove}' from {category_name}")
                                st.session_state.show_maintenance = True
                                st.rerun()

                with col3:
                    if category_name != "Other":
                        if st.button(f"🗑️ Delete Category", key=f"delete_cat_{category_name}", type="secondary"):
                            if doc_type_manager.remove_category(category_name):
                                st.success(f"Deleted category '{category_name}'")
                                st.session_state.show_maintenance = True
                                st.rerun()

        st.markdown("---")
        st.subheader("Add New Category")
        col1, col2 = st.columns(2)
        with col1:
            new_category_name = st.text_input("Category Name:", key="new_category_name")
        with col2:
            new_category_desc = st.text_input("Description:", key="new_category_desc")

        if st.button("➕ Create Category") and new_category_name and new_category_desc:
            if doc_type_manager.add_category(new_category_name, new_category_desc):
                st.success(f"Created category '{new_category_name}'")
                st.session_state.show_maintenance = True
                st.rerun()
            else:
                st.error(f"Category '{new_category_name}' already exists")

    with tab2:
        st.subheader("Keyword Mappings")
        st.markdown("Define keywords that automatically map to specific document types during ingestion.")

        mappings = doc_type_manager.get_type_mappings()
        all_types = doc_type_manager.get_all_document_types()

        if mappings:
            st.markdown("**Current Mappings:**")
            sort_col1, sort_col2, sort_col3 = st.columns([2, 2, 2])

            with sort_col1:
                if st.button("🔤 Sort by Keyword", use_container_width=True, key="sort_by_keyword"):
                    st.session_state.mapping_sort_key = "keyword"
                    st.session_state.mapping_sort_reverse = not st.session_state.get("mapping_sort_reverse", False)
                    st.session_state.show_maintenance = True
                    st.rerun()

            with sort_col2:
                if st.button("📋 Sort by Document Type", use_container_width=True, key="sort_by_doctype"):
                    st.session_state.mapping_sort_key = "doctype"
                    st.session_state.mapping_sort_reverse = not st.session_state.get("mapping_sort_reverse", False)
                    st.session_state.show_maintenance = True
                    st.rerun()

            with sort_col3:
                filter_type = st.selectbox("Filter by Type:", ["All"] + all_types, key="mapping_filter_type")

            sorted_mappings = list(mappings.items())
            if filter_type != "All":
                sorted_mappings = [(k, v) for k, v in sorted_mappings if v == filter_type]

            sort_key = st.session_state.get("mapping_sort_key", "keyword")
            sort_reverse = st.session_state.get("mapping_sort_reverse", False)

            if sort_key == "keyword":
                sorted_mappings.sort(key=lambda x: x[0].lower(), reverse=sort_reverse)
                sort_indicator = " 🔽" if sort_reverse else " 🔼"
                st.caption(f"Sorted by Keywords{sort_indicator}")
            elif sort_key == "doctype":
                sorted_mappings.sort(key=lambda x: x[1].lower(), reverse=sort_reverse)
                sort_indicator = " 🔽" if sort_reverse else " 🔼"
                st.caption(f"Sorted by Document Types{sort_indicator}")

            if not sorted_mappings:
                st.info(f"No mappings found for document type '{filter_type}'")
            else:
                st.markdown("---")
                header_col1, header_col2, header_col3 = st.columns([2, 2, 1])
                with header_col1:
                    st.markdown("**Keyword**")
                with header_col2:
                    st.markdown("**Document Type**")
                with header_col3:
                    st.markdown("**Action**")

                st.markdown("---")

                for i, (keyword, doc_type) in enumerate(sorted_mappings):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.code(keyword, language=None)
                    with col2:
                        category = doc_type_manager.get_category_for_type(doc_type)
                        st.markdown(f"**{doc_type}**")
                        st.caption(f"📂 {category}")
                    with col3:
                        if st.button("🗑️", key=f"remove_mapping_{keyword}_{i}", help=f"Remove mapping for '{keyword}'"):
                            if doc_type_manager.remove_type_mapping(keyword):
                                st.success(f"Removed mapping for '{keyword}'")
                                st.session_state.show_maintenance = True
                                st.rerun()

                st.markdown("---")
                total_mappings = len(mappings)
                shown_mappings = len(sorted_mappings)
                if filter_type != "All":
                    st.info(f"Showing {shown_mappings} of {total_mappings} mappings (filtered by '{filter_type}')")
                else:
                    st.info(f"Showing all {shown_mappings} mappings")
        else:
            st.info("No keyword mappings defined yet.")

        st.markdown("---")
        st.subheader("Add New Mapping")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_keyword = st.text_input("Keyword (e.g., 'bio', 'agenda'):", key="new_keyword")
        with col2:
            new_mapping_type = st.selectbox("Maps to Document Type:", [""] + all_types, key="new_mapping_type")
        with col3:
            st.write("")
            st.write("")
            if st.button("➕ Add Mapping") and new_keyword and new_mapping_type:
                if doc_type_manager.add_type_mapping(new_keyword, new_mapping_type):
                    st.success(f"Added mapping: '{new_keyword}' → '{new_mapping_type}'")
                    st.session_state.show_maintenance = True
                    st.rerun()

    with tab3:
        st.subheader("System Overview")

        categories = doc_type_manager.get_categories()
        mappings = doc_type_manager.get_type_mappings()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Categories", len(categories))
        with col2:
            total_types = sum(len(cat["types"]) for cat in categories.values())
            st.metric("Document Types", total_types)
        with col3:
            st.metric("Keyword Mappings", len(mappings))
        with col4:
            st.metric("Auto-suggestions", "Active" if mappings else "None")

        st.markdown("---")
        st.subheader("Category Breakdown")
        for category_name, category_data in categories.items():
            st.markdown(f"**{category_name}** ({len(category_data['types'])} types): {category_data['description']}")
            if category_data["types"]:
                st.markdown(f"*Types:* {', '.join(category_data['types'])}")

        st.markdown("---")
        st.subheader("Test Auto-Suggestion")
        test_filename = st.text_input("Test filename:", placeholder="e.g., john_smith_bio.pdf")
        if test_filename:
            suggested_type = doc_type_manager.suggest_document_type(test_filename)
            st.success(f"Suggested document type: **{suggested_type}**")

    with tab4:
        st.subheader("System Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Export Configuration**")
            if st.button("📥 Export Config"):
                config_json = doc_type_manager.export_config()
                st.download_button(
                    label="💾 Download Config JSON",
                    data=config_json,
                    file_name=f"document_types_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                )

        with col2:
            st.markdown("**Import Configuration**")
            uploaded_file = st.file_uploader("Choose config file", type=["json"])
            if uploaded_file is not None:
                config_content = uploaded_file.read().decode("utf-8")
                if st.button("📤 Import Config"):
                    if doc_type_manager.import_config(config_content):
                        st.success("Configuration imported successfully!")
                        st.session_state.show_maintenance = True
                        st.rerun()
                    else:
                        st.error("Failed to import configuration. Please check the file format.")

        st.markdown("---")
        st.subheader("Reset to Defaults")
        st.warning("⚠️ This will reset all categories, types, and mappings to default values.")
        if st.button("🔄 Reset to Defaults", type="secondary"):
            st.session_state["confirm_doctype_reset"] = True
        if st.session_state.get("confirm_doctype_reset", False):
            st.error("Are you sure? This cannot be undone.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("⚠️ Confirm Reset", type="primary"):
                    st.session_state.pop("confirm_doctype_reset", None)
                    if doc_type_manager.reset_to_defaults():
                        st.success("Configuration reset to defaults!")
                        st.session_state.show_maintenance = True
                        st.rerun()
            with col_no:
                if st.button("Cancel"):
                    st.session_state.pop("confirm_doctype_reset", None)
                    st.rerun()

    st.markdown("---")
    if st.button("✅ Close Document Type Management", use_container_width=True):
        st.session_state.show_maintenance = False
        st.rerun()
