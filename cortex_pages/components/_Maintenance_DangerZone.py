"""Shared danger-zone reset panel for Maintenance pages."""

from __future__ import annotations

import streamlit as st


def render_clean_start_danger_zone(*, db_path: str, perform_clean_start_fn) -> None:
    """Render destructive clean-start reset controls."""
    with st.expander("⚠️ Danger Zone - System Reset", expanded=False):
        st.markdown("### ⚠️ **Complete System Reset**")
        st.error("**This section contains destructive operations that cannot be undone!**")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                """
            **🚀 Clean Start Reset**

            Complete system reset function that addresses database schema issues, collection conflicts, and provides a fresh start.
            This function is specifically designed to resolve ChromaDB schema errors like 'collections.config_json_str' column missing.

            **Clean Start will:**
            - ✅ Delete entire knowledge base directory (ChromaDB)
            - ✅ Delete knowledge graph file (.gpickle)
            - ✅ Clear ALL ingestion logs and progress files
            - ✅ Remove ingested files log from database directory
            - ✅ Clear ALL staging and batch ingestion files (including failed ingests)
            - ✅ Reset working collections (working_collections.json)
            - ✅ Clear ingestion recovery metadata
            - ✅ Remove Streamlit cache and session state files
            - ✅ Clear temporary files, lock files, and state files
            - ✅ Reset database configuration paths
            - ✅ Fix ChromaDB schema conflicts and version issues
            - ✅ Provide completely fresh installation state

            **Use Clean Start when:**
            - Getting 'collections.config_json_str' schema errors
            - Collection Management shows connection errors
            - Docker vs non-Docker database conflicts
            - ChromaDB version compatibility issues
            - System appears corrupted or inconsistent
            - **Failed batch ingests** showing up in Knowledge Ingest page
            - Half-finished ingestion operations need clearing
            - Want completely fresh system without any residual files
            """
            )

        with col2:
            st.warning(
                "⚠️ **COMPLETE SYSTEM RESET**\n\n"
                "This will delete ALL data and provide a completely fresh start. "
                "All knowledge base content, collections, and configurations will be lost."
            )

            if st.button(
                "🚀 Clean Start Reset",
                use_container_width=True,
                type="secondary",
                help="⚠️ DANGER: This will delete everything!",
            ):
                st.session_state.show_confirm_clean_start = True

            if st.session_state.get("show_confirm_clean_start"):
                st.error("⚠️ **FINAL WARNING - COMPLETE SYSTEM RESET**")
                st.warning(
                    "This will delete ALL data and provide a completely fresh start. "
                    "All knowledge base content, collections, and configurations will be lost."
                )

                c1, c2 = st.columns(2)
                if c1.button("✅ YES, CLEAN START", use_container_width=True, type="primary"):
                    fresh_path = st.session_state.get("maintenance_current_db_input", db_path)
                    perform_clean_start_fn(fresh_path)
                    st.session_state.show_confirm_clean_start = False
                    st.rerun()
                if c2.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_confirm_clean_start = False
                    st.rerun()

        st.markdown("---")
        st.info(
            "💡 **Tip:** For database health issues, orphaned entries, and collection repairs, "
            "use the **Database Health Check** section above."
        )
