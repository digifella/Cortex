"""Shared reset/recovery controls for Maintenance pages."""

from __future__ import annotations

import streamlit as st


def render_reset_recovery_section(
    *,
    db_path: str,
    ingested_files_log: str,
    clear_ingestion_log_file_fn,
    perform_clean_start_fn,
    delete_ingested_document_database_fn,
    expander_title: str = "🧹 Reset & Recovery",
) -> None:
    """Render the maintenance reset and recovery control block."""
    with st.expander(expander_title, expanded=False):
        st.caption("Simplified maintenance flow: repair first, then reset only if needed.")

        st.markdown("**Step 1: Optional log reset**")
        st.caption(
            f"Clears `{ingested_files_log}` so all source files are treated as new on the next ingest scan."
        )
        clear_log_confirm = st.checkbox(
            "I understand this will cause full file re-scan",
            key="maintenance_clear_log_confirm",
        )
        if st.button(
            "🧾 Clear Ingestion Log",
            use_container_width=True,
            key="maintenance_clear_log_btn",
            disabled=not clear_log_confirm,
        ):
            clear_ingestion_log_file_fn()

        st.divider()

        st.markdown("**Step 2: Knowledge base reset**")
        reset_scope = st.radio(
            "Reset scope",
            options=["Standard Reset", "Clean Start Reset"],
            help=(
                "Standard Reset deletes ChromaDB, graph, collections, and ingest state. "
                "Clean Start also clears extended workspace/entity artifacts."
            ),
            horizontal=True,
            key="maintenance_reset_scope",
        )

        if reset_scope == "Standard Reset":
            st.info("Deletes KB artifacts: vector DB, graph, collections, ingest state files.")
        else:
            st.warning(
                "Also deletes extended artifacts (workspaces/entity profiles/structured state) "
                "in addition to standard KB artifacts."
            )

        reset_confirm = st.checkbox(
            "I understand reset actions cannot be undone",
            key="maintenance_reset_confirm",
        )
        if st.button(
            "🚀 Run Reset",
            use_container_width=True,
            type="primary",
            key="maintenance_run_reset",
            disabled=not reset_confirm,
        ):
            fresh_path = st.session_state.get("maintenance_current_db_input", db_path)
            if reset_scope == "Clean Start Reset":
                perform_clean_start_fn(fresh_path)
            else:
                delete_ingested_document_database_fn(fresh_path)
