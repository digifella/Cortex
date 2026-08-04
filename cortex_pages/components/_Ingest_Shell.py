"""Shared top-level workflow shell for Knowledge Ingest pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_ingest_page_shell(
    *,
    initialize_state,
    version_string: str,
    help_system,
    render_ollama_status_panel,
    render_recovery_panels,
    get_runtime_db_path,
    set_runtime_db_path,
    batch_state_cls,
    render_active_batch_management,
    detect_orphaned_session_from_log,
    render_orphaned_session_notice,
    render_document_type_management,
    show_collection_migration_healthcheck,
    render_sidebar_model_config,
    render_ingest_stage,
    stage_handlers: dict,
    project_root: Path,
) -> None:
    """Render the shared page shell and stage routing for ingest."""
    initialize_state()
    st.title("2. Knowledge Ingest")
    st.caption(f"Manage the knowledge base by ingesting new documents. App Version: {version_string}")

    help_system.show_help_menu()

    render_ollama_status_panel()
    render_recovery_panels()

    if st.session_state.get("show_help_modal", False):
        help_topic = st.session_state.get("help_topic", "overview")
        help_system.show_help_modal(help_topic)

    help_system.show_contextual_help("ingest")

    container_db_path = get_runtime_db_path()
    batch_manager = batch_state_cls(container_db_path)
    set_runtime_db_path(str(batch_manager.db_path))
    batch_status = batch_manager.get_status()
    stage = st.session_state.get("ingestion_stage", "config")

    if batch_status["active"] and stage not in {"analysis_running", "finalizing"}:
        render_active_batch_management(batch_manager, batch_status)

        st.markdown("---")
        if st.button("🆕 Start Fresh Ingestion", key="start_fresh", help="Clear current batch and start new ingestion"):
            batch_manager.clear_batch()
            st.success("Batch cleared. You can now start a fresh ingestion.")
            st.rerun()
    else:
        ingestion_log_path = project_root / "logs" / "ingestion.log"
        orphaned_session = detect_orphaned_session_from_log(ingestion_log_path)
        if orphaned_session and stage in {"config", "pre_analysis"}:
            render_orphaned_session_notice(orphaned_session)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "⚙️ Document Type Management",
                use_container_width=True,
                help="Manage document categories and type mappings",
            ):
                st.session_state.show_maintenance = not st.session_state.get("show_maintenance", False)
                st.rerun()

        st.markdown("---")

        if st.session_state.get("show_maintenance", False):
            render_document_type_management()
        else:
            show_collection_migration_healthcheck()
            render_sidebar_model_config()

            stage = st.session_state.get("ingestion_stage", "config")
            if stage in ["batch_processing", "analysis_running"]:
                batch_mode = st.session_state.get("batch_ingest_mode", False)
                st.info(f"🔍 **DEBUG:** Current stage: `{stage}` | Batch mode checkbox: `{batch_mode}`")

            render_ingest_stage(stage, stage_handlers)

    try:
        from cortex_engine.ui_components import render_version_footer

        render_version_footer()
    except Exception:
        pass
