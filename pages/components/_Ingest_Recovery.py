"""Shared recovery panel helpers for Knowledge Ingest pages."""

from __future__ import annotations

import streamlit as st


def render_recovery_panels(
    *,
    config_manager_cls,
    recovery_manager_cls,
    logger,
    shared_check_recovery_needed_fn,
    shared_render_recovery_section_fn,
    render_recovery_quick_actions_fn,
    recover_collection_from_ingest_log_fn,
    filter_valid_doc_ids_fn,
    collection_manager_cls,
) -> None:
    """Render quick recovery actions + recovery diagnostics with fallback."""

    def _check_recovery_needed():
        return shared_check_recovery_needed_fn(
            config_manager_cls=config_manager_cls,
            recovery_manager_cls=recovery_manager_cls,
            ttl_seconds=120,
        )

    def _render_recovery_section():
        shared_render_recovery_section_fn(
            check_recovery_needed_fn=_check_recovery_needed,
            config_manager_cls=config_manager_cls,
            maintenance_page="pages/6_Maintenance.py",
            logger=logger,
        )

    render_recovery_quick_actions_fn(
        config_manager_cls=config_manager_cls,
        filter_valid_doc_ids_fn=filter_valid_doc_ids_fn,
        collection_manager_cls=collection_manager_cls,
    )

    st.markdown("---")

    try:
        _render_recovery_section()
    except Exception as e:
        st.error(f"Recovery section failed to load: {e}")
        with st.expander("🔧 Basic Recovery Tool", expanded=False):
            st.warning("Advanced recovery features unavailable. Using basic recovery.")
            if st.button("🚀 Create Collection from All Recent Ingests"):
                try:
                    db_path = config_manager_cls().get_config().get("ai_database_path", "")
                    result = recover_collection_from_ingest_log_fn(
                        db_path=db_path,
                        collection_name="recovered_ingestion",
                        filter_valid_doc_ids_fn=filter_valid_doc_ids_fn,
                        collection_manager_cls=collection_manager_cls,
                    )
                    if not result.get("ok"):
                        st.error(result.get("error", "Basic recovery failed"))
                    else:
                        if result.get("created"):
                            st.success(f"Created collection '{result['collection_name']}'")
                        st.success(
                            f"✅ Recovered {result['recovered_doc_ids']} valid documents "
                            f"to '{result['collection_name']}' collection!"
                        )
                        if result.get("skipped_doc_ids", 0) > 0:
                            st.warning(
                                f"Skipped {result['skipped_doc_ids']} orphan/stale IDs "
                                "that were not found in the vector store."
                            )
                except Exception as recovery_error:
                    st.error(f"Basic recovery failed: {recovery_error}")
