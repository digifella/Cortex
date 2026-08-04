"""Shared database health-check controls for Maintenance pages."""

from __future__ import annotations

import streamlit as st


def render_database_health_check_section(*, db_path: str, recovery_manager_cls) -> None:
    """Render DB health scan and repair tools."""
    with st.container(border=True):
        st.subheader("🔍 Database Health Check")
        st.caption("Scan for inconsistencies between the ingestion log and ChromaDB")

        if st.button("🔍 Scan Database", use_container_width=True, key="btn_scan_database"):
            try:
                with st.spinner("Analyzing database state..."):
                    recovery_manager = recovery_manager_cls(db_path)
                    analysis = recovery_manager.analyze_ingestion_state()
                    st.session_state.health_check_results = analysis
                    st.session_state.health_check_db_path = db_path
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Database scan failed: {e}")

        if "health_check_results" in st.session_state:
            analysis = st.session_state.health_check_results
            stats = analysis.get("statistics", {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files in Log", stats.get("ingested_files_count", 0))
            with col2:
                st.metric("Files in ChromaDB", stats.get("chromadb_files_count", 0))
            with col3:
                orphaned = stats.get("orphaned_count", 0)
                st.metric(
                    "Orphaned Entries",
                    orphaned,
                    delta=f"-{orphaned}" if orphaned > 0 else None,
                    delta_color="inverse",
                )

            if orphaned > 0:
                st.warning(f"⚠️ Found {orphaned} orphaned log entries")
                st.caption(
                    "Files in the log but missing from ChromaDB. "
                    "May have failed to ingest or were manually deleted."
                )

                show_orphaned = st.checkbox(f"Show orphaned files ({orphaned})", key="show_orphaned_files")
                if show_orphaned:
                    with st.container(height=200):
                        for doc in analysis.get("orphaned_documents", [])[:50]:
                            st.text(f"• {doc['file_name']}")
                        if orphaned > 50:
                            st.caption(f"... and {orphaned - 50} more")

                if st.button(
                    "🔧 Remove Orphaned Entries",
                    type="primary",
                    use_container_width=True,
                    key="btn_fix_orphaned",
                ):
                    try:
                        with st.spinner("Cleaning up orphaned entries..."):
                            recovery_manager = recovery_manager_cls(st.session_state.health_check_db_path)
                            cleanup_result = recovery_manager.cleanup_orphaned_log_entries()

                        if cleanup_result.get("status") == "success":
                            st.success(f"✅ Removed {cleanup_result['entries_removed']} orphaned entries")
                            analysis = recovery_manager.analyze_ingestion_state()
                            st.session_state.health_check_results = analysis
                            st.rerun()
                        else:
                            st.error(f"❌ Cleanup failed: {cleanup_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Cleanup failed: {e}")

            elif stats.get("ingested_files_count", 0) > 0:
                st.success("✅ Database is healthy - no orphaned entries found")

            collection_issues = analysis.get("collection_inconsistencies", [])
            if collection_issues:
                st.warning(f"⚠️ Found {len(collection_issues)} collection inconsistencies")
                show_issues = st.checkbox(
                    f"Show issues ({len(collection_issues)})",
                    key="show_collection_issues",
                )
                if show_issues:
                    for issue in collection_issues:
                        if issue.get("type") == "missing_from_chromadb":
                            st.text(f"• {issue['collection']}: {issue['count']} missing documents")

                if st.button("🔧 Fix Collection Issues", use_container_width=True, key="btn_fix_collections"):
                    try:
                        with st.spinner("Repairing collections..."):
                            recovery_manager = recovery_manager_cls(st.session_state.health_check_db_path)
                            repair_result = recovery_manager.auto_repair_collections()

                        if repair_result.get("status") == "success":
                            st.success(f"✅ Fixed {repair_result['invalid_refs_removed']} invalid references")
                            analysis = recovery_manager.analyze_ingestion_state()
                            st.session_state.health_check_results = analysis
                            st.rerun()
                        else:
                            st.error(f"❌ Repair failed: {repair_result.get('error', 'Unknown')}")
                    except Exception as e:
                        st.error(f"❌ Repair failed: {e}")

            if st.button("🔄 Clear Results", use_container_width=True, key="btn_clear_health_results"):
                del st.session_state.health_check_results
                if "health_check_db_path" in st.session_state:
                    del st.session_state.health_check_db_path
                st.rerun()
