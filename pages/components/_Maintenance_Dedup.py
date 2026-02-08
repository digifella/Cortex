"""Shared database deduplication panel for Maintenance pages."""

from __future__ import annotations

import streamlit as st


def render_database_dedup_section(
    *,
    db_path: str,
    init_chroma_client_fn,
    collection_name: str,
    working_collection_manager_cls,
    logger,
) -> None:
    """Render duplicate analysis and cleanup controls."""
    with st.expander("🔧 Database Deduplication & Optimization", expanded=False):
        st.subheader("🔧 Database Deduplication")
        st.markdown("Remove duplicate documents from the knowledge base to improve performance and storage efficiency.")

        chroma_client = init_chroma_client_fn(db_path)
        if not chroma_client:
            st.warning("ChromaDB not accessible. Cannot perform deduplication operations.")
            return

        try:
            vector_collection = chroma_client.get_collection(name=collection_name)
            collection_mgr = working_collection_manager_cls()

            dedup_col1, dedup_col2 = st.columns([2, 1])

            with dedup_col1:
                st.markdown(
                    """
                    **What does deduplication do?**
                    - Identifies documents with identical file hashes or content
                    - Keeps the most complete version of each document
                    - Removes duplicate entries from ChromaDB
                    - Updates collections to remove references to deleted duplicates
                    """
                )

                if "dedup_analysis_results" not in st.session_state:
                    st.session_state.dedup_analysis_results = None
                if "dedup_analysis_running" not in st.session_state:
                    st.session_state.dedup_analysis_running = False

            with dedup_col2:
                if st.button(
                    "🔍 Analyze Duplicates",
                    key="analyze_duplicates_btn",
                    type="secondary",
                    use_container_width=True,
                    disabled=st.session_state.dedup_analysis_running,
                ):
                    st.session_state.dedup_analysis_running = True

                    with st.spinner("Analyzing knowledge base for duplicates... This may take a few minutes."):
                        try:
                            results = collection_mgr.deduplicate_vector_store(vector_collection, dry_run=True)
                            st.session_state.dedup_analysis_results = results

                            if results.get("status") == "analysis_complete":
                                st.success("✅ Analysis complete!")
                                st.info(
                                    f"""
                                    **Duplicate Analysis Results:**
                                    - Total documents: {results['total_documents']:,}
                                    - Duplicates found: {results['duplicates_found']:,}
                                    - Duplicate percentage: {results['duplicate_percentage']:.1f}%
                                    - Unique files: {results['unique_files']:,}
                                    - Duplicate groups: {results['duplicate_groups']:,}
                                    """
                                )
                                logger.info(
                                    "Deduplication analysis completed via Maintenance UI: "
                                    f"{results['duplicates_found']} duplicates found"
                                )
                            else:
                                st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")

                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                            logger.error(f"Maintenance UI deduplication analysis failed: {e}")
                        finally:
                            st.session_state.dedup_analysis_running = False
                            st.rerun()

            if st.session_state.dedup_analysis_results:
                results = st.session_state.dedup_analysis_results

                if results.get("status") == "analysis_complete" and results.get("duplicates_found", 0) > 0:
                    st.divider()

                    result_col1, result_col2, result_col3 = st.columns(3)

                    with result_col1:
                        st.metric("📄 Total Documents", f"{results['total_documents']:,}")
                    with result_col2:
                        st.metric("🔄 Duplicates Found", f"{results['duplicates_found']:,}")
                    with result_col3:
                        st.metric("📊 Duplicate %", f"{results['duplicate_percentage']:.1f}%")

                    st.divider()

                    st.markdown("**🧹 Cleanup Options**")

                    if results["duplicate_percentage"] > 50:
                        st.warning(
                            f"⚠️ High duplicate percentage detected ({results['duplicate_percentage']:.1f}%). "
                            "This suggests a significant duplication issue that should be resolved."
                        )
                    elif results["duplicate_percentage"] > 20:
                        st.info(
                            f"💡 Moderate duplication detected ({results['duplicate_percentage']:.1f}%). "
                            "Cleanup recommended for optimal performance."
                        )
                    else:
                        st.success(
                            f"✅ Low duplication level ({results['duplicate_percentage']:.1f}%). "
                            "Cleanup optional but will improve storage efficiency."
                        )

                    cleanup_col1, cleanup_col2 = st.columns([2, 1])

                    with cleanup_col1:
                        st.markdown(
                            f"""
                            **Cleanup will:**
                            - Remove {results['duplicates_found']:,} duplicate documents
                            - Keep the most complete version of each file
                            - Update {len(collection_mgr.get_collection_names())} collections automatically
                            - Free up storage space and improve query performance
                            """
                        )

                    with cleanup_col2:
                        if st.checkbox("I understand this action cannot be undone", key="dedup_confirm_checkbox"):
                            if st.button(
                                "🧹 Remove Duplicates",
                                key="remove_duplicates_btn",
                                type="primary",
                                use_container_width=True,
                            ):
                                with st.spinner(
                                    f"Removing {results['duplicates_found']:,} duplicate documents... "
                                    "This may take several minutes."
                                ):
                                    try:
                                        cleanup_results = collection_mgr.deduplicate_vector_store(
                                            vector_collection,
                                            dry_run=False,
                                        )

                                        if cleanup_results.get("status") == "cleanup_complete":
                                            removed_count = cleanup_results.get("removed_count", 0)
                                            st.success("✅ Deduplication complete!")
                                            st.info(
                                                f"""
                                                **Cleanup Results:**
                                                - Documents removed: {removed_count:,}
                                                - Storage space freed: ~{removed_count * 0.1:.1f} MB (estimated)
                                                - Collections updated automatically
                                                """
                                            )

                                            st.session_state.dedup_analysis_results = None

                                            logger.info(
                                                "Deduplication cleanup completed via Maintenance UI: "
                                                f"{removed_count} documents removed"
                                            )

                                            st.success(
                                                "🔄 **Recommendation:** Restart the application to ensure "
                                                "optimal performance with the cleaned database."
                                            )

                                        else:
                                            st.error(
                                                f"❌ Cleanup failed: {cleanup_results.get('error', 'Unknown error')}"
                                            )

                                    except Exception as e:
                                        st.error(f"❌ Cleanup failed: {str(e)}")
                                        logger.error(f"Maintenance UI deduplication cleanup failed: {e}")

                elif results.get("status") == "analysis_complete" and results.get("duplicates_found", 0) == 0:
                    st.success("✅ No duplicates found! Your knowledge base is already optimized.")

                elif results.get("status") == "no_documents":
                    st.info("ℹ️ No documents found in the knowledge base.")

        except Exception as e:
            st.error(f"Could not access vector collection: {e}")
