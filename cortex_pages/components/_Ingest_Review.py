"""Shared metadata review/finalization UI helpers for Knowledge Ingest pages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st


def render_metadata_review_ui(
    *,
    should_auto_finalize,
    start_automatic_finalization,
    log_failed_documents,
    get_runtime_db_path,
    serialize_staging_payload,
    project_root,
    spawn_ingest,
    start_ingest_reader,
    initialize_state,
    get_full_file_content,
    review_page_size: int,
    doc_type_options,
    proposal_outcome_options,
    get_document_type_manager,
    convert_windows_to_wsl_path,
) -> None:
    """Render staged metadata review UI and trigger finalization."""
    st.header("Review AI-Generated Metadata")

    try:
        if should_auto_finalize() and st.session_state.get("ingestion_stage") != "finalizing":
            if st.button("🔁 Retry Finalization", key="retry_finalization", help="Run finalization again from staged results"):
                start_automatic_finalization()
                st.rerun()
    except Exception:
        pass

    if "edited_staged_files" not in st.session_state or not st.session_state.edited_staged_files:
        initial_files = st.session_state.get("staged_files", [])
        failed_docs = {}

        for doc in initial_files:
            is_error = doc.get("rich_metadata", {}).get("summary", "").startswith("ERROR:")
            doc["exclude_from_final"] = is_error

            if is_error and st.session_state.get("batch_ingest_mode", False):
                file_path = doc.get("doc_posix_path", "Unknown")
                error_msg = doc.get("rich_metadata", {}).get("summary", "Unknown error")
                failed_docs[file_path] = error_msg

        if failed_docs and st.session_state.get("batch_ingest_mode", False):
            failure_log_path = log_failed_documents(failed_docs, st.session_state.db_path)
            if failure_log_path:
                st.warning(f"⚠️ {len(failed_docs)} documents failed processing and were logged to: `{failure_log_path}`")

        st.session_state.edited_staged_files = initial_files
        st.session_state.review_page = 0

    edited_files = st.session_state.edited_staged_files
    if not edited_files:
        st.success("Analysis complete, but no documents were staged for review.")
        try:
            from cortex_engine.ingest_cortex import get_staging_file_path

            wsl_db_path = convert_windows_to_wsl_path(
                st.session_state.get("db_path_input", st.session_state.get("db_path", ""))
            )
            staging_path = Path(get_staging_file_path(wsl_db_path)) if wsl_db_path else None
            if staging_path:
                st.caption(f"Staging file: `{staging_path}` (exists={staging_path.exists()})")
                if staging_path.exists():
                    try:
                        with open(staging_path, "r") as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            staged_count = len(data.get("documents", []))
                        else:
                            staged_count = len(data)
                        st.caption(f"Parsed staged documents: {staged_count}")
                        if staged_count > 0:
                            st.info("Staging file contains documents. You can retry automatic finalization.")
                            if st.button("🚀 Retry Finalization", type="primary", key="retry_finalize_host"):
                                start_automatic_finalization()
                                st.stop()
                        with st.expander("Show staging JSON (first 2KB)", expanded=False):
                            preview = json.dumps(data, indent=2)
                            st.text_area("staging_ingestion.json", value=preview[:2048], height=200)
                    except Exception as pe:
                        st.warning(f"Could not read staging file: {pe}")
        except Exception:
            pass
        st.info("Check `logs/ingestion.log` for details.")
        if st.button("⬅️ Back to Configuration", key="back_config_no_staged_docs"):
            initialize_state(force_reset=True)
            st.rerun()
        return

    batch_mode = st.session_state.get("batch_ingest_mode", False) or st.session_state.get("batch_mode_active", False)

    if batch_mode:
        st.info(
            "🔧 **Debug:** Batch mode detected - "
            f"batch_ingest_mode: {st.session_state.get('batch_ingest_mode', False)}, "
            f"batch_mode_active: {st.session_state.get('batch_mode_active', False)}"
        )

    if batch_mode:
        if not st.session_state.get("batch_auto_processed", False):
            valid_files = [doc for doc in edited_files if not doc.get("exclude_from_final", False)]
            excluded_count = len(edited_files) - len(valid_files)

            st.info(
                f"🚀 **Batch Mode:** Automatically processing {len(valid_files)} valid documents. "
                f"{excluded_count} documents excluded due to errors."
            )

            if valid_files:
                st.session_state.last_ingested_doc_ids = [
                    doc["doc_id"] for doc in valid_files if not doc.get("exclude_from_final")
                ]
                container_db_path = get_runtime_db_path()
                if not container_db_path or not Path(container_db_path).exists():
                    st.error(f"Database path is invalid or does not exist: {container_db_path}")
                    st.stop()

                staging_file_path = Path(container_db_path) / "staging_ingestion.json"
                payload = serialize_staging_payload(
                    st.session_state.edited_staged_files,
                    st.session_state.get("target_collection_name"),
                )
                with open(staging_file_path, "w") as f:
                    json.dump(payload, f, indent=2)
                st.session_state.staged_metadata = payload

                st.session_state.log_messages = ["Finalizing batch ingestion..."]
                st.session_state.ingestion_stage = "finalizing"
                st.session_state.batch_auto_processed = True

                script_path = project_root / "cortex_engine" / "ingest_cortex.py"
                command = [
                    sys.executable,
                    str(script_path),
                    "--finalize-from-staging",
                    "--db-path",
                    container_db_path,
                ]

                if st.session_state.get("skip_image_processing", False):
                    command.append("--skip-image-processing")

                st.session_state.ingestion_process = spawn_ingest(command)
                start_ingest_reader(st.session_state.ingestion_process)
                st.rerun()
            else:
                st.warning("No valid documents to process after filtering errors.")
                if st.button("⬅️ Back to Configuration", key="back_config_batch_mode"):
                    initialize_state(force_reset=True)
                    st.rerun()
            return

        st.info("🚀 **Batch Mode:** Processing has been initiated automatically.")
        return

    st.info("📋 **Manual Mode:** Please review the metadata below and proceed when ready.")
    st.info(f"Please review and approve the metadata for the **{len(edited_files)}** document(s) below.")

    valid_docs_count = len([doc for doc in edited_files if not doc.get("exclude_from_final", False)])
    if valid_docs_count > 0:
        st.info(f"💡 **Quick Option:** Skip detailed review and proceed with {valid_docs_count} valid documents")
        if st.button("⚡ Skip Review & Proceed to Finalization", type="secondary", key="quick_proceed"):
            final_files_to_process = st.session_state.edited_staged_files
            doc_ids_to_ingest = [doc["doc_id"] for doc in final_files_to_process if not doc.get("exclude_from_final")]
            st.session_state.last_ingested_doc_ids = doc_ids_to_ingest

            container_db_path = get_runtime_db_path()
            staging_file_path = Path(container_db_path) / "staging_ingestion.json"
            payload = serialize_staging_payload(final_files_to_process, st.session_state.get("target_collection_name"))
            with open(staging_file_path, "w") as f:
                json.dump(payload, f, indent=2)
            st.session_state.staged_metadata = payload

            if not container_db_path or not Path(container_db_path).exists():
                st.error(f"Database path is invalid or does not exist: {container_db_path}")
                st.stop()

            st.session_state.log_messages = ["Finalizing ingestion..."]
            st.session_state.ingestion_stage = "finalizing"
            command = [
                sys.executable,
                "-m",
                "cortex_engine.ingest_cortex",
                "--finalize-from-staging",
                "--db-path",
                container_db_path,
            ]

            if st.session_state.get("skip_image_processing", False):
                command.append("--skip-image-processing")

            st.session_state.ingestion_process = spawn_ingest(command)
            start_ingest_reader(st.session_state.ingestion_process)
            st.rerun()

    st.markdown("---")

    page = st.session_state.review_page
    start_idx, end_idx = page * review_page_size, (page + 1) * review_page_size
    paginated_files = edited_files[start_idx:end_idx]
    total_pages = -(-len(edited_files) // review_page_size) or 1

    def update_edited_state(index, field, value):
        if field == "include":
            st.session_state.edited_staged_files[index]["exclude_from_final"] = not value
        elif field == "thematic_tags":
            st.session_state.edited_staged_files[index]["rich_metadata"][field] = [
                tag.strip() for tag in value.split(",") if tag.strip()
            ]
        else:
            st.session_state.edited_staged_files[index]["rich_metadata"][field] = value

    for i, doc in enumerate(paginated_files):
        absolute_index = start_idx + i
        rich_meta = doc.get("rich_metadata", {})
        is_included = not doc.get("exclude_from_final", False)
        checkbox_label = f"**{doc.get('file_name', 'N/A')}** - {rich_meta.get('summary', 'No summary available.')}"

        new_include_val = st.checkbox(checkbox_label, value=is_included, key=f"include_{absolute_index}")
        if new_include_val != is_included:
            update_edited_state(absolute_index, "include", new_include_val)
            st.rerun()

        with st.expander("Edit Metadata & Preview"):
            filename = doc.get("file_name", "")
            doc_type_manager = get_document_type_manager()
            suggested_type = doc_type_manager.suggest_document_type(filename)

            current_doc_type = rich_meta.get("document_type", "Other")
            if suggested_type != current_doc_type and suggested_type != "Other":
                st.info(
                    f"💡 **Auto-suggestion:** Based on the filename '{filename}', "
                    f"this document might be: **{suggested_type}**"
                )
                if st.button(f"✅ Use '{suggested_type}'", key=f"suggest_{absolute_index}"):
                    update_edited_state(absolute_index, "document_type", suggested_type)
                    st.rerun()

            try:
                doc_type_index = doc_type_options.index(rich_meta.get("document_type"))
            except (ValueError, TypeError):
                doc_type_index = len(doc_type_options) - 1
            try:
                outcome_index = proposal_outcome_options.index(rich_meta.get("proposal_outcome"))
            except (ValueError, TypeError):
                outcome_index = len(proposal_outcome_options) - 1

            selected_doc_type = st.selectbox(
                "Document Type",
                options=doc_type_options,
                index=doc_type_index,
                key=f"dt_{absolute_index}",
                on_change=lambda idx=absolute_index: update_edited_state(
                    idx, "document_type", st.session_state[f"dt_{idx}"]
                ),
            )

            if selected_doc_type and selected_doc_type != "Any":
                category = doc_type_manager.get_category_for_type(selected_doc_type)
                st.caption(f"📂 Category: {category}")
            st.selectbox(
                "Proposal Outcome",
                options=proposal_outcome_options,
                index=outcome_index,
                key=f"oc_{absolute_index}",
                on_change=lambda idx=absolute_index: update_edited_state(
                    idx, "proposal_outcome", st.session_state[f"oc_{idx}"]
                ),
            )
            st.text_area(
                "Summary",
                value=rich_meta.get("summary", ""),
                key=f"sm_{absolute_index}",
                height=100,
                on_change=lambda idx=absolute_index: update_edited_state(
                    idx, "summary", st.session_state[f"sm_{idx}"]
                ),
            )
            st.text_input(
                "Thematic Tags (comma-separated)",
                value=", ".join(rich_meta.get("thematic_tags", [])),
                key=f"tg_{absolute_index}",
                on_change=lambda idx=absolute_index: update_edited_state(
                    idx, "thematic_tags", st.session_state[f"tg_{idx}"]
                ),
            )
            st.divider()
            st.text_area(
                "File Content Preview",
                get_full_file_content(doc["doc_posix_path"]),
                height=200,
                disabled=True,
                key=f"preview_{doc['doc_posix_path']}",
            )

    st.divider()
    nav_cols = st.columns([1, 1, 5])
    if page > 0:
        nav_cols[0].button(
            "⬅️ Previous",
            on_click=lambda: st.session_state.update(review_page=page - 1),
            use_container_width=True,
        )
    if end_idx < len(edited_files):
        nav_cols[1].button(
            "Next ➡️",
            on_click=lambda: st.session_state.update(review_page=page + 1),
            use_container_width=True,
        )
    nav_cols[2].write(f"Page {page + 1} of {total_pages}")

    st.divider()

    valid_docs = [doc for doc in edited_files if not doc.get("exclude_from_final", False)]
    error_docs = [doc for doc in edited_files if doc.get("exclude_from_final", False)]

    if error_docs:
        st.warning(
            f"⚠️ {len(error_docs)} documents had errors and will be excluded. "
            f"{len(valid_docs)} documents are ready for ingestion."
        )
    else:
        st.success(f"✅ All {len(valid_docs)} documents are ready for ingestion.")

    action_cols = st.columns(2)
    if action_cols[0].button("⬅️ Cancel and Go Back", use_container_width=True):
        initialize_state(force_reset=True)
        st.rerun()

    finalize_enabled = len(valid_docs) > 0
    button_text = (
        f"✅ Finalize {len(valid_docs)} Approved Documents"
        if finalize_enabled
        else "❌ No Valid Documents to Finalize"
    )

    if action_cols[1].button(button_text, use_container_width=True, type="primary", disabled=not finalize_enabled):
        final_files_to_process = st.session_state.edited_staged_files
        doc_ids_to_ingest = [doc["doc_id"] for doc in final_files_to_process if not doc.get("exclude_from_final")]
        st.session_state.last_ingested_doc_ids = doc_ids_to_ingest

        container_db_path = get_runtime_db_path()
        staging_file_path = Path(container_db_path) / "staging_ingestion.json"
        payload = serialize_staging_payload(final_files_to_process, st.session_state.get("target_collection_name"))
        with open(staging_file_path, "w") as f:
            json.dump(payload, f, indent=2)
        st.session_state.staged_metadata = payload
        if not container_db_path or not Path(container_db_path).exists():
            st.error(f"Database path is invalid or does not exist: {container_db_path}")
            st.stop()

        st.session_state.log_messages = ["Finalizing ingestion..."]
        st.session_state.ingestion_stage = "finalizing"
        command = [
            sys.executable,
            "-m",
            "cortex_engine.ingest_cortex",
            "--finalize-from-staging",
            "--db-path",
            container_db_path,
        ]

        if st.session_state.get("skip_image_processing", False):
            command.append("--skip-image-processing")
        st.session_state.ingestion_process = spawn_ingest(command)
        start_ingest_reader(st.session_state.ingestion_process)
        st.rerun()
