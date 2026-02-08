"""Shared batch-management UI helpers for Knowledge Ingest pages."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.utils import convert_to_docker_mount_path


def render_active_batch_management(
    *,
    batch_manager,
    batch_status: dict,
    auto_resume_from_batch_config,
    start_ingest_reader,
    get_ingest_lines,
    should_auto_finalize,
    start_automatic_finalization,
    initialize_state,
    logger,
) -> None:
    """Render active batch management UI and controls."""
    st.subheader("📊 Active Batch Management")
    st.markdown("### 🎛️ Batch Controls")
    st.warning("⚠️ **Batch processing is active.** Use the controls below to manage it.")

    control_col1, control_col2, control_col3, control_col4 = st.columns(4)

    with control_col1:
        if st.button("⏹️ **STOP BATCH**", key="stop_active_batch", type="secondary", use_container_width=True):
            proc = st.session_state.get("ingestion_process")
            if proc:
                try:
                    proc.terminate()
                except Exception as e:
                    logger.warning(f"Process termination failed: {e}")
                st.session_state.ingestion_process = None
            stop_ev = st.session_state.get("ingestion_reader_stop")
            if stop_ev:
                stop_ev.set()
            batch_manager.clear_batch()
            st.session_state.ingestion_stage = "config"
            st.success("✅ Batch stopped and cleared!")
            st.rerun()

    with control_col2:
        if not batch_status.get("paused", False):
            if st.button("⏸️ Pause", key="pause_active_batch", use_container_width=True):
                batch_manager.pause_batch()
                st.success("Pause request sent")
                st.rerun()
        else:
            st.info("⏸️ Paused")

    with control_col3:
        if st.button("🗑️ Clear Batch", key="clear_active_batch", use_container_width=True):
            batch_manager.clear_batch()
            st.success("Batch cleared")
            st.rerun()

    with control_col4:
        if st.button("⬅️ Back", key="back_from_active_batch", use_container_width=True):
            initialize_state(force_reset=True)
            st.rerun()

    st.markdown("---")
    r1, r2 = st.columns([1, 1])
    with r1:
        if st.button("🔄 Refresh Progress", key="refresh_active_batch_progress"):
            st.rerun()
    with r2:
        auto_refresh = st.checkbox("Auto refresh (3s)", key="auto_refresh_active_batch")
        if auto_refresh:
            time.sleep(1)
            st.rerun()

    process_obj = st.session_state.get("ingestion_process")
    if (
        batch_status.get("active", False)
        and not batch_status.get("paused", False)
        and batch_status.get("remaining", 0) > 0
        and not process_obj
    ):
        try:
            st.info("🔌 Reattaching to batch and starting processing…")
            if auto_resume_from_batch_config(batch_manager):
                st.rerun()
                return
        except Exception as e:
            logger.warning(f"Auto-start from active state failed: {e}")

    if process_obj and st.session_state.get("ingestion_output_queue") is None:
        try:
            start_ingest_reader(process_obj)
            st.info("🔌 Reattached log reader to running process…")
        except Exception as e:
            logger.warning(f"Failed to reattach reader in active mgmt: {e}")

    process_running = False
    if process_obj:
        try:
            process_running = process_obj.poll() is None
        except Exception:
            process_running = False

    if batch_status.get("is_chunked", False):
        if batch_status.get("auto_pause_after_chunks"):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Overall Progress", f"{batch_status['completed']}/{batch_status.get('total_files', 0)}")
            with c2:
                st.metric("Current Chunk", f"{batch_status['current_chunk']}/{batch_status['total_chunks']}")
            with c3:
                st.metric(
                    "Chunk Documents",
                    f"{batch_status.get('current_chunk_progress', 0)}/{batch_status.get('current_chunk_total', batch_status.get('chunk_size', 0))}",
                )
            with c4:
                st.metric(
                    "Session Chunks",
                    f"{batch_status['chunks_processed_in_session']}/{batch_status['auto_pause_after_chunks']}",
                )
            with c5:
                st.metric("Remaining Files", batch_status["remaining"])
            with c6:
                st.metric("Errors", batch_status["error_count"])
        else:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Overall Progress", f"{batch_status['completed']}/{batch_status.get('total_files', 0)}")
            with c2:
                st.metric("Current Chunk", f"{batch_status['current_chunk']}/{batch_status['total_chunks']}")
            with c3:
                st.metric(
                    "Chunk Documents",
                    f"{batch_status.get('current_chunk_progress', 0)}/{batch_status.get('current_chunk_total', batch_status.get('chunk_size', 0))}",
                )
            with c4:
                st.metric("Chunk Size", batch_status["chunk_size"])
            with c5:
                st.metric("Remaining Files", batch_status["remaining"])
            with c6:
                st.metric("Errors", batch_status["error_count"])
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Progress", f"{batch_status['completed']}/{batch_status.get('total_files', 0)}")
        with c2:
            st.metric("Completed", f"{batch_status['progress_percent']}%")
        with c3:
            st.metric("Remaining", batch_status["remaining"])
        with c4:
            st.metric("Errors", batch_status["error_count"])

    if batch_status.get("is_chunked", False):
        st.markdown("**Overall Progress**")
        st.progress(
            batch_status["progress_percent"] / 100.0,
            text=f"Total: {batch_status['completed']}/{batch_status.get('total_files', 0)} files ({batch_status['progress_percent']}%)",
        )
        st.markdown("**Current Chunk Progress**")
        chunk_docs = batch_status.get("current_chunk_progress", 0)
        chunk_total = batch_status.get("current_chunk_total", batch_status.get("chunk_size", 0))
        st.progress(
            batch_status.get("chunk_progress_percent", 0) / 100.0,
            text=f"Chunk {batch_status['current_chunk']}: {chunk_docs}/{chunk_total} documents ({batch_status.get('chunk_progress_percent', 0)}%)",
        )
    else:
        st.progress(
            batch_status["progress_percent"] / 100.0,
            text=f"Processing files... {batch_status['completed']}/{batch_status.get('total_files', 0)}",
        )

    proc = st.session_state.get("ingestion_process")
    if proc and proc.poll() is None:
        for line in get_ingest_lines(max_lines=50):
            if line.startswith("CORTEX_THROTTLE::"):
                try:
                    _, delay_str, gpu_str, cpu_str = line.split("::", 3)
                    st.session_state.current_throttle_delay = float(delay_str)
                    st.session_state.current_gpu_util = None if gpu_str == "N/A" else float(gpu_str)
                    st.session_state.current_cpu_util = None if cpu_str == "N/A" else float(cpu_str)
                    st.session_state.last_heartbeat_ts = time.time()
                except Exception:
                    pass
            elif line.startswith("CORTEX_HEARTBEAT::"):
                try:
                    _, gpu_str, cpu_str = line.split("::", 2)
                    st.session_state.current_gpu_util = None if gpu_str == "N/A" else float(gpu_str)
                    st.session_state.current_cpu_util = None if cpu_str == "N/A" else float(cpu_str)
                    st.session_state.last_heartbeat_ts = time.time()
                except Exception:
                    pass

    st.markdown("---")
    st.markdown("**⚡ Performance Tuning**")
    tc1, tc2, tc3 = st.columns([1, 1, 1])
    with tc1:
        st.metric("⏱️ Throttle Delay", f"{st.session_state.get('current_throttle_delay', 1.0):.1f}s")
    with tc2:
        gpu_val = st.session_state.get("current_gpu_util", None)
        st.metric("🎮 GPU Load", f"{gpu_val:.0f}%" if gpu_val is not None else "N/A")
    with tc3:
        cpu_val = st.session_state.get("current_cpu_util", None)
        st.metric("💻 CPU Load", f"{cpu_val:.0f}%" if cpu_val is not None else "N/A")

    st.markdown("---")
    if batch_status["paused"]:
        if batch_status.get("auto_pause_after_chunks") and batch_status.get("chunks_processed_in_session", 0) >= batch_status.get("auto_pause_after_chunks", 0):
            st.success(
                f"🎯 **Session Complete!** Processed {batch_status['chunks_processed_in_session']} chunks "
                f"({batch_status['chunks_processed_in_session'] * batch_status.get('chunk_size', 0)} files)"
            )
            st.info("Ready to start next session during your off-peak hours")
        else:
            st.warning("⏸️ Batch processing is **PAUSED**")
    else:
        st.info(f"🔄 Batch processing active (Started: {batch_status['started_at'][:19]})")

    if batch_status.get("error_count", 0) > 0:
        error_count = batch_status["error_count"]
        completed = batch_status["completed"]
        total = batch_status.get("total_files", 0)
        success_rate = round((completed / total) * 100, 1) if total > 0 else 0
        if success_rate >= 95:
            st.warning(
                f"📝 **{error_count} files skipped** during processing ({success_rate}% success rate: {completed}/{total} files completed)."
            )
        else:
            st.error(
                f"⚠️ **{error_count} errors encountered** ({success_rate}% success rate: {completed}/{total} files completed)."
            )

    if batch_status.get("remaining", 0) == 0 and batch_status.get("completed", 0) > 0:
        st.success("🎉 **Batch Processing Complete!** All files have been processed.")
        st.info(
            "Collection assignment is performed in finalization only. "
            "This avoids stale/orphan collection references."
        )
        st.session_state.last_ingested_doc_ids = []
        st.session_state.target_collection_name = ""
        batch_manager.clear_batch()

    auto_finalize_ready = should_auto_finalize() and batch_status.get("completed", 0) > 0 and batch_status.get("remaining", 0) == 0
    auto_finalize_enabled = bool(
        st.session_state.get("batch_mode_active", False) or batch_status.get("auto_finalize_enabled", False)
    )
    if auto_finalize_ready:
        st.markdown("---")
        st.info("📦 **Analysis Complete!** Your documents are ready to be added to the knowledge base.")
        fc1, fc2 = st.columns([3, 1])
        if auto_finalize_enabled and not st.session_state.get("batch_auto_finalize_started"):
            st.success("🚀 Starting automatic finalization for batch mode…")
            st.session_state.batch_auto_finalize_started = True
            st.session_state.auto_finalize_triggered = True
            start_automatic_finalization()
            st.rerun()
            return
        with fc1:
            if st.button("✅ Complete Ingestion (Finalize to Database)", type="primary", use_container_width=True, key="manual_finalize_batch"):
                with st.spinner("Finalizing documents to database..."):
                    try:
                        container_db_path = convert_to_docker_mount_path(st.session_state.db_path)
                        command = [
                            sys.executable,
                            "-m",
                            "cortex_engine.ingest_cortex",
                            "--finalize-from-staging",
                            "--db-path",
                            container_db_path,
                        ]
                        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
                        if result.returncode == 0:
                            st.success("✅ Documents successfully added to knowledge base!")
                            batch_manager.clear_batch()
                            st.session_state.manual_finalize_success = True
                            st.session_state.auto_finalize_triggered = False
                            st.session_state.ingestion_stage = "config_done"
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Finalization failed: {result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Failed to finalize: {e}")
        with fc2:
            st.caption("This will add your analyzed documents to the searchable database.")

    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        batch_is_processing = (
            batch_status.get("active", False)
            and not batch_status.get("paused", False)
            and batch_status.get("remaining", 0) > 0
        )
        is_paused = batch_status.get("paused", False)
        if batch_is_processing:
            if st.button("⏸️ Pause Processing", type="secondary", use_container_width=True, key="pause_processing_main"):
                batch_manager.pause_batch()
                st.success("⏸️ Processing paused")
                st.rerun()
        else:
            button_text = "▶️ Resume Processing"
            if batch_status.get("auto_pause_after_chunks") and is_paused and batch_status.get("chunks_processed_in_session", 0) >= batch_status.get("auto_pause_after_chunks", 0):
                button_text = "🔄 Start Next Session"
            if st.button(button_text, type="primary", use_container_width=True, key="resume_processing_main"):
                try:
                    if is_paused:
                        batch_manager.start_new_session()
                except Exception:
                    pass
                if auto_resume_from_batch_config(batch_manager):
                    st.rerun()
                else:
                    st.error("❌ Failed to resume batch automatically. Please check the logs.")
    with ac2:
        if st.button("🗑️ Clear This Batch", key="clear_active_batch", use_container_width=True):
            batch_manager.clear_batch()
            st.success("Batch cleared")
            st.rerun()
    with ac3:
        if st.button("📋 View Ingestion Logs", key="view_logs_batch", use_container_width=True):
            st.session_state.show_logs = True
            st.rerun()
    with ac4:
        if st.button("⬅️ Back to Config", key="back_to_config_batch", use_container_width=True):
            batch_manager.clear_batch()
            initialize_state(force_reset=True)
            st.rerun()

    if st.session_state.get("show_logs", False):
        st.markdown("---")
        st.subheader("📋 Recent Ingestion Logs")
        ingestion_log_path = Path(__file__).parent.parent / "logs" / "ingestion.log"
        if ingestion_log_path.exists():
            try:
                with open(ingestion_log_path, "r") as log_file:
                    log_content = log_file.read()
                if log_content.strip():
                    log_lines = log_content.strip().split("\n")
                    lc1, lc2, lc3 = st.columns([2, 1, 1])
                    with lc1:
                        line_options = [25, 50, 100, 200, len(log_lines)]
                        line_labels = ["Last 25 lines", "Last 50 lines", "Last 100 lines", "Last 200 lines", f"All {len(log_lines)} lines"]
                        selected_lines = st.selectbox(
                            "Log Display:",
                            options=line_options,
                            format_func=lambda x: line_labels[line_options.index(x)],
                            index=1,
                            key="log_display_lines",
                        )
                    with lc2:
                        if st.button("🔄 Refresh Logs", key="refresh_logs"):
                            st.rerun()
                    with lc3:
                        if st.button("❌ Hide Logs", key="hide_logs"):
                            st.session_state.show_logs = False
                            st.rerun()
                    display_lines = log_lines if selected_lines >= len(log_lines) else log_lines[-selected_lines:]
                    st.text_area(
                        "📝 Ingestion Log Output:",
                        value="\n".join(f"{i+1:4d}: {line}" for i, line in enumerate(display_lines)),
                        height=400,
                        disabled=True,
                        key="log_display_area",
                    )
                else:
                    st.info("No log entries found.")
            except Exception as e:
                st.error(f"Error reading logs: {e}")
        else:
            st.warning("Ingestion log file not found.")

    batch_is_processing = (
        batch_status.get("active", False)
        and not batch_status.get("paused", False)
        and batch_status.get("remaining", 0) > 0
    )
    if not batch_is_processing:
        st.markdown("---")
        st.subheader("⚙️ Processing Configuration")
        if batch_status.get("is_chunked", False):
            current_chunk_size = batch_status.get("chunk_size", 250)
            current_auto_pause = batch_status.get("auto_pause_after_chunks")
            st.info(
                f"✅ **Chunked Processing Active**: {current_chunk_size} files per chunk"
                + (f", auto-pause after {current_auto_pause} chunks" if current_auto_pause else ", no auto-pause")
            )
            with st.expander("🔧 **Modify Processing Settings**", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    new_chunk_size = st.selectbox(
                        "New Chunk Size:",
                        options=[100, 250, 500, 1000],
                        index=[100, 250, 500, 1000].index(current_chunk_size) if current_chunk_size in [100, 250, 500, 1000] else 1,
                        key="modify_chunk_size",
                    )
                with c2:
                    new_estimated_chunks = (batch_status.get("total_files", 0) + new_chunk_size - 1) // new_chunk_size
                    st.metric("New Total Chunks", new_estimated_chunks)
                with c3:
                    new_auto_pause = st.selectbox(
                        "Auto-pause after:",
                        options=[1, 2, 3, 5, 10, "No auto-pause"],
                        index=([1, 2, 3, 5, 10].index(current_auto_pause) if current_auto_pause and current_auto_pause in [1, 2, 3, 5, 10] else 5) if current_auto_pause else 2,
                        key="modify_auto_pause",
                    )
                if st.button("🔄 Apply New Settings", type="secondary", use_container_width=True):
                    batch_state = batch_manager.load_state()
                    if batch_state:
                        remaining_files = batch_state.get("files_remaining", [])
                        if remaining_files:
                            batch_manager.clear_batch()
                            scan_config = batch_state.get("scan_config", {})
                            auto_finalize_enabled = batch_state.get(
                                "auto_finalize_enabled",
                                scan_config.get("auto_finalize_enabled", False),
                            )
                            batch_manager.create_batch(
                                remaining_files,
                                scan_config,
                                new_chunk_size,
                                new_auto_pause if new_auto_pause != "No auto-pause" else None,
                                auto_finalize_enabled=auto_finalize_enabled,
                            )
                    st.success("✅ Settings updated.")
                    st.rerun()
        else:
            if batch_status.get("total_files", 0) > 500:
                st.warning(f"⚠️ **Large Batch**: {batch_status.get('total_files', 0)} files may cause memory issues")
                with st.expander("🔧 **Convert to Chunked Processing** (Recommended)", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        chunk_size_main = st.selectbox("Chunk Size:", options=[100, 250, 500, 1000], index=1, key="convert_chunk_size")
                    with c2:
                        estimated_chunks = (batch_status.get("total_files", 0) + chunk_size_main - 1) // chunk_size_main
                        st.metric("Estimated Chunks", estimated_chunks)
                    with c3:
                        auto_pause_chunks = st.selectbox(
                            "Auto-pause after:",
                            options=[1, 2, 3, 5, 10, "No auto-pause"],
                            index=2,
                            key="convert_auto_pause",
                        )
                    if st.button("🔄 Convert to Chunked Processing", type="secondary", use_container_width=True):
                        batch_state = batch_manager.load_state()
                        if batch_state:
                            remaining_files = batch_state.get("files_remaining", [])
                            if remaining_files:
                                batch_manager.clear_batch()
                                scan_config = batch_state.get("scan_config", {})
                                auto_finalize_enabled = batch_state.get(
                                    "auto_finalize_enabled",
                                    scan_config.get("auto_finalize_enabled", False),
                                )
                                batch_manager.create_batch(
                                    remaining_files,
                                    scan_config,
                                    chunk_size_main,
                                    auto_pause_chunks if auto_pause_chunks != "No auto-pause" else None,
                                    auto_finalize_enabled=auto_finalize_enabled,
                                )
                                st.success("✅ Chunked processing enabled!")
                                st.rerun()
            else:
                st.info("💡 **Small Batch**: No chunking needed for this batch size.")

    if process_running and not batch_status.get("paused", False):
        time.sleep(1)
        st.rerun()
