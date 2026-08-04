"""Shared processing/log UI helpers for Knowledge Ingest pages."""

from __future__ import annotations

import time

import streamlit as st

from cortex_engine.batch_manager import BatchState
from cortex_engine.utils import convert_to_docker_mount_path


def render_log_and_review_ui(
    *,
    stage_title: str,
    on_complete_stage: str,
    initialize_state,
    get_runtime_db_path,
    set_runtime_db_path,
    auto_resume_from_batch_config,
    start_ingest_reader,
    get_ingest_lines,
    load_staged_files,
    should_auto_finalize,
    start_automatic_finalization,
    max_auto_finalize_retries: int,
    logger,
) -> None:
    """Render processing logs/progress and transition to next ingest stage."""
    st.header(stage_title)

    # If the UI is in processing mode but no subprocess exists (e.g., app reload),
    # attempt an automatic resume from stored batch configuration.
    try:
        if st.session_state.get("ingestion_stage") == "analysis_running" and not st.session_state.get("ingestion_process"):
            container_db_path = convert_to_docker_mount_path(st.session_state.get("db_path", ""))
            if container_db_path:
                batch_manager = BatchState(container_db_path)
                set_runtime_db_path(str(batch_manager.db_path))
                batch_state = batch_manager.load_state()
                if batch_state and batch_state.get("files_remaining"):
                    if auto_resume_from_batch_config(batch_manager):
                        st.info("🔄 Auto-resume started after reload; continuing processing…")
                        st.rerun()
                        return
    except Exception as err:
        logger.warning(f"Auto-resume on reload skipped: {err}")

    # Add control buttons for pause/stop (always visible during processing)
    st.markdown("### 🎛️ Ingestion Controls")
    st.info("💡 **Tip:** You can stop the ingestion at any time using the Stop button below.")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("⏸️ Pause", key="pause_processing", use_container_width=True):
            container_db_path = get_runtime_db_path()
            batch_manager = BatchState(container_db_path)
            set_runtime_db_path(str(batch_manager.db_path))
            batch_manager.pause_batch()
            st.success("Pause requested")

    with col2:
        if st.button("⏹️ **STOP INGESTION**", key="stop_processing", use_container_width=True, type="secondary"):
            if st.session_state.get("ingestion_process"):
                try:
                    st.session_state.ingestion_process.terminate()
                except Exception as e:
                    logger.warning(f"Process termination failed: {e}")
                st.session_state.ingestion_process = None

            stop_ev = st.session_state.get("ingestion_reader_stop")
            if stop_ev:
                stop_ev.set()

            st.session_state.ingestion_stage = "config"
            st.warning("⚠️ Ingestion stopped by user. Returning to configuration...")
            st.rerun()

    with col3:
        if st.button("⬅️ Back", key="back_from_processing", use_container_width=True):
            initialize_state(force_reset=True)
            st.rerun()

    if "current_doc_number" not in st.session_state:
        st.session_state.current_doc_number = 0
    if "total_docs_in_batch" not in st.session_state:
        st.session_state.total_docs_in_batch = 0
    if "embedding_current" not in st.session_state:
        st.session_state.embedding_current = 0
    if "embedding_total" not in st.session_state:
        st.session_state.embedding_total = 0

    if st.session_state.total_docs_in_batch > 0:
        if st.session_state.get("finalize_started_detected", False):
            progress_text = f"🔄 Indexing batch {st.session_state.current_doc_number} of {st.session_state.total_docs_in_batch}"
            status_header = "**Finalization in Progress** (Embedding & Indexing)"
        else:
            progress_text = f"📄 Analyzing document {st.session_state.current_doc_number} of {st.session_state.total_docs_in_batch}"
            status_header = "**Analysis in Progress** (Scanning & Metadata Extraction)"

        st.markdown(f"### {status_header}")
        st.markdown(progress_text)
        progress_percent = st.session_state.current_doc_number / st.session_state.total_docs_in_batch
        progress_bar = st.progress(progress_percent, text=progress_text)

        if st.session_state.get("finalize_started_detected", False) and st.session_state.embedding_total > 0:
            embed_current = st.session_state.embedding_current
            embed_total = st.session_state.embedding_total
            embed_pct = (embed_current / embed_total * 100) if embed_total > 0 else 0
            embed_text = f"🔢 Embedding: {embed_current}/{embed_total} vectors ({embed_pct:.0f}%)"
            st.progress(embed_current / embed_total if embed_total > 0 else 0, text=embed_text)
    else:
        progress_bar = st.progress(0, text="Starting process...")

    throttle_col1, throttle_col2, throttle_col3 = st.columns([1, 1, 1])
    with throttle_col1:
        delay_val = st.session_state.get("current_throttle_delay", 0.0)
        st.metric("⏱️ Throttle Delay", f"{delay_val:.1f}s", help="Current delay between documents to reduce system load")
    with throttle_col2:
        gpu_val = st.session_state.get("current_gpu_util", None)
        if gpu_val is not None:
            gpu_delta = f"+{gpu_val - 80:.0f}%" if gpu_val > 80 else None
            st.metric("🎮 GPU Load", f"{gpu_val:.0f}%", delta=gpu_delta, delta_color="inverse", help="GPU utilization (throttle increases when high)")
        else:
            st.metric("🎮 GPU Load", "N/A", help="GPU monitoring unavailable or not yet sampled")
    with throttle_col3:
        cpu_val = st.session_state.get("current_cpu_util", None)
        if cpu_val is not None:
            cpu_delta = f"+{cpu_val - 85:.0f}%" if cpu_val > 85 else None
            st.metric("💻 CPU Load", f"{cpu_val:.0f}%", delta=cpu_delta, delta_color="inverse", help="CPU utilization (throttle increases when high)")
        else:
            st.metric("💻 CPU Load", "N/A", help="CPU monitoring unavailable or not yet sampled")

    if (gpu_val and gpu_val > 80) or (cpu_val and cpu_val > 85):
        st.warning("🎛️ **Adaptive throttling active** - System load detected, automatically slowing down to prevent freezing.")

    hb_ts = st.session_state.get("last_heartbeat_ts")
    if hb_ts:
        try:
            age = max(0, int(time.time() - hb_ts))
            st.caption(f"Last update {age}s ago")
        except Exception:
            pass

    process_still_running = False
    if st.session_state.ingestion_process:
        poll_result = st.session_state.ingestion_process.poll()
        process_still_running = poll_result is None

        if process_still_running and not st.session_state.get("ingestion_output_queue"):
            try:
                start_ingest_reader(st.session_state.ingestion_process)
                st.info("🔌 Reattached to running process; resuming live logs…")
            except Exception as reattach_err:
                logger.warning(f"Failed to reattach reader: {reattach_err}")

        for line in get_ingest_lines(max_lines=50):
            if line.startswith("CORTEX_PROGRESS::"):
                try:
                    _, progress_part, filename_part = line.split("::", 2)
                    current, total = map(int, progress_part.split("/"))
                    st.session_state.current_doc_number = current
                    st.session_state.total_docs_in_batch = total
                    st.session_state.log_messages.append(f"Processing {current}/{total}: {filename_part}")
                except (ValueError, IndexError):
                    st.session_state.log_messages.append(line)
            elif line.startswith("CORTEX_EMBEDDING::"):
                try:
                    _, progress_part, batch_info = line.split("::", 2)
                    current, total = map(int, progress_part.split("/"))
                    st.session_state.embedding_current = current
                    st.session_state.embedding_total = total
                    if current > 0:
                        pct = (current / total * 100) if total > 0 else 0
                        st.session_state.log_messages.append(f"🔢 Embedding: {current}/{total} ({pct:.0f}%)")
                except (ValueError, IndexError):
                    st.session_state.log_messages.append(line)
            elif line.startswith("CORTEX_THROTTLE::"):
                try:
                    _, delay_str, gpu_str, cpu_str = line.split("::", 3)
                    st.session_state.current_throttle_delay = float(delay_str)
                    gpu_val = None if gpu_str == "N/A" else float(gpu_str)
                    cpu_val = None if cpu_str == "N/A" else float(cpu_str)
                    st.session_state.current_gpu_util = gpu_val
                    st.session_state.current_cpu_util = cpu_val
                    st.session_state.throttle_active = bool((gpu_val is not None and gpu_val > 80) or (cpu_val is not None and cpu_val > 85))
                    st.session_state.last_heartbeat_ts = time.time()
                    gpu_disp = gpu_str if gpu_str != "N/A" else "N/A"
                    cpu_disp = cpu_str if cpu_str != "N/A" else "N/A"
                    st.session_state.log_messages.append(f"THROTTLE delay={float(delay_str):.1f}s GPU={gpu_disp}% CPU={cpu_disp}%")
                except (ValueError, IndexError):
                    pass
            elif line.startswith("CORTEX_HEARTBEAT::"):
                try:
                    _, gpu_str, cpu_str = line.split("::", 2)
                    gpu_val = None if gpu_str == "N/A" else float(gpu_str)
                    cpu_val = None if cpu_str == "N/A" else float(cpu_str)
                    st.session_state.current_gpu_util = gpu_val
                    st.session_state.current_cpu_util = cpu_val
                    st.session_state.last_heartbeat_ts = time.time()
                    gpu_disp = gpu_str if gpu_str != "N/A" else "N/A"
                    cpu_disp = cpu_str if cpu_str != "N/A" else "N/A"
                    ts = time.strftime("%H:%M:%S")
                    st.session_state.log_messages.append(f"[{ts}] HEARTBEAT GPU={gpu_disp}% CPU={cpu_disp}%")
                except (ValueError, IndexError):
                    pass
            elif line.startswith("CORTEX_STAGE::FINALIZE_START"):
                st.session_state.log_messages.append("🔄 Starting finalization (embedding documents)...")
                st.session_state.finalize_started_detected = True
            elif line.startswith("CORTEX_STAGE::FINALIZE_DONE"):
                st.session_state.log_messages.append("✅ Finalization completed successfully!")
                st.session_state.finalize_done_detected = True
                st.session_state.finalize_started_detected = False
                st.session_state.batch_auto_finalize_started = False
                st.session_state.batch_auto_processed = False
                st.session_state.batch_mode_active = False
                st.session_state.auto_finalize_triggered = False
            elif line.startswith("CORTEX_STAGE::ANALYSIS_DONE"):
                st.session_state.log_messages.append("✅ Analysis completed successfully!")
                st.session_state.analysis_done_detected = True
            else:
                st.session_state.log_messages.append(line)

        if len(st.session_state.log_messages) > 1000:
            st.session_state.log_messages = st.session_state.log_messages[-1000:]

        if not process_still_running:
            st.session_state.ingestion_process = None
            st.session_state.current_gpu_util = None
            st.session_state.current_cpu_util = None
            st.session_state.throttle_active = False
            stop_ev = st.session_state.get("ingestion_reader_stop")
            if stop_ev:
                stop_ev.set()

    log_expanded = process_still_running or st.session_state.get("ingestion_process") is not None
    with st.expander("📋 Processing Log (click to expand/collapse)", expanded=log_expanded):
        log_container = st.container(height=400, border=True)
        with log_container:
            if st.session_state.log_messages:
                st.code("\n".join(st.session_state.log_messages[-100:]), language="log")
            else:
                st.info("No log messages yet...")

    if not process_still_running and st.session_state.ingestion_process is None:
        if st.session_state.get("finalize_done_detected", False):
            st.session_state.finalize_done_detected = False
            st.session_state.ingestion_stage = "config_done"
            progress_bar.progress(1.0, text="✅ Finalization Complete!")
            st.success("✅ Finalization complete! Your knowledge base has been updated.")
            st.rerun()
            return
        if st.session_state.get("analysis_done_detected", False):
            if not st.session_state.get("finalize_started_detected", False):
                progress_bar.progress(1.0, text="✅ Analysis Complete!")
                st.success("✅ Analysis complete!")
        else:
            progress_bar.progress(1.0, text="Process completed")
            st.info("Process completed.")
    elif process_still_running:
        time.sleep(1)
        st.rerun()

    if on_complete_stage == "metadata_review" and not process_still_running:
        st.markdown("---")
        st.markdown("### 📊 Analysis Complete")

        load_staged_files()

        should_finalize = should_auto_finalize()
        finalize_done = st.session_state.get("finalize_done_detected", False)
        retry_attempts = st.session_state.get("auto_finalize_retry_attempts", 0)

        logger.info(
            f"🔍 Auto-finalize check: on_complete_stage={on_complete_stage}, "
            f"process_still_running={process_still_running}, should_finalize={should_finalize}, finalize_done={finalize_done}"
        )

        st.caption(
            "Auto-finalize status: "
            f"eligible={should_finalize}, already_done={finalize_done}, stage={on_complete_stage}"
        )

        if should_finalize and not finalize_done:
            st.session_state.auto_finalize_retry_attempts = 0
            st.success("✅ Analysis complete! Starting automatic finalization...")
            logger.info("Analysis completed successfully - starting automatic finalization")
            time.sleep(0.5)
            start_automatic_finalization()
            st.rerun()
            return

        if not should_finalize and not finalize_done and st.session_state.get("analysis_done_detected"):
            if retry_attempts < max_auto_finalize_retries:
                st.session_state.auto_finalize_retry_attempts = retry_attempts + 1
                wait_time = min(2.0, 0.5 * st.session_state.auto_finalize_retry_attempts)
                st.info(
                    f"⏳ Waiting for staged documents to finish writing ({st.session_state.auto_finalize_retry_attempts}/{max_auto_finalize_retries})…"
                )
                time.sleep(wait_time)
                st.rerun()
                return
            logger.warning("Auto-finalize gave up waiting for staging file after retries")
            st.warning("⚠️ **Auto-finalize skipped:** Staged documents were not detected after multiple checks.")

        if finalize_done:
            logger.info("Auto-finalize skipped: finalization already completed")
            st.info("ℹ️ Finalization already completed")

        st.session_state.ingestion_stage = on_complete_stage
        st.session_state.auto_finalize_retry_attempts = 0
        time.sleep(1)
        st.rerun()
