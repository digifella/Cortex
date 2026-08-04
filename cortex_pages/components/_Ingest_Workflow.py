"""Shared workflow helpers for Knowledge Ingest pages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Optional

import streamlit as st


def detect_orphaned_session_from_log(ingestion_log_path: Path, tail_lines: int = 1000) -> Optional[dict]:
    """Detect interrupted ingestion progress from recent ingestion logs."""
    if not ingestion_log_path.exists():
        return None

    try:
        with open(ingestion_log_path, "r") as log_file:
            lines = log_file.readlines()
        for line in reversed(lines[-tail_lines:]):
            if "Analyzing:" not in line or "(" not in line or "/" not in line:
                continue
            match = re.search(r"\((\d+)/(\d+)\)", line)
            if not match:
                continue
            completed = int(match.group(1))
            total = int(match.group(2))
            remaining = total - completed
            if remaining <= 0:
                return None
            return {
                "completed": completed,
                "total": total,
                "remaining": remaining,
                "progress_percent": round((completed / total) * 100, 1),
            }
    except Exception:
        return None
    return None


def render_orphaned_session_notice(orphaned_session: Optional[dict]) -> None:
    """Render banner and action controls for interrupted ingestion sessions."""
    if not orphaned_session:
        return

    st.warning("⚠️ **Interrupted Processing Detected**")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            f"""
            **Previous Progress:** {orphaned_session['completed']}/{orphaned_session['total']} files ({orphaned_session['progress_percent']}%)  
            **Estimated Remaining:** {orphaned_session['remaining']} files  
            **Status:** Processing was interrupted - resume available with new batch system
            """
        )
    with col2:
        if st.button("🔄 Enable Resume Mode", type="primary", use_container_width=True, key="enable_resume_mode"):
            st.warning("⚠️ **Manual Resume Required**")
            st.info(
                f"""
                The interrupted session ({orphaned_session['completed']}/{orphaned_session['total']} files) was from an older version.

                **To resume:**
                1. Set up the same directories and filters below
                2. The system will skip the {orphaned_session['completed']} already processed files
                3. Future batches will have full automatic resume!
                """
            )
            st.session_state.orphaned_session = orphaned_session
            st.session_state.resume_mode_enabled = True
    st.markdown("---")


def render_stage(stage: str, stage_handlers: Dict[str, Callable[[], None]], fallback_stage: str = "config") -> None:
    """Render the current ingest stage via handler lookup with safe fallback."""
    handler = stage_handlers.get(stage)
    if handler:
        handler()
        return
    st.warning(f"Unknown ingestion stage '{stage}'. Returning to '{fallback_stage}'.")
    fallback = stage_handlers.get(fallback_stage)
    if fallback:
        fallback()
