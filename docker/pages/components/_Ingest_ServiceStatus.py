"""Shared service status UI helpers for Knowledge Ingest pages."""

from __future__ import annotations

import time

import streamlit as st


def render_ollama_status_panel(cache_ttl_seconds: int = 60) -> None:
    """Render cached Ollama service status with user-facing guidance."""
    try:
        from cortex_engine.utils.ollama_utils import (
            check_ollama_service,
            get_ollama_instructions,
            get_ollama_status_message,
        )

        ollama_cache_key = "ollama_status_cache"
        ollama_cache_time_key = "ollama_status_cache_time"

        current_time = time.time()
        cached_time = st.session_state.get(ollama_cache_time_key, 0)

        if current_time - cached_time > cache_ttl_seconds:
            is_running, error_msg, resolved_url = check_ollama_service()
            st.session_state[ollama_cache_key] = (is_running, error_msg, resolved_url)
            st.session_state[ollama_cache_time_key] = current_time
        else:
            is_running, error_msg, resolved_url = st.session_state.get(
                ollama_cache_key,
                (False, "Not checked", None),
            )

        if not is_running:
            st.warning(f"⚠️ {get_ollama_status_message(is_running, error_msg)}")
            with st.expander("ℹ️ **Important: Limited AI Functionality**", expanded=False):
                st.info(
                    "**Impact:** Documents will be processed with basic metadata only. "
                    "AI-enhanced analysis, summaries, and tagging will be unavailable."
                )
                st.markdown(get_ollama_instructions())
        else:
            st.success("✅ Ollama service is running - Full AI capabilities available")
    except Exception as e:
        st.warning(f"Unable to check Ollama status: {e}")
