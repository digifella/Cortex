"""
Docker UI components (minimal but compatible with main pages)
Includes a shared version footer with environment indicator.
"""

import os
from typing import Tuple, Dict, Optional, Any
import streamlit as st
from .version_config import get_version_footer


def render_version_footer(show_divider: bool = True):
    """Render a consistent version footer with environment indicator (Docker)."""
    try:
        env = "🐳 Docker"
        footer = get_version_footer()
        if show_divider:
            st.markdown("---")
        st.caption(f"{footer} • {env}")
    except Exception:
        pass


def llm_provider_selector(task: str, key: str, help_text: str) -> Tuple[str, dict]:
    # Minimal selector stub for Docker build
    display = "Ollama (Local)"
    status = {"ok": True, "provider": "ollama"}
    return display, status

