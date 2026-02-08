"""Shared maintenance/recovery UI helpers for Knowledge Ingest pages."""

from __future__ import annotations

import time
from typing import Callable, Tuple, List

import streamlit as st


def render_maintenance_link(
    maintenance_page: str,
    *,
    button_key: str,
    caption: str = "Maintenance, reset, and deep recovery actions are available on the Maintenance page.",
    button_label: str = "🔧 Open Maintenance Page",
    button_type: str = "secondary",
    help_text: str | None = None,
) -> None:
    """Render a standard maintenance CTA used across ingest surfaces."""
    if caption:
        st.caption(caption)
    if st.button(
        button_label,
        use_container_width=True,
        type=button_type,
        key=button_key,
        help=help_text,
    ):
        st.switch_page(maintenance_page)


def check_recovery_needed(
    *,
    config_manager_cls,
    recovery_manager_cls,
    ttl_seconds: int = 120,
) -> Tuple[bool, List[str]]:
    """
    Check if recovery warnings should be shown.

    Results are cached in session state to avoid expensive repeated analysis.
    """
    cache_key = "recovery_check_cache"
    cache_time_key = "recovery_check_cache_time"
    current_time = time.time()
    cached_time = st.session_state.get(cache_time_key, 0)

    if current_time - cached_time < ttl_seconds and cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        config = st.session_state.get("cached_config")
        if not config:
            config = config_manager_cls().get_config()
        db_path = config.get("ai_database_path", "")

        if not db_path:
            result = (False, [])
            st.session_state[cache_key] = result
            st.session_state[cache_time_key] = current_time
            return result

        dismiss_key = f"recovery_dismissed_{hash(db_path)}"
        if st.session_state.get(dismiss_key, False):
            result = (False, [])
            st.session_state[cache_key] = result
            st.session_state[cache_time_key] = current_time
            return result

        analysis = recovery_manager_cls(db_path).analyze_ingestion_state()

        issues = []
        orphaned_count = analysis.get("statistics", {}).get("orphaned_count", 0)
        if orphaned_count > 10:
            issues.append(f"Found {orphaned_count} orphaned documents")

        broken_collections = analysis.get("statistics", {}).get("broken_collections", 0)
        if broken_collections > 0:
            issues.append(f"Found {broken_collections} broken collections")

        # Surface only high-priority remediation recommendations.
        for rec in analysis.get("recommendations", []) or []:
            if isinstance(rec, dict) and rec.get("priority") == "high":
                issues.append(rec.get("description", "Recovery action needed"))
                if len(issues) >= 4:
                    break

        chromadb_count = analysis.get("statistics", {}).get("chromadb_docs_count", 0)
        if len(issues) <= 1 and chromadb_count > 0 and orphaned_count < 50:
            result = (False, [])
        else:
            result = (len(issues) > 0, issues)

        st.session_state[cache_key] = result
        st.session_state[cache_time_key] = current_time
        return result
    except Exception:
        result = (False, [])
        st.session_state[cache_key] = result
        st.session_state[cache_time_key] = current_time
        return result


def render_recovery_section(
    *,
    check_recovery_needed_fn: Callable[[], Tuple[bool, List[str]]],
    config_manager_cls,
    maintenance_page: str,
    logger,
) -> None:
    """Render recovery warning/CTA area."""
    try:
        config = config_manager_cls().get_config()
        db_path = config.get("ai_database_path", "")
        if not db_path:
            return

        recovery_needed, issues = check_recovery_needed_fn()
        show_maintenance = st.session_state.get("show_recovery_maintenance", False)

        if recovery_needed and not show_maintenance:
            with st.container():
                st.warning(f"⚠️ **Database maintenance may be needed:** {', '.join(issues[:2])}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("🔧 **Open Recovery Tools**", type="primary", use_container_width=True):
                        st.session_state.show_recovery_maintenance = True
                        st.rerun()
                with col2:
                    if st.button("🚫 Dismiss (Hide Until Next Issue)", use_container_width=True):
                        dismiss_key = f"recovery_dismissed_{hash(db_path)}"
                        st.session_state[dismiss_key] = True
                        st.rerun()
        elif not recovery_needed and not show_maintenance:
            render_maintenance_link(
                maintenance_page,
                button_key="ingest_open_maintenance_recovery_idle",
                caption="No critical recovery issues detected. Use Maintenance for optional repair/reset tools.",
                button_type="secondary",
                help_text="Open maintenance and recovery tools",
            )

        if show_maintenance or recovery_needed:
            st.warning("⚠️ **Database issues detected!** Advanced recovery tools are available on the **Maintenance** page.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔧 Go to Maintenance", use_container_width=True, type="primary"):
                    st.switch_page(maintenance_page)
            with col2:
                if st.button("Dismiss Warning", use_container_width=True):
                    dismiss_key = f"recovery_dismissed_{hash(db_path)}"
                    st.session_state[dismiss_key] = True
                    st.rerun()
    except Exception as e:
        logger.error(f"Recovery section render failed: {e}")
        st.error(f"⚠️ Recovery section error: {e}")
