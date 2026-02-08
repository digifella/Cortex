"""Shared database path scan/save controls for Maintenance pages."""

from __future__ import annotations

import os

import streamlit as st


def render_database_path_tools(
    *,
    current_input: str,
    docker_mode: bool,
    convert_windows_to_wsl_path_fn,
    config_manager_cls,
    discovered_paths_key: str = "discovered_db_paths",
) -> None:
    """Render quick-scan + save path controls and discovered-path picker."""

    def scan_candidates():
        cands = []
        drives = list("CDEFGHIJKLMNOPQRSTUVWXYZ")
        for d in drives:
            base = f"{d}:/ai_databases"
            wsl = convert_windows_to_wsl_path_fn(base)
            kb = os.path.join(wsl, "knowledge_hub_db")
            if os.path.isdir(wsl) or os.path.isdir(kb):
                cands.append(base)
        if docker_mode:
            for p in ["/app/data/ai_databases", "/data", "/workspace/data/ai_databases"]:
                if os.path.isdir(p) or os.path.isdir(os.path.join(p, "knowledge_hub_db")):
                    cands.append(p)
        return sorted(set(cands))

    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button("🔎 Scan Common Locations", use_container_width=True, key="scan_db_locations"):
            st.session_state[discovered_paths_key] = scan_candidates()
            st.rerun()
    with cols[1]:
        if st.button("💾 Save Path", use_container_width=True, key="save_db_path"):
            try:
                config_manager_cls().update_config({"ai_database_path": current_input})
                if "maintenance_config" in st.session_state and st.session_state.maintenance_config:
                    st.session_state.maintenance_config["db_path"] = current_input
                st.success("✅ Database path saved")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save path: {e}")

    if st.session_state.get(discovered_paths_key):
        choice = st.selectbox(
            "Discovered database locations",
            st.session_state[discovered_paths_key],
            help="Select a discovered location to populate the field above.",
        )
        if st.button("Use Selected Location", key="use_selected_db_loc"):
            try:
                config_manager_cls().update_config({"ai_database_path": choice})
                if "maintenance_config" in st.session_state and st.session_state.maintenance_config:
                    st.session_state.maintenance_config["db_path"] = choice
                st.success(f"✅ Path set to {choice}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to set path: {e}")
