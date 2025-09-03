# Minimal session state manager for Docker pages

import streamlit as st
from .config_manager import ConfigManager


def initialize_app_session_state():
    """Initialize core session state values with config-backed defaults."""
    cfg = ConfigManager().get_config()

    # Paths refreshed from config
    st.session_state.db_path_input = cfg.get("ai_database_path", "")
    st.session_state.knowledge_source_path = cfg.get("knowledge_source_path", "")

    # Collections cached structure placeholder; actual collections loaded by pages via manager
    if 'collections' not in st.session_state:
        st.session_state.collections = {}

    # General defaults
    defaults = {
        "model_provider": "Local",
        "openai_api_key": "",
        "selected_collection": "default",
        "new_collection_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

