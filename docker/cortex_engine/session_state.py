# ## File: cortex_engine/session_state.py
# Version: 3.2.0 (Circular Import Fix)
# Date: 2025-07-16
# Purpose: Centralized session state management for the Cortex Suite Streamlit app.
#          - CRITICAL FIX (v3.2.0): Resolved a circular import error by moving
#            the `WorkingCollectionManager` import inside the function where it
#            is used (deferred import).

import streamlit as st
from .config import DEFAULT_EXCLUSION_PATTERNS_STR
from .config_manager import ConfigManager
# CRITICAL FIX: The import below has been removed from the top level.
# from .collection_manager import WorkingCollectionManager

def initialize_app_session_state():
    """
    Initializes and refreshes all shared session state variables.
    Paths are always re-loaded from the persistent config file to ensure
    they are up-to-date across pages.
    """
    # CRITICAL FIX: Import is deferred to runtime inside the function.
    from .collection_manager import WorkingCollectionManager

    config_manager = ConfigManager()
    persistent_config = config_manager.get_config()

    # --- Paths (Always refresh from config file) ---
    st.session_state.db_path_input = persistent_config.get("ai_database_path", "")
    st.session_state.knowledge_source_path = persistent_config.get("knowledge_source_path", "")

    # --- Collections (Always refresh from file) ---
    st.session_state.collections = WorkingCollectionManager().collections

    # --- Other state (Initialize only if not present) ---
    defaults = {
        "model_provider": "Local",
        "openai_api_key": "",
        "selected_collection": "default",
        "new_collection_name": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val