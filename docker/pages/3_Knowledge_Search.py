"""
Docker wrapper for Knowledge Search page.
Delegates to the main pages/3_Knowledge_Search.py implementation to avoid divergence.
"""

import importlib.util
from pathlib import Path
import streamlit as st


def _load_main_search_module():
    # Resolve project root: docker/pages -> project_root
    project_root = Path(__file__).resolve().parents[2]
    main_page = project_root / "pages" / "3_Knowledge_Search.py"
    if not main_page.exists():
        st.error(f"Main Knowledge Search page not found at {main_page}")
        return None
    spec = importlib.util.spec_from_file_location("knowledge_search_main", str(main_page))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


module = _load_main_search_module()
if module and hasattr(module, "main"):
    module.main()
else:
    st.error("Failed to load Knowledge Search page implementation.")
