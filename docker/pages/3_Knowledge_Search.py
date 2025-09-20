"""
Docker Knowledge Search loader with fallback:
- Try to load main project page (../.. / pages / 3_Knowledge_Search.py)
- If missing in container, load local docker implementation (3_Knowledge_Search_impl.py)
"""

import importlib.util
from pathlib import Path
import streamlit as st


def _load_from_path(path: Path, module_name: str):
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    except Exception as e:
        st.error(f"Failed to load module from {path}: {e}")
    return None


def _load_search_module():
    project_root = Path(__file__).resolve().parents[2]
    primary = project_root / "pages" / "3_Knowledge_Search.py"
    if primary.exists():
        return _load_from_path(primary, "knowledge_search_primary")

    # Fallback to docker local implementation
    fallback = Path(__file__).resolve().parent / "3_Knowledge_Search_impl.py"
    if fallback.exists():
        return _load_from_path(fallback, "knowledge_search_docker")

    st.error(f"Knowledge Search implementation not found. Expected one of:\n- {primary}\n- {fallback}")
    return None


module = _load_search_module()
if module and hasattr(module, "main"):
    module.main()
