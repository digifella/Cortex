# ## File: Cortex_Suite.py
# Date: 2026-08-04
# Purpose: Application shell and grouped navigation for Cortex Suite

from pathlib import Path

import streamlit as st

from cortex_ui.navigation import NAVIGATION_SECTIONS


PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Cortex Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def page(path: str, title: str, icon: str, *, default: bool = False) -> st.Page:
    """Create a consistently configured Cortex navigation entry."""
    if not (PROJECT_ROOT / path).is_file():
        raise FileNotFoundError(f"Cortex page is missing: {path}")
    return st.Page(
        path,
        title=title,
        icon=icon,
        default=default,
    )


navigation = {}
for section_index, (section_title, definitions) in enumerate(NAVIGATION_SECTIONS):
    navigation[section_title] = [
        page(path, title, icon, default=section_index == 0 and page_index == 0)
        for page_index, (path, title, icon) in enumerate(definitions)
    ]

selected_page = st.navigation(navigation)
selected_page.run()
