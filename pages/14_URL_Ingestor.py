# ## File: pages/14_URL_Ingestor.py
# Version: v6.0.11
# Date: 2026-04-02
# Purpose: Legacy wrapper page for the shared URL PDF ingestor UI.

import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.url_ingestor_ui import render_url_ingestor_ui

st.set_page_config(page_title="URL PDF Ingestor", layout="wide", page_icon="🌐")

st.info("The URL PDF ingestor now also lives inside Document Extract as a dedicated tab.")
render_url_ingestor_ui(standalone=True)
