from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Researcher Assistant", layout="wide", page_icon="🔎")

_PAGE_PATH = Path(__file__).resolve().parent / "7_Document_Extract.py"
_SPEC = importlib.util.spec_from_file_location("cortex_document_extract_page_for_researcher", _PAGE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load shared document/research page module: {_PAGE_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

PAGE_VERSION = _MODULE.PAGE_VERSION


def main() -> None:
    st.title("Researcher Assistant")
    st.caption(
        f"Version: {PAGE_VERSION} • Extract included-study evidence, resolve citations, retrieve papers, and assemble researcher packages"
    )

    tab_included_study, tab_resolver, tab_retriever, tab_study_miner, tab_url_ingest = st.tabs(
        [
            "Included Study Extractor",
            "Research Resolver",
            "Paper Retriever",
            "Study Miner",
            "URL PDF Ingestor",
        ]
    )

    with tab_included_study:
        _MODULE._render_included_study_extractor_tab()

    with tab_resolver:
        _MODULE._render_research_resolver_tab(include_retrieval_review=False)

    with tab_retriever:
        _MODULE._render_research_retrieval_review_stage()

    with tab_study_miner:
        _MODULE._render_study_miner_tab()

    with tab_url_ingest:
        _MODULE._render_url_ingestor_tab()


if __name__ == "__main__":
    main()

try:
    from cortex_engine.ui_components import render_version_footer

    render_version_footer()
except Exception:
    pass
