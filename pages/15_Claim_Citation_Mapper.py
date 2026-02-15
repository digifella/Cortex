"""
Claim Citation Mapper
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from cortex_engine.claim_mapper import map_claims_to_evidence
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.config_manager import ConfigManager
from cortex_engine.ui_theme import apply_theme
from cortex_engine.utils import resolve_db_root_path
from cortex_engine.version_config import VERSION_STRING


def _resolve_db_root() -> Path:
    cfg = ConfigManager().get_config()
    db_path = cfg.get("ai_database_path")
    db_root = resolve_db_root_path(db_path)
    if not db_root:
        raise RuntimeError("AI database path is not configured.")
    return db_root


st.set_page_config(page_title="Claim Citation Mapper", layout="wide", page_icon="🧷")
apply_theme()

st.title("Claim Citation Mapper")
st.caption(f"Version: {VERSION_STRING}")
st.markdown("Map draft claims to supporting evidence from your knowledge base.")

default_text = st.session_state.get("claim_mapper_input", "")
draft_text = st.text_area(
    "Draft Text",
    value=default_text,
    height=240,
    placeholder="Paste draft paragraphs here...",
)

collection_mgr = WorkingCollectionManager()
collection_names = collection_mgr.get_collection_names()
default_collections = ["default"] if "default" in collection_names else []
selected_collections = st.multiselect(
    "Collections Scope (optional)",
    options=collection_names,
    default=default_collections,
    help="If selected, mapping will only consider documents from these collections.",
)

c1, c2 = st.columns(2)
top_k = c1.slider("Evidence per claim", min_value=1, max_value=8, value=3, step=1)
support_threshold = c2.slider("Support threshold", min_value=0.2, max_value=0.9, value=0.42, step=0.02)

run = st.button("Map Claims", type="primary")

if run:
    if not draft_text.strip():
        st.warning("Please paste draft text first.")
    else:
        try:
            db_root = _resolve_db_root()
            chroma_db = db_root / "knowledge_hub_db"
            result = map_claims_to_evidence(
                db_path=str(chroma_db),
                text=draft_text,
                top_k=top_k,
                selected_collections=selected_collections,
                support_threshold=support_threshold,
            )
            st.session_state["claim_mapper_result"] = result
            st.session_state["claim_mapper_input"] = draft_text
            st.session_state["claim_mapper_db_root"] = str(db_root)
        except Exception as e:
            st.error(f"Mapping failed: {e}")

result = st.session_state.get("claim_mapper_result")
if result:
    summary = result.get("summary", {})
    m1, m2, m3 = st.columns(3)
    m1.metric("Supported", int(summary.get("supported", 0)))
    m2.metric("Weak", int(summary.get("weak", 0)))
    m3.metric("Unsupported", int(summary.get("unsupported", 0)))

    rows = []
    for claim in result.get("claims", []):
        top = claim.get("evidence", [])
        best = top[0] if top else {}
        rows.append(
            {
                "claim_id": claim.get("claim_id"),
                "status": claim.get("status"),
                "best_score": claim.get("best_score"),
                "best_source_file": best.get("source_file", ""),
                "claim_text": claim.get("claim_text", ""),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No claims detected from input text.")

    with st.expander("Detailed Evidence", expanded=False):
        for claim in result.get("claims", []):
            st.markdown(f"**{claim.get('claim_id')} • {claim.get('status')} • score={claim.get('best_score')}**")
            st.write(claim.get("claim_text", ""))
            for ev in claim.get("evidence", []):
                st.caption(f"Source: {ev.get('source_file', '')} | doc_id={ev.get('doc_id', '')} | score={ev.get('score')}")
                st.write(ev.get("snippet", ""))
            st.markdown("---")

    if st.button("Save Mapping JSON"):
        try:
            db_root = Path(st.session_state.get("claim_mapper_db_root", ""))
            out_dir = db_root / "claim_maps"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"claim_map_{time.strftime('%Y%m%d_%H%M%S')}.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            st.success(f"Saved: {out_path}")
        except Exception as e:
            st.error(f"Save failed: {e}")
