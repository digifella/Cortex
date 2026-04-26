# ## File: pages/18_Private_Vault_GraphRAG.py
# Version: v6.0.11
# Date: 2026-04-22
# Purpose: Local-only public/private vault GraphRAG search and maintenance UI.

import subprocess

import streamlit as st

from cortex_engine.private_vault_rag import (
    PRIVATE_VAULT,
    PUBLIC_VAULT,
    build_vault_graph,
    load_vault_note,
    markdown_for_streamlit,
    run_vault_indexer,
    search_vault,
    vault_graph_stats,
    vault_index_stats,
)
from cortex_engine.ui_theme import apply_theme
from cortex_engine.version_config import VERSION_STRING


st.set_page_config(page_title="Vault GraphRAG", layout="wide")
apply_theme()


def _status_cards():
    private_index = vault_index_stats("private")
    public_index = vault_index_stats("public")
    private_graph = vault_graph_stats("private")
    public_graph = vault_graph_stats("public")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Private Chunks", f"{private_index.get('chunks', 0):,}")
    col2.metric("Public Chunks", f"{public_index.get('chunks', 0):,}")
    col3.metric("Private Graph Docs", f"{private_graph.get('documents', 0):,}")
    col4.metric("Public Graph Docs", f"{public_graph.get('documents', 0):,}")

    with st.expander("Index and graph paths", expanded=False):
        st.write(f"Private vault: `{PRIVATE_VAULT}`")
        st.write(f"Public vault: `{PUBLIC_VAULT}`")
        st.write(f"Vector DB: `{private_index.get('path') or public_index.get('path')}`")
        st.write(f"Private graph: `{private_graph.get('graph_path')}`")
        st.write(f"Public graph: `{public_graph.get('graph_path')}`")
        for stats in [private_index, public_index]:
            if stats.get("error"):
                st.warning(f"{stats.get('vault')} index: {stats['error']}")
            elif stats.get("warning"):
                st.caption(f"{stats.get('vault')} index: {stats['warning']}")
        for stats in [private_graph, public_graph]:
            if stats.get("mtime"):
                st.caption(f"{stats.get('vault')} graph last built: {stats['mtime']}")


def _maintenance_panel():
    with st.expander("Maintenance", expanded=False):
        st.caption("Reindex updates the selected Chroma collection. Rebuild graph refreshes Obsidian links, tags, and folder relationships.")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        maintenance_vault = col1.radio("Vault", ["private", "public"], horizontal=True)
        full_index = col2.checkbox("Full reindex", value=False)

        if col3.button("Rebuild graph", use_container_width=True):
            with st.status(f"Building {maintenance_vault} vault graph...", expanded=True) as status:
                stats = build_vault_graph(maintenance_vault)
                st.json(stats)
                status.update(label=f"{maintenance_vault.title()} graph rebuilt", state="complete")
            st.rerun()

        if col4.button("Run indexer", use_container_width=True):
            try:
                with st.status(f"Running {maintenance_vault} vault indexer...", expanded=True) as status:
                    completed = run_vault_indexer(maintenance_vault, full=full_index)
                    if completed.stdout:
                        st.code(completed.stdout[-6000:])
                    if completed.stderr:
                        st.code(completed.stderr[-6000:])
                    if completed.returncode == 0:
                        status.update(label="Private indexer completed", state="complete")
                    else:
                        status.update(label="Private indexer failed", state="error")
            except subprocess.TimeoutExpired:
                st.error("Private indexer timed out.")
            st.rerun()


def _search_panel():
    st.subheader("Search")
    with st.form("vault_search"):
        query = st.text_input("Vault query", placeholder="latest note ingested, RCSI robotics, Gemini embeddings...")
        col1, col2, col3, col4 = st.columns([1.2, 1, 1, 2])
        target = col1.radio("Vault", ["private", "public", "both"], horizontal=True)
        mode = col2.radio("Mode", ["rapid", "rich"], horizontal=True, index=0)
        timeout = col3.number_input("Timeout seconds", min_value=10, max_value=300, value=60, step=10)
        submitted = col4.form_submit_button("Search vault", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Enter a query first.")
            return

        try:
            with st.status(f"Searching {target} vault...", expanded=False):
                payload = search_vault(query.strip(), target=target, mode=mode, timeout=int(timeout))
        except subprocess.TimeoutExpired:
            st.error("Vault search timed out. Use rapid mode or rebuild the selected index.")
            return
        except Exception as exc:
            st.error(f"Vault search failed: {exc}")
            return

        st.session_state.vault_last_payload = payload
        results = payload.get("results", [])
        if results:
            st.session_state.vault_selected_note = {"vault": results[0].vault, "source_file": results[0].source_file}

    payload = st.session_state.get("vault_last_payload")
    if not payload:
        return

    results = payload.get("results", [])
    if not results:
        st.info("No matching private notes found.")
        return

    st.markdown("#### Answer")
    st.write(payload.get("answer") or "No answer returned.")

    left, right = st.columns([0.38, 0.62], gap="large")
    with left:
        st.markdown(f"#### Retrieved Notes ({len(results)})")
        for idx, result in enumerate(results, start=1):
            selected_note = st.session_state.get("vault_selected_note") or {}
            selected = selected_note.get("vault") == result.vault and selected_note.get("source_file") == result.source_file
            label = f"{idx}. [{result.vault}] {result.title}"
            if st.button(label, key=f"vault_note_select_{idx}_{result.vault}_{result.source_file}", use_container_width=True, type="primary" if selected else "secondary"):
                st.session_state.vault_selected_note = {"vault": result.vault, "source_file": result.source_file}
                st.rerun()
            st.caption(f"{result.score:.3f} | {result.vault} | {result.source_file}")
            st.write(result.snippet[:360])
            if result.graph_context:
                st.caption("Linked: " + ", ".join(result.graph_context[:3]))

    with right:
        _render_selected_note()


def _render_selected_note():
    selected_note = st.session_state.get("vault_selected_note") or {}
    source_file = selected_note.get("source_file")
    vault = selected_note.get("vault", "private")
    if not source_file:
        st.info("Select a retrieved note to open it.")
        return

    try:
        note = load_vault_note(source_file, vault)
    except Exception as exc:
        st.error(f"Could not open note: {exc}")
        return

    st.markdown(f"### {note.title}")
    st.caption(f"{vault} | {note.source_file}")

    col1, col2, col3 = st.columns([1, 1, 2])
    col1.link_button("Open in Obsidian", note.obsidian_uri, use_container_width=True)
    with col2.popover("Metadata", use_container_width=True):
        st.write(f"Path: `{note.abs_path}`")
        if note.metadata:
            st.json(note.metadata)
        else:
            st.caption("No frontmatter metadata.")
    with col3.popover("Links", use_container_width=True):
        if note.wikilinks:
            st.write(", ".join(note.wikilinks))
        else:
            st.caption("No Obsidian wikilinks found in this note.")
        if note.tags:
            st.markdown("Tags")
            st.write(", ".join(note.tags))

    if note.related_documents or note.neighbours:
        st.markdown("#### Related Notes And Graph Context")
        doc_neighbours = note.related_documents or [n for n in note.neighbours if n.source_file]
        other_neighbours = [n for n in note.neighbours if not n.source_file]

        if doc_neighbours:
            for idx, neighbour in enumerate(doc_neighbours[:6], start=1):
                if st.button(
                    f"{neighbour.title} ({neighbour.relationship})",
                    key=f"vault_related_{vault}_{idx}_{neighbour.source_file}",
                    use_container_width=True,
                ):
                    st.session_state.vault_selected_note = {"vault": vault, "source_file": neighbour.source_file}
                    st.rerun()
        if other_neighbours:
            st.caption(", ".join(f"{n.node_type}: {n.title}" for n in other_neighbours[:10]))
    else:
        st.caption("No graph neighbours found. Rebuild the graph after notes are linked or tagged.")

    st.markdown("#### Note")
    body = markdown_for_streamlit(note.body) or "_Empty note body._"
    if len(body) > 25000:
        show_full = st.checkbox("Show full note", value=False)
        if not show_full:
            st.info(f"Showing the first 25,000 characters of a {len(body):,}-character note.")
            body = body[:25000] + "\n\n..."
    st.markdown(body)


def _legacy_results_view(results):
    for result in results:
        with st.expander(f"{result.title} | {result.source_file}", expanded=False):
            st.caption(f"Score: {result.score:.3f} | Vault: {result.vault}")
            st.write(result.snippet)
            if result.graph_context:
                st.markdown("Graph context")
                st.write(", ".join(result.graph_context))
            else:
                st.caption("No private graph neighbours found for this note.")


def main():
    st.title("Vault GraphRAG")
    st.caption(f"Local public/private vault search via NemoClaw RAG and Cortex graph helpers. Cortex {VERSION_STRING}.")
    _status_cards()
    _maintenance_panel()
    _search_panel()


if __name__ == "__main__":
    main()
