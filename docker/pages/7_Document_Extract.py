# ## File: pages/7_Document_Extract.py
# Version: v5.8.0
# Date: 2026-01-29
# Purpose: Document extraction tools — Textifier (document to Markdown) and Anonymizer.

import streamlit as st
import sys
from pathlib import Path
import os
import tempfile
import time
import zipfile
import io
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core modules
from cortex_engine.anonymizer import DocumentAnonymizer, AnonymizationMapping
from cortex_engine.utils import get_logger, convert_windows_to_wsl_path
from cortex_engine.config_manager import ConfigManager

# Set up logging
logger = get_logger(__name__)

# Page configuration
st.set_page_config(page_title="Document Extract", layout="wide", page_icon="📝")

# Page metadata
PAGE_VERSION = "v5.8.0"


# ======================================================================
# Shared helpers
# ======================================================================

def _get_knowledge_base_files(extensions: List[str]) -> List[Path]:
    """Return files from knowledge base directories matching given extensions."""
    config_manager = ConfigManager()
    config = config_manager.get_config()

    possible_dirs = []
    if config.get("db_path"):
        base_path = Path(convert_windows_to_wsl_path(config["db_path"]))
        possible_dirs.extend([
            base_path / "documents",
            base_path / "source_documents",
            base_path.parent / "documents",
            base_path.parent / "source_documents",
        ])
    possible_dirs.extend([
        project_root / "documents",
        project_root / "source_documents",
        project_root / "test_documents",
    ])

    files = []
    for dir_path in possible_dirs:
        if dir_path.exists():
            for fp in dir_path.glob("**/*"):
                if fp.is_file() and fp.suffix.lower() in extensions:
                    files.append(fp)
    return files


def _file_input_widget(key_prefix: str, allowed_types: List[str], label: str = "Choose a document:"):
    """Render upload / browse KB widget. Returns selected file path or None."""
    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Browse Knowledge Base"],
        key=f"{key_prefix}_method",
    )

    selected_file = None

    if input_method == "Upload File":
        uploaded = st.file_uploader(label, type=allowed_types, key=f"{key_prefix}_upload",
                                    accept_multiple_files=(st.session_state.get(f"{key_prefix}_batch", False)))
        if uploaded:
            files = uploaded if isinstance(uploaded, list) else [uploaded]
            temp_dir = Path(tempfile.gettempdir()) / f"cortex_{key_prefix}"
            temp_dir.mkdir(exist_ok=True, mode=0o755)
            paths = []
            for uf in files:
                dest = str(temp_dir / f"upload_{int(time.time())}_{uf.name}")
                with open(dest, "wb") as f:
                    f.write(uf.getvalue())
                os.chmod(dest, 0o644)
                paths.append(dest)
            if len(paths) == 1:
                selected_file = paths[0]
                st.success(f"Uploaded: {files[0].name}")
            else:
                selected_file = paths  # list for batch
                st.success(f"Uploaded {len(paths)} files")
    else:
        knowledge_files = _get_knowledge_base_files([f".{t}" for t in allowed_types])
        if knowledge_files:
            names = [f"{f.name} ({f.parent.name})" for f in knowledge_files]
            idx = st.selectbox("Select document:", range(len(names)),
                               format_func=lambda x: names[x], index=None,
                               placeholder="Choose a document...", key=f"{key_prefix}_kb")
            if idx is not None:
                selected_file = str(knowledge_files[idx])
                st.success(f"Selected: {knowledge_files[idx].name}")
        else:
            st.warning("No documents found in knowledge base directories")
            st.info("Try uploading a file instead")

    return selected_file


# ======================================================================
# Textifier tab
# ======================================================================

def _render_textifier_tab():
    """Render the Textifier tool UI."""
    st.markdown("Convert PDF, DOCX, or PPTX documents to rich Markdown with optional AI image descriptions.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Input")

        use_vision = st.toggle("Use Vision Model for images", value=True, key="txt_vision")
        batch_mode = st.toggle("Batch mode (multi-file)", value=False, key="txt_batch_toggle")
        st.session_state["textifier_batch"] = batch_mode

        selected = _file_input_widget("textifier", ["pdf", "docx", "pptx"])

    with col2:
        st.header("Output")

        if selected:
            files_to_process = selected if isinstance(selected, list) else [selected]
            total_files = len(files_to_process)

            if st.button("Convert to Markdown", type="primary", use_container_width=True):
                from cortex_engine.textifier import DocumentTextifier

                results = {}
                progress = st.progress(0.0, "Starting conversion...")
                status_text = st.empty()

                for file_idx, fpath in enumerate(files_to_process):
                    fname = Path(fpath).stem
                    file_base = file_idx / total_files
                    file_span = 1.0 / total_files

                    def _on_progress(frac, msg, _base=file_base, _span=file_span, _name=Path(fpath).name):
                        overall = min(_base + frac * _span, 1.0)
                        label = f"[{_name}] {msg}" if total_files > 1 else msg
                        progress.progress(overall, label)

                    if total_files > 1:
                        status_text.info(f"File {file_idx + 1}/{total_files}: {Path(fpath).name}")

                    textifier = DocumentTextifier(use_vision=use_vision, on_progress=_on_progress)
                    try:
                        md_content = textifier.textify_file(fpath)
                        results[f"{fname}_textified.md"] = md_content
                    except Exception as e:
                        st.error(f"Failed to convert {Path(fpath).name}: {e}")
                        logger.error(f"Textifier error for {fpath}: {e}", exc_info=True)

                progress.progress(1.0, "Done!")
                status_text.empty()

                if results:
                    st.session_state["textifier_results"] = results

        # Display results
        results = st.session_state.get("textifier_results")
        if results:
            st.divider()
            st.subheader("Results")

            if len(results) == 1:
                name, content = next(iter(results.items()))
                st.download_button("Download Markdown", content, file_name=name,
                                   mime="text/markdown", use_container_width=True)
                with st.expander("Preview", expanded=True):
                    st.markdown(content[:5000] + ("\n\n*... truncated ...*" if len(content) > 5000 else ""))
            else:
                # Zip download for batch
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, content in results.items():
                        zf.writestr(name, content)
                buf.seek(0)
                st.download_button("Download All (ZIP)", buf.getvalue(),
                                   file_name="textified_documents.zip",
                                   mime="application/zip", use_container_width=True)
                for name, content in results.items():
                    with st.expander(name):
                        st.markdown(content[:3000] + ("\n\n*... truncated ...*" if len(content) > 3000 else ""))
        elif selected:
            st.info("Click **Convert to Markdown** to process your document")
        else:
            st.info("Select a document from the left panel to get started")


# ======================================================================
# Anonymizer tab (original logic preserved)
# ======================================================================

def _render_anonymizer_tab():
    """Render the Anonymizer tool UI (original Document Anonymizer logic)."""
    st.markdown("Replace identifying information with generic placeholders for privacy protection.")

    if "anonymizer_results" not in st.session_state:
        st.session_state.anonymizer_results = {}
    if "current_anonymization" not in st.session_state:
        st.session_state.current_anonymization = None

    with st.expander("About Document Anonymizer", expanded=False):
        st.markdown("""
        **Protect sensitive information** by replacing identifying details with generic placeholders.

        **Features:**
        - **Smart Entity Detection**: Automatically finds people, companies, and locations
        - **Consistent Replacement**: Same entity always gets the same placeholder
        - **Multiple Formats**: PDF, Word, and text file support

        **Replacement Examples:**
        - People: John Smith -> Person A
        - Companies: Acme Corp -> Company 1
        - Contact Info: emails -> [EMAIL], phones -> [PHONE]
        """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Document Input")
        selected_file = _file_input_widget("anonymizer", ["pdf", "docx", "txt"])

        if selected_file and not isinstance(selected_file, list):
            st.divider()
            st.subheader("Anonymization Settings")
            confidence_threshold = st.slider(
                "Entity Detection Confidence:",
                min_value=0.1, max_value=0.9, value=0.3, step=0.1,
                help="Lower values detect more entities (may include false positives)",
            )
            st.session_state.confidence_threshold = confidence_threshold

    with col2:
        st.header("Anonymization Process")

        if selected_file and not isinstance(selected_file, list):
            file_path = Path(selected_file)
            st.markdown(f"**File:** `{file_path.name}`")

            if st.button("Start Anonymization", type="primary", use_container_width=True):
                progress_bar = st.progress(0, "Initializing anonymization...")
                status_container = st.container()

                try:
                    with status_container:
                        st.info("Reading document content...")
                    progress_bar.progress(25, "Reading document...")

                    anonymizer = DocumentAnonymizer()
                    mapping = AnonymizationMapping()

                    with status_container:
                        st.info("Detecting entities and sensitive information...")
                    progress_bar.progress(50, "Detecting entities...")

                    result_path, result_mapping = anonymizer.anonymize_single_file(
                        input_path=selected_file,
                        output_path=None,
                        mapping=mapping,
                        confidence_threshold=st.session_state.confidence_threshold,
                    )

                    with status_container:
                        st.info("Applying anonymization...")
                    progress_bar.progress(75, "Anonymizing content...")

                    st.session_state.current_anonymization = {
                        "original_file": selected_file,
                        "anonymized_file": result_path,
                        "mapping": result_mapping,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    progress_bar.progress(100, "Anonymization complete!")
                    with status_container:
                        st.success("**Anonymization completed successfully!**")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("People", len([k for k, v in result_mapping.mappings.items() if v.startswith("Person")]))
                        with col_b:
                            st.metric("Companies", len([k for k, v in result_mapping.mappings.items() if v.startswith("Company")]))
                        with col_c:
                            st.metric("Projects", len([k for k, v in result_mapping.mappings.items() if v.startswith("Project")]))

                except Exception as e:
                    progress_bar.progress(0, "Anonymization failed")
                    st.error(f"**Anonymization failed:** {str(e)}")
                    logger.error(f"Anonymization error: {e}", exc_info=True)

        if st.session_state.current_anonymization:
            st.divider()
            st.subheader("Anonymization Results")
            result = st.session_state.current_anonymization

            col_orig, col_anon = st.columns(2)
            with col_orig:
                st.markdown("**Original File:**")
                st.code(Path(result["original_file"]).name)
            with col_anon:
                st.markdown("**Anonymized File:**")
                st.code(Path(result["anonymized_file"]).name)

            st.markdown("### Download Results")
            try:
                with open(result["anonymized_file"], "r", encoding="utf-8") as f:
                    anonymized_content = f.read()

                st.download_button(
                    label="Download Anonymized Document",
                    data=anonymized_content,
                    file_name=Path(result["anonymized_file"]).name,
                    mime="text/plain",
                    use_container_width=True,
                )

                mapping_content = _generate_mapping_report(result["mapping"])
                st.download_button(
                    label="Download Mapping Reference",
                    data=mapping_content,
                    file_name=f"anonymization_mapping_{int(time.time())}.txt",
                    mime="text/plain",
                    help="Reference file showing original -> anonymized mappings (keep secure!)",
                )

                with st.expander("Preview Anonymized Content", expanded=False):
                    preview = anonymized_content[:2000]
                    if len(anonymized_content) > 2000:
                        preview += "\n\n... [Content truncated for preview] ..."
                    st.text_area("Preview:", preview, height=300)

                if result["mapping"].mappings:
                    with st.expander("Entity Mappings", expanded=False):
                        import pandas as pd
                        rows = []
                        for original, anon in result["mapping"].mappings.items():
                            rows.append({"Original": original, "Anonymized": anon,
                                         "Type": _get_entity_type(anon)})
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Could not load results: {str(e)}")

        elif selected_file and not isinstance(selected_file, list):
            st.info("Click **Start Anonymization** to process your document")
        else:
            st.info("Select a document from the left panel to get started")


def _generate_mapping_report(mapping: AnonymizationMapping) -> str:
    """Generate a formatted mapping report."""
    report = [
        "ANONYMIZATION MAPPING REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "WARNING: KEEP THIS FILE SECURE AND SEPARATE FROM ANONYMIZED DOCUMENTS",
        "",
    ]
    if not mapping.mappings:
        report.append("No entity mappings found.")
        return "\n".join(report)

    groups = {"Person": [], "Company": [], "Project": []}
    other = []
    for original, anon in mapping.mappings.items():
        placed = False
        for prefix in groups:
            if anon.startswith(prefix):
                groups[prefix].append((original, anon))
                placed = True
                break
        if not placed:
            other.append((original, anon))

    labels = {"Person": "PEOPLE", "Company": "COMPANIES", "Project": "PROJECTS"}
    for prefix, items in groups.items():
        if items:
            report.append(f"{labels[prefix]}:")
            for orig, anon in sorted(items):
                report.append(f"  {orig} -> {anon}")
            report.append("")
    if other:
        report.append("OTHER:")
        for orig, anon in sorted(other):
            report.append(f"  {orig} -> {anon}")

    return "\n".join(report)


def _get_entity_type(anonymized: str) -> str:
    for prefix, label in [("Person", "Person"), ("Company", "Company"),
                          ("Project", "Project"), ("Location", "Location")]:
        if anonymized.startswith(prefix):
            return label
    return "Other"


# ======================================================================
# Main
# ======================================================================

def main():
    st.title("Document Extract")
    st.caption(f"Version: {PAGE_VERSION} • Document conversion and privacy tools")

    tab_textifier, tab_anonymizer = st.tabs(["Textifier", "Anonymizer"])

    with tab_textifier:
        _render_textifier_tab()

    with tab_anonymizer:
        _render_anonymizer_tab()


if __name__ == "__main__":
    main()

# Consistent version footer
try:
    from cortex_engine.ui_components import render_version_footer
    render_version_footer()
except Exception:
    pass
