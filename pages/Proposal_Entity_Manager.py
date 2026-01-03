"""
Proposal Entity Manager
Version: 1.0.0
Date: 2026-01-03

Purpose: Manage organizational entities for tender responses.
Create entity profiles, select source documents, and extract structured data
on a per-entity basis.

Key Features:
- Create/edit/delete entity profiles
- Browse KB and select source folders/documents
- Extract structured data per entity
- View extraction status and data completeness
- Support multiple entities (longboardfella, Deakin, Escient, etc.)
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import asyncio

# Set page config
st.set_page_config(
    page_title="Entity Manager",
    page_icon="🏢",
    layout="wide"
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortex_engine.entity_manager import EntityManager, EntityProfile, EntityType, ExtractionStatus
from cortex_engine.kb_navigator import KBNavigator
from cortex_engine.tender_data_extractor import TenderDataExtractor
from cortex_engine.adaptive_model_manager import AdaptiveModelManager
from cortex_engine.config_manager import ConfigManager
from cortex_engine.ui_theme import apply_theme, section_header
from cortex_engine.utils import (
    convert_windows_to_wsl_path,
    get_logger
)
import chromadb
from chromadb.config import Settings as ChromaSettings
import networkx as nx

# Apply theme
apply_theme()

# Set up logging
logger = get_logger(__name__)

# Page version
PAGE_VERSION = "v1.0.0"


def initialize_session_state():
    """Initialize session state variables."""
    if 'entity_manager' not in st.session_state:
        st.session_state.entity_manager = None

    if 'kb_navigator' not in st.session_state:
        st.session_state.kb_navigator = None

    if 'selected_entity_id' not in st.session_state:
        st.session_state.selected_entity_id = None

    if 'show_create_form' not in st.session_state:
        st.session_state.show_create_form = False

    if 'selected_folders' not in st.session_state:
        st.session_state.selected_folders = []

    if 'selected_doc_ids' not in st.session_state:
        st.session_state.selected_doc_ids = set()


def get_entity_status_emoji(status: ExtractionStatus) -> str:
    """Get emoji for extraction status."""
    status_emojis = {
        ExtractionStatus.NEVER: "⚪",
        ExtractionStatus.EXTRACTING: "🔄",
        ExtractionStatus.COMPLETE: "✅",
        ExtractionStatus.STALE: "⚠️",
        ExtractionStatus.ERROR: "❌"
    }
    return status_emojis.get(status, "❓")


def render_entity_list(entity_manager: EntityManager):
    """Render list of existing entities."""
    section_header("🏢", "Your Entities", "Manage organizational profiles for tender responses")

    entities = entity_manager.list_entities()

    if not entities:
        st.info("📭 No entities created yet. Click '+ Create New Entity' to get started.")
        return

    # Display entities
    for entity in sorted(entities, key=lambda e: e.created_date, reverse=True):
        status_emoji = get_entity_status_emoji(entity.extraction_status)

        with st.expander(
            f"{status_emoji} {entity.entity_name} ({entity.entity_type.value})",
            expanded=st.session_state.selected_entity_id == entity.entity_id
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Type:** {entity.entity_type.value}")
                if entity.description:
                    st.markdown(f"**Description:** {entity.description}")

                # Status
                status_text = entity.extraction_status.value.replace('_', ' ').title()
                if entity.extraction_status == ExtractionStatus.COMPLETE:
                    age_text = f"{entity.age_days} days ago" if entity.age_days else "recently"
                    st.success(f"✅ Extracted {age_text}")
                elif entity.extraction_status == ExtractionStatus.STALE:
                    st.warning(f"⚠️ Stale (extracted {entity.age_days} days ago)")
                elif entity.extraction_status == ExtractionStatus.ERROR:
                    st.error(f"❌ Error: {entity.extraction_error}")
                else:
                    st.info(f"Status: {status_text}")

            with col2:
                st.metric("Source Documents", entity.source_document_count)
                st.metric("Folders", len(entity.source_folders))

            # Data completeness
            if entity.data_completeness:
                st.markdown("**Data Completeness:**")
                cols = st.columns(4)
                for i, (category, has_data) in enumerate(entity.data_completeness.items()):
                    with cols[i % 4]:
                        icon = "✓" if has_data else "✗"
                        st.caption(f"{icon} {category.title()}")

            # Actions
            st.divider()
            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                if st.button("👁️ View Data", key=f"view_{entity.entity_id}"):
                    st.session_state.selected_entity_id = entity.entity_id

            with col_b:
                if st.button("🔄 Re-Extract", key=f"extract_{entity.entity_id}"):
                    st.session_state.extract_entity_id = entity.entity_id
                    st.rerun()

            with col_c:
                if st.button("✏️ Edit Sources", key=f"edit_{entity.entity_id}"):
                    st.session_state.edit_entity_id = entity.entity_id
                    st.rerun()

            with col_d:
                if st.button("🗑️ Delete", key=f"delete_{entity.entity_id}"):
                    if entity_manager.delete_entity(entity.entity_id):
                        st.success(f"Deleted {entity.entity_name}")
                        st.rerun()


def render_create_entity_form(entity_manager: EntityManager, kb_navigator: KBNavigator):
    """Render form to create new entity."""
    section_header("➕", "Create New Entity", "Set up a new organizational profile")

    with st.form("create_entity_form"):
        # Basic info
        col1, col2 = st.columns(2)

        with col1:
            entity_name = st.text_input(
                "Entity Name",
                placeholder="e.g., longboardfella consulting pty ltd",
                help="Official name of the organization"
            )

        with col2:
            entity_type = st.selectbox(
                "Entity Type",
                options=[t.value for t in EntityType],
                format_func=lambda x: x.replace('_', ' ').title()
            )

        description = st.text_area(
            "Description (Optional)",
            placeholder="e.g., Primary trading entity for consulting services",
            height=80
        )

        st.divider()

        # Source selection
        st.markdown("### 📁 Select Source Documents")
        st.caption("Choose KB folders containing data for this entity")

        # Folder search
        search_query = st.text_input(
            "Search folders",
            placeholder="e.g., longboardfella",
            help="Search for folders by name"
        )

        if search_query:
            # Get folder suggestions
            suggestions = kb_navigator.get_folder_suggestions(search_query, limit=20)

            if suggestions:
                st.markdown("**Matching Folders:**")

                for folder_path in suggestions:
                    folder_node = kb_navigator.get_folder_node(folder_path)
                    doc_count = folder_node.get_total_document_count() if folder_node else 0

                    col_check, col_info = st.columns([1, 4])

                    with col_check:
                        is_selected = folder_path in st.session_state.selected_folders
                        if st.checkbox(
                            f"{doc_count} docs",
                            value=is_selected,
                            key=f"folder_select_{folder_path}"
                        ):
                            if folder_path not in st.session_state.selected_folders:
                                st.session_state.selected_folders.append(folder_path)
                        else:
                            if folder_path in st.session_state.selected_folders:
                                st.session_state.selected_folders.remove(folder_path)

                    with col_info:
                        st.caption(f"📁 `{folder_path}`")
            else:
                st.warning("No folders found matching your search")

        # Show selected folders
        if st.session_state.selected_folders:
            st.success(f"✅ Selected {len(st.session_state.selected_folders)} folder(s)")

            # Calculate total documents
            total_docs = 0
            for folder_path in st.session_state.selected_folders:
                docs = kb_navigator.filter_by_folder(folder_path, include_subfolders=True)
                total_docs += len(docs)

            st.info(f"📊 Total documents in selected folders: {total_docs}")

        # Submit
        submitted = st.form_submit_button("💾 Create Entity", type="primary")

        if submitted:
            if not entity_name:
                st.error("Please enter an entity name")
            elif not st.session_state.selected_folders:
                st.error("Please select at least one source folder")
            else:
                try:
                    # Get document IDs from selected folders
                    all_doc_ids = []
                    for folder_path in st.session_state.selected_folders:
                        docs = kb_navigator.filter_by_folder(folder_path, include_subfolders=True)
                        all_doc_ids.extend([doc.doc_id for doc in docs])

                    # Create entity
                    entity = entity_manager.create_entity(
                        entity_name=entity_name,
                        entity_type=EntityType(entity_type),
                        description=description or None,
                        source_folders=st.session_state.selected_folders,
                        source_document_ids=all_doc_ids
                    )

                    # Update document count
                    entity_manager.update_entity(
                        entity.entity_id,
                        source_document_count=len(all_doc_ids)
                    )

                    st.success(f"✅ Created entity: {entity_name}")
                    st.session_state.selected_folders = []
                    st.session_state.show_create_form = False
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to create entity: {str(e)}")
                    logger.error(f"Entity creation failed: {e}", exc_info=True)


def main():
    """Main page function."""
    st.title("🏢 Proposal Entity Manager")
    st.caption(f"Version: {PAGE_VERSION}")

    st.markdown("""
    Manage organizational entities for tender responses. Each entity has its own
    structured data extracted from selected KB documents.
    """)

    # Initialize session state
    initialize_session_state()

    # Get config
    config_manager = ConfigManager()
    config = config_manager.get_config()
    db_path = config.get('ai_database_path')

    if not db_path:
        st.error("❌ Database path not configured. Please set it in the main settings.")
        return

    # Normalize path
    wsl_db_path = convert_windows_to_wsl_path(db_path)

    # Initialize managers
    try:
        if st.session_state.entity_manager is None:
            st.session_state.entity_manager = EntityManager(Path(wsl_db_path))

        if st.session_state.kb_navigator is None:
            st.session_state.kb_navigator = KBNavigator(
                db_path=wsl_db_path,
                collection_name=config.get('collection_name', 'knowledge_hub')
            )

        entity_manager = st.session_state.entity_manager
        kb_navigator = st.session_state.kb_navigator

    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")
        logger.error(f"Initialization failed: {e}", exc_info=True)
        return

    # KB Statistics
    with st.expander("📊 Knowledge Base Statistics", expanded=False):
        stats = kb_navigator.get_statistics()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Documents", stats['total_documents'])
        with col2:
            st.metric("Total Folders", stats['total_folders'])
        with col3:
            st.metric("Max Folder Depth", stats['max_folder_depth'])

    st.divider()

    # Main content
    if st.session_state.show_create_form:
        render_create_entity_form(entity_manager, kb_navigator)

        if st.button("← Back to Entity List"):
            st.session_state.show_create_form = False
            st.session_state.selected_folders = []
            st.rerun()

    else:
        # Entity list
        render_entity_list(entity_manager)

        # Create button
        st.divider()
        if st.button("➕ Create New Entity", type="primary"):
            st.session_state.show_create_form = True
            st.rerun()

    # Handle extraction trigger
    if hasattr(st.session_state, 'extract_entity_id'):
        entity_id = st.session_state.extract_entity_id
        del st.session_state.extract_entity_id

        entity = entity_manager.get_entity(entity_id)
        if entity:
            st.info(f"🔄 Starting extraction for {entity.entity_name}...")

            try:
                # Initialize extractor
                chroma_db_path = str(Path(wsl_db_path) / "knowledge_hub_db")
                db_settings = ChromaSettings(anonymized_telemetry=False)
                chroma_client = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
                collection = chroma_client.get_collection(config.get('collection_name', 'knowledge_hub'))

                # Load knowledge graph
                graph_path = Path(wsl_db_path) / "knowledge_cortex.gpickle"
                knowledge_graph = nx.read_gpickle(graph_path) if graph_path.exists() else None

                model_manager = AdaptiveModelManager()
                extractor = TenderDataExtractor(
                    vector_index=collection,
                    knowledge_graph=knowledge_graph,
                    model_manager=model_manager,
                    db_path=Path(wsl_db_path)
                )

                # Update status
                entity_manager.update_extraction_status(entity_id, ExtractionStatus.EXTRACTING)

                # Run extraction
                progress_placeholder = st.empty()

                def progress_callback(message):
                    progress_placeholder.info(message)

                # TODO: Update extractor to support entity_id and document filtering
                structured = asyncio.run(extractor.extract_all_structured_data(progress_callback))

                # Save to entity's file
                import json
                with open(entity.structured_data_file, 'w') as f:
                    json.dump(structured.to_json_serializable(), f, indent=2)

                # Update status
                entity_manager.update_extraction_status(entity_id, ExtractionStatus.COMPLETE)
                entity_manager.update_data_completeness(
                    entity_id,
                    {
                        "organization": structured.organization is not None,
                        "insurances": len(structured.insurances) > 0,
                        "qualifications": len(structured.team_qualifications) > 0,
                        "projects": len(structured.projects) > 0
                    }
                )

                st.success(f"✅ Extraction complete for {entity.entity_name}!")
                st.rerun()

            except Exception as e:
                entity_manager.update_extraction_status(
                    entity_id,
                    ExtractionStatus.ERROR,
                    error=str(e)
                )
                st.error(f"❌ Extraction failed: {str(e)}")
                logger.error(f"Extraction failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
