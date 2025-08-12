"""
Idea Generator - Novel Concept Synthesis from Knowledge Collections

This module implements the Idea Generator feature that guides users through a structured
ideation process based on the Double Diamond methodology (Discover, Define, Develop, Deliver).
"""

import streamlit as st
import sys
from pathlib import Path

# Set page config first (must be called before any other Streamlit commands)
st.set_page_config(page_title="Idea Generator", layout="wide", page_icon="💡")

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Fix NLTK circular import issue by setting environment variable before imports
import os
# Create NLTK data directory if it doesn't exist
nltk_data_dir = '/tmp/nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)
os.environ['NLTK_DATA'] = nltk_data_dir

from cortex_engine.idea_generator import IdeaGenerator
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.config import EMBED_MODEL, COLLECTION_NAME
from cortex_engine.utils import convert_windows_to_wsl_path, get_logger
from cortex_engine.session_state import initialize_app_session_state

# Vector index components
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings
import os

logger = get_logger(__name__)

def load_vector_index(db_path):
    """Load the vector index from the knowledge base."""
    try:
        if not db_path or not db_path.strip():
            logger.error("Database path is empty or None")
            return None
        
        logger.info(f"Loading vector index from database path: {db_path}")
        wsl_db_path = convert_windows_to_wsl_path(db_path)
        chroma_db_path = os.path.join(wsl_db_path, "knowledge_hub_db")
        
        logger.info(f"Converted to WSL path: {wsl_db_path}")
        logger.info(f"ChromaDB path: {chroma_db_path}")
        
        if not os.path.isdir(chroma_db_path):
            logger.error(f"Knowledge base directory not found at '{chroma_db_path}'")
            return None
        
        logger.info("Configuring models...")
        
        # Check if Ollama is available for idea generation
        from cortex_engine.utils.ollama_utils import check_ollama_service, format_ollama_error_for_user
        
        is_running, error_msg = check_ollama_service()
        if not is_running:
            st.error("🚫 **Idea Generator Unavailable**")
            st.markdown(format_ollama_error_for_user("Idea Generation", error_msg))
            st.stop()
        
        # Configure models
        Settings.llm = Ollama(model="mistral", request_timeout=120.0)
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        
        logger.info("Loading ChromaDB...")
        # Load database
        db_settings = ChromaSettings(anonymized_telemetry=False)
        db = chromadb.PersistentClient(path=chroma_db_path, settings=db_settings)
        chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        
        logger.info("Creating vector store...")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=chroma_db_path)
        
        logger.info("Loading index from storage...")
        index = load_index_from_storage(storage_context)
        
        # Verify the index is properly loaded
        if index is None:
            logger.error("Index loaded as None")
            return None
        
        if not hasattr(index, 'vector_store'):
            logger.error("Index does not have vector_store attribute")
            return None
        
        if index.vector_store is None:
            logger.error("Index vector_store is None")
            return None
        
        logger.info(f"Vector index loaded successfully from '{chroma_db_path}'")
        return index
        
    except Exception as e:
        logger.error(f"Failed to load vector index: {e}", exc_info=True)
        return None

def main():
    """Main Idea Generator interface"""
    
    # Initialize session state first
    initialize_app_session_state()
    
    st.title("💡 Idea Generator")
    st.markdown("### Transform your knowledge into innovative concepts")
    
    st.markdown("""
    The Idea Generator synthesizes novel ideas from your curated knowledge collections using the 
    **Double Diamond methodology**: Discover → Define → Develop → Deliver
    """)
    
    # Load vector index and graph manager
    vector_index = None
    graph_manager = None
    
    if hasattr(st.session_state, 'db_path_input') and st.session_state.db_path_input:
        with st.spinner("🔄 Loading knowledge base and graph..."):
            vector_index = load_vector_index(st.session_state.db_path_input)
            
            # Load graph manager
            try:
                from cortex_engine.graph_manager import EnhancedGraphManager
                wsl_db_path = convert_windows_to_wsl_path(st.session_state.db_path_input)
                graph_file_path = os.path.join(wsl_db_path, "knowledge_cortex.gpickle")
                graph_manager = EnhancedGraphManager(graph_file_path)
                logger.info("Graph manager loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load graph manager: {e}")
                graph_manager = None
        
        if vector_index is None:
            st.error("❌ Failed to load knowledge base. Please ensure the database path is correct and the knowledge base exists.")
            st.info("💡 **Tip**: Go to Knowledge Search page first to configure the database path.")
            st.code(f"Database path: {st.session_state.db_path_input}")
            return
        else:
            status_parts = ["✅ Knowledge base loaded successfully!"]
            if graph_manager:
                status_parts.append("📊 Knowledge graph loaded!")
            else:
                status_parts.append("⚠️ Knowledge graph not available (entity filtering disabled)")
            st.success(" | ".join(status_parts))
    else:
        st.error("❌ Database path not configured. Please configure the database path first.")
        st.info("💡 **Tip**: Go to Knowledge Search page first to configure the database path.")
        return
    
    # Initialize session state
    if "idea_gen_phase" not in st.session_state:
        st.session_state.idea_gen_phase = "setup"
    
    # Configuration Section
    st.markdown("---")
    st.subheader("🔧 Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Collection Selector
        try:
            collection_mgr = WorkingCollectionManager()
            collection_names = collection_mgr.get_collection_names()
            if collection_names:
                selected_collection = st.selectbox(
                    "📚 Select Knowledge Collection", 
                    options=collection_names, 
                    key="selected_collection",
                    help="Choose the knowledge collection to use as the foundation for idea generation"
                )
            else:
                st.warning("⚠️ No collections found. Please create a collection first in Collection Management.")
                selected_collection = None
        except Exception as e:
            st.error(f"❌ Error loading collections: {e}")
            selected_collection = None
    
    with col2:
        # Model Selector
        llm_provider = st.selectbox(
            "🤖 Select LLM Provider", 
            options=["Local (Ollama)", "Cloud (Gemini)", "Cloud (OpenAI)"], 
            key="llm_provider",
            help="Choose the AI model to power the ideation process"
        )
    
    # Clear any existing automatic analysis if collection changes
    if "last_selected_collection" not in st.session_state:
        st.session_state.last_selected_collection = None
    
    if st.session_state.last_selected_collection != selected_collection:
        # Collection changed, clear previous analysis
        if "collection_analysis" in st.session_state:
            del st.session_state.collection_analysis
        if "identified_themes" in st.session_state:
            del st.session_state.identified_themes
        if "theme_identification_complete" in st.session_state:
            del st.session_state.theme_identification_complete
        st.session_state.last_selected_collection = selected_collection
    
    # Collection Analysis and Filtering Section
    if selected_collection and selected_collection != "default":
        st.markdown("---")
        st.subheader("🔍 Collection Analysis & Filtering")
        
        # Only analyze when user requests it, not automatically
        if st.button("🔍 Analyze Collection", type="secondary", use_container_width=True):
            with st.spinner("🔄 Analyzing collection..."):
                idea_gen_temp = IdeaGenerator(vector_index, graph_manager)
                collection_analysis = idea_gen_temp.analyze_collection_for_filters(selected_collection)
                st.session_state.collection_analysis = collection_analysis
                st.rerun()
        
        # Show analysis results if available
        collection_analysis = st.session_state.get("collection_analysis", {})
        
        if collection_analysis and "error" not in collection_analysis:
            # Display collection statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                total_docs = collection_analysis.get("total_documents", 0)
                st.metric("📚 Total Documents", total_docs)
            with col2:
                doc_types = collection_analysis.get("document_types", [])
                st.metric("📑 Document Types", len(doc_types))
            with col3:
                tags = collection_analysis.get("thematic_tags", [])
                st.metric("🏷️ Unique Tags", len(tags))
            
            # Filter Controls
            st.markdown("#### 🎯 Apply Filters (Optional)")
            st.markdown("Narrow your analysis to specific document types, outcomes, or themes:")
            
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                # Document type filter
                doc_types = collection_analysis.get("document_types", [])
                doc_type_options = ["Any"] + doc_types
                selected_doc_type = st.selectbox(
                    "📑 Document Type",
                    options=doc_type_options,
                    key="filter_doc_type",
                    help="Focus on specific document types (e.g., 'Final Report' for completed work)"
                )
                
                # Proposal outcome filter
                outcomes = collection_analysis.get("proposal_outcomes", [])
                outcome_options = ["Any"] + outcomes
                selected_outcome = st.selectbox(
                    "🎯 Proposal Outcome",
                    options=outcome_options,
                    key="filter_outcome",
                    help="Focus on successful projects (Won) or specific outcomes"
                )
                
                # Entity-based filtering (if graph manager is available)
                if graph_manager:
                    try:
                        entity_options = idea_gen_temp.get_entity_filter_options(selected_collection)
                        
                        if entity_options.get('consultants'):
                            consultant_options = ["Any"] + list(entity_options['consultants'])
                            st.selectbox("👤 Consultant/Author", 
                                       options=consultant_options,
                                       key="filter_consultant",
                                       help="Filter by document author")
                    except Exception as e:
                        logger.warning(f"Consultant filtering unavailable: {e}")
            
            with filter_col2:
                # Thematic tags filter
                thematic_tags = collection_analysis.get("thematic_tags", [])
                if thematic_tags:
                    selected_tags = st.multiselect(
                        "🏷️ Thematic Tags",
                        options=thematic_tags,
                        key="filter_tags",
                        help="Select specific themes or technologies to focus on"
                    )
                else:
                    st.info("No thematic tags found in this collection")
                    selected_tags = []
                
                # Entity filtering continued (client organizations)
                if graph_manager:
                    try:
                        entity_options = idea_gen_temp.get_entity_filter_options(selected_collection)
                        
                        if entity_options.get('clients'):
                            client_options = ["Any"] + list(entity_options['clients'])
                            st.selectbox("🏢 Client Organization", 
                                       options=client_options,
                                       key="filter_client",
                                       help="Filter by client organization")
                    except Exception as e:
                        logger.warning(f"Client filtering unavailable: {e}")
                
                # Quick filter buttons
                st.markdown("**Quick Filters:**")
                if st.button("📋 Final Reports Only", use_container_width=True, key="filter_final_reports"):
                    st.session_state.filter_doc_type = "Final Report"
                    st.rerun()
                
                if st.button("🏆 Successful Projects Only", use_container_width=True, key="filter_successful_projects"):
                    st.session_state.filter_outcome = "Won"
                    st.rerun()
            
            # Filter preview
            active_filters = []
            if selected_doc_type != "Any":
                active_filters.append(f"Document Type: {selected_doc_type}")
            if selected_outcome != "Any":
                active_filters.append(f"Outcome: {selected_outcome}")
            if selected_tags:
                active_filters.append(f"Tags: {', '.join(selected_tags)}")
            if st.session_state.get("filter_consultant", "Any") != "Any":
                active_filters.append(f"Consultant: {st.session_state.filter_consultant}")
            if st.session_state.get("filter_client", "Any") != "Any":
                active_filters.append(f"Client: {st.session_state.filter_client}")
            
            if active_filters:
                st.info(f"🎯 **Active Filters:** {' | '.join(active_filters)}")
                
                # Calculate filtered document count
                filters_for_preview = {}
                if selected_doc_type != "Any":
                    filters_for_preview["document_type"] = selected_doc_type
                if selected_outcome != "Any":
                    filters_for_preview["proposal_outcome"] = selected_outcome
                if selected_tags:
                    filters_for_preview["thematic_tags"] = selected_tags
                if st.session_state.get("filter_consultant", "Any") != "Any":
                    filters_for_preview["consultant"] = st.session_state.filter_consultant
                if st.session_state.get("filter_client", "Any") != "Any":
                    filters_for_preview["client"] = st.session_state.filter_client
                
                # Estimate filtered count (simplified)
                filtered_count = collection_analysis.get("total_documents", 0)
                if selected_doc_type != "Any":
                    stats = collection_analysis.get("statistics", {})
                    type_dist = stats.get("document_type_distribution", {})
                    filtered_count = type_dist.get(selected_doc_type, 0)
                
                st.caption(f"📊 Estimated documents after filtering: ~{filtered_count}")
            else:
                st.caption("ℹ️ No filters applied - analyzing entire collection")
            
            # Theme Identification Step (after collection analysis & filtering)
            st.markdown("---")
            st.subheader("🎨 Step 1: Theme Identification")
            st.markdown("Now let's identify the major themes in your collection based on the analysis and any filters applied.")
            
            theme_identification_ready = selected_collection and selected_collection != "default"
            
            if st.button("🔍 Identify Major Themes", type="primary", disabled=not theme_identification_ready, use_container_width=True, key="identify_themes_btn"):
                with st.spinner("🔄 Analyzing collection for major themes..."):
                    try:
                        idea_gen = IdeaGenerator(vector_index, graph_manager)
                        
                        # Debug: Verify which collection is being used
                        st.info(f"🔍 Debug: Selected collection = '{selected_collection}'")
                        
                        # Get basic collection content for theme analysis
                        doc_ids = idea_gen.collection_mgr.get_doc_ids_by_name(selected_collection)
                        st.info(f"📊 Debug: Found {len(doc_ids) if doc_ids else 0} document IDs in collection '{selected_collection}'")
                        
                        if doc_ids:
                            # Get a sample of documents for theme identification
                            collection_content = idea_gen._get_collection_content(selected_collection, None)
                            st.info(f"📄 Debug: Retrieved {len(collection_content)} documents for theme analysis")
                            
                            # Debug: Show sample of collection content and thematic tags
                            if collection_content:
                                st.info(f"🔬 Debug: Analyzing thematic tags from {len(collection_content)} documents...")
                                
                                # Show first 3 documents' tags
                                for i, doc in enumerate(collection_content[:3]):
                                    title = doc.get('title', 'No title')
                                    tags = doc.get('metadata', {}).get('thematic_tags', 'No tags')
                                    st.info(f"📄 Doc {i+1}: '{title}' → Tags: '{tags}'")
                                
                                # Count all unique tags to see what we're working with
                                all_tags = []
                                for doc in collection_content:
                                    tags_str = doc.get('metadata', {}).get('thematic_tags', '')
                                    if tags_str:
                                        doc_tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                                        all_tags.extend(doc_tags)
                                
                                from collections import Counter
                                tag_counts = Counter(all_tags)
                                top_raw_tags = tag_counts.most_common(10)
                                st.info(f"🏷️ Debug: Top 10 raw tags: {top_raw_tags}")
                            
                            # Extract themes using the enhanced theme visualizer
                            # Use a direct import to avoid any attribute issues
                            from cortex_engine.theme_visualizer import ThemeNetworkVisualizer
                            theme_visualizer = ThemeNetworkVisualizer()
                            theme_analysis = theme_visualizer.extract_themes_from_discovery(
                                {"themes": [], "opportunities": []}, collection_content
                            )
                            
                            # Debug: Show theme analysis results
                            total_themes = len(theme_analysis.get("themes", []))
                            st.info(f"🎨 Debug: Extracted {total_themes} themes from collection")
                            
                            st.session_state.identified_themes = theme_analysis
                            st.session_state.theme_identification_complete = True
                            st.rerun()
                        else:
                            st.error("❌ Selected collection is empty. Please choose a different collection.")
                    except Exception as e:
                        st.error(f"❌ Theme identification failed: {e}")
            
            # Display identified themes with selection interface
            if st.session_state.get("theme_identification_complete") and "identified_themes" in st.session_state:
                theme_data = st.session_state.identified_themes
                
                st.success("✅ **Major Themes Identified!**")
                
                if "themes" in theme_data and theme_data["themes"]:
                    st.markdown("### 🎨 Select Themes for Ideation")
                    st.markdown("Review the themes below and **select only the ones relevant** for your ideation session:")
                    
                    # Initialize selected themes in session state if not exists
                    if "selected_themes_for_ideation" not in st.session_state:
                        # By default, select the top meaningful themes (skip obvious generic ones)
                        default_selected = []
                        for theme_info in theme_data["themes"][:15]:  # Check top 15
                            if isinstance(theme_info, dict):
                                theme_name = theme_info.get("theme", "")
                                # Auto-exclude obvious organizational/generic terms
                                exclude_terms = {'cenitex', 'meeting', 'conference', 'collaboration', 'presentation', 'government', 'department'}
                                if theme_name.lower() not in exclude_terms and len(theme_name.split()) >= 2:
                                    default_selected.append(theme_name)
                        
                        st.session_state.selected_themes_for_ideation = default_selected[:8]  # Limit to 8 themes
                    
                    # Theme selection interface
                    theme_col1, theme_col2 = st.columns([3, 1])
                    
                    with theme_col1:
                        st.markdown("**📋 Available Themes (select/deselect as needed):**")
                        
                        # Display themes with checkboxes
                        available_themes = []
                        for theme_info in theme_data["themes"][:20]:  # Show top 20 themes
                            if isinstance(theme_info, dict):
                                theme_name = theme_info.get("theme", "Unknown Theme")
                                frequency = theme_info.get("frequency", 0)
                                available_themes.append((theme_name, frequency))
                        
                        # Create checkboxes for themes in a grid
                        checkbox_cols = st.columns(2)
                        selected_themes = []
                        
                        for i, (theme_name, frequency) in enumerate(available_themes):
                            col_idx = i % 2
                            with checkbox_cols[col_idx]:
                                is_selected = st.checkbox(
                                    f"**{theme_name}** _{frequency}x_", 
                                    value=theme_name in st.session_state.selected_themes_for_ideation,
                                    key=f"theme_select_{i}_{theme_name}",
                                    help=f"Theme appears {frequency} times in collection"
                                )
                                
                                if is_selected:
                                    selected_themes.append(theme_name)
                        
                        # Update session state with selected themes
                        st.session_state.selected_themes_for_ideation = selected_themes
                    
                    with theme_col2:
                        st.markdown("**✅ Selected Themes:**")
                        if st.session_state.selected_themes_for_ideation:
                            for i, theme in enumerate(st.session_state.selected_themes_for_ideation, 1):
                                st.markdown(f"{i}. {theme}")
                            
                            theme_count = len(st.session_state.selected_themes_for_ideation)
                            if theme_count > 10:
                                st.warning(f"⚠️ {theme_count} themes selected. Consider reducing to 5-8 for focused ideation.")
                            elif theme_count < 3:
                                st.info(f"💡 Only {theme_count} theme(s) selected. Consider adding 2-3 more for richer ideation.")
                            else:
                                st.success(f"✅ {theme_count} themes selected - good for focused ideation!")
                        else:
                            st.warning("⚠️ No themes selected. Please select at least 2-3 themes for ideation.")
                        
                        # Quick selection buttons
                        st.markdown("**🚀 Quick Actions:**")
                        
                        if st.button("🔄 Reset Selection", key="reset_theme_selection"):
                            # Reset to smart defaults
                            default_selected = []
                            for theme_info in theme_data["themes"][:15]:
                                if isinstance(theme_info, dict):
                                    theme_name = theme_info.get("theme", "")
                                    exclude_terms = {'cenitex', 'meeting', 'conference', 'collaboration', 'presentation', 'government', 'department'}
                                    if theme_name.lower() not in exclude_terms and len(theme_name.split()) >= 2:
                                        default_selected.append(theme_name)
                            st.session_state.selected_themes_for_ideation = default_selected[:8]
                            st.rerun()
                        
                        if st.button("🎯 Select Top 5", key="select_top5_themes"):
                            # Select top 5 meaningful themes
                            top_themes = []
                            for theme_info in theme_data["themes"][:10]:
                                if isinstance(theme_info, dict):
                                    theme_name = theme_info.get("theme", "")
                                    exclude_terms = {'cenitex', 'meeting', 'conference', 'collaboration', 'presentation'}
                                    if theme_name.lower() not in exclude_terms and len(top_themes) < 5:
                                        top_themes.append(theme_name)
                            st.session_state.selected_themes_for_ideation = top_themes
                            st.rerun()
                        
                        if st.button("❌ Clear All", key="clear_all_themes"):
                            st.session_state.selected_themes_for_ideation = []
                            st.rerun()
                
                # Show advanced theme network visualization
                if st.session_state.selected_themes_for_ideation:
                    st.markdown("### 🌐 Theme Relationship Network")
                    st.markdown("Explore how your selected themes connect and relate to each other:")
                    
                    # Layout selection
                    viz_col1, viz_col2 = st.columns([3, 1])
                    
                    with viz_col2:
                        from cortex_engine.advanced_theme_network import AdvancedThemeNetworkVisualizer
                        advanced_visualizer = AdvancedThemeNetworkVisualizer()
                        layout_options = advanced_visualizer.get_layout_options()
                        
                        selected_layout = st.selectbox(
                            "🎨 Layout Style",
                            options=[opt["value"] for opt in layout_options],
                            format_func=lambda x: next(opt["label"] for opt in layout_options if opt["value"] == x),
                            key="theme_layout_selection",
                            help="Choose how themes are arranged in the network"
                        )
                        
                        # Show layout description
                        layout_desc = next(opt["description"] for opt in layout_options if opt["value"] == selected_layout)
                        st.caption(f"💡 {layout_desc}")
                    
                    with viz_col1:
                        try:
                            # Filter theme data to selected themes only
                            selected_themes = set(st.session_state.selected_themes_for_ideation)
                            
                            # Create filtered theme data with only selected themes
                            filtered_themes = [
                                theme_info for theme_info in theme_data.get("themes", [])
                                if isinstance(theme_info, dict) and theme_info.get("theme", "") in selected_themes
                            ]
                            
                            # Filter cooccurrence data
                            filtered_cooccurrence = {
                                pair: weight for pair, weight in theme_data.get("cooccurrence", {}).items()
                                if pair[0] in selected_themes and pair[1] in selected_themes
                            }
                            
                            # Filter metrics
                            filtered_metrics = {}
                            for metric_type, metric_data in theme_data.get("metrics", {}).items():
                                if isinstance(metric_data, dict):
                                    filtered_metrics[metric_type] = {
                                        theme: value for theme, value in metric_data.items()
                                        if theme in selected_themes
                                    }
                                else:
                                    filtered_metrics[metric_type] = metric_data
                            
                            filtered_theme_data = {
                                "themes": filtered_themes,
                                "cooccurrence": filtered_cooccurrence,
                                "metrics": filtered_metrics
                            }
                            
                            # Create the advanced network visualization
                            network_fig = advanced_visualizer.create_research_rabbit_network(
                                filtered_theme_data,
                                layout_algorithm=selected_layout,
                                max_themes=len(selected_themes)
                            )
                            
                            st.plotly_chart(network_fig, use_container_width=True, key="advanced_theme_network")
                            
                        except Exception as e:
                            logger.error(f"Advanced theme visualization failed: {e}")
                            st.warning(f"⚠️ Advanced visualization unavailable: {e}")
                            
                            # Fallback to basic visualization
                            try:
                                from cortex_engine.theme_visualizer import ThemeNetworkVisualizer
                                basic_visualizer = ThemeNetworkVisualizer()
                                
                                if "visualization_data" in theme_data:
                                    # Filter visualization data to only show selected themes
                                    selected_themes = set(st.session_state.selected_themes_for_ideation)
                                    filtered_viz_data = {
                                        "nodes": [node for node in theme_data["visualization_data"]["nodes"] 
                                                 if node.get("label") in selected_themes],
                                        "edges": [edge for edge in theme_data["visualization_data"]["edges"]
                                                 if edge.get("source") in selected_themes and edge.get("target") in selected_themes]
                                    }
                                    
                                    filtered_theme_data = theme_data.copy()
                                    filtered_theme_data["visualization_data"] = filtered_viz_data
                                    
                                    basic_fig = basic_visualizer.create_interactive_network(
                                        filtered_theme_data, 
                                        title=f"Selected Themes Network ({len(selected_themes)} themes)"
                                    )
                                    st.plotly_chart(basic_fig, use_container_width=True, key="basic_theme_fallback")
                                
                            except Exception as fallback_e:
                                st.error(f"❌ Theme visualization failed: {fallback_e}")
                    
                    # Optional: Show theme importance chart alongside network
                    if st.checkbox("📊 Show Theme Importance Ranking", key="show_importance_chart"):
                        try:
                            importance_fig = advanced_visualizer.create_theme_importance_chart(filtered_themes)
                            st.plotly_chart(importance_fig, use_container_width=True, key="theme_importance_chart")
                        except Exception as e:
                            st.warning(f"Importance chart unavailable: {e}")
                
                # Check if ready for ideation
                if st.session_state.selected_themes_for_ideation:
                    st.markdown("---")
                    st.markdown("### 🚀 Ready for Ideation")
                    st.success(f"✅ {len(st.session_state.selected_themes_for_ideation)} themes selected! You can now proceed with the full ideation process below.")
                else:
                    st.markdown("---")
                    st.warning("⚠️ Please select at least 2-3 themes above before proceeding to ideation.")
    
        elif selected_collection and selected_collection != "default":
            st.info("👆 Click 'Identify Major Themes' to start the ideation process")
            return  # Don't show ideation section until themes are identified
    
    else:
        st.info("📌 Please select a knowledge collection (other than 'default') to begin theme identification.")
        return  # Don't show ideation section until collection is selected
    
    # Innovation Parameters Section (shown after theme identification)
    if not st.session_state.get("theme_identification_complete"):
        return
    
    # Input Section
    st.markdown("---")
    st.subheader("💭 Innovation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        seed_ideas = st.text_area(
            "🌱 Seed Ideas", 
            key="seed_ideas", 
            placeholder="Enter any initial ideas, questions, or themes to guide the process...\n\nExample: 'How can we improve customer experience using AI?'",
            height=100
        )
        
        constraints = st.text_area(
            "⚠️ Constraints", 
            key="constraints", 
            placeholder="Define any limitations or boundaries...\n\nExample: 'Budget under $50k, must be implementable within 6 months'",
            height=100
        )
    
    with col2:
        innovation_goals = st.text_area(
            "🎯 Innovation Goals", 
            key="innovation_goals", 
            placeholder="What do you want to achieve?...\n\nExample: 'Develop a new product line, improve efficiency by 30%'",
            height=100
        )
        
        # Research options
        st.markdown("**🌐 Research Options**")
        allow_research = st.checkbox(
            "Allow Internet Research", 
            key="allow_research",
            help="Enable web research to supplement your knowledge collection"
        )
    
    st.markdown("---")
    st.subheader("🎯 Step 2: Generate Problem Statements")
    st.markdown("Transform your selected themes into specific, actionable problem statements that can drive innovation.")
    
    # Validation
    can_start = (selected_collection and selected_collection != "default" and 
                st.session_state.get("selected_themes_for_ideation", []))
    
    if not can_start:
        if not selected_collection or selected_collection == "default":
            st.info("📌 Please select a knowledge collection first.")
        elif not st.session_state.get("selected_themes_for_ideation", []):
            st.info("📌 Please select themes in Step 1 first.")
        return
    
    # Generate Problem Statements Button
    if st.button("🎯 Generate Problem Statements", type="primary", disabled=not can_start, use_container_width=True):
        with st.spinner("🎯 Generating problem statements from selected themes..."):
            try:
                # Build filters from UI state
                filters = {}
                if selected_collection and hasattr(st.session_state, 'filter_doc_type'):
                    if st.session_state.get('filter_doc_type', 'Any') != 'Any':
                        filters['document_type'] = st.session_state.filter_doc_type
                    if st.session_state.get('filter_outcome', 'Any') != 'Any':
                        filters['proposal_outcome'] = st.session_state.filter_outcome
                    if st.session_state.get('filter_tags', []):
                        filters['thematic_tags'] = st.session_state.filter_tags
                    if st.session_state.get('filter_consultant', 'Any') != 'Any':
                        filters['consultant'] = st.session_state.filter_consultant
                    if st.session_state.get('filter_client', 'Any') != 'Any':
                        filters['client'] = st.session_state.filter_client
                
                # Get selected themes for focused discovery
                selected_themes = st.session_state.get("selected_themes_for_ideation", [])
                
                if not selected_themes:
                    st.error("❌ No themes selected for ideation. Please go back and select at least 2-3 themes.")
                    return
                
                idea_gen = IdeaGenerator(vector_index, graph_manager)
                problem_results = idea_gen.generate_problem_statements(
                    collection_name=selected_collection,
                    selected_themes=selected_themes,
                    seed_ideas=seed_ideas,
                    constraints=constraints,
                    goals=innovation_goals,
                    llm_provider=llm_provider,
                    filters=filters if filters else None
                )
                st.session_state.problem_statements = problem_results
                st.session_state.idea_gen_phase = "problems_complete"
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error during discovery phase: {e}")
    
    # Display Problem Statement Results
    if st.session_state.idea_gen_phase == "problems_complete" and "problem_statements" in st.session_state:
        st.markdown("---")
        st.subheader("🎯 Generated Problem Statements")
        
        results = st.session_state.problem_statements
        
        if isinstance(results, dict):
            if results.get("status") == "success":
                # Display analysis metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📚 Documents Analyzed", results.get("document_count", 0))
                with col2:
                    st.metric("🎨 Themes Used", len(results.get("selected_themes", [])))
                with col3:
                    st.metric("🤖 Analysis Model", results.get("analysis_model", "Unknown"))
                
                # Show selected themes
                if results.get("selected_themes"):
                    st.info(f"🎯 **Focus Themes:** {', '.join(results['selected_themes'])}")
                
                # Show applied filters if any
                if results.get("filtered_analysis") and results.get("filters_applied"):
                    st.info(f"🔍 **Applied Filters:** {results['filters_applied']}")
                
                st.markdown("---")
                
                # Display problem statements
                if "problem_statements" in results and results["problem_statements"]:
                    st.markdown("### 🎯 Problem Statements by Theme")
                    
                    problem_statements = results["problem_statements"]
                    for i, theme_problems in enumerate(problem_statements, 1):
                        theme = theme_problems.get("theme", f"Theme {i}")
                        problems = theme_problems.get("problems", [])
                        
                        with st.container(border=True):
                            st.markdown(f"**🎨 {theme}**")
                            
                            for j, problem in enumerate(problems, 1):
                                st.markdown(f"**{j}.** {problem}")
                            
                            # Add selection checkbox for each problem
                            if f"selected_problems_{i}" not in st.session_state:
                                st.session_state[f"selected_problems_{i}"] = []
                            
                            selected_for_theme = st.multiselect(
                                f"Select problems for ideation",
                                options=problems,
                                default=st.session_state[f"selected_problems_{i}"],
                                key=f"problems_select_{i}_{theme}",
                                help="Choose which problems to focus on for idea generation"
                            )
                            st.session_state[f"selected_problems_{i}"] = selected_for_theme
                
                # Summary section
                if "summary" in results:
                    st.markdown("### 📋 Analysis Summary")
                    st.markdown(results["summary"])
                
                # Collect all selected problems for next step
                all_selected_problems = []
                for i, theme_problems in enumerate(results.get("problem_statements", []), 1):
                    selected = st.session_state.get(f"selected_problems_{i}", [])
                    all_selected_problems.extend(selected)
                
                st.session_state.selected_problems_for_ideation = all_selected_problems
                
                # Next step
                st.markdown("---")
                if all_selected_problems:
                    st.success(f"✅ {len(all_selected_problems)} problem statement(s) selected for ideation!")
                else:
                    st.warning("⚠️ Please select at least one problem statement above to proceed to idea generation.")
            
            elif results.get("status") == "error":
                st.error(f"❌ Problem statement generation failed: {results.get('error', 'Unknown error')}")
            
            else:
                st.warning("⚠️ Unexpected response format. Please try again.")
        
        else:
            st.error("❌ Invalid response received. Please try again.")
    
    # Step 3: Idea Generation
    if (st.session_state.get("selected_problems_for_ideation") and 
        st.session_state.idea_gen_phase == "problems_complete"):
        
        st.markdown("---")
        st.subheader("💡 Step 3: Generate Ideas")
        st.markdown("Transform your selected problem statements into innovative, actionable solutions.")
        
        selected_problems = st.session_state.selected_problems_for_ideation
        
        # Show selected problems
        with st.expander("📋 Selected Problem Statements", expanded=False):
            for i, problem in enumerate(selected_problems, 1):
                st.markdown(f"**{i}.** {problem}")
        
        # Idea generation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            num_ideas_per_problem = st.slider(
                "Ideas per problem statement",
                min_value=2,
                max_value=10,
                value=5,
                help="How many ideas to generate for each problem statement"
            )
            
            creativity_level = st.selectbox(
                "Creativity Level",
                options=["Practical", "Balanced", "Highly Creative"],
                index=1,
                help="Balance between practical feasibility and creative innovation"
            )
        
        with col2:
            idea_focus = st.multiselect(
                "Focus Areas (optional)",
                options=[
                    "Technology Solutions",
                    "Process Improvements", 
                    "Service Innovations",
                    "Cost Reduction",
                    "User Experience",
                    "Automation",
                    "Integration",
                    "Risk Mitigation",
                    "Sustainability",
                    "Collaboration"
                ],
                help="Select specific areas to focus the idea generation"
            )
            
            include_implementation = st.checkbox(
                "Include implementation notes",
                value=True,
                help="Add brief implementation guidance for each idea"
            )
        
        # Generate Ideas Button
        if st.button("🚀 Generate Ideas", type="primary", use_container_width=True, key="generate_ideas_btn"):
            with st.spinner(f"🧠 Generating {num_ideas_per_problem} ideas for each of {len(selected_problems)} problem statements..."):
                try:
                    idea_gen = IdeaGenerator(vector_index, graph_manager)
                    
                    # Get the collection and themes for context
                    collection_name = st.session_state.get("selected_collection")
                    selected_themes = st.session_state.get("selected_themes_for_ideation", [])
                    
                    # Generate ideas
                    ideas_result = idea_gen.generate_ideas_from_problems(
                        problem_statements=selected_problems,
                        collection_name=collection_name,
                        themes=selected_themes,
                        num_ideas_per_problem=num_ideas_per_problem,
                        creativity_level=creativity_level,
                        focus_areas=idea_focus,
                        include_implementation=include_implementation,
                        llm_provider=llm_provider
                    )
                    
                    st.session_state.generated_ideas = ideas_result
                    st.session_state.idea_gen_phase = "ideas_complete"
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Idea generation failed: {e}")
    
    # Display Generated Ideas
    if st.session_state.idea_gen_phase == "ideas_complete" and "generated_ideas" in st.session_state:
        st.markdown("---")
        st.subheader("💡 Generated Ideas")
        
        ideas_result = st.session_state.generated_ideas
        
        if isinstance(ideas_result, dict) and ideas_result.get("status") == "success":
            # Display summary metrics
            total_ideas = sum(len(group.get("ideas", [])) for group in ideas_result.get("idea_groups", []))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💡 Total Ideas", total_ideas)
            with col2:
                st.metric("📋 Problem Statements", len(ideas_result.get("idea_groups", [])))
            with col3:
                st.metric("🎨 Themes", len(st.session_state.get("selected_themes_for_ideation", [])))
            
            st.markdown("---")
            
            # Display ideas grouped by problem statement
            for i, idea_group in enumerate(ideas_result.get("idea_groups", []), 1):
                problem = idea_group.get("problem_statement", f"Problem {i}")
                ideas = idea_group.get("ideas", [])
                
                with st.container(border=True):
                    st.markdown(f"### 🎯 Problem {i}")
                    st.markdown(f"**{problem}**")
                    
                    if ideas:
                        st.markdown(f"**💡 Generated Ideas ({len(ideas)}):**")
                        
                        for j, idea in enumerate(ideas, 1):
                            if isinstance(idea, dict):
                                idea_title = idea.get("title", f"Idea {j}")
                                idea_description = idea.get("description", "No description")
                                implementation = idea.get("implementation", "")
                                
                                with st.expander(f"💡 {j}. {idea_title}", expanded=False):
                                    st.markdown(f"**Description:** {idea_description}")
                                    if implementation:
                                        st.markdown(f"**Implementation:** {implementation}")
                            else:
                                st.markdown(f"**{j}.** {idea}")
                    else:
                        st.info("No ideas generated for this problem statement.")
            
            # Summary
            if "summary" in ideas_result:
                st.markdown("### 📋 Generation Summary")
                st.markdown(ideas_result["summary"])
            
            # Next steps
            st.markdown("---")
            st.success("🎉 **Idea generation complete!** You now have innovative solutions for your selected problem statements.")
            st.info("💡 **Next Steps:** Review the ideas, select the most promising ones, and consider developing them into detailed proposals or implementation plans.")
            
        else:
            st.error(f"❌ Idea generation failed: {ideas_result.get('error', 'Unknown error')}")

# Run the main application
if __name__ == "__main__":
    main()
