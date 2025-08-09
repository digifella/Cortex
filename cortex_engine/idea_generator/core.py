# ## File: cortex_engine/idea_generator/core.py  
# Version: 1.0.0
# Date: 2025-08-08
# Purpose: Core IdeaGenerator class - extracted from monolithic module

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..collection_manager import WorkingCollectionManager
from ..graph_manager import EnhancedGraphManager
from ..utils.logging_utils import get_logger
from ..theme_visualizer import ThemeNetworkVisualizer
from .double_diamond import DoubleDiamondProcessor
from .agents import IdeationAgents
from .export import IdeaExporter

logger = get_logger(__name__)

class IdeaGenerator:
    """
    Core engine for generating innovative ideas from knowledge collections.
    
    Implements the Double Diamond methodology:
    - Discover: Analyze collection for themes and opportunities
    - Define: Formulate specific problem statements  
    - Develop: Generate diverse solutions using multi-agent ideation
    - Deliver: Create structured, actionable reports
    """
    
    def __init__(self, vector_index=None, graph_manager=None):
        """Initialize the Idea Generator with required components."""
        self.collection_mgr = WorkingCollectionManager()
        self.vector_index = vector_index
        self.graph_manager = graph_manager
        self.theme_visualizer = ThemeNetworkVisualizer()
        
        # Initialize specialized processors
        self.diamond_processor = DoubleDiamondProcessor(vector_index, graph_manager)
        self.ideation_agents = IdeationAgents()
        self.exporter = IdeaExporter()
        
    def run_discovery(self, collection_name: str, seed_ideas: str = "", 
                     constraints: str = "", goals: str = "", 
                     research: bool = False, llm_provider: str = "Local (Ollama)",
                     filters: Optional[Dict[str, Any]] = None,
                     selected_themes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute the Discovery phase of the Double Diamond process.
        
        Args:
            collection_name: Name of the working collection to analyze
            seed_ideas: Initial ideas or themes to guide analysis
            constraints: Limitations or boundaries to consider
            goals: Innovation objectives and desired outcomes
            research: Whether to supplement with web research
            llm_provider: AI model to use for analysis
            filters: Optional filtering criteria
            selected_themes: Pre-selected themes to focus on
            
        Returns:
            Dict containing discovery results with themes, opportunities, and analysis
        """
        return self.diamond_processor.run_discovery(
            collection_name, seed_ideas, constraints, goals, 
            research, llm_provider, filters, selected_themes
        )
    
    def run_define(self, discovery_results: Dict[str, Any], 
                   focus_themes: List[str], problem_scope: str = "",
                   llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """Execute the Define phase of the Double Diamond process."""
        return self.diamond_processor.run_define(
            discovery_results, focus_themes, problem_scope, llm_provider
        )
    
    def run_develop(self, define_results: Dict[str, Any],
                    innovation_approach: str = "incremental",
                    ideation_modes: List[str] = None,
                    llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """Execute the Develop phase of the Double Diamond process."""
        return self.diamond_processor.run_develop(
            define_results, innovation_approach, ideation_modes, llm_provider
        )
    
    def run_deliver(self, develop_results: Dict[str, Any],
                    implementation_focus: str = "balanced",
                    llm_provider: str = "Local (Ollama)") -> Dict[str, Any]:
        """Execute the Deliver phase of the Double Diamond process."""
        return self.diamond_processor.run_deliver(
            develop_results, implementation_focus, llm_provider
        )
        
    def export_results(self, phase_results: Dict[str, Any], 
                      output_dir: str, filename_prefix: str = "idea_session") -> Dict[str, str]:
        """Export idea generation results to files."""
        return self.exporter.export_results(phase_results, output_dir, filename_prefix)
    
    def _validate_collection(self, collection_name: str) -> bool:
        """Validate that collection exists and has content."""
        collections = self.collection_mgr.get_collection_names()
        if collection_name not in collections:
            logger.warning(f"Collection '{collection_name}' not found")
            return False
            
        doc_ids = self.collection_mgr.get_doc_ids_by_name(collection_name)
        if not doc_ids:
            logger.warning(f"Collection '{collection_name}' is empty")
            return False
            
        return True
    
    def _get_collection_content(self, collection_name: str, 
                              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve and optionally filter collection document content."""
        try:
            # Get document IDs for the collection
            doc_ids = self.collection_mgr.get_doc_ids_by_name(collection_name)
            if not doc_ids:
                return []
            
            # Retrieve document content from vector store
            if not self.vector_index:
                logger.warning("No vector index available for content retrieval")
                return []
            
            collection_docs = []
            for doc_id in doc_ids:
                try:
                    # Query for document by ID
                    doc_results = self.vector_index.query(
                        query_str=f"doc_id:{doc_id}",
                        similarity_top_k=1,
                        filters={"doc_id": doc_id} if hasattr(self.vector_index, 'query') else None
                    )
                    
                    if doc_results and hasattr(doc_results, 'source_nodes') and doc_results.source_nodes:
                        node = doc_results.source_nodes[0]
                        doc_content = {
                            "doc_id": doc_id,
                            "title": node.metadata.get("file_name", "Unknown"),
                            "content": node.text[:2000],  # Limit content size
                            "document_type": node.metadata.get("document_type", "Unknown"),
                            "proposal_outcome": node.metadata.get("proposal_outcome", "N/A"),
                            "thematic_tags": node.metadata.get("thematic_tags", []),
                            "entities": node.metadata.get("extracted_entities", []),
                            "metadata": node.metadata
                        }
                        
                        # Apply filters if provided
                        if filters and not self._passes_filters(doc_content, filters):
                            continue
                            
                        collection_docs.append(doc_content)
                        
                except Exception as e:
                    logger.warning(f"Failed to retrieve content for doc {doc_id}: {e}")
                    continue
            
            return collection_docs
            
        except Exception as e:
            logger.error(f"Error retrieving collection content: {e}")
            return []
    
    def _passes_filters(self, doc_content: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document passes the applied filters."""
        try:
            # Document type filter
            if "document_type" in filters and filters["document_type"]:
                if doc_content.get("document_type") != filters["document_type"]:
                    return False
            
            # Proposal outcome filter
            if "proposal_outcome" in filters and filters["proposal_outcome"]:
                if doc_content.get("proposal_outcome") != filters["proposal_outcome"]:
                    return False
            
            # Thematic tags filter
            if "thematic_tags" in filters and filters["thematic_tags"]:
                doc_tags = doc_content.get("thematic_tags", [])
                if not any(tag in doc_tags for tag in filters["thematic_tags"]):
                    return False
            
            # Client filter (check entities for organizations)
            if "client_filter" in filters and filters["client_filter"]:
                entities = doc_content.get("entities", [])
                client_found = any(
                    entity.get("entity_type") == "organization" and 
                    filters["client_filter"].lower() in entity.get("name", "").lower()
                    for entity in entities
                )
                if not client_found:
                    return False
            
            # Consultant filter (check entities for people)
            if "consultant_filter" in filters and filters["consultant_filter"]:
                entities = doc_content.get("entities", [])
                consultant_found = any(
                    entity.get("entity_type") == "person" and 
                    filters["consultant_filter"].lower() in entity.get("name", "").lower()
                    for entity in entities
                )
                if not consultant_found:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error applying filters: {e}")
            return True  # Include document if filter evaluation fails