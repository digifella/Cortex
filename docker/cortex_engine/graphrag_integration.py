# cortex_engine/graphrag_integration.py
# 25_07_25
# V1.0 Enhanced GraphRAG Integration Bridge
"""
Integration layer for the enhanced GraphRAG capabilities with existing Cortex Suite components.
Provides backward-compatible interface while enabling advanced graph-based features.
"""

from typing import List, Dict, Optional, Any
import os
from pathlib import Path

from cortex_engine.graph_manager import EnhancedGraphManager
from cortex_engine.graph_query import GraphQueryEngine
from cortex_engine.entity_extractor import EntityExtractor
from cortex_engine.utils import get_logger, convert_windows_to_wsl_path
from cortex_engine.config import get_db_path

logger = get_logger(__name__)

class GraphRAGIntegration:
    """
    Integration layer for enhanced GraphRAG capabilities.
    Provides a unified interface for existing Cortex components to access graph features.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize GraphRAG integration with database path."""
        self.db_path = db_path or get_db_path()
        
        # Convert Windows path to WSL if needed
        self.db_path = convert_windows_to_wsl_path(self.db_path)
        
        # Initialize graph components
        self.graph_file_path = os.path.join(self.db_path, "knowledge_cortex.gpickle")
        self.graph_manager = EnhancedGraphManager(self.graph_file_path)
        self.entity_extractor = EntityExtractor()
        
        # Graph query engine will be initialized when vector index is available
        self.graph_query_engine = None
        
        logger.info(f"GraphRAG integration initialized with db_path: {self.db_path}")
    
    def initialize_query_engine(self, vector_index):
        """Initialize the graph query engine with vector index."""
        self.graph_query_engine = GraphQueryEngine(self.graph_manager, vector_index)
        logger.info("GraphRAG query engine initialized")
    
    def enhanced_search(self, query: str, vector_index, use_graph_context: bool = True, 
                       max_hops: int = 2) -> List[Dict]:
        """
        Perform enhanced search using GraphRAG capabilities.
        Backward-compatible wrapper for existing search functionality.
        """
        if not self.graph_query_engine:
            self.initialize_query_engine(vector_index)
        
        try:
            return self.graph_query_engine.hybrid_search(
                query=query,
                use_graph_context=use_graph_context,
                max_hops=max_hops
            )
        except Exception as e:
            logger.warning(f"GraphRAG search failed, falling back to vector search: {e}")
            # Fallback to standard vector search
            return vector_index.as_retriever(similarity_top_k=10).retrieve(query)
    
    def expand_query(self, query: str, max_expansions: int = 5) -> Dict[str, Any]:
        """
        Expand query using graph relationships.
        Returns expansion suggestions and related entities.
        """
        if not self.graph_query_engine:
            logger.warning("Graph query engine not initialized, cannot expand query")
            return {
                'original_query': query,
                'expanded_terms': [],
                'related_entities': {},
                'suggested_queries': [query],
                'expansion_reasoning': ['GraphRAG not available']
            }
        
        return self.graph_query_engine.expand_query_with_graph_context(query, max_expansions)
    
    def get_document_context(self, doc_id: str, max_hops: int = 2) -> Dict:
        """Get comprehensive graph context for a document."""
        if not self.graph_query_engine:
            return {'error': 'Graph query engine not initialized'}
        
        return self.graph_query_engine._get_multi_hop_context(doc_id, max_hops)
    
    def analyze_search_coverage(self, query: str, search_results: List[Dict]) -> Dict:
        """Analyze how well search results cover the query."""
        if not self.graph_query_engine:
            return {'error': 'Graph query engine not initialized'}
        
        return self.graph_query_engine.analyze_query_coverage(query, search_results)
    
    def get_entity_suggestions(self, partial_name: str, entity_type: Optional[str] = None) -> List[str]:
        """Get entity name suggestions based on partial input."""
        suggestions = []
        partial_lower = partial_name.lower()
        
        # Search through graph entities
        if entity_type:
            entity_set = self.graph_manager.entity_index.get(entity_type, set())
        else:
            entity_set = set()
            for entities in self.graph_manager.entity_index.values():
                entity_set.update(entities)
        
        for entity_id in entity_set:
            entity_name = entity_id.split(':', 1)[1] if ':' in entity_id else entity_id
            if partial_lower in entity_name.lower():
                suggestions.append(entity_name)
        
        return sorted(suggestions)[:10]  # Return top 10 suggestions
    
    def get_graph_statistics(self) -> Dict:
        """Get comprehensive graph statistics."""
        stats = self.graph_manager.get_graph_stats()
        
        # Add query engine statistics if available
        if self.graph_query_engine:
            stats['pagerank_cache_size'] = len(self.graph_query_engine._pagerank_cache)
            stats['community_cache_size'] = len(self.graph_query_engine._community_cache)
        
        return stats
    
    def extract_entities_from_document(self, text: str, metadata: Dict) -> Dict:
        """Extract entities and relationships from document text."""
        try:
            entities, relationships = self.entity_extractor.extract_entities_and_relationships(text, metadata)
            
            return {
                'entities': [entity.dict() for entity in entities],
                'relationships': [rel.dict() for rel in relationships],
                'extraction_success': True
            }
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {
                'entities': [],
                'relationships': [],
                'extraction_success': False,
                'error': str(e)
            }
    
    def update_graph_from_extraction(self, extraction_result: Dict, doc_filename: str) -> bool:
        """Update graph with extracted entities and relationships."""
        try:
            if not extraction_result.get('extraction_success', False):
                return False
            
            # Add entities to graph
            for entity_data in extraction_result['entities']:
                entity_id = f"{entity_data['entity_type']}:{entity_data['name']}"
                self.graph_manager.add_entity(
                    entity_id=entity_id,
                    entity_type=entity_data['entity_type'],
                    confidence=entity_data.get('confidence', 1.0),
                    aliases=entity_data.get('aliases', []),
                    extraction_method=entity_data.get('extraction_method', 'unknown')
                )
            
            # Add document as entity
            self.graph_manager.add_entity(
                entity_id=doc_filename,
                entity_type='document',
                confidence=1.0
            )
            
            # Add relationships to graph
            for rel_data in extraction_result['relationships']:
                # Determine if source/target are entities or documents
                source = rel_data['source']
                target = rel_data['target']
                
                # Convert entity names to IDs if needed
                if target == doc_filename:
                    target = doc_filename
                else:
                    # Try to find matching entity
                    entity_id = self.graph_manager.get_entity_by_name(target)
                    if entity_id:
                        target = entity_id
                
                if source != doc_filename:
                    entity_id = self.graph_manager.get_entity_by_name(source)
                    if entity_id:
                        source = entity_id
                
                self.graph_manager.add_relationship(
                    source=source,
                    target=target,
                    relationship_type=rel_data['relationship_type'],
                    strength=rel_data.get('strength_indicators', {}).get('calculated_strength', 1.0),
                    confidence=rel_data.get('confidence', 1.0),
                    context=rel_data.get('context', ''),
                    evidence=rel_data.get('evidence', [])
                )
            
            # Save updated graph
            self.graph_manager.save_graph()
            logger.info(f"Graph updated with entities and relationships from {doc_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update graph from extraction: {e}")
            return False
    
    def get_semantic_search_results(self, query: str, vector_index, 
                                   similarity_threshold: float = 0.6) -> List[Dict]:
        """Perform semantic search with graph-enhanced results."""
        if not self.graph_query_engine:
            self.initialize_query_engine(vector_index)
        
        try:
            return self.graph_query_engine.semantic_similarity_search(query, similarity_threshold)
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            # Fallback to regular enhanced search
            return self.enhanced_search(query, vector_index, use_graph_context=True)
    
    def find_related_documents(self, doc_id: str, max_results: int = 10) -> List[Dict]:
        """Find documents related to the given document through graph connections."""
        if doc_id not in self.graph_manager.graph:
            return []
        
        related_docs = []
        
        # Get multi-hop context
        context = self.get_document_context(doc_id, max_hops=2)
        
        # Extract related documents from context
        for related_doc in context.get('related_documents', []):
            related_docs.append({
                'document': related_doc['document'],
                'connection_type': related_doc['connection_type'],
                'connection_entity': related_doc.get('connection_entity', 'unknown')
            })
        
        return related_docs[:max_results]
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on GraphRAG components."""
        health_status = {
            'graph_manager': 'healthy',
            'entity_extractor': 'healthy',
            'graph_query_engine': 'not_initialized',
            'graph_file_exists': os.path.exists(self.graph_file_path),
            'graph_nodes': self.graph_manager.graph.number_of_nodes(),
            'graph_edges': self.graph_manager.graph.number_of_edges(),
            'issues': []
        }
        
        # Check graph query engine
        if self.graph_query_engine:
            health_status['graph_query_engine'] = 'healthy'
        
        # Check for potential issues
        if health_status['graph_nodes'] == 0:
            health_status['issues'].append('Graph has no nodes - may need to run ingestion')
        
        if not health_status['graph_file_exists']:
            health_status['issues'].append('Graph file does not exist - will be created on first use')
        
        return health_status

# Global instance for easy access across the application
_graphrag_instance = None

def get_graphrag_integration(db_path: Optional[str] = None) -> GraphRAGIntegration:
    """Get or create the global GraphRAG integration instance."""
    global _graphrag_instance
    
    if _graphrag_instance is None:
        _graphrag_instance = GraphRAGIntegration(db_path)
    
    return _graphrag_instance

def reset_graphrag_integration():
    """Reset the global GraphRAG integration instance (useful for testing)."""
    global _graphrag_instance
    _graphrag_instance = None