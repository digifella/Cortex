# cortex_engine/graph_query.py
# 25_07_25
# V2.0 Enhanced GraphRAG with Multi-hop Traversal
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict, deque
import networkx as nx
from cortex_engine.graph_manager import EnhancedGraphManager
from cortex_engine.utils import get_logger

logger = get_logger(__name__)

# Query expansion cache to avoid recomputing expansions for similar queries
_expansion_cache = {}

class GraphQueryEngine:
    def __init__(self, graph_manager: EnhancedGraphManager, vector_index):
        self.graph = graph_manager
        self.vector_index = vector_index
        self._pagerank_cache = {}
        self._community_cache = {}
        self._expansion_cache = {}
    
    def hybrid_search(self, query: str, use_graph_context: bool = True, max_hops: int = 2) -> List[Dict]:
        """Perform enhanced hybrid search with multi-hop graph traversal and ranking."""
        # First, do standard vector search
        vector_results = self.vector_index.as_retriever(similarity_top_k=15).retrieve(query)
        
        if not use_graph_context:
            return vector_results
        
        logger.info(f"Enhancing {len(vector_results)} vector results with GraphRAG (max_hops={max_hops})")
        
        # Extract query entities for graph-guided expansion
        query_entities = self._extract_query_entities(query)
        
        # Get graph-based document scoring
        doc_scores = self._calculate_graph_document_scores()
        
        # Enhance results with comprehensive graph context
        enhanced_results = []
        seen_docs = set()
        
        for result in vector_results:
            doc_id = result.metadata.get('doc_id', result.metadata.get('file_name', ''))
            
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            
            # Multi-hop graph context
            graph_context = self._get_multi_hop_context(doc_id, max_hops)
            
            # Graph-based relevance boost
            graph_score = doc_scores.get(doc_id, 0.0)
            
            # Entity-based query relevance
            entity_boost = self._calculate_entity_relevance(doc_id, query_entities)
            
            # Combine scores (vector similarity + graph centrality + entity relevance)
            combined_score = result.score + (graph_score * 0.3) + (entity_boost * 0.2)
            
            result.metadata.update({
                'graph_context': graph_context,
                'graph_score': graph_score,
                'entity_relevance': entity_boost,
                'combined_score': combined_score
            })
            
            enhanced_results.append(result)
        
        # Add graph-discovered documents that weren't in vector results
        graph_discovered = self._discover_documents_via_graph(query_entities, seen_docs, max_results=5)
        enhanced_results.extend(graph_discovered)
        
        # Re-rank by combined score
        enhanced_results.sort(key=lambda x: x.metadata.get('combined_score', x.score), reverse=True)
        
        logger.info(f"GraphRAG enhanced search: {len(enhanced_results)} total results")
        return enhanced_results[:15]  # Return top 15
    
    def expand_query_with_graph_context(self, original_query: str, max_expansions: int = 5) -> Dict[str, any]:
        """Expand query using graph relationships and entity connections."""
        expansion_result = {
            'original_query': original_query,
            'expanded_terms': [],
            'related_entities': {},
            'suggested_queries': [],
            'expansion_reasoning': []
        }
        
        # Extract entities from original query
        query_entities = self._extract_query_entities(original_query)
        
        # Find related terms through graph traversal
        related_terms = set()
        
        for entity_type, entities in query_entities.items():
            for entity_id in entities:
                if entity_id in self.graph.graph:
                    # Find related entities through graph connections
                    related_entities = self._find_related_entities(entity_id, max_hops=2)
                    
                    for related_entity, distance in related_entities.items():
                        # Extract meaningful terms from entity names
                        entity_name = related_entity.split(':', 1)[1] if ':' in related_entity else related_entity
                        terms = self._extract_meaningful_terms(entity_name)
                        related_terms.update(terms)
                        
                        # Store in structured format
                        if entity_type not in expansion_result['related_entities']:
                            expansion_result['related_entities'][entity_type] = []
                        
                        expansion_result['related_entities'][entity_type].append({
                            'entity': entity_name,
                            'distance': distance,
                            'terms': terms
                        })
        
        # Rank and select best expansion terms
        term_scores = self._score_expansion_terms(related_terms, original_query)
        top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:max_expansions]
        
        expansion_result['expanded_terms'] = [term for term, score in top_terms]
        
        # Generate suggested query variations
        expansion_result['suggested_queries'] = self._generate_query_suggestions(
            original_query, expansion_result['expanded_terms']
        )
        
        # Provide reasoning for expansions
        expansion_result['expansion_reasoning'] = self._generate_expansion_reasoning(
            query_entities, expansion_result['related_entities']
        )
        
        logger.info(f"Query expansion generated {len(expansion_result['expanded_terms'])} terms and {len(expansion_result['suggested_queries'])} suggestions")
        return expansion_result
    
    def _find_related_entities(self, entity_id: str, max_hops: int = 2) -> Dict[str, int]:
        """Find entities related to the given entity within max_hops."""
        related = {}
        visited = {entity_id}
        queue = deque([(entity_id, 0)])
        
        while queue:
            current_entity, depth = queue.popleft()
            
            if depth >= max_hops:
                continue
            
            # Explore neighbors
            for neighbor in self.graph.graph.neighbors(current_entity):
                if neighbor not in visited and not neighbor.endswith('.pdf'):
                    visited.add(neighbor)
                    related[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))
            
            # Explore predecessors
            for predecessor in self.graph.graph.predecessors(current_entity):
                if predecessor not in visited and not predecessor.endswith('.pdf'):
                    visited.add(predecessor)
                    related[predecessor] = depth + 1
                    queue.append((predecessor, depth + 1))
        
        return related
    
    def _extract_meaningful_terms(self, entity_name: str) -> List[str]:
        """Extract meaningful terms from entity names for query expansion."""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Split on various delimiters
        terms = re.split(r'[\s_-]+', entity_name.lower())
        
        # Filter out stop words and very short terms
        meaningful_terms = [
            term for term in terms 
            if len(term) > 2 and term not in stop_words and term.isalpha()
        ]
        
        return meaningful_terms
    
    def _score_expansion_terms(self, terms: Set[str], original_query: str) -> Dict[str, float]:
        """Score potential expansion terms based on relevance and graph connectivity."""
        term_scores = {}
        original_terms = set(original_query.lower().split())
        
        for term in terms:
            score = 0.0
            
            # Avoid terms already in the original query
            if term in original_terms:
                continue
            
            # Score based on term frequency in graph
            term_frequency = self._count_term_in_graph(term)
            score += min(1.0, term_frequency / 10.0)  # Normalize frequency score
            
            # Score based on semantic similarity (simple word overlap)
            similarity_score = self._calculate_term_similarity(term, original_query)
            score += similarity_score
            
            # Boost technical terms and proper nouns
            if term[0].isupper() or len(term) > 6:
                score += 0.2
            
            term_scores[term] = score
        
        return term_scores
    
    def _count_term_in_graph(self, term: str) -> int:
        """Count how often a term appears in graph entity names."""
        count = 0
        term_lower = term.lower()
        
        for node in self.graph.graph.nodes():
            node_name = node.split(':', 1)[1] if ':' in node else node
            if term_lower in node_name.lower():
                count += 1
        
        return count
    
    def _calculate_term_similarity(self, term: str, query: str) -> float:
        """Calculate similarity between a term and the original query."""
        query_words = set(query.lower().split())
        term_lower = term.lower()
        
        # Simple character overlap similarity
        max_similarity = 0.0
        for query_word in query_words:
            if len(query_word) > 2 and len(term_lower) > 2:
                overlap = len(set(query_word).intersection(set(term_lower)))
                total_chars = len(set(query_word).union(set(term_lower)))
                if total_chars > 0:
                    similarity = overlap / total_chars
                    max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _generate_query_suggestions(self, original_query: str, expansion_terms: List[str]) -> List[str]:
        """Generate suggested query variations using expansion terms."""
        suggestions = []
        
        # Add individual expansion terms
        for term in expansion_terms[:3]:
            suggestions.append(f"{original_query} {term}")
        
        # Combine multiple expansion terms
        if len(expansion_terms) >= 2:
            suggestions.append(f"{original_query} {expansion_terms[0]} {expansion_terms[1]}")
        
        # Alternative phrasings
        if expansion_terms:
            suggestions.append(f"{expansion_terms[0]} {original_query}")
            suggestions.append(f"related to {original_query} and {expansion_terms[0]}")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _generate_expansion_reasoning(self, query_entities: Dict[str, Set[str]], 
                                   related_entities: Dict[str, List[Dict]]) -> List[str]:
        """Generate human-readable reasoning for query expansions."""
        reasoning = []
        
        for entity_type, entities in query_entities.items():
            if entities and entity_type in related_entities:
                entity_names = [e.split(':', 1)[1] if ':' in e else e for e in entities]
                reasoning.append(
                    f"Found {len(related_entities[entity_type])} related {entity_type} "
                    f"connected to: {', '.join(entity_names[:2])}"
                )
        
        if not reasoning:
            reasoning.append("Expansion based on general graph connectivity patterns")
        
        return reasoning

    def _extract_query_entities(self, query: str) -> Dict[str, Set[str]]:
        """Extract potential entities from query text for graph expansion."""
        entities = {'people': set(), 'organizations': set(), 'projects': set()}
        
        # Simple entity extraction from query
        query_lower = query.lower()
        
        # Map singular entity types to plural keys
        entity_type_map = {
            'person': 'people',
            'organization': 'organizations', 
            'project': 'projects'
        }
        
        # Check against known entities in graph
        for entity_type in ['person', 'organization', 'project']:
            if entity_type in self.graph.entity_index:
                for entity_id in self.graph.entity_index[entity_type]:
                    entity_name = entity_id.split(':', 1)[1].lower()
                    if entity_name in query_lower or any(word in query_lower for word in entity_name.split()):
                        plural_key = entity_type_map[entity_type]
                        entities[plural_key].add(entity_id)
        
        return entities
    
    def _get_multi_hop_context(self, doc_id: str, max_hops: int) -> Dict:
        """Get comprehensive multi-hop graph context for a document."""
        context = {
            'direct_entities': {'authors': [], 'clients': [], 'projects': []},
            'related_documents': [],
            'collaboration_network': [],
            'graph_distance': {},
            'centrality_scores': {}
        }
        
        if doc_id not in self.graph.graph:
            return context
        
        # BFS traversal for multi-hop relationships
        visited = {doc_id}
        queue = deque([(doc_id, 0)])
        
        while queue and len(visited) < 50:  # Limit to prevent explosion
            current_node, depth = queue.popleft()
            
            if depth >= max_hops:
                continue
            
            # Explore neighbors
            for neighbor in self.graph.graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    
                    # Record relationship
                    edge_data = self.graph.graph.edges[current_node, neighbor]
                    rel_type = edge_data.get('relationship_type', 'unknown')
                    
                    # Categorize by entity type and relationship
                    if depth == 0:  # Direct relationships
                        if neighbor.startswith('person:') and rel_type == 'authored':
                            context['direct_entities']['authors'].append(neighbor)
                        elif neighbor.startswith('organization:') and rel_type == 'client_of':
                            context['direct_entities']['clients'].append(neighbor)
                        elif neighbor.startswith('project:'):
                            context['direct_entities']['projects'].append(neighbor)
                    
                    # Track graph distances
                    context['graph_distance'][neighbor] = depth + 1
            
            # Also check incoming edges
            for predecessor in self.graph.graph.predecessors(current_node):
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append((predecessor, depth + 1))
                    context['graph_distance'][predecessor] = depth + 1
        
        # Find related documents through shared entities
        for author in context['direct_entities']['authors']:
            author_name = author.split(':', 1)[1] if ':' in author else author
            related_docs = self.graph.query_consultant_projects(author_name)
            for doc_info in related_docs:
                if doc_info['document'] != doc_id:
                    context['related_documents'].append({
                        'document': doc_info['document'],
                        'connection_type': 'shared_author',
                        'connection_entity': author
                    })
        
        # Find collaboration network
        for author in context['direct_entities']['authors']:
            author_name = author.split(':', 1)[1] if ':' in author else author
            collaborators = self.graph.query_consultant_collaborators(author_name)
            context['collaboration_network'].extend(list(collaborators))
        
        return context
    
    def _calculate_graph_document_scores(self) -> Dict[str, float]:
        """Calculate graph-based importance scores for documents using PageRank."""
        if not self.graph.graph.nodes():
            return {}
        
        # Use cached PageRank if available
        graph_hash = hash(frozenset(self.graph.graph.edges()))
        if graph_hash in self._pagerank_cache:
            return self._pagerank_cache[graph_hash]
        
        try:
            # Calculate PageRank on the entire graph
            pagerank_scores = nx.pagerank(self.graph.graph, weight='relationship_strength', max_iter=100)
            
            # Filter to document nodes only
            doc_scores = {}
            for node, score in pagerank_scores.items():
                # Documents are nodes that don't start with entity prefixes
                if not any(node.startswith(prefix) for prefix in ['person:', 'organization:', 'project:']):
                    doc_scores[node] = score
            
            self._pagerank_cache[graph_hash] = doc_scores
            logger.info(f"Calculated PageRank scores for {len(doc_scores)} documents")
            return doc_scores
            
        except Exception as e:
            logger.warning(f"PageRank calculation failed: {e}")
            return {}
    
    def _calculate_entity_relevance(self, doc_id: str, query_entities: Dict[str, Set[str]]) -> float:
        """Calculate how relevant a document is based on query entities."""
        if doc_id not in self.graph.graph:
            return 0.0
        
        relevance_score = 0.0
        
        # Check direct entity connections
        for neighbor in self.graph.graph.neighbors(doc_id):
            for entity_type, entities in query_entities.items():
                if neighbor in entities:
                    # Weight by relationship type
                    edge_data = self.graph.graph.edges[doc_id, neighbor]
                    rel_type = edge_data.get('relationship_type', '')
                    
                    if rel_type == 'authored':
                        relevance_score += 1.0
                    elif rel_type == 'client_of':
                        relevance_score += 0.8
                    elif rel_type == 'documented_in':
                        relevance_score += 0.6
                    else:
                        relevance_score += 0.3
        
        return relevance_score
    
    def _discover_documents_via_graph(self, query_entities: Dict[str, Set[str]], 
                                    excluded_docs: Set[str], max_results: int = 5) -> List[Dict]:
        """Discover additional relevant documents through graph traversal."""
        discovered = []
        candidates = set()
        
        # Find documents connected to query entities
        for entity_type, entities in query_entities.items():
            for entity in entities:
                if entity in self.graph.graph:
                    for neighbor in self.graph.graph.neighbors(entity):
                        # Check if it's a document (not another entity)
                        if (not any(neighbor.startswith(prefix) for prefix in ['person:', 'organization:', 'project:']) 
                            and neighbor not in excluded_docs):
                            candidates.add(neighbor)
        
        # Score and rank candidates
        scored_candidates = []
        for doc_id in candidates:
            relevance = self._calculate_entity_relevance(doc_id, query_entities)
            if relevance > 0:
                scored_candidates.append((doc_id, relevance))
        
        # Sort by relevance and return top results
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for doc_id, score in scored_candidates[:max_results]:
            # Create a mock result object for discovered documents
            mock_result = type('MockResult', (), {
                'score': score * 0.5,  # Lower than vector results
                'metadata': {
                    'doc_id': doc_id,
                    'file_name': doc_id,
                    'discovery_method': 'graph_traversal',
                    'combined_score': score * 0.5
                }
            })()
            discovered.append(mock_result)
        
        return discovered
    
    def analyze_query_coverage(self, query: str, search_results: List[Dict]) -> Dict[str, any]:
        """Analyze how well search results cover the query based on graph relationships."""
        coverage_analysis = {
            'query_entities_found': {},
            'coverage_score': 0.0,
            'missing_aspects': [],
            'well_covered_aspects': [],
            'relationship_coverage': {}
        }
        
        # Extract query entities
        query_entities = self._extract_query_entities(query)
        
        # Analyze coverage for each entity type
        total_coverage = 0.0
        entity_types_count = 0
        
        for entity_type, entities in query_entities.items():
            if not entities:
                continue
                
            entity_types_count += 1
            entity_coverage = 0.0
            found_entities = set()
            
            # Check which query entities appear in results
            for result in search_results:
                graph_context = result.metadata.get('graph_context', {})
                direct_entities = graph_context.get('direct_entities', {})
                
                # Check for direct matches
                if entity_type.rstrip('s') + 's' in direct_entities:
                    result_entities = direct_entities[entity_type.rstrip('s') + 's']
                    for result_entity in result_entities:
                        entity_name = result_entity.split(':', 1)[1] if ':' in result_entity else result_entity
                        for query_entity in entities:
                            query_name = query_entity.split(':', 1)[1] if ':' in query_entity else query_entity
                            if query_name.lower() == entity_name.lower():
                                found_entities.add(query_entity)
            
            # Calculate coverage for this entity type
            if entities:
                entity_coverage = len(found_entities) / len(entities)
                total_coverage += entity_coverage
                
                coverage_analysis['query_entities_found'][entity_type] = {
                    'requested': len(entities),
                    'found': len(found_entities),
                    'coverage': entity_coverage,
                    'found_entities': list(found_entities)
                }
                
                if entity_coverage > 0.7:
                    coverage_analysis['well_covered_aspects'].append(entity_type)
                elif entity_coverage < 0.3:
                    coverage_analysis['missing_aspects'].append(entity_type)
        
        # Overall coverage score
        if entity_types_count > 0:
            coverage_analysis['coverage_score'] = total_coverage / entity_types_count
        
        # Analyze relationship coverage
        relationship_types = defaultdict(int)
        for result in search_results:
            graph_context = result.metadata.get('graph_context', {})
            for related_doc in graph_context.get('related_documents', []):
                connection_type = related_doc.get('connection_type', 'unknown')
                relationship_types[connection_type] += 1
        
        coverage_analysis['relationship_coverage'] = dict(relationship_types)
        
        return coverage_analysis
    
    def get_document_centrality_analysis(self, doc_id: str) -> Dict:
        """Get comprehensive centrality analysis for a document."""
        if doc_id not in self.graph.graph:
            return {'error': 'Document not found in graph'}
        
        try:
            # Calculate various centrality measures
            degree_centrality = nx.degree_centrality(self.graph.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph.graph, k=100)  # Sample for large graphs
            closeness_centrality = nx.closeness_centrality(self.graph.graph)
            
            return {
                'degree_centrality': degree_centrality.get(doc_id, 0),
                'betweenness_centrality': betweenness_centrality.get(doc_id, 0), 
                'closeness_centrality': closeness_centrality.get(doc_id, 0),
                'total_connections': self.graph.graph.degree(doc_id),
                'neighbor_types': self._analyze_neighbor_types(doc_id)
            }
        except Exception as e:
            logger.error(f"Centrality analysis failed for {doc_id}: {e}")
            return {'error': f'Analysis failed: {e}'}
    
    def _analyze_neighbor_types(self, doc_id: str) -> Dict[str, int]:
        """Analyze the types of entities connected to a document."""
        neighbor_types = defaultdict(int)
        
        for neighbor in self.graph.graph.neighbors(doc_id):
            if neighbor.startswith('person:'):
                neighbor_types['people'] += 1
            elif neighbor.startswith('organization:'):
                neighbor_types['organizations'] += 1
            elif neighbor.startswith('project:'):
                neighbor_types['projects'] += 1
            else:
                neighbor_types['documents'] += 1
        
        return dict(neighbor_types)
    
    def semantic_similarity_search(self, query: str, similarity_threshold: float = 0.6) -> List[Dict]:
        """Perform similarity search using graph structure and entity relationships."""
        # Start with expanded query
        expansion_result = self.expand_query_with_graph_context(query)
        
        # Create expanded query string
        expanded_query = query + " " + " ".join(expansion_result['expanded_terms'])
        
        # Perform hybrid search with expanded query
        results = self.hybrid_search(expanded_query, use_graph_context=True, max_hops=3)
        
        # Add expansion metadata to results
        for result in results:
            result.metadata['query_expansion'] = {
                'original_query': query,
                'expanded_terms': expansion_result['expanded_terms'],
                'expansion_reasoning': expansion_result['expansion_reasoning']
            }
        
        # Filter by similarity threshold if needed
        filtered_results = [
            result for result in results 
            if result.metadata.get('combined_score', result.score) >= similarity_threshold
        ]
        
        logger.info(f"Semantic similarity search returned {len(filtered_results)} results above threshold {similarity_threshold}")
        return filtered_results