# cortex_engine/graph_manager.py (enhanced version)
import os
import networkx as nx
import pickle
from typing import List, Dict, Set, Optional
from collections import defaultdict
import logging

class EnhancedGraphManager:
    def __init__(self, graph_file_path: str):
        self.graph_file_path = graph_file_path
        logging.info(f"Initializing EnhancedGraphManager with path: {graph_file_path}")
        
        if os.path.exists(self.graph_file_path):
            try:
                with open(self.graph_file_path, 'rb') as f:
                    self.graph = pickle.load(f)
                logging.info(f"Loaded existing graph with {self.graph.number_of_nodes()} nodes")
            except Exception as e:
                logging.error(f"Failed to load graph: {e}")
                self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()
            logging.info("Created new empty graph")
        
        # Initialize indices for efficient querying
        self._build_indices()
    
    def _build_indices(self):
        """Build indices for efficient querying."""
        self.entity_index = defaultdict(set)  # entity_type -> set of node_ids
        self.relationship_index = defaultdict(set)  # relationship_type -> set of edge tuples
        
        for node, data in self.graph.nodes(data=True):
            if 'entity_type' in data:
                self.entity_index[data['entity_type']].add(node)
        
        for source, target, data in self.graph.edges(data=True):
            if 'relationship_type' in data:
                self.relationship_index[data['relationship_type']].add((source, target))
    
    def add_entity(self, entity_id: str, entity_type: str, **attributes):
        """Add an entity node with automatic indexing."""
        self.graph.add_node(entity_id, entity_type=entity_type, **attributes)
        self.entity_index[entity_type].add(entity_id)
    
    def add_relationship(self, source: str, target: str, relationship_type: str, strength: float = 1.0, **attributes):
        """Add a relationship with automatic indexing and relationship strength."""
        # Calculate relationship strength based on type and frequency
        if 'relationship_strength' not in attributes:
            attributes['relationship_strength'] = self._calculate_relationship_strength(relationship_type, strength)
        
        self.graph.add_edge(source, target, relationship_type=relationship_type, **attributes)
        self.relationship_index[relationship_type].add((source, target))
    
    def query_consultant_projects(self, consultant_name: str) -> List[Dict]:
        """Find all projects and reports a consultant worked on."""
        results = []
        
        # Find by exact match
        consultant_id = f"person:{consultant_name}"
        
        if consultant_id in self.graph:
            # Find all documents authored by this consultant
            for neighbor in self.graph.neighbors(consultant_id):
                edge_data = self.graph.edges[consultant_id, neighbor]
                if edge_data.get('relationship_type') == 'authored':
                    if neighbor in self.graph:
                        node_data = self.graph.nodes[neighbor]
                        results.append({
                            'document': neighbor,
                            'type': node_data.get('document_type', 'Unknown'),
                            'metadata': node_data
                        })
        
        return results
    
    def query_consultant_collaborators(self, consultant_name: str) -> Set[str]:
        """Find all people who collaborated with a consultant."""
        collaborators = set()
        consultant_id = f"person:{consultant_name}"
        
        if consultant_id in self.graph:
            for neighbor in self.graph.neighbors(consultant_id):
                edge_data = self.graph.edges[consultant_id, neighbor]
                if edge_data.get('relationship_type') == 'collaborated_with':
                    collaborators.add(neighbor)
            
            # Also check reverse relationships
            for predecessor in self.graph.predecessors(consultant_id):
                edge_data = self.graph.edges[predecessor, consultant_id]
                if edge_data.get('relationship_type') == 'collaborated_with':
                    collaborators.add(predecessor)
        
        return collaborators
    
    def query_client_projects(self, client_name: str) -> List[Dict]:
        """Find all work done for a specific client."""
        results = []
        client_id = f"organization:{client_name}"
        
        if client_id in self.graph:
            # Find all documents that have this organization as a client
            for neighbor in self.graph.neighbors(client_id):
                edge_data = self.graph.edges[client_id, neighbor]
                if edge_data.get('relationship_type') == 'client_of':
                    if neighbor in self.graph:
                        node_data = self.graph.nodes[neighbor]
                        results.append({
                            'document': neighbor,
                            'metadata': node_data
                        })
        
        return results
    
    def get_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> Optional[str]:
        """Find entity ID by name, optionally filtered by type."""
        if entity_type:
            potential_id = f"{entity_type}:{name}"
            if potential_id in self.graph:
                return potential_id
        else:
            # Search all entity types
            for etype in ['person', 'organization', 'project']:
                potential_id = f"{etype}:{name}"
                if potential_id in self.graph:
                    return potential_id
        return None
    
    def get_popular_document_types_by_client(self) -> Dict[str, Dict[str, int]]:
        """Analyze which document types are most common for different clients."""
        client_doc_types = defaultdict(lambda: defaultdict(int))
        
        for node in self.entity_index['organization']:
            projects = self.query_client_projects(node.split(':', 1)[1])
            for project in projects:
                doc_type = project['metadata'].get('document_type', 'Unknown')
                client_doc_types[node][doc_type] += 1
        
        return dict(client_doc_types)
    
    def _calculate_relationship_strength(self, relationship_type: str, base_strength: float = 1.0) -> float:
        """Calculate relationship strength based on type and context."""
        strength_weights = {
            'authored': 1.5,        # Strong authorship connection
            'client_of': 1.2,       # Important business relationship  
            'collaborated_with': 1.0, # Collaboration strength
            'documented_in': 0.8,   # Project documentation
            'mentioned_in': 0.5     # Weaker mention relationship
        }
        return base_strength * strength_weights.get(relationship_type, 1.0)
    
    def get_relationship_strength_distribution(self) -> Dict[str, Dict[str, float]]:
        """Analyze relationship strength distribution across the graph."""
        strength_stats = defaultdict(lambda: {'total': 0, 'count': 0, 'average': 0})
        
        for source, target, data in self.graph.edges(data=True):
            rel_type = data.get('relationship_type', 'unknown')
            strength = data.get('relationship_strength', 1.0)
            
            strength_stats[rel_type]['total'] += strength
            strength_stats[rel_type]['count'] += 1
        
        # Calculate averages
        for rel_type, stats in strength_stats.items():
            if stats['count'] > 0:
                stats['average'] = stats['total'] / stats['count']
        
        return dict(strength_stats)
    
    def find_strongest_connections(self, entity_id: str, limit: int = 10) -> List[Dict]:
        """Find the strongest connections for an entity."""
        if entity_id not in self.graph:
            return []
        
        connections = []
        
        # Check outgoing edges
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.edges[entity_id, neighbor]
            strength = edge_data.get('relationship_strength', 1.0)
            connections.append({
                'target': neighbor,
                'relationship_type': edge_data.get('relationship_type', 'unknown'),
                'strength': strength,
                'direction': 'outgoing'
            })
        
        # Check incoming edges  
        for predecessor in self.graph.predecessors(entity_id):
            edge_data = self.graph.edges[predecessor, entity_id]
            strength = edge_data.get('relationship_strength', 1.0)
            connections.append({
                'target': predecessor,
                'relationship_type': edge_data.get('relationship_type', 'unknown'),
                'strength': strength,
                'direction': 'incoming'
            })
        
        # Sort by strength and return top connections
        connections.sort(key=lambda x: x['strength'], reverse=True)
        return connections[:limit]
    
    def detect_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        """Detect communities in the graph using modularity-based clustering."""
        try:
            # Convert to undirected for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Use NetworkX's community detection (requires networkx >= 2.8)
            communities = nx.community.greedy_modularity_communities(undirected_graph, resolution=resolution)
            
            # Create mapping of node -> community_id
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            
            return community_map
            
        except Exception as e:
            logging.warning(f"Community detection failed: {e}")
            return {}
    
    def get_entity_influence_score(self, entity_id: str) -> Dict[str, float]:
        """Calculate influence score for an entity based on graph position."""
        if entity_id not in self.graph:
            return {'error': 'Entity not found'}
        
        try:
            # Calculate multiple centrality measures
            degree_centrality = self.graph.degree(entity_id, weight='relationship_strength')
            
            # Normalize by total nodes for comparison
            normalized_degree = degree_centrality / max(1, self.graph.number_of_nodes() - 1)
            
            # Calculate local clustering coefficient
            clustering = nx.clustering(self.graph.to_undirected(), entity_id, weight='relationship_strength')
            
            # Calculate influence as weighted combination
            influence_score = (normalized_degree * 0.6) + (clustering * 0.4)
            
            return {
                'influence_score': influence_score,
                'degree_centrality': normalized_degree,
                'clustering_coefficient': clustering,
                'total_connections': self.graph.degree(entity_id)
            }
            
        except Exception as e:
            logging.error(f"Influence calculation failed for {entity_id}: {e}")
            return {'error': f'Calculation failed: {e}'}
    
    def save_graph(self):
        """Save the graph to a pickle file."""
        try:
            with open(self.graph_file_path, 'wb') as f:
                pickle.dump(self.graph, f)
            logging.info(f"Graph saved successfully to {self.graph_file_path}")
        except Exception as e:
            logging.error(f"Failed to save graph: {e}")
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the graph."""
        # Calculate advanced graph metrics
        relationship_strengths = self.get_relationship_strength_distribution()
        
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'entity_counts': {etype: len(nodes) for etype, nodes in self.entity_index.items()},
            'relationship_counts': {rtype: len(edges) for rtype, edges in self.relationship_index.items()},
            'relationship_strengths': relationship_strengths,
            'graph_density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
            'number_of_components': nx.number_weakly_connected_components(self.graph)
        }
        return stats