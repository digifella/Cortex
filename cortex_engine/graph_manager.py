from pathlib import Path
import networkx as nx
import pickle
from typing import Any
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


class EnhancedGraphManager:
    def __init__(self, graph_path: str):
        self.graph_path = Path(graph_path)
        self.graph = nx.Graph()
        self._entity_index = {}  # Cache for entity_type -> set of entity_ids
        try:
            if self.graph_path.exists():
                with open(self.graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
                self._rebuild_entity_index()
        except Exception as e:
            logger.warning(f"Failed to load graph: {e}; starting with empty graph")

    def _rebuild_entity_index(self) -> None:
        """Rebuild entity index from graph nodes."""
        self._entity_index = {}
        for node_id, attrs in self.graph.nodes(data=True):
            entity_type = attrs.get('entity_type', 'unknown')
            if entity_type not in self._entity_index:
                self._entity_index[entity_type] = set()
            self._entity_index[entity_type].add(node_id)

    @property
    def entity_index(self) -> dict:
        """Get index of entities by type."""
        return self._entity_index

    def get_graph_stats(self) -> dict:
        """Get comprehensive graph statistics."""
        stats = {
            'total_entities': self.graph.number_of_nodes(),
            'total_relationships': self.graph.number_of_edges(),
            'entity_types': {},
            'relationship_types': {},
            'connected_components': nx.number_connected_components(self.graph) if self.graph.number_of_nodes() > 0 else 0,
        }

        # Count entities by type
        for node_id, attrs in self.graph.nodes(data=True):
            entity_type = attrs.get('entity_type', 'unknown')
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1

        # Count relationships by type
        for u, v, attrs in self.graph.edges(data=True):
            rel_type = attrs.get('relationship', 'related_to')
            stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1

        return stats

    def add_entity(self, entity_id: str, entity_type: str, **attrs: Any) -> None:
        data = {"entity_type": entity_type}
        data.update(attrs or {})
        self.graph.add_node(entity_id, **data)
        # Update entity index
        if entity_type not in self._entity_index:
            self._entity_index[entity_type] = set()
        self._entity_index[entity_type].add(entity_id)

    def add_relationship(self, source_id: str, target_id: str, relationship: str = "related_to", **attrs: Any) -> None:
        data = {"relationship": relationship}
        data.update(attrs or {})
        self.graph.add_edge(source_id, target_id, **data)

    def save_graph(self) -> None:
        try:
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.graph_path, 'wb') as f:
                pickle.dump(self.graph, f)
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

