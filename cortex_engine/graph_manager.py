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
        try:
            if self.graph_path.exists():
                with open(self.graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load graph: {e}; starting with empty graph")

    def add_entity(self, entity_id: str, entity_type: str, **attrs: Any) -> None:
        data = {"entity_type": entity_type}
        data.update(attrs or {})
        self.graph.add_node(entity_id, **data)

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

