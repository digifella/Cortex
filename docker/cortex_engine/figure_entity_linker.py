"""
Figure Entity Linker for Cortex Suite
Phase 2 Enhancement: Links figures to knowledge graph entities for semantic search.

Version: 1.0.0
Date: 2026-01-13
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

import spacy
from llama_index.core import Document

from .utils.logging_utils import get_logger

logger = get_logger(__name__)

# Load spaCy model for NER (lazy loading)
_nlp = None


def _get_nlp():
    """Lazy load spaCy NLP model."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model for entity extraction")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            _nlp = None
    return _nlp


class FigureEntityLinker:
    """
    Links figures to knowledge graph entities using VLM descriptions.

    Features:
    - Extracts entities (people, organizations, projects) from VLM descriptions
    - Matches entities against knowledge graph using fuzzy matching
    - Adds `figure_entities` metadata for enhanced retrieval
    - Enables queries like "show figures related to [person/org]"
    """

    def __init__(self, knowledge_graph: Any = None):
        """
        Initialize figure entity linker.

        Args:
            knowledge_graph: NetworkX knowledge graph (optional, loaded on demand)
        """
        self.knowledge_graph = knowledge_graph
        self.nlp = _get_nlp()

    def process_document(self, document: Document) -> Document:
        """
        Process a document to link figures to knowledge graph entities.

        Args:
            document: LlamaIndex Document with potential figure metadata

        Returns:
            Document with enhanced `figure_entities` metadata
        """
        if not self._has_figures(document):
            return document

        logger.debug(f"Processing figures for entity linking: {document.metadata.get('file_name', 'document')}")

        # Extract figures from metadata
        figures = self._extract_figures(document)

        if not figures:
            return document

        # Extract and link entities for each figure
        figure_entity_links = []

        for figure in figures:
            vlm_description = figure.get('vlm_description', '')

            if not vlm_description:
                continue

            # Extract entities from VLM description
            entities = self._extract_entities_from_text(vlm_description)

            if entities:
                figure_link = {
                    'figure_index': figure.get('index', 0),
                    'figure_page': figure.get('page'),
                    'entities': entities,
                    'entity_types': list(set(e['type'] for e in entities))
                }
                figure_entity_links.append(figure_link)

                logger.debug(
                    f"Figure {figure.get('index')} linked to {len(entities)} entities: "
                    f"{[e['text'] for e in entities[:3]]}"
                )

        # Add entity links to metadata
        if figure_entity_links:
            document.metadata['figure_entities'] = figure_entity_links
            document.metadata['has_figure_entities'] = True

            logger.info(
                f"Linked {len(figure_entity_links)} figures to knowledge graph entities "
                f"in {document.metadata.get('file_name', 'document')}"
            )

        return document

    def _has_figures(self, document: Document) -> bool:
        """Check if document contains figures with VLM descriptions."""
        metadata = document.metadata

        if 'docling_figures' not in metadata:
            return False

        try:
            figures = metadata['docling_figures']
            if isinstance(figures, str):
                figures = json.loads(figures)

            # Check if any figures have VLM descriptions
            return any('vlm_description' in fig for fig in figures)

        except (json.JSONDecodeError, TypeError, KeyError):
            return False

    def _extract_figures(self, document: Document) -> List[Dict[str, Any]]:
        """Extract figure data from document metadata."""
        try:
            figures = document.metadata.get('docling_figures', [])
            if isinstance(figures, str):
                figures = json.loads(figures)
            return figures
        except (json.JSONDecodeError, TypeError):
            return []

    def _extract_entities_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from text using spaCy.

        Args:
            text: VLM description or other text

        Returns:
            List of entity dictionaries with 'text', 'type', 'start', 'end'
        """
        if not self.nlp:
            logger.warning("spaCy NLP not available, skipping entity extraction")
            return []

        entities = []

        try:
            doc = self.nlp(text)

            for ent in doc.ents:
                # Filter relevant entity types
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
                    entity = {
                        'text': ent.text,
                        'type': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    }

                    # Optional: Match against knowledge graph if available
                    if self.knowledge_graph:
                        matched_node = self._match_entity_to_graph(ent.text, ent.label_)
                        if matched_node:
                            entity['graph_node_id'] = matched_node

                    entities.append(entity)

        except Exception as e:
            logger.warning(f"Error extracting entities from text: {e}")

        return entities

    def _match_entity_to_graph(
        self,
        entity_text: str,
        entity_type: str
    ) -> Optional[str]:
        """
        Match an entity to a knowledge graph node using fuzzy matching.

        Args:
            entity_text: Entity text (e.g., "John Smith")
            entity_type: spaCy entity type (e.g., "PERSON")

        Returns:
            Node ID if match found, None otherwise
        """
        if not self.knowledge_graph:
            return None

        try:
            # Map spaCy types to graph node types
            type_mapping = {
                'PERSON': 'person',
                'ORG': 'organization',
                'GPE': 'organization',  # Treat geopolitical entities as orgs
                'PRODUCT': 'project',
                'EVENT': 'project',
                'WORK_OF_ART': 'project'
            }

            graph_type = type_mapping.get(entity_type)
            if not graph_type:
                return None

            # Search for matching nodes in graph
            entity_lower = entity_text.lower()

            for node, data in self.knowledge_graph.nodes(data=True):
                if data.get('type') != graph_type:
                    continue

                node_name = data.get('name', '').lower()

                # Exact match
                if node_name == entity_lower:
                    return node

                # Fuzzy match (simple substring check)
                if entity_lower in node_name or node_name in entity_lower:
                    return node

        except Exception as e:
            logger.warning(f"Error matching entity to graph: {e}")

        return None

    def link_batch_documents(
        self,
        documents: List[Document],
        knowledge_graph: Any = None
    ) -> List[Document]:
        """
        Process a batch of documents for figure entity linking.

        Args:
            documents: List of LlamaIndex Documents
            knowledge_graph: Optional knowledge graph to use for matching

        Returns:
            List of documents with enhanced figure_entities metadata
        """
        if knowledge_graph:
            self.knowledge_graph = knowledge_graph

        linked_docs = []
        linked_count = 0

        for doc in documents:
            linked_doc = self.process_document(doc)
            linked_docs.append(linked_doc)

            if linked_doc.metadata.get('has_figure_entities'):
                linked_count += 1

        logger.info(f"Entity linking: {linked_count}/{len(documents)} documents with linked figures")

        return linked_docs


def create_figure_entity_linker(
    knowledge_graph: Any = None
) -> FigureEntityLinker:
    """
    Factory function to create figure entity linker.

    Args:
        knowledge_graph: Optional NetworkX knowledge graph

    Returns:
        Configured FigureEntityLinker instance
    """
    return FigureEntityLinker(knowledge_graph=knowledge_graph)


def load_knowledge_graph_for_linking(graph_path: str) -> Optional[Any]:
    """
    Load knowledge graph from disk for entity linking.

    Args:
        graph_path: Path to .gpickle knowledge graph file

    Returns:
        NetworkX graph if successful, None otherwise
    """
    try:
        import networkx as nx
        from pathlib import Path

        path = Path(graph_path)
        if not path.exists():
            logger.warning(f"Knowledge graph not found at {graph_path}")
            return None

        graph = nx.read_gpickle(str(path))
        logger.info(f"✅ Loaded knowledge graph with {graph.number_of_nodes()} nodes for entity linking")
        return graph

    except Exception as e:
        logger.warning(f"Could not load knowledge graph for entity linking: {e}")
        return None
