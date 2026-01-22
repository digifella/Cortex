"""
Table-Aware Chunking Enhancer for Cortex Suite
Phase 2 Enhancement: Smart document chunking that preserves table integrity.

Version: 1.0.0
Date: 2026-01-13
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from .utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TableLocation:
    """Represents a table's location in the document."""
    start_idx: int
    end_idx: int
    page: Optional[int] = None
    table_data: Optional[Dict[str, Any]] = None
    caption: Optional[str] = None


class TableChunkingEnhancer:
    """
    Enhanced document chunker that preserves table integrity.

    Features:
    - Detects tables using Docling provenance metadata
    - Splits text at table boundaries (never mid-table)
    - Adds context window around tables
    - Generates table-specific formatted text for better embeddings
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        table_context_sentences: int = 2
    ):
        """
        Initialize table-aware chunker.

        Args:
            chunk_size: Target chunk size for regular text
            chunk_overlap: Overlap between regular text chunks
            table_context_sentences: Number of sentences to include before/after tables
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_context_sentences = table_context_sentences

        # Default sentence splitter for non-table text
        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_document(self, document: Document) -> List[Document]:
        """
        Process a document with table-aware chunking.

        Args:
            document: LlamaIndex Document with potential table metadata

        Returns:
            List of chunked Document objects with preserved tables
        """
        # Check if document has Docling structure metadata
        has_tables = self._has_tables(document)

        if not has_tables:
            # No tables - use standard chunking
            logger.debug(f"No tables detected in {document.metadata.get('file_name', 'document')}")
            return self._standard_chunk(document)

        logger.debug(f"Processing document with table-aware chunking: {document.metadata.get('file_name', 'document')}")

        # Extract table locations from provenance
        table_locations = self._extract_table_locations(document)

        if not table_locations:
            # Metadata indicated tables but we couldn't locate them
            logger.warning(f"Table metadata found but locations unavailable - using standard chunking")
            return self._standard_chunk(document)

        # Perform table-aware chunking
        return self._table_aware_chunk(document, table_locations)

    def _has_tables(self, document: Document) -> bool:
        """Check if document contains tables based on Docling metadata."""
        metadata = document.metadata

        # Check docling_structure metadata
        if 'docling_structure' in metadata:
            try:
                structure = metadata['docling_structure']
                if isinstance(structure, str):
                    structure = json.loads(structure)
                return structure.get('has_tables', False)
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        # Fallback: check provenance for table elements
        if 'docling_provenance' in metadata:
            try:
                provenance = metadata['docling_provenance']
                if isinstance(provenance, str):
                    provenance = json.loads(provenance)

                elements = provenance.get('elements', [])
                return any(elem.get('type') == 'table' for elem in elements)
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        return False

    def _extract_table_locations(self, document: Document) -> List[TableLocation]:
        """
        Extract table locations from Docling provenance metadata.

        Returns list of TableLocation objects with character offsets in the text.
        """
        locations = []

        try:
            provenance = document.metadata.get('docling_provenance', {})
            if isinstance(provenance, str):
                provenance = json.loads(provenance)

            elements = provenance.get('elements', [])

            # Find table elements
            for elem in elements:
                if elem.get('type') == 'table':
                    # Use text_sample to find approximate location in full text
                    text_sample = elem.get('text_sample', '')

                    if text_sample:
                        # Find this table in the full document text
                        start_idx = document.text.find(text_sample)
                        if start_idx != -1:
                            # Estimate end index (this is approximate)
                            # In production, we'd need more precise boundaries
                            end_idx = start_idx + len(text_sample) + 500  # Add buffer

                            location = TableLocation(
                                start_idx=start_idx,
                                end_idx=min(end_idx, len(document.text)),
                                page=elem.get('page'),
                                table_data=elem,
                                caption=self._extract_table_caption(document.text, start_idx)
                            )
                            locations.append(location)

            logger.debug(f"Found {len(locations)} table locations")

        except Exception as e:
            logger.warning(f"Error extracting table locations: {e}")

        return sorted(locations, key=lambda x: x.start_idx)

    def _extract_table_caption(self, text: str, table_start: int) -> Optional[str]:
        """Extract table caption from surrounding text."""
        # Look for caption in preceding 200 characters
        search_start = max(0, table_start - 200)
        preceding_text = text[search_start:table_start]

        # Common caption patterns
        caption_patterns = [
            r'Table\s+\d+[:\.]?\s*([^\n]+)',
            r'TABLE\s+\d+[:\.]?\s*([^\n]+)',
        ]

        for pattern in caption_patterns:
            match = re.search(pattern, preceding_text)
            if match:
                return match.group(1).strip()

        return None

    def _table_aware_chunk(
        self,
        document: Document,
        table_locations: List[TableLocation]
    ) -> List[Document]:
        """
        Chunk document while preserving table integrity.

        Strategy:
        1. Split document into segments: [text_before_table1][table1][text_between][table2]...
        2. Chunk text segments normally
        3. Keep tables as single chunks with context
        """
        chunks = []
        text = document.text
        last_end = 0

        for table_loc in table_locations:
            # Chunk text before this table
            if last_end < table_loc.start_idx:
                text_before = text[last_end:table_loc.start_idx]
                text_chunks = self._chunk_text_segment(text_before, document.metadata)
                chunks.extend(text_chunks)

            # Create table chunk with context
            table_chunk = self._create_table_chunk(
                document,
                table_loc,
                text
            )
            chunks.append(table_chunk)

            last_end = table_loc.end_idx

        # Chunk remaining text after last table
        if last_end < len(text):
            text_after = text[last_end:]
            text_chunks = self._chunk_text_segment(text_after, document.metadata)
            chunks.extend(text_chunks)

        logger.debug(f"Table-aware chunking produced {len(chunks)} chunks ({len(table_locations)} tables)")
        return chunks

    def _chunk_text_segment(
        self,
        text: str,
        base_metadata: Dict[str, Any]
    ) -> List[Document]:
        """Chunk a text segment using standard sentence splitter."""
        if not text.strip():
            return []

        # Create temporary document for chunking
        temp_doc = Document(text=text, metadata=base_metadata.copy())

        # Use sentence splitter
        chunks = self.sentence_splitter.get_nodes_from_documents([temp_doc])

        # Convert nodes back to Documents
        doc_chunks = []
        for node in chunks:
            doc_chunk = Document(
                text=node.text,
                metadata={**base_metadata, 'chunk_type': 'text'}
            )
            doc_chunks.append(doc_chunk)

        return doc_chunks

    def _create_table_chunk(
        self,
        document: Document,
        table_loc: TableLocation,
        full_text: str
    ) -> Document:
        """
        Create a table chunk with surrounding context.

        Includes:
        - Context before table (N sentences)
        - Table content (formatted for embeddings)
        - Context after table (N sentences)
        - Enhanced metadata
        """
        # Extract table text
        table_text = full_text[table_loc.start_idx:table_loc.end_idx]

        # Add context before table
        context_before = self._extract_context_before(
            full_text,
            table_loc.start_idx,
            self.table_context_sentences
        )

        # Add context after table
        context_after = self._extract_context_after(
            full_text,
            table_loc.end_idx,
            self.table_context_sentences
        )

        # Format table for better embeddings
        formatted_table = self._format_table_for_embedding(table_text, table_loc)

        # Combine into chunk text
        chunk_parts = []
        if context_before:
            chunk_parts.append(f"Context: {context_before}")

        if table_loc.caption:
            chunk_parts.append(f"\n\n{table_loc.caption}\n")

        chunk_parts.append(formatted_table)

        if context_after:
            chunk_parts.append(f"\n\nContext: {context_after}")

        chunk_text = "\n".join(chunk_parts)

        # Enhanced metadata
        chunk_metadata = document.metadata.copy()
        chunk_metadata.update({
            'chunk_type': 'table',
            'table_page': table_loc.page,
            'table_caption': table_loc.caption,
            'has_table': True
        })

        return Document(text=chunk_text, metadata=chunk_metadata)

    def _extract_context_before(
        self,
        text: str,
        position: int,
        num_sentences: int
    ) -> str:
        """Extract N sentences before a position."""
        # Look back up to 500 characters
        search_start = max(0, position - 500)
        preceding_text = text[search_start:position]

        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+\s+', preceding_text)

        # Take last N sentences
        context_sentences = sentences[-num_sentences:] if len(sentences) >= num_sentences else sentences

        return ' '.join(context_sentences).strip()

    def _extract_context_after(
        self,
        text: str,
        position: int,
        num_sentences: int
    ) -> str:
        """Extract N sentences after a position."""
        # Look ahead up to 500 characters
        search_end = min(len(text), position + 500)
        following_text = text[position:search_end]

        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', following_text)

        # Take first N sentences
        context_sentences = sentences[:num_sentences]

        return ' '.join(context_sentences).strip()

    def _format_table_for_embedding(
        self,
        table_text: str,
        table_loc: TableLocation
    ) -> str:
        """
        Format table text for optimal embedding generation.

        Adds semantic structure to help embeddings understand tabular data.
        """
        # Basic formatting - in production, parse actual table structure
        formatted = f"[TABLE]\n{table_text}\n[/TABLE]"

        return formatted

    def _standard_chunk(self, document: Document) -> List[Document]:
        """Fall back to standard chunking for documents without tables."""
        nodes = self.sentence_splitter.get_nodes_from_documents([document])

        # Convert nodes to Documents
        chunks = []
        for node in nodes:
            chunk_doc = Document(
                text=node.text,
                metadata={**document.metadata, 'chunk_type': 'text'}
            )
            chunks.append(chunk_doc)

        return chunks


def create_table_aware_chunker(
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    table_context_sentences: int = 2
) -> TableChunkingEnhancer:
    """
    Factory function to create table-aware chunker.

    Args:
        chunk_size: Target size for text chunks
        chunk_overlap: Overlap between text chunks
        table_context_sentences: Context sentences around tables

    Returns:
        Configured TableChunkingEnhancer instance
    """
    return TableChunkingEnhancer(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        table_context_sentences=table_context_sentences
    )
