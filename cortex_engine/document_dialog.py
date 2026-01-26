# ## File: cortex_engine/document_dialog.py
# Version: 1.0.0
# Date: 2026-01-26
# Purpose: Conversational Q&A engine for document collections.
#          Enables multi-turn conversations with RAG-retrieved context
#          from ingested document collections.

import os
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings as ChromaSettings

from .utils import get_logger, convert_to_docker_mount_path
from .config import COLLECTION_NAME, QWEN3_VL_RERANKER_ENABLED, QWEN3_VL_RERANKER_TOP_K
from .embedding_service import embed_query
from .collection_manager import WorkingCollectionManager
from .document_summarizer import SUMMARIZER_MODELS

logger = get_logger(__name__)


@dataclass
class DialogMessage:
    """Represents a single message in a dialog conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[Dict] = field(default_factory=list)  # Citations for assistant messages

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "sources": self.sources
        }


@dataclass
class DialogSession:
    """Manages a conversation session with a document collection."""
    session_id: str
    collection_name: str
    doc_ids: List[str]
    messages: List[DialogMessage] = field(default_factory=list)
    model_name: str = "mistral:latest"
    created_at: datetime = field(default_factory=datetime.now)

    def get_conversation_context(self, max_turns: int = 3) -> str:
        """
        Get recent conversation history for context.

        Args:
            max_turns: Maximum number of Q&A pairs to include

        Returns:
            Formatted conversation history string
        """
        if not self.messages:
            return ""

        # Get last N turns (each turn = user + assistant)
        recent_messages = self.messages[-(max_turns * 2):]

        context_parts = []
        for msg in recent_messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {msg.content}")

        return "\n".join(context_parts)

    def add_message(self, role: str, content: str, sources: List[Dict] = None):
        """Add a message to the session."""
        self.messages.append(DialogMessage(
            role=role,
            content=content,
            sources=sources or []
        ))

    def to_dict(self) -> Dict:
        """Convert session to dictionary for export."""
        return {
            "session_id": self.session_id,
            "collection_name": self.collection_name,
            "doc_ids": self.doc_ids,
            "model_name": self.model_name,
            "created_at": self.created_at.isoformat(),
            "messages": [msg.to_dict() for msg in self.messages]
        }


@dataclass
class DialogResponse:
    """Response from a dialog query."""
    success: bool
    answer: str
    sources: List[Dict] = field(default_factory=list)
    context_chunks: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None


class DocumentDialogEngine:
    """
    Conversational Q&A engine for document collections.

    Enables multi-turn conversations with documents using:
    - Semantic search via ChromaDB embeddings
    - Optional neural reranking for precision
    - Conversation history for follow-up questions
    - Source citations for transparency
    """

    def __init__(self, model_name: str = "mistral:latest", db_path: str = ""):
        """
        Initialize the dialog engine.

        Args:
            model_name: Ollama model name for generation
            db_path: Path to the AI database directory
        """
        self.model_name = model_name
        self.db_path = db_path
        self._chroma_client = None
        self._vector_collection = None
        self._collection_manager = None

    def _get_chroma_collection(self):
        """Get or initialize ChromaDB collection."""
        if self._vector_collection is not None:
            return self._vector_collection

        if not self.db_path:
            raise ValueError("Database path not configured")

        # Resolve path for Docker/WSL compatibility
        safe_path = convert_to_docker_mount_path(self.db_path)
        chroma_db_path = os.path.join(safe_path, "knowledge_hub_db")

        if not os.path.isdir(chroma_db_path):
            raise FileNotFoundError(f"Knowledge base not found: {chroma_db_path}")

        settings = ChromaSettings(anonymized_telemetry=False)
        self._chroma_client = chromadb.PersistentClient(path=chroma_db_path, settings=settings)
        self._vector_collection = self._chroma_client.get_collection(COLLECTION_NAME)

        logger.info(f"Connected to ChromaDB collection: {COLLECTION_NAME}")
        return self._vector_collection

    def _get_collection_manager(self) -> WorkingCollectionManager:
        """Get or initialize collection manager."""
        if self._collection_manager is None:
            self._collection_manager = WorkingCollectionManager()
        return self._collection_manager

    def initialize_session(self, collection_name: str) -> DialogSession:
        """
        Initialize a new dialog session for a collection.

        Args:
            collection_name: Name of the working collection

        Returns:
            New DialogSession instance
        """
        mgr = self._get_collection_manager()
        doc_ids = mgr.get_doc_ids_by_name(collection_name)

        if not doc_ids:
            logger.warning(f"Collection '{collection_name}' is empty or not found")

        session = DialogSession(
            session_id=str(uuid.uuid4()),
            collection_name=collection_name,
            doc_ids=doc_ids,
            model_name=self.model_name
        )

        logger.info(f"Initialized dialog session {session.session_id} with {len(doc_ids)} documents")
        return session

    def query(
        self,
        question: str,
        session: DialogSession,
        top_k: int = 5,
        use_reranker: bool = True
    ) -> DialogResponse:
        """
        Query the collection with a question.

        Args:
            question: User's question
            session: Current dialog session
            top_k: Number of chunks to retrieve
            use_reranker: Whether to apply neural reranking

        Returns:
            DialogResponse with answer and sources
        """
        start_time = time.time()

        try:
            if not session.doc_ids:
                return DialogResponse(
                    success=False,
                    answer="",
                    error="This collection has no documents. Please add documents to the collection first."
                )

            # Step 1: Retrieve relevant chunks from the collection
            chunks = self._retrieve_relevant_chunks(question, session.doc_ids, top_k * 2)

            if not chunks:
                return DialogResponse(
                    success=False,
                    answer="",
                    error="No relevant content found in the collection for this question."
                )

            # Step 2: Optionally rerank for precision
            if use_reranker and QWEN3_VL_RERANKER_ENABLED:
                chunks = self._rerank_chunks(question, chunks, top_k)
            else:
                chunks = chunks[:top_k]

            # Step 3: Build context from chunks
            context = self._build_context(chunks)

            # Step 4: Generate response with conversation history
            conversation_history = session.get_conversation_context(max_turns=3)
            answer = self._generate_response(question, context, conversation_history)

            # Step 5: Extract and format sources
            sources = self._extract_sources(chunks)

            # Add messages to session
            session.add_message("user", question)
            session.add_message("assistant", answer, sources)

            processing_time = time.time() - start_time

            return DialogResponse(
                success=True,
                answer=answer,
                sources=sources,
                context_chunks=len(chunks),
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Dialog query failed: {e}")
            return DialogResponse(
                success=False,
                answer="",
                error=str(e),
                processing_time=time.time() - start_time
            )

    def _retrieve_relevant_chunks(
        self,
        question: str,
        doc_ids: List[str],
        top_k: int
    ) -> List[Dict]:
        """
        Retrieve relevant chunks from ChromaDB filtered by document IDs.

        Args:
            question: Query text
            doc_ids: List of document IDs to search within
            top_k: Number of results to retrieve

        Returns:
            List of chunk dictionaries with content and metadata
        """
        collection = self._get_chroma_collection()

        # Generate query embedding
        query_embedding = embed_query(question)

        # Query with doc_id filter
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 100),  # ChromaDB limit
            where={"doc_id": {"$in": doc_ids}},
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        if results and results.get("documents"):
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for i, doc in enumerate(documents):
                chunk = {
                    "content": doc,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else 0.0,
                    "score": 1.0 - (distances[i] if i < len(distances) else 0.0)  # Convert distance to similarity
                }
                chunks.append(chunk)

        logger.info(f"Retrieved {len(chunks)} chunks for query")
        return chunks

    def _rerank_chunks(
        self,
        question: str,
        chunks: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Apply neural reranking to chunks for better precision.

        Args:
            question: Query text
            chunks: List of chunks to rerank
            top_k: Number of results to return after reranking

        Returns:
            Reranked and filtered chunks
        """
        try:
            from .graph_query import rerank_search_results

            reranked = rerank_search_results(
                query=question,
                results=chunks,
                top_k=top_k,
                text_key="content"
            )

            logger.info(f"Reranked {len(chunks)} chunks to top {len(reranked)}")
            return reranked

        except ImportError:
            logger.warning("Reranker not available, using original ranking")
            return chunks[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using original ranking")
            return chunks[:top_k]

    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            source_name = metadata.get("file_name", metadata.get("doc_id", f"Source {i}"))

            context_parts.append(f"[Source {i}: {source_name}]\n{content}")

        return "\n\n---\n\n".join(context_parts)

    def _generate_response(
        self,
        question: str,
        context: str,
        conversation_history: str
    ) -> str:
        """
        Generate response using LLM with context and conversation history.

        Args:
            question: Current user question
            context: Retrieved document context
            conversation_history: Previous conversation turns

        Returns:
            Generated answer string
        """
        import requests

        # Build prompt
        history_section = ""
        if conversation_history:
            history_section = f"""
PREVIOUS CONVERSATION:
{conversation_history}

---

"""

        prompt = f"""You are a helpful document analyst having a conversation with a user about their document collection. Answer questions based ONLY on the provided document context. Cite sources using [Source N] notation.

{history_section}DOCUMENT CONTEXT:
---
{context}
---

CURRENT QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the document context above
- Reference specific sources using [Source 1], [Source 2], etc.
- If the answer isn't in the documents, say so clearly
- Be concise but thorough
- Use markdown formatting for readability
- Consider the conversation history for context on follow-up questions

ANSWER:"""

        try:
            # Get model context window
            model_config = SUMMARIZER_MODELS.get(self.model_name, {})
            max_tokens = min(2000, model_config.get("context_window", 8192) // 4)

            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0.3,
                    "num_predict": max_tokens,
                    "num_gpu": -1
                }
            }

            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()

            result = response.json()
            if "response" in result:
                return result["response"].strip()
            else:
                raise Exception(f"Unexpected response format: {result}")

        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama - ensure it's running on localhost:11434")
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out - try a shorter question or smaller model")
        except Exception as e:
            raise Exception(f"LLM generation failed: {e}")

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract source information from chunks for citation.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of source dictionaries with file info
        """
        sources = []
        seen_files = set()

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            file_name = metadata.get("file_name", "Unknown")
            doc_id = metadata.get("doc_id", "")

            # Deduplicate by file
            if file_name not in seen_files:
                seen_files.add(file_name)
                sources.append({
                    "source_number": i,
                    "file_name": file_name,
                    "doc_id": doc_id,
                    "file_path": metadata.get("doc_posix_path", ""),
                    "document_type": metadata.get("document_type", "Unknown"),
                    "score": chunk.get("score", chunk.get("rerank_score", 0.0))
                })

        return sources

    def export_conversation(
        self,
        session: DialogSession,
        format: str = "markdown"
    ) -> str:
        """
        Export conversation to specified format.

        Args:
            session: Dialog session to export
            format: Export format ('markdown' or 'json')

        Returns:
            Formatted conversation string
        """
        if format == "json":
            return json.dumps(session.to_dict(), indent=2)

        # Markdown format
        lines = [
            f"# Document Dialog: {session.collection_name}",
            f"",
            f"**Session ID:** {session.session_id}",
            f"**Created:** {session.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Model:** {session.model_name}",
            f"**Documents:** {len(session.doc_ids)}",
            f"",
            "---",
            ""
        ]

        for msg in session.messages:
            timestamp = msg.timestamp.strftime('%H:%M')
            if msg.role == "user":
                lines.append(f"## Question ({timestamp})")
                lines.append(f"")
                lines.append(msg.content)
            else:
                lines.append(f"## Answer ({timestamp})")
                lines.append(f"")
                lines.append(msg.content)
                if msg.sources:
                    lines.append("")
                    lines.append("**Sources:**")
                    for src in msg.sources:
                        lines.append(f"- [{src.get('source_number', '?')}] {src.get('file_name', 'Unknown')}")
            lines.append("")
            lines.append("---")
            lines.append("")

        lines.append("")
        lines.append(f"*Exported from Cortex Suite Document Dialog*")

        return "\n".join(lines)

    def get_collection_documents(self, collection_name: str) -> List[Dict]:
        """
        Get document metadata for a collection.

        Args:
            collection_name: Name of the working collection

        Returns:
            List of document metadata dictionaries
        """
        mgr = self._get_collection_manager()
        doc_ids = mgr.get_doc_ids_by_name(collection_name)

        if not doc_ids:
            return []

        try:
            collection = self._get_chroma_collection()
            results = collection.get(
                where={"doc_id": {"$in": doc_ids}},
                include=["metadatas"]
            )

            # Deduplicate by doc_id
            docs = {}
            for meta in results.get("metadatas", []):
                doc_id = meta.get("doc_id", "")
                if doc_id and doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "file_name": meta.get("file_name", "Unknown"),
                        "document_type": meta.get("document_type", "Unknown"),
                        "file_path": meta.get("doc_posix_path", "")
                    }

            return list(docs.values())

        except Exception as e:
            logger.warning(f"Failed to get collection documents: {e}")
            return []
