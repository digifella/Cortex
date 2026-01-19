"""
Evidence Retriever for Intelligent Proposal Completion

Version: 1.0.0
Date: 2026-01-19

Purpose: Retrieves relevant evidence from a nominated knowledge collection
to support substantive proposal responses.

Key features:
- Searches nominated collection only (not entire knowledge base)
- Query reformulation based on question type
- Uses Qwen3-VL reranker for precision when available
- Extracts and scores relevant passages
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

from .field_classifier import QuestionType
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class Evidence:
    """A piece of evidence retrieved from the knowledge collection."""
    text: str                    # The evidence text/passage
    source_doc: str              # Source document name/path
    source_chunk_id: str         # Chunk ID for reference
    relevance_score: float       # 0-1 relevance to the question
    doc_type: Optional[str]      # e.g., "case_study", "policy", "capability"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSearchResult:
    """Result of an evidence search."""
    question: str
    question_type: QuestionType
    evidence: List[Evidence]
    collection_name: str
    total_candidates: int        # How many docs were searched
    search_query: str            # The reformulated query used


class EvidenceRetriever:
    """
    Retrieves relevant evidence from knowledge collections for proposal responses.

    Integrates with existing knowledge search infrastructure and uses
    Qwen3-VL reranker for precision when available.
    """

    # Query prefixes by question type to improve search quality
    QUERY_PREFIXES = {
        QuestionType.CAPABILITY: "experience delivering projects expertise in",
        QuestionType.METHODOLOGY: "approach methodology process for",
        QuestionType.VALUE_PROPOSITION: "benefits impact outcomes value of",
        QuestionType.COMPLIANCE: "certification policy compliance standards for",
        QuestionType.INNOVATION: "innovative solution technology approach to",
        QuestionType.RISK: "risk mitigation strategy contingency for",
        QuestionType.PERSONNEL: "team expertise qualifications staff for",
        QuestionType.PRICING: "pricing cost structure rates for",
        QuestionType.GENERAL: "",
    }

    # Keywords to boost by question type
    BOOST_KEYWORDS = {
        QuestionType.CAPABILITY: ["delivered", "completed", "achieved", "experience", "track record", "successfully"],
        QuestionType.METHODOLOGY: ["approach", "process", "steps", "methodology", "framework", "phases"],
        QuestionType.VALUE_PROPOSITION: ["benefit", "impact", "outcome", "value", "improvement", "savings"],
        QuestionType.COMPLIANCE: ["compliant", "certified", "accredited", "meets", "standards", "ISO"],
        QuestionType.INNOVATION: ["innovative", "novel", "unique", "improved", "developed", "created"],
        QuestionType.RISK: ["risk", "mitigation", "contingency", "backup", "fallback", "manage"],
        QuestionType.PERSONNEL: ["team", "qualified", "experienced", "expertise", "specialist", "senior"],
        QuestionType.PRICING: ["cost", "rate", "price", "fee", "budget", "estimate"],
        QuestionType.GENERAL: [],
    }

    def __init__(self, db_path: str):
        """
        Initialize the evidence retriever.

        Args:
            db_path: Path to the knowledge database
        """
        self.db_path = db_path
        self._search_engine = None
        self._collection_manager = None

        logger.info(f"EvidenceRetriever initialized for {db_path}")

    def _get_search_engine(self):
        """Lazy-load the search engine to avoid circular imports."""
        if self._search_engine is None:
            from .graph_query import GraphQueryEngine
            self._search_engine = GraphQueryEngine(self.db_path)
        return self._search_engine

    def _get_collection_manager(self):
        """Lazy-load the collection manager."""
        if self._collection_manager is None:
            from .collection_manager import WorkingCollectionManager
            self._collection_manager = WorkingCollectionManager()
        return self._collection_manager

    def find_evidence(
        self,
        question: str,
        question_type: QuestionType,
        collection_name: Optional[str],
        max_results: int = 5,
        use_reranker: bool = True
    ) -> EvidenceSearchResult:
        """
        Find relevant evidence from a collection or the entire knowledge base.

        Args:
            question: The question to find evidence for
            question_type: Type of question (affects query reformulation)
            collection_name: Name of the collection to search, or None for entire KB
            max_results: Maximum evidence pieces to return
            use_reranker: Whether to use Qwen3-VL reranker for precision

        Returns:
            EvidenceSearchResult with ranked evidence
        """
        # Determine if searching entire KB or specific collection
        search_entire_kb = collection_name is None or collection_name == ""
        display_name = "Entire Knowledge Base" if search_entire_kb else collection_name

        logger.info(f"Searching '{display_name}' for evidence on: {question[:50]}...")

        # Get document IDs (None means no filter = entire KB)
        doc_ids = None
        total_candidates = 0

        if not search_entire_kb:
            collection_mgr = self._get_collection_manager()
            doc_ids = collection_mgr.get_doc_ids_by_name(collection_name)

            if not doc_ids:
                logger.warning(f"Collection '{collection_name}' is empty or doesn't exist")
                return EvidenceSearchResult(
                    question=question,
                    question_type=question_type,
                    evidence=[],
                    collection_name=collection_name,
                    total_candidates=0,
                    search_query=question
                )
            total_candidates = len(doc_ids)

        # Reformulate query for better search
        search_query = self._build_search_query(question, question_type)

        # Search with optional collection filter
        search_engine = self._get_search_engine()

        try:
            # Determine search parameters
            candidates = max_results * 3  # Over-fetch for filtering

            # Use hybrid search with optional reranker
            # Pass doc_id_filter only if we have a collection filter
            results = search_engine.hybrid_search(
                query=search_query,
                top_k=candidates,
                doc_id_filter=set(doc_ids) if doc_ids else None,
                use_reranker=use_reranker
            )

            # Convert to Evidence objects
            evidence_list = self._convert_to_evidence(results, question_type)

            # Score and rank evidence
            scored_evidence = self._score_evidence(evidence_list, question, question_type)

            # Return top results
            return EvidenceSearchResult(
                question=question,
                question_type=question_type,
                evidence=scored_evidence[:max_results],
                collection_name=display_name,
                total_candidates=total_candidates if doc_ids else len(results),
                search_query=search_query
            )

        except Exception as e:
            logger.error(f"Evidence search failed: {e}")
            # Fallback to simpler search
            return self._fallback_search(question, question_type, collection_name, doc_ids, max_results)

    def _build_search_query(self, question: str, question_type: QuestionType) -> str:
        """
        Reformulate the question into an effective search query.

        Adds type-specific prefixes and keywords to improve retrieval.
        """
        prefix = self.QUERY_PREFIXES.get(question_type, "")

        # Extract key terms from question (simple approach)
        # Remove common question words
        stop_words = {
            "please", "describe", "detail", "explain", "outline", "provide",
            "the", "a", "an", "and", "or", "of", "to", "for", "in", "on",
            "your", "our", "how", "what", "which", "that", "this", "will"
        }

        words = question.lower().split()
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]

        # Combine prefix with key terms
        query = f"{prefix} {' '.join(key_terms[:10])}"  # Limit to 10 key terms

        return query.strip()

    def _convert_to_evidence(
        self,
        search_results: List[Dict[str, Any]],
        question_type: QuestionType
    ) -> List[Evidence]:
        """Convert search results to Evidence objects."""
        evidence_list = []

        for result in search_results:
            # Extract text content
            text = result.get('content', result.get('text', ''))
            if not text:
                continue

            # Get metadata
            metadata = result.get('metadata', {})
            source_doc = metadata.get('source', metadata.get('filename', 'Unknown'))
            chunk_id = result.get('id', metadata.get('chunk_id', 'unknown'))

            # Determine doc type from metadata
            doc_type = self._infer_doc_type(metadata, text)

            # Initial relevance score from search
            relevance = result.get('score', result.get('relevance', 0.5))
            if isinstance(relevance, str):
                try:
                    relevance = float(relevance)
                except:
                    relevance = 0.5

            evidence = Evidence(
                text=text[:2000],  # Limit text length
                source_doc=source_doc,
                source_chunk_id=str(chunk_id),
                relevance_score=min(relevance, 1.0),
                doc_type=doc_type,
                metadata=metadata
            )
            evidence_list.append(evidence)

        return evidence_list

    def _infer_doc_type(self, metadata: Dict, text: str) -> Optional[str]:
        """Infer document type from metadata and content."""
        # Check metadata first
        doc_type = metadata.get('document_type', metadata.get('doc_type'))
        if doc_type:
            return doc_type

        # Infer from filename
        source = metadata.get('source', '').lower()
        if 'case' in source or 'project' in source:
            return 'case_study'
        elif 'policy' in source or 'procedure' in source:
            return 'policy'
        elif 'capability' in source or 'profile' in source:
            return 'capability_statement'
        elif 'cv' in source or 'resume' in source:
            return 'personnel'
        elif 'proposal' in source:
            return 'past_proposal'

        # Infer from content
        text_lower = text.lower()[:500]
        if 'project overview' in text_lower or 'deliverables' in text_lower:
            return 'case_study'
        elif 'policy' in text_lower or 'procedure' in text_lower:
            return 'policy'

        return 'document'

    def _score_evidence(
        self,
        evidence_list: List[Evidence],
        question: str,
        question_type: QuestionType
    ) -> List[Evidence]:
        """
        Re-score evidence based on relevance to question and type.

        Boosts evidence containing type-specific keywords.
        """
        boost_keywords = self.BOOST_KEYWORDS.get(question_type, [])
        question_lower = question.lower()

        for evidence in evidence_list:
            text_lower = evidence.text.lower()
            score = evidence.relevance_score

            # Boost for keyword matches
            keyword_matches = sum(1 for kw in boost_keywords if kw in text_lower)
            if keyword_matches > 0:
                score += 0.05 * min(keyword_matches, 3)  # Max 0.15 boost

            # Boost for question term overlap
            question_terms = set(question_lower.split())
            text_terms = set(text_lower.split())
            overlap = len(question_terms & text_terms)
            if overlap > 2:
                score += 0.05 * min(overlap - 2, 3)  # Max 0.15 boost

            # Boost for substantive content (not just headers/fragments)
            if len(evidence.text) > 200:
                score += 0.05

            # Cap at 1.0
            evidence.relevance_score = min(score, 1.0)

        # Sort by relevance
        evidence_list.sort(key=lambda e: -e.relevance_score)

        return evidence_list

    def _fallback_search(
        self,
        question: str,
        question_type: QuestionType,
        collection_name: Optional[str],
        doc_ids: Optional[List[str]],
        max_results: int
    ) -> EvidenceSearchResult:
        """
        Fallback search when main search fails.

        Uses simpler direct ChromaDB search.
        """
        logger.info("Using fallback search method")
        display_name = collection_name or "Entire Knowledge Base"

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from .config import COLLECTION_NAME

            chroma_path = Path(self.db_path) / "knowledge_hub_db"
            client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            collection = client.get_collection(COLLECTION_NAME)

            # Simple query - no filter if searching entire KB
            results = collection.query(
                query_texts=[question],
                n_results=max_results * 2,
                where={"doc_id": {"$in": doc_ids}} if doc_ids else None
            )

            evidence_list = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0.5

                    evidence = Evidence(
                        text=doc[:2000],
                        source_doc=metadata.get('source', 'Unknown'),
                        source_chunk_id=str(results['ids'][0][i]) if results['ids'] else 'unknown',
                        relevance_score=1 - min(distance, 1.0),  # Convert distance to similarity
                        doc_type=self._infer_doc_type(metadata, doc),
                        metadata=metadata
                    )
                    evidence_list.append(evidence)

            return EvidenceSearchResult(
                question=question,
                question_type=question_type,
                evidence=evidence_list[:max_results],
                collection_name=display_name,
                total_candidates=len(doc_ids) if doc_ids else len(evidence_list),
                search_query=question
            )

        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return EvidenceSearchResult(
                question=question,
                question_type=question_type,
                evidence=[],
                collection_name=display_name,
                total_candidates=0,
                search_query=question
            )

    def get_collection_summary(self, collection_name: str) -> Dict[str, Any]:
        """
        Get summary of a collection's contents for user display.

        Returns:
            Dict with document count, types, and sample titles
        """
        collection_mgr = self._get_collection_manager()
        doc_ids = collection_mgr.get_doc_ids_by_name(collection_name)

        if not doc_ids:
            return {
                'name': collection_name,
                'document_count': 0,
                'doc_types': {},
                'sample_titles': []
            }

        # Get sample metadata
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from .config import COLLECTION_NAME

            chroma_path = Path(self.db_path) / "knowledge_hub_db"
            client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            collection = client.get_collection(COLLECTION_NAME)
            results = collection.get(
                ids=doc_ids[:50],  # Sample first 50
                include=['metadatas']
            )

            doc_types = {}
            sample_titles = []

            if results and results['metadatas']:
                for metadata in results['metadatas']:
                    doc_type = metadata.get('document_type', 'document')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

                    source = metadata.get('source', '')
                    if source and source not in sample_titles:
                        sample_titles.append(source)
                        if len(sample_titles) >= 10:
                            break

            return {
                'name': collection_name,
                'document_count': len(doc_ids),
                'doc_types': doc_types,
                'sample_titles': sample_titles
            }

        except Exception as e:
            logger.error(f"Failed to get collection summary: {e}")
            return {
                'name': collection_name,
                'document_count': len(doc_ids),
                'doc_types': {},
                'sample_titles': []
            }
