"""
Universal Knowledge Assistant - Unified Interface for Knowledge Work
Version: 1.0.0
Date: 2026-01-01

Purpose: Single interface that combines:
- Internal knowledge search (RAG + GraphRAG)
- External research (academic papers, web)
- AI-powered synthesis and ideation
- Real-time streaming results

Replaces: AI Assisted Research + Knowledge Synthesizer
Workflow: Input → Intent Classification → Parallel Execution → Streaming Synthesis
"""

import asyncio
import json
from typing import Dict, List, Optional, AsyncIterator, Literal, Any
from dataclasses import dataclass
from enum import Enum
import re

from .adaptive_model_manager import get_model_manager, TaskType
from .utils.logging_utils import get_logger
from .utils.modern_ollama_llm import ModernOllamaLLM
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import QueryBundle

logger = get_logger(__name__)


class IntentType(str, Enum):
    """User intent classification for query routing."""
    RESEARCH = "research"          # Deep research on a topic (internal + external)
    SYNTHESIS = "synthesis"        # Synthesize knowledge from internal sources
    IDEATION = "ideation"         # Generate new ideas from knowledge base
    QUESTION = "question"          # Direct question answering
    EXPLORATION = "exploration"    # Open-ended exploration


class DepthLevel(str, Enum):
    """Depth of analysis requested."""
    QUICK = "quick"          # Fast overview (1-2 min, top results only)
    THOROUGH = "thorough"    # Balanced analysis (3-5 min, comprehensive)
    DEEP = "deep"            # Extensive research (10+ min, exhaustive)


@dataclass
class SourceConfig:
    """Configuration for knowledge sources to query."""
    internal_rag: bool = True        # Query internal knowledge base
    internal_graph: bool = True      # Query knowledge graph
    external_papers: bool = False    # Query academic papers (Semantic Scholar)
    external_web: bool = False       # Query web sources
    external_video: bool = False     # Query YouTube videos


@dataclass
class SearchResult:
    """Result from a knowledge source."""
    source_type: str              # "internal_rag", "internal_graph", "external_paper", etc.
    title: str                    # Result title
    content: str                  # Result content/excerpt
    relevance_score: float        # Relevance to query (0-1)
    metadata: Dict[str, Any]      # Additional metadata (authors, date, etc.)
    url: Optional[str] = None     # URL if applicable


@dataclass
class SynthesisChunk:
    """Chunk of synthesized knowledge (for streaming)."""
    content: str                  # Text content
    chunk_type: Literal["text", "source", "theme", "insight", "summary"]
    metadata: Optional[Dict] = None


class UniversalKnowledgeAssistant:
    """
    Unified knowledge assistant that intelligently routes queries across
    multiple knowledge sources and synthesizes results in real-time.
    """

    def __init__(
        self,
        rag_index: Optional[VectorStoreIndex] = None,
        graph_manager: Optional[Any] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize the Universal Knowledge Assistant.

        Args:
            rag_index: LlamaIndex vector store index for internal RAG
            graph_manager: GraphManager instance for knowledge graph queries
            collection_name: Name of the working collection to use
        """
        self.model_manager = get_model_manager()
        self.rag_index = rag_index
        self.graph_manager = graph_manager
        self.collection_name = collection_name

        # Model instances (lazy-loaded)
        self._router_llm = None
        self._power_llm = None

    async def _get_router_llm(self) -> ModernOllamaLLM:
        """Get or create fast router LLM for classification."""
        if self._router_llm is None:
            model_name = await self.model_manager.recommend_model(
                TaskType.ROUTER,
                preference="fastest"
            )
            logger.info(f"Using router model: {model_name}")
            self._router_llm = ModernOllamaLLM(model=model_name)
        return self._router_llm

    async def _get_power_llm(self, task_type: TaskType = TaskType.RESEARCH) -> ModernOllamaLLM:
        """Get or create power LLM for analysis/synthesis."""
        if self._power_llm is None:
            model_name = await self.model_manager.recommend_model(
                task_type,
                preference="best"
            )
            logger.info(f"Using power model for {task_type}: {model_name}")
            self._power_llm = ModernOllamaLLM(model=model_name)
        return self._power_llm

    async def classify_intent(self, user_input: str) -> IntentType:
        """
        Classify user intent to route the query appropriately.

        Args:
            user_input: The user's input text

        Returns:
            Classified intent type
        """
        router_llm = await self._get_router_llm()

        prompt = f"""Classify the user's intent into one of these categories:
- RESEARCH: Deep research on a topic, wants comprehensive analysis
- SYNTHESIS: Wants to synthesize/combine knowledge from existing sources
- IDEATION: Wants to generate new ideas or innovative solutions
- QUESTION: Direct question that needs a specific answer
- EXPLORATION: Open-ended exploration of a topic

User input: "{user_input}"

Respond with just the category name (e.g., "RESEARCH"), nothing else."""

        try:
            response = await router_llm.acomplete(prompt)
            intent_str = response.text.strip().upper()

            # Map response to IntentType
            for intent in IntentType:
                if intent.value.upper() in intent_str:
                    logger.info(f"Classified intent: {intent.value}")
                    return intent

            # Default to QUESTION if unclear
            logger.warning(f"Could not classify intent, defaulting to QUESTION. Response: {intent_str}")
            return IntentType.QUESTION

        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return IntentType.QUESTION

    async def search_internal_rag(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search the internal RAG system.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results from RAG
        """
        if not self.rag_index:
            logger.warning("No RAG index available")
            return []

        try:
            logger.info(f"Searching internal RAG: '{query}' (top_k={top_k})")

            # Create query bundle
            query_bundle = QueryBundle(query_str=query)

            # Perform retrieval
            retriever = self.rag_index.as_retriever(similarity_top_k=top_k)
            nodes = await asyncio.to_thread(retriever.retrieve, query_bundle)

            results = []
            for node in nodes:
                results.append(SearchResult(
                    source_type="internal_rag",
                    title=node.metadata.get("title", "Untitled Document"),
                    content=node.text,
                    relevance_score=node.score if hasattr(node, 'score') else 0.0,
                    metadata=node.metadata
                ))

            logger.info(f"Found {len(results)} RAG results")
            return results

        except Exception as e:
            logger.error(f"Error searching internal RAG: {e}")
            return []

    async def search_internal_graph(
        self,
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search the knowledge graph for related entities and relationships.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results from knowledge graph
        """
        if not self.graph_manager:
            logger.warning("No graph manager available")
            return []

        try:
            logger.info(f"Searching knowledge graph: '{query}' (top_k={top_k})")

            # Extract potential entities from query
            # This is simplified - in production, use NER
            entities = [word for word in query.split() if len(word) > 3]

            results = []
            for entity in entities[:top_k]:
                # Get related nodes from graph
                related = await asyncio.to_thread(
                    self.graph_manager.get_related_entities,
                    entity
                ) if hasattr(self.graph_manager, 'get_related_entities') else []

                if related:
                    results.append(SearchResult(
                        source_type="internal_graph",
                        title=f"Knowledge about '{entity}'",
                        content=f"Related entities: {', '.join(related[:10])}",
                        relevance_score=0.8,
                        metadata={"entity": entity, "related": related}
                    ))

            logger.info(f"Found {len(results)} graph results")
            return results

        except Exception as e:
            logger.error(f"Error searching knowledge graph: {e}")
            return []

    async def search_external_papers(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search academic papers via Semantic Scholar API.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results from academic papers
        """
        try:
            import aiohttp

            logger.info(f"Searching academic papers: '{query}' (top_k={top_k})")

            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": top_k,
                "fields": "title,abstract,authors,year,citationCount,url"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        return []

                    data = await response.json()
                    papers = data.get("data", [])

                    results = []
                    for paper in papers:
                        authors = ", ".join([a.get("name", "Unknown") for a in paper.get("authors", [])[:3]])
                        results.append(SearchResult(
                            source_type="external_paper",
                            title=paper.get("title", "Untitled Paper"),
                            content=paper.get("abstract", "No abstract available"),
                            relevance_score=0.7,  # Semantic Scholar doesn't provide scores
                            metadata={
                                "authors": authors,
                                "year": paper.get("year"),
                                "citations": paper.get("citationCount", 0),
                                "paper_id": paper.get("paperId")
                            },
                            url=paper.get("url")
                        ))

                    logger.info(f"Found {len(results)} academic papers")
                    return results

        except Exception as e:
            logger.error(f"Error searching academic papers: {e}")
            return []

    async def parallel_search(
        self,
        query: str,
        sources: SourceConfig,
        depth: DepthLevel
    ) -> Dict[str, List[SearchResult]]:
        """
        Execute parallel searches across all enabled sources.

        Args:
            query: Search query
            sources: Configuration of which sources to search
            depth: Depth level (affects top_k)

        Returns:
            Dictionary mapping source types to results
        """
        # Determine top_k based on depth
        top_k_map = {
            DepthLevel.QUICK: 5,
            DepthLevel.THOROUGH: 10,
            DepthLevel.DEEP: 20
        }
        top_k = top_k_map[depth]

        # Build list of search tasks
        tasks = []
        task_names = []

        if sources.internal_rag:
            tasks.append(self.search_internal_rag(query, top_k))
            task_names.append("internal_rag")

        if sources.internal_graph:
            tasks.append(self.search_internal_graph(query, top_k))
            task_names.append("internal_graph")

        if sources.external_papers:
            tasks.append(self.search_external_papers(query, top_k // 2))
            task_names.append("external_papers")

        # Execute all searches in parallel
        logger.info(f"Executing {len(tasks)} parallel searches: {task_names}")
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize results by source type
        all_results = {}
        for task_name, results in zip(task_names, results_list):
            if isinstance(results, Exception):
                logger.error(f"Search failed for {task_name}: {results}")
                all_results[task_name] = []
            else:
                all_results[task_name] = results

        total_results = sum(len(r) for r in all_results.values())
        logger.info(f"Parallel search complete: {total_results} total results")

        return all_results

    async def synthesize_stream(
        self,
        query: str,
        search_results: Dict[str, List[SearchResult]],
        intent: IntentType,
        depth: DepthLevel
    ) -> AsyncIterator[SynthesisChunk]:
        """
        Synthesize search results into coherent knowledge stream.

        Args:
            query: Original user query
            search_results: Results from parallel search
            intent: Classified user intent
            depth: Depth level requested

        Yields:
            Synthesis chunks as they're generated
        """
        # Get appropriate power LLM for task
        task_map = {
            IntentType.RESEARCH: TaskType.RESEARCH,
            IntentType.SYNTHESIS: TaskType.SYNTHESIS,
            IntentType.IDEATION: TaskType.IDEATION,
            IntentType.QUESTION: TaskType.ANALYSIS,
            IntentType.EXPLORATION: TaskType.RESEARCH
        }
        power_llm = await self._get_power_llm(task_map[intent])

        # Count total results
        total_results = sum(len(results) for results in search_results.values())

        # Yield initial status
        yield SynthesisChunk(
            content=f"Found {total_results} relevant sources. Analyzing...",
            chunk_type="text",
            metadata={"status": "analyzing"}
        )

        # Build context from search results
        context_parts = []
        for source_type, results in search_results.items():
            if results:
                context_parts.append(f"\n## {source_type.replace('_', ' ').title()} Results\n")
                for i, result in enumerate(results[:10], 1):  # Limit context
                    context_parts.append(f"{i}. **{result.title}**")
                    context_parts.append(f"   {result.content[:300]}...")  # Truncate long content
                    context_parts.append("")

        context = "\n".join(context_parts)

        # Build synthesis prompt based on intent
        prompt = self._build_synthesis_prompt(query, context, intent, depth)

        # Stream synthesis from LLM
        logger.info(f"Streaming synthesis for {intent.value} query")

        try:
            async for chunk in power_llm.astream_complete(prompt):
                # Yield text chunks as they arrive
                yield SynthesisChunk(
                    content=chunk.text,
                    chunk_type="text"
                )

        except Exception as e:
            logger.error(f"Error during synthesis streaming: {e}")
            yield SynthesisChunk(
                content=f"\n\n⚠️ Error during synthesis: {str(e)}",
                chunk_type="text",
                metadata={"error": str(e)}
            )

        # Yield sources at the end
        yield SynthesisChunk(
            content="\n\n## Sources\n",
            chunk_type="source"
        )

        for source_type, results in search_results.items():
            if results:
                for result in results[:5]:  # Top 5 sources
                    source_text = f"- **{result.title}**"
                    if result.url:
                        source_text += f" - [Link]({result.url})"
                    source_text += "\n"

                    yield SynthesisChunk(
                        content=source_text,
                        chunk_type="source",
                        metadata={"source": result.metadata}
                    )

    def _build_synthesis_prompt(
        self,
        query: str,
        context: str,
        intent: IntentType,
        depth: DepthLevel
    ) -> str:
        """Build synthesis prompt based on intent and depth."""

        base_instruction = {
            IntentType.RESEARCH: "Provide a comprehensive research synthesis on the topic.",
            IntentType.SYNTHESIS: "Synthesize the knowledge from multiple sources into coherent insights.",
            IntentType.IDEATION: "Generate innovative ideas based on the knowledge provided.",
            IntentType.QUESTION: "Answer the question directly and concisely.",
            IntentType.EXPLORATION: "Explore the topic from multiple angles and perspectives."
        }[intent]

        depth_instruction = {
            DepthLevel.QUICK: "Keep the response concise (2-3 paragraphs).",
            DepthLevel.THOROUGH: "Provide a balanced, thorough analysis (4-6 paragraphs).",
            DepthLevel.DEEP: "Provide an extensive, comprehensive analysis with multiple sections."
        }[depth]

        prompt = f"""You are a knowledge synthesis AI assistant. {base_instruction}

User Query: "{query}"

{depth_instruction}

## Available Knowledge

{context}

## Your Response

Synthesize the above knowledge to address the user's query. Be clear, insightful, and well-structured.
Use markdown formatting for better readability. Do not mention the sources in the main text - they will be listed separately."""

        return prompt

    async def process_query(
        self,
        user_input: str,
        sources: Optional[SourceConfig] = None,
        depth: DepthLevel = DepthLevel.THOROUGH,
        auto_classify_intent: bool = True
    ) -> AsyncIterator[SynthesisChunk]:
        """
        Main entry point: Process a user query end-to-end with streaming results.

        Args:
            user_input: The user's question/topic
            sources: Which knowledge sources to query (None = auto-select)
            depth: How thorough the analysis should be
            auto_classify_intent: Whether to auto-classify intent

        Yields:
            Synthesis chunks in real-time
        """
        logger.info(f"Processing query: '{user_input[:100]}...' (depth={depth.value})")

        # Step 1: Classify intent (if enabled)
        if auto_classify_intent:
            yield SynthesisChunk(
                content="Analyzing your query...\n\n",
                chunk_type="text",
                metadata={"status": "classifying"}
            )

            intent = await self.classify_intent(user_input)

            yield SynthesisChunk(
                content=f"Intent: {intent.value.title()}\n\n",
                chunk_type="text",
                metadata={"intent": intent.value}
            )
        else:
            intent = IntentType.QUESTION

        # Step 2: Determine sources (if not specified)
        if sources is None:
            sources = SourceConfig(
                internal_rag=True,
                internal_graph=True,
                external_papers=(intent == IntentType.RESEARCH and depth != DepthLevel.QUICK)
            )

        # Step 3: Parallel search
        yield SynthesisChunk(
            content="Searching knowledge sources...\n\n",
            chunk_type="text",
            metadata={"status": "searching"}
        )

        search_results = await self.parallel_search(user_input, sources, depth)

        # Step 4: Stream synthesis
        async for chunk in self.synthesize_stream(user_input, search_results, intent, depth):
            yield chunk

        logger.info("Query processing complete")


# Convenience function for quick usage
async def quick_knowledge_query(
    query: str,
    rag_index: Optional[VectorStoreIndex] = None,
    depth: DepthLevel = DepthLevel.QUICK
) -> str:
    """
    Quick knowledge query without streaming (returns complete response).

    Args:
        query: User query
        rag_index: Optional RAG index
        depth: Depth level

    Returns:
        Complete synthesis as string
    """
    assistant = UniversalKnowledgeAssistant(rag_index=rag_index)

    result_parts = []
    async for chunk in assistant.process_query(query, depth=depth):
        result_parts.append(chunk.content)

    return "".join(result_parts)
