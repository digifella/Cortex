# ## File: async_query.py
# Version: 1.0.0
# Date: 2025-07-27
# Purpose: Async query engine for high-performance concurrent searches.
#          Provides parallel vector search, graph queries, and hybrid retrieval.

import asyncio
import aiofiles
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
import os
import time
from datetime import datetime
import json

from pydantic import BaseModel
import chromadb
import networkx as nx

from .utils.logging_utils import get_logger
from .graph_manager import EnhancedGraphManager
from .query_cortex import setup_models, FINAL_SYNTHESIS_PROMPT
from .config import COLLECTION_NAME, KB_LLM_MODEL

logger = get_logger(__name__)

class AsyncQueryConfig(BaseModel):
    """Configuration for async query operations"""
    max_concurrent_queries: int = 5
    max_results_per_query: int = 10
    similarity_threshold: float = 0.7
    enable_graph_context: bool = True
    enable_hybrid_search: bool = True
    query_timeout: float = 30.0
    use_semantic_reranking: bool = True

class AsyncQueryResult(BaseModel):
    """Result of async query operation"""
    query: str
    results: List[Dict[str, Any]] = []
    graph_context: Optional[Dict] = None
    synthesis: Optional[str] = None
    total_results: int = 0
    processing_time: float = 0.0
    vector_search_time: float = 0.0
    graph_search_time: float = 0.0
    synthesis_time: float = 0.0
    status: str = "success"
    error: Optional[str] = None

class AsyncSearchEngine:
    """High-performance async search and query engine"""
    
    def __init__(self, 
                 db_path: str, 
                 config: Optional[AsyncQueryConfig] = None):
        self.db_path = db_path
        self.config = config or AsyncQueryConfig()
        self.chroma_client = None
        self.collection = None
        self.graph_manager = None
        self.executor = None
        self.llm = None
        
    async def initialize(self) -> None:
        """Initialize async search components"""
        try:
            logger.info("🔍 Initializing async search engine...")
            
            # Initialize thread pool
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_queries)
            
            # Initialize components in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._initialize_sync_components)
            
            logger.info("✅ Async search engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize async search engine: {e}")
            raise
    
    def _initialize_sync_components(self) -> None:
        """Initialize synchronous components (run in executor)"""
        # Setup models
        setup_models()
        
        # Initialize ChromaDB
        chroma_path = os.path.join(self.db_path, "knowledge_hub_db")
        if os.path.exists(chroma_path):
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            try:
                self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            except Exception as e:
                logger.warning(f"ChromaDB collection not found: {e}")
        
        # Initialize graph manager
        if self.config.enable_graph_context:
            graph_path = os.path.join(self.db_path, "knowledge_cortex.gpickle")
            if os.path.exists(graph_path):
                self.graph_manager = EnhancedGraphManager(graph_path)
    
    async def search_async(self, 
                          query: str,
                          search_type: str = "hybrid",
                          include_synthesis: bool = True) -> AsyncQueryResult:
        """
        Perform async search with optional synthesis
        
        Args:
            query: Search query string
            search_type: Type of search ("vector", "graph", "hybrid")
            include_synthesis: Whether to synthesize results
            
        Returns:
            AsyncQueryResult with search results and metadata
        """
        start_time = time.time()
        result = AsyncQueryResult(query=query)
        
        try:
            logger.debug(f"🔍 Starting async search: '{query[:50]}...'")
            
            # Perform search based on type
            if search_type == "vector":
                await self._vector_search_async(query, result)
            elif search_type == "graph":
                await self._graph_search_async(query, result)
            elif search_type == "hybrid":
                await self._hybrid_search_async(query, result)
            else:
                raise ValueError(f"Unknown search type: {search_type}")
            
            # Perform synthesis if requested
            if include_synthesis and result.results:
                await self._synthesize_results_async(query, result)
            
            result.processing_time = time.time() - start_time
            result.total_results = len(result.results)
            
            logger.debug(f"✅ Search completed in {result.processing_time:.2f}s: {result.total_results} results")
            
            return result
            
        except Exception as e:
            result.status = "error"
            result.error = str(e)
            result.processing_time = time.time() - start_time
            logger.error(f"❌ Search error: {e}")
            return result
    
    async def _vector_search_async(self, query: str, result: AsyncQueryResult) -> None:
        """Perform async vector search"""
        if not self.collection:
            raise ValueError("ChromaDB collection not available")
        
        start_time = time.time()
        
        # Run vector search in executor
        loop = asyncio.get_event_loop()
        search_results = await loop.run_in_executor(
            self.executor,
            self._vector_search_sync,
            query
        )
        
        result.vector_search_time = time.time() - start_time
        result.results = search_results
    
    def _vector_search_sync(self, query: str) -> List[Dict[str, Any]]:
        """Synchronous vector search (runs in executor)"""
        try:
            search_result = self.collection.query(
                query_texts=[query],
                n_results=self.config.max_results_per_query,
                include=['documents', 'metadatas', 'distances']
            )
            
            results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                search_result['documents'][0],
                search_result['metadatas'][0],
                search_result['distances'][0]
            )):
                if distance <= (1.0 - self.config.similarity_threshold):
                    results.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity': 1.0 - distance,
                        'rank': i + 1,
                        'source': 'vector'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    async def _graph_search_async(self, query: str, result: AsyncQueryResult) -> None:
        """Perform async graph search"""
        if not self.graph_manager:
            raise ValueError("Graph manager not available")
        
        start_time = time.time()
        
        # Run graph search in executor
        loop = asyncio.get_event_loop()
        graph_results = await loop.run_in_executor(
            self.executor,
            self._graph_search_sync,
            query
        )
        
        result.graph_search_time = time.time() - start_time
        result.results = graph_results.get('results', [])
        result.graph_context = graph_results.get('context', {})
    
    def _graph_search_sync(self, query: str) -> Dict[str, Any]:
        """Synchronous graph search (runs in executor)"""
        try:
            # Extract entities from query
            entities = self.graph_manager.find_related_entities(query)
            
            results = []
            context = {
                'entities_found': len(entities),
                'relationships': [],
                'paths': []
            }
            
            for entity in entities[:self.config.max_results_per_query]:
                entity_info = self.graph_manager.get_entity_context(entity)
                if entity_info:
                    results.append({
                        'content': f"Entity: {entity}",
                        'metadata': entity_info,
                        'similarity': 0.8,  # Graph matches are considered high relevance
                        'rank': len(results) + 1,
                        'source': 'graph'
                    })
                    
                    # Add relationships to context
                    relationships = self.graph_manager.get_entity_relationships(entity)
                    context['relationships'].extend(relationships)
            
            return {
                'results': results,
                'context': context
            }
            
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return {'results': [], 'context': {}}
    
    async def _hybrid_search_async(self, query: str, result: AsyncQueryResult) -> None:
        """Perform hybrid vector + graph search"""
        # Run vector and graph searches concurrently
        vector_task = self._vector_search_async(query, AsyncQueryResult(query=query))
        graph_task = self._graph_search_async(query, AsyncQueryResult(query=query))
        
        vector_result, graph_result = await asyncio.gather(
            vector_task, graph_task, return_exceptions=True
        )
        
        # Combine results
        combined_results = []
        
        # Add vector results
        if not isinstance(vector_result, Exception) and hasattr(vector_result, 'results'):
            combined_results.extend(vector_result.results)
            result.vector_search_time = vector_result.vector_search_time
        
        # Add graph results
        if not isinstance(graph_result, Exception) and hasattr(graph_result, 'results'):
            combined_results.extend(graph_result.results)
            result.graph_search_time = graph_result.graph_search_time
            result.graph_context = graph_result.graph_context
        
        # Remove duplicates and rank by similarity
        seen_content = set()
        unique_results = []
        
        for res in sorted(combined_results, key=lambda x: x.get('similarity', 0), reverse=True):
            content_hash = hash(res.get('content', ''))
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(res)
        
        result.results = unique_results[:self.config.max_results_per_query]
    
    async def _synthesize_results_async(self, query: str, result: AsyncQueryResult) -> None:
        """Synthesize search results into coherent answer"""
        if not result.results:
            return
        
        start_time = time.time()
        
        # Prepare context from results
        context_parts = []
        for i, res in enumerate(result.results):
            source = res.get('metadata', {}).get('file_name', f"Source {i+1}")
            content = res.get('content', '')
            context_parts.append(f"[{source}]: {content}")
        
        context = "\n\n".join(context_parts)
        
        # Run synthesis in executor
        loop = asyncio.get_event_loop()
        synthesis = await loop.run_in_executor(
            self.executor,
            self._synthesize_sync,
            query,
            context
        )
        
        result.synthesis = synthesis
        result.synthesis_time = time.time() - start_time
    
    def _synthesize_sync(self, query: str, context: str) -> str:
        """Synchronous result synthesis (runs in executor)"""
        try:
            from llama_index.core import Settings
            
            prompt = FINAL_SYNTHESIS_PROMPT.format(
                question=query,
                context=context
            )
            
            response = Settings.llm.complete(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return f"Error synthesizing results: {str(e)}"
    
    async def batch_search_async(self, 
                               queries: List[str],
                               search_type: str = "hybrid",
                               include_synthesis: bool = True) -> List[AsyncQueryResult]:
        """
        Perform multiple searches concurrently
        
        Args:
            queries: List of query strings
            search_type: Type of search for all queries
            include_synthesis: Whether to synthesize all results
            
        Returns:
            List of AsyncQueryResult objects
        """
        logger.info(f"🔍 Starting batch search for {len(queries)} queries")
        
        # Create semaphore to limit concurrent queries
        semaphore = asyncio.Semaphore(self.config.max_concurrent_queries)
        
        async def search_with_semaphore(query: str) -> AsyncQueryResult:
            async with semaphore:
                return await self.search_async(query, search_type, include_synthesis)
        
        # Run all searches concurrently
        tasks = [search_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch search error for query {i}: {result}")
                # Create error result
                error_result = AsyncQueryResult(
                    query=queries[i],
                    status="error",
                    error=str(result)
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)
        
        logger.info(f"✅ Batch search completed: {len(valid_results)} results")
        return valid_results
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        try:
            stats = {
                'total_documents': 0,
                'total_entities': 0,
                'collection_exists': False,
                'graph_exists': False,
                'last_updated': None
            }
            
            # ChromaDB stats
            if self.collection:
                stats['total_documents'] = self.collection.count()
                stats['collection_exists'] = True
            
            # Graph stats
            if self.graph_manager:
                graph_stats = self.graph_manager.get_stats()
                stats['total_entities'] = graph_stats.get('entities', 0)
                stats['graph_exists'] = True
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting search stats: {e}")
            return {}
    
    async def cleanup(self) -> None:
        """Cleanup async resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            logger.info("🧹 Async search engine cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Convenience functions

async def search_knowledge_base_async(db_path: str,
                                    query: str,
                                    config: Optional[AsyncQueryConfig] = None,
                                    search_type: str = "hybrid") -> AsyncQueryResult:
    """
    High-level async knowledge base search function
    
    Args:
        db_path: Path to the knowledge database
        query: Search query string
        config: Optional search configuration
        search_type: Type of search ("vector", "graph", "hybrid")
        
    Returns:
        AsyncQueryResult with search results
    """
    engine = AsyncSearchEngine(db_path, config)
    
    try:
        await engine.initialize()
        result = await engine.search_async(query, search_type)
        return result
    finally:
        await engine.cleanup()

async def batch_search_knowledge_base_async(db_path: str,
                                          queries: List[str],
                                          config: Optional[AsyncQueryConfig] = None,
                                          search_type: str = "hybrid") -> List[AsyncQueryResult]:
    """
    High-level async batch search function
    
    Args:
        db_path: Path to the knowledge database
        queries: List of search queries
        config: Optional search configuration
        search_type: Type of search for all queries
        
    Returns:
        List of AsyncQueryResult objects
    """
    engine = AsyncSearchEngine(db_path, config)
    
    try:
        await engine.initialize()
        results = await engine.batch_search_async(queries, search_type)
        return results
    finally:
        await engine.cleanup()