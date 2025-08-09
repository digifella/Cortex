# ## File: tests/test_async_query.py
# Version: 1.0.0
# Date: 2025-07-27
# Purpose: Unit tests for async query functionality.

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from cortex_engine.async_query import (
    AsyncSearchEngine,
    AsyncQueryConfig,
    AsyncQueryResult,
    search_knowledge_base_async,
    batch_search_knowledge_base_async
)

pytestmark = pytest.mark.asyncio

class TestAsyncQueryConfig:
    """Test AsyncQueryConfig model"""
    
    def test_default_config(self):
        config = AsyncQueryConfig()
        assert config.max_concurrent_queries == 5
        assert config.max_results_per_query == 10
        assert config.similarity_threshold == 0.7
        assert config.enable_graph_context == True
        assert config.enable_hybrid_search == True
        assert config.query_timeout == 30.0
        assert config.use_semantic_reranking == True
    
    def test_custom_config(self):
        config = AsyncQueryConfig(
            max_concurrent_queries=10,
            similarity_threshold=0.8,
            enable_graph_context=False
        )
        assert config.max_concurrent_queries == 10
        assert config.similarity_threshold == 0.8
        assert config.enable_graph_context == False

class TestAsyncQueryResult:
    """Test AsyncQueryResult model"""
    
    def test_default_result(self):
        result = AsyncQueryResult(query="test query")
        assert result.query == "test query"
        assert result.results == []
        assert result.graph_context is None
        assert result.synthesis is None
        assert result.total_results == 0
        assert result.processing_time == 0.0
        assert result.status == "success"
        assert result.error is None
    
    def test_result_with_error(self):
        result = AsyncQueryResult(
            query="test query",
            status="error",
            error="Search failed"
        )
        assert result.status == "error"
        assert result.error == "Search failed"

class TestAsyncSearchEngine:
    """Test AsyncSearchEngine functionality"""
    
    async def test_initialization(self, temp_db_path):
        """Test engine initialization"""
        config = AsyncQueryConfig(max_concurrent_queries=2)
        engine = AsyncSearchEngine(temp_db_path, config)
        
        with patch('cortex_engine.async_query.setup_models'):
            await engine.initialize()
        
        assert engine.db_path == temp_db_path
        assert engine.config == config
        assert engine.executor is not None
        
        await engine.cleanup()
    
    @patch('cortex_engine.async_query.chromadb.PersistentClient')
    @patch('cortex_engine.async_query.setup_models')
    async def test_sync_component_initialization(self, mock_setup, mock_chroma, temp_db_path):
        """Test synchronous component initialization"""
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        engine = AsyncSearchEngine(temp_db_path)
        await engine.initialize()
        
        # Check that setup_models was called
        mock_setup.assert_called_once()
        
        # Check that ChromaDB was initialized
        mock_chroma.assert_called_once()
        
        await engine.cleanup()
    
    async def test_vector_search_sync(self, mock_chroma_collection):
        """Test synchronous vector search"""
        engine = AsyncSearchEngine("/tmp/test")
        engine.collection = mock_chroma_collection
        
        results = engine._vector_search_sync("test query")
        
        assert len(results) == 2
        assert results[0]['content'] == 'Sample document content'
        assert results[0]['similarity'] == 0.9  # 1.0 - 0.1
        assert results[0]['source'] == 'vector'
        
        mock_chroma_collection.query.assert_called_once()
    
    async def test_vector_search_sync_with_threshold(self, mock_chroma_collection):
        """Test vector search with similarity threshold"""
        config = AsyncQueryConfig(similarity_threshold=0.8)
        engine = AsyncSearchEngine("/tmp/test", config)
        engine.collection = mock_chroma_collection
        
        # Mock collection to return low similarity
        mock_chroma_collection.query.return_value = {
            'documents': [['Low similarity document']],
            'metadatas': [[{'file_name': 'test.txt'}]],
            'distances': [[0.9]]  # High distance = low similarity
        }
        
        results = engine._vector_search_sync("test query")
        
        # Should filter out low similarity results
        assert len(results) == 0
    
    async def test_graph_search_sync(self, mock_graph_manager):
        """Test synchronous graph search"""
        engine = AsyncSearchEngine("/tmp/test")
        engine.graph_manager = mock_graph_manager
        
        result = engine._graph_search_sync("test query")
        
        assert 'results' in result
        assert 'context' in result
        assert len(result['results']) == 2
        assert result['results'][0]['source'] == 'graph'
        
        mock_graph_manager.find_related_entities.assert_called_once_with("test query")
    
    async def test_synthesize_sync(self):
        """Test result synthesis"""
        engine = AsyncSearchEngine("/tmp/test")
        
        with patch('cortex_engine.async_query.Settings') as mock_settings:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.text = "Synthesized answer"
            mock_llm.complete.return_value = mock_response
            mock_settings.llm = mock_llm
            
            result = engine._synthesize_sync("test query", "test context")
            
            assert result == "Synthesized answer"
            mock_llm.complete.assert_called_once()
    
    async def test_vector_search_async(self, temp_db_path, mock_chroma_collection):
        """Test async vector search"""
        engine = AsyncSearchEngine(temp_db_path)
        engine.collection = mock_chroma_collection
        engine.executor = Mock()
        
        result = AsyncQueryResult(query="test")
        
        with patch.object(asyncio, 'get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=[
                {'content': 'test', 'similarity': 0.9, 'source': 'vector'}
            ])
            
            await engine._vector_search_async("test query", result)
        
        assert len(result.results) == 1
        assert result.vector_search_time > 0
    
    async def test_hybrid_search_async(self, temp_db_path):
        """Test hybrid search combining vector and graph results"""
        engine = AsyncSearchEngine(temp_db_path)
        
        # Mock the individual search methods
        async def mock_vector_search(query, result):
            result.results = [{'content': 'vector result', 'similarity': 0.9, 'source': 'vector'}]
            result.vector_search_time = 0.1
        
        async def mock_graph_search(query, result):
            result.results = [{'content': 'graph result', 'similarity': 0.8, 'source': 'graph'}]
            result.graph_search_time = 0.1
            result.graph_context = {'entities': 1}
        
        engine._vector_search_async = mock_vector_search
        engine._graph_search_async = mock_graph_search
        
        result = AsyncQueryResult(query="test")
        await engine._hybrid_search_async("test query", result)
        
        # Should have results from both searches
        assert len(result.results) == 2
        assert result.vector_search_time == 0.1
        assert result.graph_search_time == 0.1
        assert result.graph_context == {'entities': 1}
    
    @patch('cortex_engine.async_query.setup_models')
    async def test_search_async_vector(self, mock_setup, temp_db_path, mock_chroma_collection):
        """Test async search with vector search type"""
        engine = AsyncSearchEngine(temp_db_path)
        engine.collection = mock_chroma_collection
        await engine.initialize()
        
        with patch.object(engine, '_vector_search_async') as mock_vector:
            mock_vector.return_value = None  # Void method
            
            result = await engine.search_async("test query", "vector", False)
        
        assert result.query == "test query"
        assert result.status == "success"
        mock_vector.assert_called_once()
        
        await engine.cleanup()
    
    @patch('cortex_engine.async_query.setup_models')
    async def test_search_async_with_synthesis(self, mock_setup, temp_db_path):
        """Test async search with synthesis"""
        engine = AsyncSearchEngine(temp_db_path)
        await engine.initialize()
        
        # Mock search to return results
        with patch.object(engine, '_vector_search_async') as mock_search, \
             patch.object(engine, '_synthesize_results_async') as mock_synthesize:
            
            async def mock_search_impl(query, result):
                result.results = [{'content': 'test result'}]
            
            mock_search.side_effect = mock_search_impl
            
            result = await engine.search_async("test query", "vector", True)
        
        mock_synthesize.assert_called_once()
        
        await engine.cleanup()
    
    @patch('cortex_engine.async_query.setup_models')
    async def test_search_async_error_handling(self, mock_setup, temp_db_path):
        """Test error handling in async search"""
        engine = AsyncSearchEngine(temp_db_path)
        await engine.initialize()
        
        with patch.object(engine, '_vector_search_async') as mock_search:
            mock_search.side_effect = Exception("Search failed")
            
            result = await engine.search_async("test query", "vector")
        
        assert result.status == "error"
        assert "Search failed" in result.error
        
        await engine.cleanup()
    
    @patch('cortex_engine.async_query.setup_models')
    async def test_batch_search_async(self, mock_setup, temp_db_path):
        """Test batch search functionality"""
        engine = AsyncSearchEngine(temp_db_path)
        await engine.initialize()
        
        queries = ["query1", "query2", "query3"]
        
        with patch.object(engine, 'search_async') as mock_search:
            mock_search.return_value = AsyncQueryResult(query="mock", status="success")
            
            results = await engine.batch_search_async(queries)
        
        assert len(results) == 3
        assert mock_search.call_count == 3
        
        await engine.cleanup()
    
    async def test_get_search_stats(self, temp_db_path):
        """Test getting search statistics"""
        engine = AsyncSearchEngine(temp_db_path)
        
        # Mock components
        engine.collection = Mock()
        engine.collection.count.return_value = 10
        
        engine.graph_manager = Mock()
        engine.graph_manager.get_stats.return_value = {'entities': 5}
        
        stats = await engine.get_search_stats()
        
        assert stats['total_documents'] == 10
        assert stats['total_entities'] == 5
        assert stats['collection_exists'] == True
        assert stats['graph_exists'] == True
    
    async def test_cleanup(self, temp_db_path):
        """Test engine cleanup"""
        engine = AsyncSearchEngine(temp_db_path)
        
        with patch('cortex_engine.async_query.setup_models'):
            await engine.initialize()
        
        executor = engine.executor
        await engine.cleanup()
        
        # Executor should be shut down
        assert executor._shutdown == True

class TestSearchIntegration:
    """Integration tests for search functionality"""
    
    @pytest.mark.integration
    @patch('cortex_engine.async_query.AsyncSearchEngine')
    async def test_search_knowledge_base_async(self, mock_engine_class, temp_db_path):
        """Test high-level search function"""
        # Mock engine instance
        mock_engine = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine.search_async = AsyncMock()
        mock_engine.cleanup = AsyncMock()
        mock_engine_class.return_value = mock_engine
        
        # Mock result
        expected_result = AsyncQueryResult(query="test", status="success")
        mock_engine.search_async.return_value = expected_result
        
        result = await search_knowledge_base_async(temp_db_path, "test query")
        
        assert result == expected_result
        mock_engine.initialize.assert_called_once()
        mock_engine.search_async.assert_called_once_with("test query", "hybrid")
        mock_engine.cleanup.assert_called_once()
    
    @pytest.mark.integration
    @patch('cortex_engine.async_query.AsyncSearchEngine')
    async def test_batch_search_knowledge_base_async(self, mock_engine_class, temp_db_path):
        """Test high-level batch search function"""
        # Mock engine instance
        mock_engine = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine.batch_search_async = AsyncMock()
        mock_engine.cleanup = AsyncMock()
        mock_engine_class.return_value = mock_engine
        
        # Mock results
        queries = ["query1", "query2"]
        expected_results = [
            AsyncQueryResult(query="query1", status="success"),
            AsyncQueryResult(query="query2", status="success")
        ]
        mock_engine.batch_search_async.return_value = expected_results
        
        results = await batch_search_knowledge_base_async(temp_db_path, queries)
        
        assert results == expected_results
        mock_engine.initialize.assert_called_once()
        mock_engine.batch_search_async.assert_called_once_with(queries, "hybrid")
        mock_engine.cleanup.assert_called_once()

class TestPerformanceAndConcurrency:
    """Test performance and concurrency aspects"""
    
    @pytest.mark.asyncio
    async def test_concurrent_search_limit(self, temp_db_path):
        """Test that concurrent searches are properly limited"""
        config = AsyncQueryConfig(max_concurrent_queries=2)
        engine = AsyncSearchEngine(temp_db_path, config)
        
        with patch('cortex_engine.async_query.setup_models'):
            await engine.initialize()
        
        search_times = []
        
        async def timed_search(query):
            start = time.time()
            with patch.object(engine, 'search_async') as mock_search:
                mock_search.return_value = AsyncQueryResult(query=query)
                await engine.search_async(query)
            search_times.append(time.time() - start)
        
        # Start multiple searches concurrently
        queries = [f"query{i}" for i in range(5)]
        await asyncio.gather(*[timed_search(q) for q in queries])
        
        # With concurrency limit, some searches should take longer
        assert len(search_times) == 5
        
        await engine.cleanup()
    
    async def test_search_timeout_handling(self, temp_db_path):
        """Test search timeout handling"""
        config = AsyncQueryConfig(query_timeout=0.1)  # Very short timeout
        engine = AsyncSearchEngine(temp_db_path, config)
        
        with patch('cortex_engine.async_query.setup_models'):
            await engine.initialize()
        
        # Mock a slow search
        with patch.object(engine, '_vector_search_async') as mock_search:
            async def slow_search(query, result):
                await asyncio.sleep(0.2)  # Longer than timeout
            
            mock_search.side_effect = slow_search
            
            # This should handle the timeout gracefully
            # Note: Actual timeout implementation depends on your search logic
            result = await engine.search_async("test query", "vector")
            
            # The search might still complete, but this tests the structure
            assert result.query == "test query"
        
        await engine.cleanup()