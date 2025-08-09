# ## File: tests/test_async_ingest.py
# Version: 1.0.0
# Date: 2025-07-27
# Purpose: Unit tests for async ingestion functionality.

import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from cortex_engine.async_ingest import (
    AsyncIngestionEngine, 
    AsyncIngestionConfig, 
    AsyncIngestionResult,
    ingest_documents_async
)

pytestmark = pytest.mark.asyncio

class TestAsyncIngestionConfig:
    """Test AsyncIngestionConfig model"""
    
    def test_default_config(self):
        config = AsyncIngestionConfig()
        assert config.max_concurrent_files == 5
        assert config.max_concurrent_entities == 10
        assert config.chunk_size == 100
        assert config.use_process_pool == True
        assert config.max_workers is None
    
    def test_custom_config(self):
        config = AsyncIngestionConfig(
            max_concurrent_files=10,
            use_process_pool=False,
            max_workers=4
        )
        assert config.max_concurrent_files == 10
        assert config.use_process_pool == False
        assert config.max_workers == 4

class TestAsyncIngestionResult:
    """Test AsyncIngestionResult model"""
    
    def test_default_result(self):
        result = AsyncIngestionResult()
        assert result.success_count == 0
        assert result.error_count == 0
        assert result.skipped_count == 0
        assert result.processed_files == []
        assert result.errors == []
        assert result.total_entities == 0
        assert result.total_relationships == 0
        assert result.processing_time == 0.0

class TestAsyncIngestionEngine:
    """Test AsyncIngestionEngine functionality"""
    
    async def test_initialization(self, temp_db_path):
        """Test engine initialization"""
        config = AsyncIngestionConfig(use_process_pool=False)
        engine = AsyncIngestionEngine(temp_db_path, config)
        
        await engine.initialize()
        
        assert engine.db_path == temp_db_path
        assert engine.config == config
        assert engine.executor is not None
        
        await engine.cleanup()
    
    async def test_initialization_with_process_pool(self, temp_db_path):
        """Test engine initialization with process pool"""
        config = AsyncIngestionConfig(use_process_pool=True, max_workers=2)
        engine = AsyncIngestionEngine(temp_db_path, config)
        
        await engine.initialize()
        
        assert engine.executor is not None
        
        await engine.cleanup()
    
    @patch('cortex_engine.async_ingest.EntityExtractor')
    @patch('cortex_engine.async_ingest.EnhancedGraphManager')  
    async def test_sync_component_initialization(self, mock_graph, mock_extractor, temp_db_path):
        """Test synchronous component initialization"""
        config = AsyncIngestionConfig(use_process_pool=False)
        engine = AsyncIngestionEngine(temp_db_path, config)
        
        await engine.initialize()
        
        # Check that components were initialized
        mock_extractor.assert_called_once()
        mock_graph.assert_called_once()
        
        await engine.cleanup()
    
    async def test_load_processed_files_async_empty(self, temp_db_path):
        """Test loading empty processed files log"""
        engine = AsyncIngestionEngine(temp_db_path)
        
        log_path = os.path.join(temp_db_path, "test.log")
        result = await engine._load_processed_files_async(log_path)
        
        assert result == {}
    
    async def test_load_processed_files_async_with_data(self, temp_db_path):
        """Test loading processed files log with data"""
        engine = AsyncIngestionEngine(temp_db_path)
        
        log_path = os.path.join(temp_db_path, "test.log")
        test_data = {"file1.txt": "hash1", "file2.txt": "hash2"}
        
        # Create test log file
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            import json
            json.dump(test_data, f)
        
        result = await engine._load_processed_files_async(log_path)
        
        assert result == test_data
    
    async def test_should_process_file_new_file(self, temp_db_path, create_test_files):
        """Test should_process_file for new file"""
        engine = AsyncIngestionEngine(temp_db_path)
        
        file_path = list(create_test_files.values())[0]
        processed_files = {}
        
        should_process = await engine._should_process_file(file_path, processed_files)
        assert should_process == True
    
    async def test_should_process_file_exclusion(self, temp_db_path):
        """Test should_process_file with exclusion patterns"""
        engine = AsyncIngestionEngine(temp_db_path)
        
        file_path = "/tmp/test_invoice.pdf"
        processed_files = {}
        exclusion_patterns = ["invoice", "temp"]
        
        should_process = await engine._should_process_file(
            file_path, processed_files, exclusion_patterns
        )
        assert should_process == False
    
    async def test_update_processed_files_log_async(self, temp_db_path, create_test_files):
        """Test updating processed files log"""
        engine = AsyncIngestionEngine(temp_db_path)
        
        log_path = os.path.join(temp_db_path, "test.log")
        processed_files = list(create_test_files.values())[:2]
        
        await engine._update_processed_files_log_async(log_path, processed_files)
        
        # Verify log was created and contains data
        assert os.path.exists(log_path)
        
        loaded_data = await engine._load_processed_files_async(log_path)
        assert len(loaded_data) == 2
        for file_path in processed_files:
            assert file_path in loaded_data
    
    @patch('cortex_engine.async_ingest.get_document_content')
    async def test_process_file_sync(self, mock_get_content, temp_db_path, create_test_files):
        """Test synchronous file processing"""
        config = AsyncIngestionConfig(use_process_pool=False)
        engine = AsyncIngestionEngine(temp_db_path, config)
        await engine.initialize()
        
        # Mock content extraction
        mock_get_content.return_value = {
            'content': 'Test document content',
            'metadata': {}
        }
        
        # Mock entity extractor
        engine.entity_extractor = Mock()
        engine.entity_extractor.extract_entities_from_text.return_value = {
            'entities': [],
            'relationships': []
        }
        
        # Mock graph manager
        engine.graph_manager = Mock()
        
        # Mock collection
        engine.collection = Mock()
        
        file_path = list(create_test_files.values())[0]
        result = engine._process_file_sync(file_path)
        
        assert result['file_path'] == file_path
        assert 'doc_id' in result
        assert 'entities' in result
        assert 'relationships' in result
        
        await engine.cleanup()
    
    async def test_get_ingestion_stats(self, temp_db_path):
        """Test getting ingestion statistics"""
        config = AsyncIngestionConfig(use_process_pool=False)
        engine = AsyncIngestionEngine(temp_db_path, config)
        
        # Mock components
        engine.collection = Mock()
        engine.collection.count.return_value = 5
        
        engine.graph_manager = Mock()
        engine.graph_manager.get_stats.return_value = {
            'entities': 10,
            'relationships': 15
        }
        
        stats = await engine.get_ingestion_stats()
        
        assert stats['total_documents'] == 5
        assert stats['total_entities'] == 10
        assert stats['total_relationships'] == 15
        assert 'database_size' in stats
    
    async def test_cleanup(self, temp_db_path):
        """Test engine cleanup"""
        config = AsyncIngestionConfig(use_process_pool=False)
        engine = AsyncIngestionEngine(temp_db_path, config)
        
        await engine.initialize()
        executor = engine.executor
        
        await engine.cleanup()
        
        # Executor should be shut down
        assert executor._shutdown == True

class TestAsyncIngestionIntegration:
    """Integration tests for async ingestion"""
    
    @pytest.mark.integration
    async def test_process_documents_empty_list(self, temp_db_path):
        """Test processing empty document list"""
        config = AsyncIngestionConfig(use_process_pool=False)
        engine = AsyncIngestionEngine(temp_db_path, config)
        await engine.initialize()
        
        result = await engine.process_documents_async([])
        
        assert result.success_count == 0
        assert result.error_count == 0
        assert result.skipped_count == 0
        
        await engine.cleanup()
    
    @pytest.mark.integration  
    @patch('cortex_engine.async_ingest.get_document_content')
    async def test_process_documents_success(self, mock_get_content, temp_db_path, create_test_files):
        """Test successful document processing"""
        config = AsyncIngestionConfig(use_process_pool=False, max_concurrent_files=1)
        engine = AsyncIngestionEngine(temp_db_path, config)
        await engine.initialize()
        
        # Mock content extraction
        mock_get_content.return_value = {
            'content': 'Test document content',
            'metadata': {}
        }
        
        # Mock entity extractor
        engine.entity_extractor = Mock()
        engine.entity_extractor.extract_entities_from_text.return_value = {
            'entities': [],
            'relationships': []
        }
        
        # Mock graph manager
        engine.graph_manager = Mock()
        
        # Mock collection
        engine.collection = Mock()
        
        file_paths = list(create_test_files.values())[:2]
        result = await engine.process_documents_async(file_paths)
        
        assert result.success_count == 2
        assert result.error_count == 0
        assert len(result.processed_files) == 2
        
        await engine.cleanup()
    
    @pytest.mark.integration
    async def test_process_documents_with_progress_callback(self, temp_db_path, create_test_files):
        """Test document processing with progress callback"""
        config = AsyncIngestionConfig(use_process_pool=False)
        engine = AsyncIngestionEngine(temp_db_path, config)
        await engine.initialize()
        
        # Mock components for successful processing
        engine.entity_extractor = Mock()
        engine.entity_extractor.extract_entities_from_text.return_value = {
            'entities': [],
            'relationships': []
        }
        engine.graph_manager = Mock()
        engine.collection = Mock()
        
        progress_calls = []
        
        async def mock_progress_callback(progress, result):
            progress_calls.append((progress, result.success_count + result.error_count))
        
        with patch('cortex_engine.async_ingest.get_document_content') as mock_get_content:
            mock_get_content.return_value = {
                'content': 'Test content',
                'metadata': {}
            }
            
            file_paths = list(create_test_files.values())[:2]
            result = await engine.process_documents_async(
                file_paths, 
                progress_callback=mock_progress_callback
            )
        
        # Check that progress callback was called
        assert len(progress_calls) == 2
        assert progress_calls[-1][0] == 1.0  # Final progress should be 100%
        
        await engine.cleanup()

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.mark.integration
    @patch('cortex_engine.async_ingest.AsyncIngestionEngine')
    async def test_ingest_documents_async(self, mock_engine_class, temp_db_path):
        """Test high-level ingest_documents_async function"""
        # Mock engine instance
        mock_engine = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine.process_documents_async = AsyncMock()
        mock_engine.cleanup = AsyncMock()
        mock_engine_class.return_value = mock_engine
        
        # Mock result
        expected_result = AsyncIngestionResult(success_count=1)
        mock_engine.process_documents_async.return_value = expected_result
        
        result = await ingest_documents_async(temp_db_path, ["test.txt"])
        
        assert result == expected_result
        mock_engine.initialize.assert_called_once()
        mock_engine.process_documents_async.assert_called_once()
        mock_engine.cleanup.assert_called_once()
    
    async def test_async_progress_callback(self, caplog):
        """Test default async progress callback"""
        from cortex_engine.async_ingest import async_progress_callback
        
        result = AsyncIngestionResult(success_count=5, error_count=1, skipped_count=2)
        
        await async_progress_callback(0.75, result)
        
        # Check that log message was created
        assert "Progress: 75.0%" in caplog.text
        assert "Success: 5" in caplog.text