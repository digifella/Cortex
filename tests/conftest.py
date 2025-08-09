# ## File: tests/conftest.py
# Version: 1.0.0
# Date: 2025-07-27
# Purpose: Pytest configuration and fixtures for Cortex Suite tests.

import pytest
import asyncio
import tempfile
import shutil
import os
import json
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, AsyncMock

import chromadb
from fastapi.testclient import TestClient

from cortex_engine.config_manager import ConfigManager
from cortex_engine.async_ingest import AsyncIngestionEngine, AsyncIngestionConfig
from cortex_engine.async_query import AsyncSearchEngine, AsyncQueryConfig
from cortex_engine.backup_manager import BackupManager
from cortex_engine.graph_manager import EnhancedGraphManager

# Test configuration
TEST_DB_PATH = None
TEST_CONFIG = {
    "ai_database_path": None,
    "last_db_path": None,
    "knowledge_source_path": "/tmp/test_knowledge",
    "last_docs_path": "/tmp/test_docs"
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Provide a temporary database path for testing"""
    global TEST_DB_PATH
    temp_dir = tempfile.mkdtemp(prefix="cortex_test_")
    TEST_DB_PATH = temp_dir
    
    # Update test config
    TEST_CONFIG["ai_database_path"] = temp_dir
    TEST_CONFIG["last_db_path"] = temp_dir
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    TEST_DB_PATH = None

@pytest.fixture
def mock_config_manager() -> ConfigManager:
    """Mock ConfigManager for testing"""
    config_manager = Mock(spec=ConfigManager)
    config_manager.get_config.return_value = TEST_CONFIG.copy()
    config_manager.save_config = Mock()
    return config_manager

@pytest.fixture
def sample_documents() -> Dict[str, str]:
    """Sample document content for testing"""
    return {
        "doc1.txt": "This is a sample document about artificial intelligence and machine learning. It discusses neural networks and deep learning algorithms.",
        "doc2.txt": "Project Cortex is a knowledge management system. It uses vector databases and graph structures for information retrieval.",
        "doc3.txt": "The research paper analyzes the impact of GraphRAG on enterprise knowledge systems. Authors: John Smith, Jane Doe.",
        "doc4.md": "# Technical Documentation\n\nThis document explains the system architecture and implementation details.",
        "image_desc.txt": "Image description: A diagram showing the system architecture with components connected by arrows."
    }

@pytest.fixture
def create_test_files(temp_db_path: str, sample_documents: Dict[str, str]) -> Generator[Dict[str, str], None, None]:
    """Create test files in temporary directory"""
    test_files = {}
    docs_dir = Path(temp_db_path) / "test_docs"
    docs_dir.mkdir(exist_ok=True)
    
    for filename, content in sample_documents.items():
        file_path = docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        test_files[filename] = str(file_path)
    
    yield test_files
    
    # Cleanup handled by temp_db_path fixture

@pytest.fixture
async def ingestion_engine(temp_db_path: str) -> AsyncIngestionEngine:
    """Create and initialize async ingestion engine for testing"""
    config = AsyncIngestionConfig(
        max_concurrent_files=2,
        max_concurrent_entities=5,
        use_process_pool=False  # Use threads for testing
    )
    
    engine = AsyncIngestionEngine(temp_db_path, config)
    await engine.initialize()
    
    yield engine
    
    await engine.cleanup()

@pytest.fixture
async def search_engine(temp_db_path: str) -> AsyncSearchEngine:
    """Create and initialize async search engine for testing"""
    config = AsyncQueryConfig(
        max_concurrent_queries=2,
        max_results_per_query=5,
        enable_graph_context=False,  # Disable for simpler testing
        enable_hybrid_search=False
    )
    
    engine = AsyncSearchEngine(temp_db_path, config)
    await engine.initialize()
    
    yield engine
    
    await engine.cleanup()

@pytest.fixture
def backup_manager(temp_db_path: str) -> BackupManager:
    """Create backup manager for testing"""
    return BackupManager(temp_db_path)

@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection for testing"""
    collection = Mock()
    collection.count.return_value = 10
    collection.query.return_value = {
        'documents': [['Sample document content', 'Another document']],
        'metadatas': [[{'file_name': 'test.txt'}, {'file_name': 'test2.txt'}]],
        'distances': [[0.1, 0.3]]
    }
    collection.add = Mock()
    return collection

@pytest.fixture
def mock_graph_manager():
    """Mock graph manager for testing"""
    graph_manager = Mock(spec=EnhancedGraphManager)
    graph_manager.get_stats.return_value = {
        'entities': 15,
        'relationships': 25,
        'nodes': 15,
        'edges': 25
    }
    graph_manager.find_related_entities.return_value = ['entity1', 'entity2']
    graph_manager.get_entity_context.return_value = {
        'type': 'person',
        'properties': {'name': 'John Doe'}
    }
    graph_manager.get_entity_relationships.return_value = ['knows', 'works_with']
    graph_manager.add_entity = Mock()
    graph_manager.add_relationship = Mock()
    graph_manager.save_graph = Mock()
    return graph_manager

@pytest.fixture
def api_client(temp_db_path: str, mock_config_manager: ConfigManager):
    """Create FastAPI test client"""
    # Import here to avoid circular imports
    from api.main import app, config_manager, db_path
    
    # Override global variables for testing
    app.dependency_overrides[lambda: config_manager] = lambda: mock_config_manager
    app.dependency_overrides[lambda: db_path] = lambda: temp_db_path
    
    with TestClient(app) as client:
        yield client
    
    # Clear overrides
    app.dependency_overrides.clear()

@pytest.fixture
def sample_entities():
    """Sample entity data for testing"""
    return [
        {
            'name': 'John Smith',
            'type': 'person',
            'properties': {'role': 'researcher'}
        },
        {
            'name': 'Jane Doe',
            'type': 'person',
            'properties': {'role': 'author'}
        },
        {
            'name': 'Project Cortex',
            'type': 'project',
            'properties': {'status': 'active'}
        }
    ]

@pytest.fixture
def sample_relationships():
    """Sample relationship data for testing"""
    return [
        {
            'source': 'John Smith',
            'target': 'Jane Doe',
            'relationship': 'collaborates_with'
        },
        {
            'source': 'John Smith',
            'target': 'Project Cortex',
            'relationship': 'works_on'
        }
    ]

# Test markers
pytest.mark.asyncio = pytest.mark.asyncio
pytest.mark.integration = pytest.mark.integration
pytest.mark.unit = pytest.mark.unit
pytest.mark.api = pytest.mark.api

# Async test utilities
async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
    """Wait for a condition to become true"""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return True
        await asyncio.sleep(interval)
    
    return False

def create_mock_file(path: str, content: str = "test content"):
    """Create a mock file for testing"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    return path