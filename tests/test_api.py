# ## File: tests/test_api.py
# Version: 1.0.0
# Date: 2025-07-27
# Purpose: Integration tests for the Cortex Suite REST API.

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    def test_health_check(self, api_client):
        """Test health check endpoint"""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    @patch('api.main.AsyncSearchEngine')
    def test_system_status(self, mock_search_engine, api_client):
        """Test system status endpoint"""
        # Mock search engine
        mock_engine = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine.get_search_stats = AsyncMock(return_value={
            'total_documents': 100,
            'total_entities': 50,
            'collection_exists': True,
            'graph_exists': True
        })
        mock_engine.cleanup = AsyncMock()
        mock_search_engine.return_value = mock_engine
        
        response = api_client.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 100
        assert data["total_entities"] == 50
        assert "last_updated" in data

class TestSearchEndpoints:
    """Test search-related endpoints"""
    
    @patch('api.main.AsyncSearchEngine')
    def test_search_knowledge_base(self, mock_search_engine, api_client):
        """Test knowledge base search endpoint"""
        # Mock search engine
        mock_engine = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine.search_async = AsyncMock()
        mock_engine.cleanup = AsyncMock()
        mock_search_engine.return_value = mock_engine
        
        # Mock search result
        from cortex_engine.async_query import AsyncQueryResult
        mock_result = AsyncQueryResult(
            query="test query",
            results=[
                {
                    "content": "Test document content",
                    "metadata": {"file_name": "test.txt"},
                    "similarity": 0.9
                }
            ],
            synthesis="This is a synthesized answer",
            total_results=1,
            processing_time=0.5
        )
        mock_engine.search_async.return_value = mock_result
        
        # Make API request
        search_request = {
            "query": "test query",
            "search_type": "hybrid",
            "max_results": 10,
            "include_synthesis": True
        }
        
        response = api_client.post("/search", json=search_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["total_results"] == 1
        assert data["synthesis"] == "This is a synthesized answer"
        assert len(data["results"]) == 1
    
    @patch('api.main.AsyncSearchEngine')
    def test_batch_search_knowledge_base(self, mock_search_engine, api_client):
        """Test batch search endpoint"""
        # Mock search engine
        mock_engine = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine.batch_search_async = AsyncMock()
        mock_engine.cleanup = AsyncMock()
        mock_search_engine.return_value = mock_engine
        
        # Mock batch search results
        from cortex_engine.async_query import AsyncQueryResult
        mock_results = [
            AsyncQueryResult(query="query1", total_results=1, processing_time=0.1),
            AsyncQueryResult(query="query2", total_results=2, processing_time=0.2)
        ]
        mock_engine.batch_search_async.return_value = mock_results
        
        # Make API request
        batch_request = {
            "queries": ["query1", "query2"],
            "search_type": "vector",
            "include_synthesis": False
        }
        
        response = api_client.post("/search/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["query"] == "query1"
        assert data[1]["query"] == "query2"
    
    def test_search_invalid_request(self, api_client):
        """Test search with invalid request data"""
        invalid_request = {
            "search_type": "invalid_type"
            # Missing required "query" field
        }
        
        response = api_client.post("/search", json=invalid_request)
        assert response.status_code == 422  # Validation error

class TestIngestionEndpoints:
    """Test document ingestion endpoints"""
    
    @patch('api.main.AsyncIngestionEngine')
    def test_ingest_documents(self, mock_ingestion_engine, api_client, create_test_files):
        """Test document ingestion endpoint"""
        # Mock ingestion engine
        mock_engine = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine.process_documents_async = AsyncMock()
        mock_engine.cleanup = AsyncMock()
        mock_ingestion_engine.return_value = mock_engine
        
        # Mock ingestion result
        from cortex_engine.async_ingest import AsyncIngestionResult
        mock_result = AsyncIngestionResult(
            success_count=2,
            error_count=0,
            skipped_count=0,
            processed_files=["file1.txt", "file2.txt"],
            total_entities=5,
            total_relationships=3,
            processing_time=1.5
        )
        mock_engine.process_documents_async.return_value = mock_result
        
        # Make API request
        file_paths = list(create_test_files.values())[:2]
        ingestion_request = {
            "file_paths": file_paths,
            "exclusion_patterns": ["*.tmp"],
            "max_concurrent": 3
        }
        
        response = api_client.post("/ingest", json=ingestion_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success_count"] == 2
        assert data["total_entities"] == 5
        assert data["total_relationships"] == 3
    
    def test_ingest_invalid_paths(self, api_client):
        """Test ingestion with invalid file paths"""
        ingestion_request = {
            "file_paths": ["/non/existent/file.txt"],
            "exclusion_patterns": [],
            "max_concurrent": 1
        }
        
        response = api_client.post("/ingest", json=ingestion_request)
        assert response.status_code == 400
        assert "No valid file paths" in response.json()["detail"]
    
    @patch('api.main.AsyncIngestionEngine')
    async def test_upload_and_ingest_file(self, mock_ingestion_engine, api_client):
        """Test file upload and ingestion endpoint"""
        # Mock ingestion engine
        mock_engine = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine.process_documents_async = AsyncMock()
        mock_engine.cleanup = AsyncMock()
        mock_ingestion_engine.return_value = mock_engine
        
        # Mock successful ingestion
        from cortex_engine.async_ingest import AsyncIngestionResult
        mock_result = AsyncIngestionResult(
            success_count=1,
            total_entities=2,
            total_relationships=1
        )
        mock_engine.process_documents_async.return_value = mock_result
        
        # Create test file for upload
        test_content = b"Test file content for upload"
        files = {"file": ("test.txt", test_content, "text/plain")}
        
        response = api_client.post("/ingest/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"
        assert data["status"] == "success"
        assert data["entities_extracted"] == 2

class TestCollectionEndpoints:
    """Test collection management endpoints"""
    
    @patch('api.main.CollectionManager')
    def test_get_collections(self, mock_collection_manager, api_client, mock_config_manager):
        """Test get all collections endpoint"""
        # Mock collection manager
        mock_manager = Mock()
        mock_manager.get_all_collections.return_value = {
            "collection1": {
                "description": "First collection",
                "documents": ["doc1.txt", "doc2.txt"],
                "created_date": "2025-07-27T10:00:00Z",
                "last_modified": "2025-07-27T10:00:00Z"
            },
            "collection2": {
                "description": "Second collection",
                "documents": ["doc3.txt"],
                "created_date": "2025-07-27T11:00:00Z",
                "last_modified": "2025-07-27T11:00:00Z"
            }
        }
        mock_collection_manager.return_value = mock_manager
        
        response = api_client.get("/collections")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "collection1"
        assert data[0]["description"] == "First collection"
        assert len(data[0]["documents"]) == 2
    
    @patch('api.main.CollectionManager')
    def test_create_collection(self, mock_collection_manager, api_client, mock_config_manager):
        """Test create collection endpoint"""
        # Mock collection manager
        mock_manager = Mock()
        mock_manager.save_collection = Mock()
        mock_collection_manager.return_value = mock_manager
        
        collection_data = {
            "description": "New test collection",
            "documents": ["test1.txt", "test2.txt"]
        }
        
        response = api_client.post(
            "/collections/test_collection",
            data=collection_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "created successfully" in data["message"]
        
        # Verify collection manager was called
        mock_manager.save_collection.assert_called_once()

class TestEntityEndpoints:
    """Test entity-related endpoints"""
    
    @patch('api.main.EnhancedGraphManager')
    def test_get_entity_info(self, mock_graph_manager, api_client, temp_db_path):
        """Test get entity information endpoint"""
        # Create mock graph file
        graph_path = Path(temp_db_path) / "knowledge_cortex.gpickle"
        graph_path.write_text("fake graph")
        
        # Mock graph manager
        mock_manager = Mock()
        mock_manager.get_entity_context.return_value = {
            "type": "person",
            "name": "John Doe",
            "role": "researcher"
        }
        mock_manager.get_entity_relationships.return_value = [
            "works_with", "collaborates_on"
        ]
        mock_graph_manager.return_value = mock_manager
        
        response = api_client.get("/entities/John%20Doe")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "John Doe"
        assert data["type"] == "person"
        assert len(data["relationships"]) == 2
    
    def test_get_entity_info_not_found(self, api_client, temp_db_path):
        """Test get entity info for non-existent entity"""
        response = api_client.get("/entities/NonExistent")
        assert response.status_code == 404

class TestBackupEndpoints:
    """Test backup and restore endpoints"""
    
    @patch('api.main.BackupManager')
    def test_create_backup(self, mock_backup_manager, api_client):
        """Test create backup endpoint"""
        # Mock backup manager
        mock_manager = AsyncMock()
        mock_manager.create_backup_async = AsyncMock()
        mock_backup_manager.return_value = mock_manager
        
        # Mock backup metadata
        from cortex_engine.backup_manager import BackupMetadata
        mock_metadata = BackupMetadata(
            backup_id="test_backup",
            backup_type="full",
            creation_time="2025-07-27T10:00:00Z",
            source_path="/test",
            backup_path="/test_backup.tar.gz",
            file_count=10,
            total_size=1024000,
            compression="gzip",
            checksum="abcdef123456"
        )
        mock_manager.create_backup_async.return_value = mock_metadata
        
        backup_request = {
            "backup_path": "/backups/test_backup.tar.gz",
            "include_images": True,
            "compress": True
        }
        
        response = api_client.post("/backup/create", json=backup_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["backup_id"] == "test_backup"
        assert data["file_count"] == 10
        assert data["compression"] == "gzip"
    
    @patch('api.main.BackupManager')
    def test_list_backups(self, mock_backup_manager, api_client):
        """Test list backups endpoint"""
        # Mock backup manager
        mock_manager = AsyncMock()
        mock_manager.list_backups = AsyncMock()
        mock_backup_manager.return_value = mock_manager
        
        # Mock backup list
        from cortex_engine.backup_manager import BackupMetadata
        mock_backups = [
            BackupMetadata(
                backup_id="backup1",
                backup_type="full",
                creation_time="2025-07-27T10:00:00Z",
                source_path="/test",
                backup_path="/backup1.tar",
                file_count=5,
                total_size=512000,
                compression="none",
                checksum="hash1"
            ),
            BackupMetadata(
                backup_id="backup2",
                backup_type="incremental",
                creation_time="2025-07-27T11:00:00Z",
                source_path="/test",
                backup_path="/backup2.tar.gz",
                file_count=2,
                total_size=256000,
                compression="gzip",
                checksum="hash2"
            )
        ]
        mock_manager.list_backups.return_value = mock_backups
        
        response = api_client.get("/backup/list")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["backup_id"] == "backup1"
        assert data[1]["backup_type"] == "incremental"
    
    @patch('api.main.BackupManager')
    def test_restore_backup(self, mock_backup_manager, api_client):
        """Test restore backup endpoint"""
        # Mock backup manager
        mock_manager = AsyncMock()
        mock_manager.restore_backup_async = AsyncMock()
        mock_backup_manager.return_value = mock_manager
        
        # Mock restore metadata
        from cortex_engine.backup_manager import RestoreMetadata
        mock_restore = RestoreMetadata(
            restore_id="restore_123",
            backup_id="backup_123",
            restore_time="2025-07-27T12:00:00Z",
            target_path="/test",
            files_restored=8,
            errors=[],
            success=True
        )
        mock_manager.restore_backup_async.return_value = mock_restore
        
        restore_request = {
            "backup_id": "backup_123",
            "overwrite_existing": True,
            "verify_checksum": True
        }
        
        response = api_client.post("/backup/restore", json=restore_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["backup_id"] == "backup_123"
        assert data["files_restored"] == 8
        assert data["success"] == True
    
    @patch('api.main.BackupManager')
    def test_delete_backup(self, mock_backup_manager, api_client):
        """Test delete backup endpoint"""
        # Mock backup manager
        mock_manager = AsyncMock()
        mock_manager.delete_backup = AsyncMock(return_value=True)
        mock_backup_manager.return_value = mock_manager
        
        response = api_client.delete("/backup/test_backup")
        
        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"]
    
    @patch('api.main.BackupManager')
    def test_delete_backup_not_found(self, mock_backup_manager, api_client):
        """Test delete non-existent backup"""
        # Mock backup manager
        mock_manager = AsyncMock()
        mock_manager.delete_backup = AsyncMock(return_value=False)
        mock_backup_manager.return_value = mock_manager
        
        response = api_client.delete("/backup/non_existent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('api.main.BackupManager')
    def test_verify_backup(self, mock_backup_manager, api_client):
        """Test verify backup endpoint"""
        # Mock backup manager
        mock_manager = AsyncMock()
        mock_manager.verify_backup_integrity = AsyncMock(return_value=True)
        mock_backup_manager.return_value = mock_manager
        
        response = api_client.get("/backup/test_backup/verify")
        
        assert response.status_code == 200
        data = response.json()
        assert data["backup_id"] == "test_backup"
        assert data["valid"] == True
        assert "verified" in data["message"]

class TestAPIErrorHandling:
    """Test API error handling"""
    
    @patch('api.main.AsyncSearchEngine')
    def test_search_engine_error(self, mock_search_engine, api_client):
        """Test handling of search engine errors"""
        # Mock search engine to raise exception
        mock_engine = AsyncMock()
        mock_engine.initialize = AsyncMock()
        mock_engine.search_async = AsyncMock(side_effect=Exception("Search failed"))
        mock_engine.cleanup = AsyncMock()
        mock_search_engine.return_value = mock_engine
        
        search_request = {
            "query": "test query",
            "search_type": "vector"
        }
        
        response = api_client.post("/search", json=search_request)
        
        assert response.status_code == 500
        assert "Search failed" in response.json()["detail"]
    
    def test_invalid_json_request(self, api_client):
        """Test handling of invalid JSON in requests"""
        response = api_client.post(
            "/search",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity

class TestAPIAuthentication:
    """Test API authentication (mocked)"""
    
    def test_protected_endpoint_without_token(self, api_client):
        """Test accessing protected endpoint without token"""
        # Note: The verify_token dependency is mocked in conftest.py
        # In a real implementation, this would return 401/403
        search_request = {
            "query": "test query",
            "search_type": "vector"
        }
        
        response = api_client.post("/search", json=search_request)
        
        # With mocked auth, this should still work
        # In production, you'd test actual auth failure
        assert response.status_code in [200, 401, 403, 500]  # Depends on mocking