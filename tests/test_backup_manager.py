# ## File: tests/test_backup_manager.py
# Version: 1.0.0
# Date: 2025-07-27
# Purpose: Unit tests for backup and restore functionality.

import pytest
import asyncio
import os
import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from cortex_engine.backup_manager import (
    BackupManager,
    BackupMetadata,
    RestoreMetadata,
    create_knowledge_base_backup,
    restore_knowledge_base_backup
)

pytestmark = pytest.mark.asyncio

class TestBackupMetadata:
    """Test BackupMetadata dataclass"""
    
    def test_backup_metadata_creation(self):
        metadata = BackupMetadata(
            backup_id="test_backup",
            backup_type="full",
            creation_time="2025-07-27T10:00:00Z",
            source_path="/test/source",
            backup_path="/test/backup.tar.gz",
            file_count=10,
            total_size=1024,
            compression="gzip",
            checksum="abcdef123456"
        )
        
        assert metadata.backup_id == "test_backup"
        assert metadata.backup_type == "full"
        assert metadata.file_count == 10
        assert metadata.version == "1.0.0"  # Default value
        assert metadata.includes_images == True  # Default value

class TestRestoreMetadata:
    """Test RestoreMetadata dataclass"""
    
    def test_restore_metadata_creation(self):
        metadata = RestoreMetadata(
            restore_id="restore_123",
            backup_id="backup_123",
            restore_time="2025-07-27T10:00:00Z",
            target_path="/test/target",
            files_restored=5,
            errors=[],
            success=True
        )
        
        assert metadata.restore_id == "restore_123"
        assert metadata.backup_id == "backup_123"
        assert metadata.files_restored == 5
        assert metadata.success == True

class TestBackupManager:
    """Test BackupManager functionality"""
    
    def test_backup_manager_initialization(self, temp_db_path):
        """Test backup manager initialization"""
        manager = BackupManager(temp_db_path)
        
        assert str(manager.db_path) == temp_db_path
        assert manager.backup_dir == Path(temp_db_path) / "backups"
        assert manager.backup_metadata_file == Path(temp_db_path) / "backup_metadata.json"
        
        # Check that backup directory was created
        assert manager.backup_dir.exists()
    
    def test_core_paths_configuration(self, temp_db_path):
        """Test that core paths are properly configured"""
        manager = BackupManager(temp_db_path)
        
        assert "knowledge_hub_db" in manager.core_paths
        assert "knowledge_cortex.gpickle" in manager.core_paths
        assert "backup_metadata.json" in manager.core_paths
        assert "ingested_files.log" in manager.core_paths
    
    async def test_load_processed_files_async_empty(self, temp_db_path):
        """Test loading empty processed files log"""
        manager = BackupManager(temp_db_path)
        
        non_existent_path = os.path.join(temp_db_path, "non_existent.log")
        result = await manager._load_processed_files_async(non_existent_path)
        
        assert result == {}
    
    async def test_load_processed_files_async_with_data(self, temp_db_path):
        """Test loading processed files log with data"""
        manager = BackupManager(temp_db_path)
        
        log_path = os.path.join(temp_db_path, "test.log")
        test_data = {"file1.txt": "hash1", "file2.txt": "hash2"}
        
        # Create test log file
        with open(log_path, 'w') as f:
            json.dump(test_data, f)
        
        result = await manager._load_processed_files_async(log_path)
        assert result == test_data
    
    async def test_calculate_file_checksum(self, temp_db_path):
        """Test file checksum calculation"""
        manager = BackupManager(temp_db_path)
        
        # Create a test file
        test_file = Path(temp_db_path) / "test.txt"
        test_content = "test content for checksum"
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        checksum = await manager._calculate_file_checksum(test_file)
        
        # Verify checksum is a valid SHA256 hash
        assert len(checksum) == 64
        assert all(c in '0123456789abcdef' for c in checksum)
    
    async def test_has_existing_files_empty(self, temp_db_path):
        """Test has_existing_files with empty directory"""
        manager = BackupManager(temp_db_path)
        
        empty_dir = Path(temp_db_path) / "empty"
        empty_dir.mkdir()
        
        has_files = await manager._has_existing_files(empty_dir)
        assert has_files == False
    
    async def test_has_existing_files_with_core_files(self, temp_db_path):
        """Test has_existing_files with core files present"""
        manager = BackupManager(temp_db_path)
        
        # Create a core file
        test_dir = Path(temp_db_path) / "test_target"
        test_dir.mkdir()
        (test_dir / "knowledge_hub_db").mkdir()
        
        has_files = await manager._has_existing_files(test_dir)
        assert has_files == True
    
    async def test_get_files_to_backup_full(self, temp_db_path):
        """Test getting files for full backup"""
        manager = BackupManager(temp_db_path)
        
        # Create some test files
        (Path(temp_db_path) / "knowledge_hub_db").mkdir()
        (Path(temp_db_path) / "knowledge_hub_db" / "test.txt").write_text("test")
        (Path(temp_db_path) / "knowledge_cortex.gpickle").write_text("graph")
        
        files = await manager._get_files_to_backup("full", True)
        
        assert len(files) >= 2
        file_names = [f.name for f in files]
        assert "test.txt" in file_names
        assert "knowledge_cortex.gpickle" in file_names
    
    async def test_get_files_to_backup_exclude_images(self, temp_db_path):
        """Test getting files for backup excluding images"""
        manager = BackupManager(temp_db_path)
        
        # Create test files including images
        knowledge_dir = Path(temp_db_path) / "knowledge_hub_db"
        knowledge_dir.mkdir()
        images_dir = knowledge_dir / "images"
        images_dir.mkdir()
        
        (knowledge_dir / "test.txt").write_text("test")
        (images_dir / "image.png").write_text("fake image")
        
        files_with_images = await manager._get_files_to_backup("full", True)
        files_without_images = await manager._get_files_to_backup("full", False)
        
        # With images should have more files
        assert len(files_with_images) > len(files_without_images)
    
    async def test_create_backup_archive(self, temp_db_path):
        """Test creating backup archive"""
        manager = BackupManager(temp_db_path)
        
        # Create test files
        test_file1 = Path(temp_db_path) / "test1.txt"
        test_file2 = Path(temp_db_path) / "test2.txt"
        test_file1.write_text("content1")
        test_file2.write_text("content2")
        
        files = [test_file1, test_file2]
        archive_path = Path(temp_db_path) / "test_backup.tar"
        
        total_size = await manager._create_backup_archive(files, archive_path, False)
        
        assert archive_path.exists()
        assert total_size > 0
        
        # Verify archive contents
        with tarfile.open(archive_path, 'r') as tar:
            members = tar.getnames()
            assert "test1.txt" in members
            assert "test2.txt" in members
    
    async def test_create_backup_archive_compressed(self, temp_db_path):
        """Test creating compressed backup archive"""
        manager = BackupManager(temp_db_path)
        
        # Create test file
        test_file = Path(temp_db_path) / "test.txt"
        test_file.write_text("test content" * 100)  # Larger content for compression
        
        files = [test_file]
        archive_path = Path(temp_db_path) / "test_backup.tar.gz"
        
        total_size = await manager._create_backup_archive(files, archive_path, True)
        
        assert archive_path.exists()
        assert total_size > 0
        
        # Compressed file should be smaller than uncompressed
        uncompressed_archive = Path(temp_db_path) / "test_backup_uncompressed.tar"
        await manager._create_backup_archive(files, uncompressed_archive, False)
        
        assert archive_path.stat().st_size < uncompressed_archive.stat().st_size
    
    async def test_extract_backup_archive(self, temp_db_path):
        """Test extracting backup archive"""
        manager = BackupManager(temp_db_path)
        
        # Create and populate archive
        archive_path = Path(temp_db_path) / "test.tar"
        with tarfile.open(archive_path, 'w') as tar:
            # Create a temporary file to add to archive
            temp_file = Path(temp_db_path) / "temp.txt"
            temp_file.write_text("test content")
            tar.add(temp_file, arcname="restored.txt")
            temp_file.unlink()  # Remove temp file
        
        # Extract to target directory
        target_dir = Path(temp_db_path) / "extract_target"
        target_dir.mkdir()
        
        files_restored, errors = await manager._extract_backup_archive(
            str(archive_path), target_dir, True
        )
        
        assert files_restored == 1
        assert len(errors) == 0
        assert (target_dir / "restored.txt").exists()
        assert (target_dir / "restored.txt").read_text() == "test content"
    
    async def test_extract_backup_archive_no_overwrite(self, temp_db_path):
        """Test extracting archive without overwriting existing files"""
        manager = BackupManager(temp_db_path)
        
        # Create archive
        archive_path = Path(temp_db_path) / "test.tar"
        with tarfile.open(archive_path, 'w') as tar:
            temp_file = Path(temp_db_path) / "temp.txt"
            temp_file.write_text("archive content")
            tar.add(temp_file, arcname="existing.txt")
            temp_file.unlink()
        
        # Create target directory with existing file
        target_dir = Path(temp_db_path) / "extract_target"
        target_dir.mkdir()
        existing_file = target_dir / "existing.txt"
        existing_file.write_text("original content")
        
        files_restored, errors = await manager._extract_backup_archive(
            str(archive_path), target_dir, False  # Don't overwrite
        )
        
        assert files_restored == 0
        assert len(errors) == 1
        assert "exists and overwrite disabled" in errors[0]
        assert existing_file.read_text() == "original content"  # Unchanged
    
    async def test_save_and_load_backup_metadata(self, temp_db_path):
        """Test saving and loading backup metadata"""
        manager = BackupManager(temp_db_path)
        
        metadata = BackupMetadata(
            backup_id="test_backup",
            backup_type="full",
            creation_time="2025-07-27T10:00:00Z",
            source_path=temp_db_path,
            backup_path="/test/backup.tar",
            file_count=5,
            total_size=1024,
            compression="none",
            checksum="test_checksum"
        )
        
        # Save metadata
        await manager._save_backup_metadata(metadata)
        
        # Load metadata
        loaded_metadata = await manager._load_backup_metadata("test_backup")
        
        assert loaded_metadata is not None
        assert loaded_metadata.backup_id == "test_backup"
        assert loaded_metadata.backup_type == "full"
        assert loaded_metadata.file_count == 5
    
    async def test_list_backups_empty(self, temp_db_path):
        """Test listing backups when none exist"""
        manager = BackupManager(temp_db_path)
        
        backups = await manager.list_backups()
        assert backups == []
    
    async def test_list_backups_with_data(self, temp_db_path):
        """Test listing backups with existing data"""
        manager = BackupManager(temp_db_path)
        
        # Create test metadata file
        metadata_list = [
            {
                "backup_id": "backup1",
                "backup_type": "full",
                "creation_time": "2025-07-27T10:00:00Z",
                "source_path": temp_db_path,
                "backup_path": "/test/backup1.tar",
                "file_count": 5,
                "total_size": 1024,
                "compression": "none",
                "checksum": "checksum1",
                "version": "1.0.0",
                "includes_images": True,
                "parent_backup_id": None
            }
        ]
        
        with open(manager.backup_metadata_file, 'w') as f:
            json.dump(metadata_list, f)
        
        backups = await manager.list_backups()
        
        assert len(backups) == 1
        assert backups[0].backup_id == "backup1"
        assert backups[0].backup_type == "full"

class TestBackupOperations:
    """Test full backup operations"""
    
    async def test_create_backup_async_validation_error(self, temp_db_path):
        """Test backup creation with validation error"""
        manager = BackupManager(temp_db_path)
        
        with pytest.raises(Exception) as exc_info:
            await manager.create_backup_async(backup_type="invalid_type")
        
        assert "Invalid backup type" in str(exc_info.value)
    
    @patch('cortex_engine.backup_manager.BackupManager._get_files_to_backup')
    @patch('cortex_engine.backup_manager.BackupManager._create_backup_archive')
    @patch('cortex_engine.backup_manager.BackupManager._calculate_file_checksum')
    async def test_create_backup_async_success(self, mock_checksum, mock_archive, mock_files, temp_db_path):
        """Test successful backup creation"""
        manager = BackupManager(temp_db_path)
        
        # Mock the helper methods
        test_files = [Path(temp_db_path) / "test.txt"]
        mock_files.return_value = test_files
        mock_archive.return_value = 1024
        mock_checksum.return_value = "test_checksum"
        
        metadata = await manager.create_backup_async(
            backup_name="test_backup",
            description="Test backup"
        )
        
        assert metadata.backup_id == "test_backup"
        assert metadata.backup_type == "full"
        assert metadata.file_count == 1
        assert metadata.total_size == 1024
        assert metadata.checksum == "test_checksum"
        assert metadata.description == "Test backup"
        
        # Verify methods were called
        mock_files.assert_called_once()
        mock_archive.assert_called_once()
        mock_checksum.assert_called_once()
    
    async def test_restore_backup_async_not_found(self, temp_db_path):
        """Test restore with non-existent backup"""
        manager = BackupManager(temp_db_path)
        
        with pytest.raises(Exception) as exc_info:
            await manager.restore_backup_async("non_existent_backup")
        
        assert "Backup not found" in str(exc_info.value)
    
    async def test_delete_backup_success(self, temp_db_path):
        """Test successful backup deletion"""
        manager = BackupManager(temp_db_path)
        
        # Create a test backup file and metadata
        backup_dir = manager.backup_dir / "test_backup"
        backup_dir.mkdir()
        backup_file = backup_dir / "test_backup.tar"
        backup_file.write_text("fake backup")
        
        # Create metadata
        metadata = BackupMetadata(
            backup_id="test_backup",
            backup_type="full",
            creation_time="2025-07-27T10:00:00Z",
            source_path=temp_db_path,
            backup_path=str(backup_file),
            file_count=1,
            total_size=100,
            compression="none",
            checksum="test"
        )
        await manager._save_backup_metadata(metadata)
        
        # Delete the backup
        success = await manager.delete_backup("test_backup")
        
        assert success == True
        assert not backup_file.exists()
        
        # Verify metadata was updated
        remaining_backups = await manager.list_backups()
        assert len(remaining_backups) == 0
    
    async def test_delete_backup_not_found(self, temp_db_path):
        """Test deleting non-existent backup"""
        manager = BackupManager(temp_db_path)
        
        success = await manager.delete_backup("non_existent")
        assert success == False
    
    async def test_get_backup_info(self, temp_db_path):
        """Test getting backup information"""
        manager = BackupManager(temp_db_path)
        
        # Create test metadata
        metadata = BackupMetadata(
            backup_id="test_backup",
            backup_type="full",
            creation_time="2025-07-27T10:00:00Z",
            source_path=temp_db_path,
            backup_path="/test/backup.tar",
            file_count=5,
            total_size=1024,
            compression="gzip",
            checksum="test_checksum"
        )
        await manager._save_backup_metadata(metadata)
        
        info = await manager.get_backup_info("test_backup")
        
        assert info is not None
        assert info.backup_id == "test_backup"
        assert info.file_count == 5
    
    async def test_get_backup_info_not_found(self, temp_db_path):
        """Test getting info for non-existent backup"""
        manager = BackupManager(temp_db_path)
        
        info = await manager.get_backup_info("non_existent")
        assert info is None
    
    async def test_cleanup_old_backups(self, temp_db_path):
        """Test cleaning up old backups"""
        manager = BackupManager(temp_db_path)
        
        # Create multiple backup metadata entries
        for i in range(5):
            metadata = BackupMetadata(
                backup_id=f"backup_{i}",
                backup_type="full",
                creation_time=f"2025-07-{27-i:02d}T10:00:00Z",  # Different dates
                source_path=temp_db_path,
                backup_path=f"/test/backup_{i}.tar",
                file_count=1,
                total_size=100,
                compression="none",
                checksum=f"checksum_{i}"
            )
            await manager._save_backup_metadata(metadata)
        
        # Cleanup, keeping only 3 backups
        deleted_count = await manager.cleanup_old_backups(keep_count=3)
        
        assert deleted_count == 2  # Should delete 2 oldest backups
        
        remaining_backups = await manager.list_backups()
        assert len(remaining_backups) == 3

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @patch('cortex_engine.backup_manager.BackupManager')
    async def test_create_knowledge_base_backup(self, mock_manager_class, temp_db_path):
        """Test high-level backup creation function"""
        # Mock manager instance
        mock_manager = AsyncMock()
        mock_metadata = BackupMetadata(
            backup_id="test",
            backup_type="full", 
            creation_time="2025-07-27T10:00:00Z",
            source_path=temp_db_path,
            backup_path="/test.tar",
            file_count=1,
            total_size=100,
            compression="gzip",
            checksum="test"
        )
        mock_manager.create_backup_async.return_value = mock_metadata
        mock_manager_class.return_value = mock_manager
        
        result = await create_knowledge_base_backup(temp_db_path, "test_backup")
        
        assert result == mock_metadata
        mock_manager.create_backup_async.assert_called_once_with(
            backup_name="test_backup",
            backup_type="full",
            include_images=True,
            compress=True
        )
    
    @patch('cortex_engine.backup_manager.BackupManager')
    async def test_restore_knowledge_base_backup(self, mock_manager_class, temp_db_path):
        """Test high-level backup restore function"""
        # Mock manager instance
        mock_manager = AsyncMock()
        mock_restore_metadata = RestoreMetadata(
            restore_id="restore_123",
            backup_id="backup_123",
            restore_time="2025-07-27T10:00:00Z",
            target_path=temp_db_path,
            files_restored=5,
            errors=[],
            success=True
        )
        mock_manager.restore_backup_async.return_value = mock_restore_metadata
        mock_manager_class.return_value = mock_manager
        
        result = await restore_knowledge_base_backup(temp_db_path, "backup_123")
        
        assert result == mock_restore_metadata
        mock_manager.restore_backup_async.assert_called_once_with(
            backup_id="backup_123",
            overwrite_existing=False
        )