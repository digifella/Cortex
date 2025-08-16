# ## File: backup_manager.py
# Version: 1.0.0
# Date: 2025-07-27
# Purpose: Comprehensive backup and restore system for Cortex Suite knowledge base.
#          Supports full database backup, incremental backups, and point-in-time recovery.

import os
import shutil
import json
import tarfile
import hashlib
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import logging
from dataclasses import dataclass, asdict
import tempfile

from .utils.logging_utils import get_logger
from .utils.file_utils import get_file_hash
from .exceptions import *

logger = get_logger(__name__)

@dataclass
class BackupMetadata:
    """Metadata for backup operations"""
    backup_id: str
    backup_type: str  # 'full', 'incremental', 'differential'
    creation_time: str
    source_path: str
    backup_path: str
    file_count: int
    total_size: int
    compression: str
    checksum: str
    version: str = "1.0.0"
    description: Optional[str] = None
    includes_images: bool = True
    parent_backup_id: Optional[str] = None  # For incremental backups

@dataclass
class RestoreMetadata:
    """Metadata for restore operations"""
    restore_id: str
    backup_id: str
    restore_time: str
    target_path: str
    files_restored: int
    errors: List[str]
    success: bool

class BackupManager:
    """Comprehensive backup and restore manager for knowledge base"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.backup_metadata_file = self.db_path / "backup_metadata.json"
        self.restore_log_file = self.db_path / "restore_log.json"
        
        # Ensure backup directory structure
        self.backup_dir = self.db_path / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Core paths to backup
        self.core_paths = [
            "knowledge_hub_db",      # ChromaDB vector store
            "knowledge_cortex.gpickle",  # Knowledge graph
            "backup_metadata.json",   # Backup metadata
            "ingested_files.log"     # Processed files log
        ]
        
        # Optional paths
        self.optional_paths = [
            "images",               # Image store (if exists)
            "temp_uploads",         # Temporary uploads (usually excluded)
        ]
    
    async def create_backup_async(self,
                                backup_name: Optional[str] = None,
                                backup_type: str = "full",
                                include_images: bool = True,
                                compress: bool = True,
                                description: Optional[str] = None,
                                parent_backup_id: Optional[str] = None) -> BackupMetadata:
        """
        Create a comprehensive backup of the knowledge base
        
        Args:
            backup_name: Optional custom backup name
            backup_type: Type of backup ('full', 'incremental', 'differential')
            include_images: Whether to include image files
            compress: Whether to compress the backup
            description: Optional backup description
            parent_backup_id: Parent backup ID for incremental backups
            
        Returns:
            BackupMetadata object with backup information
        """
        try:
            backup_id = backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"📦 Starting {backup_type} backup: {backup_id}")
            
            # Validate backup type
            if backup_type not in ['full', 'incremental', 'differential']:
                raise ValidationError(f"Invalid backup type: {backup_type}")
            
            # Create backup directory
            backup_path = self.backup_dir / backup_id
            backup_path.mkdir(exist_ok=True)
            
            # Determine files to backup
            files_to_backup = await self._get_files_to_backup(
                backup_type, include_images, parent_backup_id
            )
            
            if not files_to_backup:
                raise ValidationError("No files found to backup")
            
            # Create backup archive
            archive_path = backup_path / f"{backup_id}.tar"
            if compress:
                archive_path = backup_path / f"{backup_id}.tar.gz"
            
            total_size = await self._create_backup_archive(
                files_to_backup, archive_path, compress
            )
            
            # Generate checksum
            checksum = await self._calculate_file_checksum(archive_path)
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                creation_time=datetime.now(timezone.utc).isoformat(),
                source_path=str(self.db_path),
                backup_path=str(archive_path),
                file_count=len(files_to_backup),
                total_size=total_size,
                compression="gzip" if compress else "none",
                checksum=checksum,
                description=description,
                includes_images=include_images,
                parent_backup_id=parent_backup_id
            )
            
            # Save metadata
            await self._save_backup_metadata(metadata)
            
            logger.info(f"✅ Backup completed: {backup_id} ({total_size:,} bytes)")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            raise CortexException(f"Backup operation failed: {str(e)}")
    
    async def restore_backup_async(self,
                                 backup_id: str,
                                 target_path: Optional[str] = None,
                                 overwrite_existing: bool = False,
                                 verify_checksum: bool = True) -> RestoreMetadata:
        """
        Restore a backup to the knowledge base
        
        Args:
            backup_id: ID of the backup to restore
            target_path: Optional custom target path (defaults to original db_path)
            overwrite_existing: Whether to overwrite existing files
            verify_checksum: Whether to verify backup integrity
            
        Returns:
            RestoreMetadata object with restore information
        """
        try:
            restore_id = f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"📥 Starting restore: {backup_id} -> {restore_id}")
            
            # Load backup metadata
            backup_metadata = await self._load_backup_metadata(backup_id)
            if not backup_metadata:
                raise ValidationError(f"Backup not found: {backup_id}")
            
            # Verify backup integrity
            if verify_checksum:
                await self._verify_backup_integrity(backup_metadata)
            
            # Prepare target path
            target = Path(target_path) if target_path else self.db_path
            target.mkdir(parents=True, exist_ok=True)
            
            # Check for existing files
            if not overwrite_existing and await self._has_existing_files(target):
                raise ValidationError(
                    "Target directory contains existing files. Use overwrite_existing=True to proceed."
                )
            
            # Perform restore
            files_restored, errors = await self._extract_backup_archive(
                backup_metadata.backup_path, target, overwrite_existing
            )
            
            # Create restore metadata
            restore_metadata = RestoreMetadata(
                restore_id=restore_id,
                backup_id=backup_id,
                restore_time=datetime.now(timezone.utc).isoformat(),
                target_path=str(target),
                files_restored=files_restored,
                errors=errors,
                success=len(errors) == 0
            )
            
            # Save restore log
            await self._save_restore_log(restore_metadata)
            
            if restore_metadata.success:
                logger.info(f"✅ Restore completed: {files_restored} files restored")
            else:
                logger.warning(f"⚠️ Restore completed with {len(errors)} errors")
            
            return restore_metadata
            
        except Exception as e:
            logger.error(f"❌ Restore failed: {e}")
            raise CortexException(f"Restore operation failed: {str(e)}")
    
    async def list_backups(self) -> List[BackupMetadata]:
        """List all available backups with metadata"""
        try:
            if not self.backup_metadata_file.exists():
                return []
            
            async with aiofiles.open(self.backup_metadata_file, 'r') as f:
                content = await f.read()
                metadata_list = json.loads(content) if content.strip() else []
            
            return [BackupMetadata(**meta) for meta in metadata_list]
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup and its metadata"""
        try:
            logger.info(f"🗑️ Deleting backup: {backup_id}")
            
            # Load current metadata
            metadata_list = await self.list_backups()
            backup_to_delete = None
            
            for metadata in metadata_list:
                if metadata.backup_id == backup_id:
                    backup_to_delete = metadata
                    break
            
            if not backup_to_delete:
                raise ValidationError(f"Backup not found: {backup_id}")
            
            # Delete backup files
            backup_path = Path(backup_to_delete.backup_path)
            if backup_path.exists():
                backup_path.unlink()
            
            # Delete backup directory if empty
            backup_dir = backup_path.parent
            if backup_dir.exists() and not any(backup_dir.iterdir()):
                backup_dir.rmdir()
            
            # Update metadata
            updated_metadata = [m for m in metadata_list if m.backup_id != backup_id]
            await self._save_backup_metadata_list(updated_metadata)
            
            logger.info(f"✅ Backup deleted: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting backup: {e}")
            return False
    
    async def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get detailed information about a specific backup"""
        metadata_list = await self.list_backups()
        for metadata in metadata_list:
            if metadata.backup_id == backup_id:
                return metadata
        return None
    
    async def verify_backup_integrity(self, backup_id: str) -> bool:
        """Verify the integrity of a backup"""
        try:
            backup_metadata = await self._load_backup_metadata(backup_id)
            if not backup_metadata:
                return False
            
            await self._verify_backup_integrity(backup_metadata)
            return True
            
        except Exception as e:
            logger.error(f"Backup integrity check failed: {e}")
            return False
    
    async def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Clean up old backups, keeping only the most recent ones"""
        try:
            metadata_list = await self.list_backups()
            
            # Sort by creation time (newest first)
            metadata_list.sort(
                key=lambda x: x.creation_time, 
                reverse=True
            )
            
            # Delete old backups
            deleted_count = 0
            for metadata in metadata_list[keep_count:]:
                if await self.delete_backup(metadata.backup_id):
                    deleted_count += 1
            
            logger.info(f"🧹 Cleaned up {deleted_count} old backups")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
            return 0
    
    # Private helper methods
    
    async def _get_files_to_backup(self,
                                 backup_type: str,
                                 include_images: bool,
                                 parent_backup_id: Optional[str] = None) -> List[Path]:
        """Get list of files to include in backup"""
        files_to_backup = []
        
        # Always include core paths
        for core_path in self.core_paths:
            full_path = self.db_path / core_path
            if full_path.exists():
                if full_path.is_file():
                    files_to_backup.append(full_path)
                else:
                    # Add all files in directory
                    for file_path in full_path.rglob('*'):
                        if file_path.is_file():
                            files_to_backup.append(file_path)
        
        # Include images if requested
        if include_images:
            images_path = self.db_path / "knowledge_hub_db" / "images"
            if images_path.exists():
                for file_path in images_path.rglob('*'):
                    if file_path.is_file():
                        files_to_backup.append(file_path)
        
        # For incremental backups, filter by modification time
        if backup_type in ['incremental', 'differential'] and parent_backup_id:
            parent_metadata = await self._load_backup_metadata(parent_backup_id)
            if parent_metadata:
                parent_time = datetime.fromisoformat(parent_metadata.creation_time.replace('Z', '+00:00'))
                filtered_files = []
                
                for file_path in files_to_backup:
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                    if file_mtime > parent_time:
                        filtered_files.append(file_path)
                
                files_to_backup = filtered_files
        
        return files_to_backup
    
    async def _create_backup_archive(self,
                                   files: List[Path],
                                   archive_path: Path,
                                   compress: bool) -> int:
        """Create backup archive from file list"""
        mode = "w:gz" if compress else "w"
        total_size = 0
        
        with tarfile.open(archive_path, mode) as tar:
            for file_path in files:
                # Calculate relative path within db_path
                rel_path = file_path.relative_to(self.db_path)
                tar.add(file_path, arcname=rel_path)
                total_size += file_path.stat().st_size
        
        return total_size
    
    async def _extract_backup_archive(self,
                                    archive_path: str,
                                    target_path: Path,
                                    overwrite: bool) -> Tuple[int, List[str]]:
        """Extract backup archive to target path"""
        archive = Path(archive_path)
        files_restored = 0
        errors = []
        
        try:
            with tarfile.open(archive, 'r') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        target_file = target_path / member.name
                        
                        if target_file.exists() and not overwrite:
                            errors.append(f"File exists and overwrite disabled: {member.name}")
                            continue
                        
                        try:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            tar.extract(member, target_path)
                            files_restored += 1
                        except Exception as e:
                            errors.append(f"Failed to extract {member.name}: {str(e)}")
            
        except Exception as e:
            errors.append(f"Archive extraction failed: {str(e)}")
        
        return files_restored, errors
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while True:
                chunk = await f.read(8192)
                if not chunk:
                    break
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _verify_backup_integrity(self, metadata: BackupMetadata) -> None:
        """Verify backup file integrity using checksum"""
        backup_path = Path(metadata.backup_path)
        
        if not backup_path.exists():
            raise ValidationError(f"Backup file not found: {backup_path}")
        
        current_checksum = await self._calculate_file_checksum(backup_path)
        
        if current_checksum != metadata.checksum:
            raise ValidationError(f"Backup integrity check failed for {metadata.backup_id}")
    
    async def _has_existing_files(self, target_path: Path) -> bool:
        """Check if target path contains existing files"""
        for core_path in self.core_paths:
            full_path = target_path / core_path
            if full_path.exists():
                return True
        return False
    
    async def _save_backup_metadata(self, metadata: BackupMetadata) -> None:
        """Save backup metadata to file"""
        metadata_list = await self.list_backups()
        metadata_list.append(metadata)
        await self._save_backup_metadata_list(metadata_list)
    
    async def _save_backup_metadata_list(self, metadata_list: List[BackupMetadata]) -> None:
        """Save complete metadata list to file"""
        metadata_dicts = [asdict(meta) for meta in metadata_list]
        
        async with aiofiles.open(self.backup_metadata_file, 'w') as f:
            await f.write(json.dumps(metadata_dicts, indent=2))
    
    async def _load_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Load specific backup metadata"""
        metadata_list = await self.list_backups()
        for metadata in metadata_list:
            if metadata.backup_id == backup_id:
                return metadata
        return None
    
    async def _save_restore_log(self, restore_metadata: RestoreMetadata) -> None:
        """Save restore operation log"""
        restore_log = []
        
        if self.restore_log_file.exists():
            async with aiofiles.open(self.restore_log_file, 'r') as f:
                content = await f.read()
                restore_log = json.loads(content) if content.strip() else []
        
        restore_log.append(asdict(restore_metadata))
        
        async with aiofiles.open(self.restore_log_file, 'w') as f:
            await f.write(json.dumps(restore_log, indent=2))

# Convenience functions

async def create_knowledge_base_backup(db_path: str,
                                     backup_name: Optional[str] = None,
                                     include_images: bool = True,
                                     compress: bool = True) -> BackupMetadata:
    """
    High-level function to create a full knowledge base backup
    
    Args:
        db_path: Path to the knowledge database
        backup_name: Optional custom backup name
        include_images: Whether to include image files
        compress: Whether to compress the backup
        
    Returns:
        BackupMetadata object with backup information
    """
    backup_manager = BackupManager(db_path)
    return await backup_manager.create_backup_async(
        backup_name=backup_name,
        backup_type="full",
        include_images=include_images,
        compress=compress
    )

async def restore_knowledge_base_backup(db_path: str,
                                      backup_id: str,
                                      overwrite_existing: bool = False) -> RestoreMetadata:
    """
    High-level function to restore a knowledge base backup
    
    Args:
        db_path: Path to the knowledge database
        backup_id: ID of the backup to restore
        overwrite_existing: Whether to overwrite existing files
        
    Returns:
        RestoreMetadata object with restore information
    """
    backup_manager = BackupManager(db_path)
    return await backup_manager.restore_backup_async(
        backup_id=backup_id,
        overwrite_existing=overwrite_existing
    )