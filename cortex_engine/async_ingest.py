# ## File: async_ingest.py
# Version: 1.0.0
# Date: 2025-07-27
# Purpose: Async document ingestion module for high-performance batch processing.
#          Provides concurrent document processing, entity extraction, and knowledge graph updates.

import asyncio
import aiofiles
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
import hashlib
from datetime import datetime
import json

from pydantic import BaseModel, ValidationError
import chromadb

from .utils.logging_utils import get_logger
from .utils.file_utils import get_file_hash
from .ingest_cortex import (
    RichMetadata, DocumentMetadata,
    load_processed_files_log, get_document_content,
)
from .entity_extractor import EntityExtractor
from .graph_manager import EnhancedGraphManager
from .config import INGESTED_FILES_LOG, COLLECTION_NAME
from .embedding_service import embed_texts

logger = get_logger(__name__)

class AsyncIngestionConfig(BaseModel):
    """Configuration for async ingestion operations"""
    max_concurrent_files: int = 5
    max_concurrent_entities: int = 10
    chunk_size: int = 100
    use_process_pool: bool = True
    max_workers: Optional[int] = None

class AsyncIngestionResult(BaseModel):
    """Result of async ingestion operation"""
    success_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    processed_files: List[str] = []
    errors: List[Dict[str, str]] = []
    total_entities: int = 0
    total_relationships: int = 0
    processing_time: float = 0.0

class AsyncIngestionEngine:
    """High-performance async document ingestion engine"""
    
    def __init__(self, 
                 db_path: str, 
                 config: Optional[AsyncIngestionConfig] = None):
        self.db_path = db_path
        self.config = config or AsyncIngestionConfig()
        self.entity_extractor = None
        self.graph_manager = None
        self.chroma_client = None
        self.collection = None
        self.executor = None
        
    async def initialize(self) -> None:
        """Initialize async components and connections"""
        try:
            logger.info("🚀 Initializing async ingestion engine...")
            
            # Initialize thread/process pools
            if self.config.use_process_pool:
                self.executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
            else:
                self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            
            # Initialize components in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._initialize_sync_components)
            
            logger.info("✅ Async ingestion engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize async ingestion engine: {e}")
            raise
    
    def _initialize_sync_components(self) -> None:
        """Initialize synchronous components (run in executor)"""
        # Initialize entity extractor
        self.entity_extractor = EntityExtractor()
        
        # Initialize graph manager
        graph_path = os.path.join(self.db_path, "knowledge_cortex.gpickle")
        self.graph_manager = EnhancedGraphManager(graph_path)
        
        # Initialize ChromaDB
        chroma_path = os.path.join(self.db_path, "knowledge_hub_db")
        os.makedirs(chroma_path, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
        except ValueError as e:
            logger.info(f"Collection {COLLECTION_NAME} not found, creating new one: {e}")
            self.collection = self.chroma_client.create_collection(COLLECTION_NAME)
        except Exception as e:
            logger.error(f"Failed to get or create collection {COLLECTION_NAME}: {e}")
            raise RuntimeError(f"Cannot initialize ChromaDB collection: {e}") from e
    
    async def process_documents_async(self, 
                                    file_paths: List[str],
                                    exclusion_patterns: List[str] = None,
                                    progress_callback: Optional[callable] = None) -> AsyncIngestionResult:
        """
        Process multiple documents concurrently with async I/O
        
        Args:
            file_paths: List of file paths to process
            exclusion_patterns: Patterns to exclude during processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            AsyncIngestionResult with processing statistics
        """
        start_time = datetime.now()
        result = AsyncIngestionResult()
        
        try:
            logger.info(f"🔄 Starting async processing of {len(file_paths)} documents")
            
            # Load processed files log
            log_path = os.path.join(self.db_path, INGESTED_FILES_LOG)
            processed_files = await self._load_processed_files_async(log_path)
            
            # Filter files that need processing
            files_to_process = []
            for file_path in file_paths:
                if await self._should_process_file(file_path, processed_files, exclusion_patterns):
                    files_to_process.append(file_path)
                else:
                    result.skipped_count += 1
            
            logger.info(f"📄 Processing {len(files_to_process)} new/updated files")
            
            # Process files in concurrent batches
            semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
            
            async def process_file_with_semaphore(file_path: str) -> Tuple[bool, Optional[Dict]]:
                async with semaphore:
                    return await self._process_single_file_async(file_path)
            
            # Create tasks for concurrent processing
            tasks = [process_file_with_semaphore(fp) for fp in files_to_process]
            
            # Process with progress tracking
            for i, task in enumerate(asyncio.as_completed(tasks)):
                success, file_result = await task
                
                if success:
                    result.success_count += 1
                    if file_result:
                        result.processed_files.append(file_result.get('file_path', ''))
                        result.total_entities += len(file_result.get('entities', []))
                        result.total_relationships += len(file_result.get('relationships', []))
                else:
                    result.error_count += 1
                    if file_result:
                        result.errors.append(file_result)
                
                # Progress callback
                if progress_callback:
                    progress = (i + 1) / len(tasks)
                    await progress_callback(progress, result)
            
            # Update processed files log
            await self._update_processed_files_log_async(log_path, files_to_process)
            
            result.processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Async processing completed: {result.success_count} success, "
                       f"{result.error_count} errors, {result.skipped_count} skipped")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Critical error in async processing: {e}")
            result.processing_time = (datetime.now() - start_time).total_seconds()
            raise
    
    async def _process_single_file_async(self, file_path: str) -> Tuple[bool, Optional[Dict]]:
        """Process a single file asynchronously"""
        try:
            logger.debug(f"📝 Processing file: {file_path}")
            
            # Run CPU-intensive work in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._process_file_sync, 
                file_path
            )
            
            return True, result
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return False, {
                'file_path': file_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _process_file_sync(self, file_path: str) -> Dict:
        """Synchronous file processing (runs in executor)"""
        # Get document content
        content_result = get_document_content(file_path)
        if not content_result or not content_result.get('content'):
            raise ValueError(f"Failed to extract content from {file_path}")
        
        # Extract entities and relationships
        entities = []
        relationships = []
        
        if self.entity_extractor:
            extraction_result = self.entity_extractor.extract_entities_from_text(
                content_result['content']
            )
            entities = extraction_result.get('entities', [])
            relationships = extraction_result.get('relationships', [])
        
        # Create document with metadata
        doc_metadata = DocumentMetadata(
            doc_id=get_file_hash(file_path),
            doc_posix_path=str(Path(file_path).as_posix()),
            file_name=os.path.basename(file_path),
            last_modified_date=datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).isoformat(),
            extracted_entities=[e.dict() if hasattr(e, 'dict') else e for e in entities],
            extracted_relationships=[r.dict() if hasattr(r, 'dict') else r for r in relationships]
        )
        
        # Prepare metadata consistent with primary ingestion
        flat_metadata = {
            "doc_id": doc_metadata.doc_id,
            "file_name": doc_metadata.file_name,
            "doc_posix_path": doc_metadata.doc_posix_path,
            "last_modified_date": doc_metadata.last_modified_date,
            # Minimal rich fields to keep search filters stable
            "document_type": "Other",
            "proposal_outcome": "N/A",
        }

        # Add to vector store with explicit embedding
        if self.collection:
            embedding = embed_texts([content_result['content']])[0]
            self.collection.add(
                documents=[content_result['content']],
                metadatas=[flat_metadata],
                embeddings=[embedding],
                ids=[doc_metadata.doc_id]
            )
        
        # Update knowledge graph
        if self.graph_manager:
            for entity in entities:
                self.graph_manager.add_entity(entity)
            for relationship in relationships:
                self.graph_manager.add_relationship(relationship)
            self.graph_manager.save_graph()
        
        return {
            'file_path': file_path,
            'doc_id': doc_metadata.doc_id,
            'entities': entities,
            'relationships': relationships,
            'content_length': len(content_result['content'])
        }
    
    async def _load_processed_files_async(self, log_path: str) -> Dict[str, str]:
        """Asynchronously load processed files log"""
        try:
            if not os.path.exists(log_path):
                return {}
            
            async with aiofiles.open(log_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content.strip() else {}
        except Exception as e:
            logger.warning(f"Could not load processed files log: {e}")
            return {}
    
    async def _update_processed_files_log_async(self, log_path: str, processed_files: List[str]) -> None:
        """Asynchronously update processed files log"""
        try:
            existing_log = await self._load_processed_files_async(log_path)
            
            for file_path in processed_files:
                existing_log[file_path] = get_file_hash(file_path)
            
            async with aiofiles.open(log_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(existing_log, indent=2))
                
        except Exception as e:
            logger.error(f"Failed to update processed files log: {e}")
    
    async def _should_process_file(self, 
                                 file_path: str, 
                                 processed_files: Dict[str, str],
                                 exclusion_patterns: List[str] = None) -> bool:
        """Check if file should be processed"""
        # Check exclusion patterns
        if exclusion_patterns:
            file_name = os.path.basename(file_path)
            for pattern in exclusion_patterns:
                if pattern in file_name:
                    return False
        
        # Check if already processed with same hash
        current_hash = get_file_hash(file_path)
        return processed_files.get(file_path) != current_hash
    
    async def get_ingestion_stats(self) -> Dict:
        """Get current ingestion statistics"""
        try:
            stats = {
                'total_documents': 0,
                'total_entities': 0,
                'total_relationships': 0,
                'database_size': 0,
                'last_updated': None
            }
            
            # Get ChromaDB stats
            if self.collection:
                stats['total_documents'] = self.collection.count()
            
            # Get graph stats
            if self.graph_manager:
                graph_stats = self.graph_manager.get_stats()
                stats['total_entities'] = graph_stats.get('entities', 0)
                stats['total_relationships'] = graph_stats.get('relationships', 0)
            
            # Get database size
            if os.path.exists(self.db_path):
                stats['database_size'] = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(self.db_path)
                    for filename in filenames
                )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting ingestion stats: {e}")
            return {}
    
    async def cleanup(self) -> None:
        """Cleanup async resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            logger.info("🧹 Async ingestion engine cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Convenience functions for easy integration

async def ingest_documents_async(db_path: str,
                               file_paths: List[str],
                               config: Optional[AsyncIngestionConfig] = None,
                               progress_callback: Optional[callable] = None) -> AsyncIngestionResult:
    """
    High-level async document ingestion function
    
    Args:
        db_path: Path to the knowledge database
        file_paths: List of document paths to ingest
        config: Optional ingestion configuration
        progress_callback: Optional progress callback function
        
    Returns:
        AsyncIngestionResult with processing statistics
    """
    engine = AsyncIngestionEngine(db_path, config)
    
    try:
        await engine.initialize()
        result = await engine.process_documents_async(
            file_paths, 
            progress_callback=progress_callback
        )
        return result
    finally:
        await engine.cleanup()

async def async_progress_callback(progress: float, result: AsyncIngestionResult) -> None:
    """Default progress callback for async ingestion"""
    logger.info(f"📊 Progress: {progress:.1%} - "
               f"Success: {result.success_count}, "
               f"Errors: {result.error_count}, "
               f"Skipped: {result.skipped_count}")
