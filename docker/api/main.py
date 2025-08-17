# ## File: api/main.py
# Version: 1.0.0
# Date: 2025-07-27
# Purpose: FastAPI REST API for Cortex Suite external integration.
#          Provides comprehensive endpoints for knowledge management and search.

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import asyncio
import json
import logging
from pathlib import Path
import os
from datetime import datetime
import uuid

from cortex_engine.async_ingest import AsyncIngestionEngine, AsyncIngestionConfig, AsyncIngestionResult
from cortex_engine.async_query import AsyncSearchEngine, AsyncQueryConfig, AsyncQueryResult
from cortex_engine.config_manager import ConfigManager
from cortex_engine.collection_manager import WorkingCollectionManager
from cortex_engine.utils.logging_utils import get_logger
from cortex_engine.utils.path_utils import convert_windows_to_wsl_path
from cortex_engine.exceptions import *
from cortex_engine.backup_manager import BackupManager, BackupMetadata, RestoreMetadata

# Configure logging
logger = get_logger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Cortex Suite API",
    description="REST API for AI-powered knowledge management and proposal generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - Configure for production security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default
        "http://localhost:3000",  # React dev server
        "http://127.0.0.1:8501",
        # Add your production domains here
        # "https://yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global state
config_manager = None
db_path = None

# Pydantic models for API

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    search_type: str = Field("hybrid", description="Search type: vector, graph, or hybrid")
    max_results: int = Field(10, description="Maximum number of results")
    include_synthesis: bool = Field(True, description="Include AI synthesis of results")
    similarity_threshold: float = Field(0.7, description="Minimum similarity threshold")

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    synthesis: Optional[str] = None
    total_results: int
    processing_time: float
    status: str = "success"
    error: Optional[str] = None

class BatchSearchRequest(BaseModel):
    queries: List[str] = Field(..., description="List of search queries")
    search_type: str = Field("hybrid", description="Search type for all queries")
    include_synthesis: bool = Field(True, description="Include synthesis for all results")

class IngestionRequest(BaseModel):
    file_paths: List[str] = Field(..., description="List of file paths to ingest")
    exclusion_patterns: List[str] = Field(default_factory=list, description="File patterns to exclude")
    max_concurrent: int = Field(5, description="Maximum concurrent file processing")

class IngestionResponse(BaseModel):
    success_count: int
    error_count: int
    skipped_count: int
    processed_files: List[str]
    errors: List[Dict[str, str]]
    total_entities: int
    total_relationships: int
    processing_time: float

class EntityInfo(BaseModel):
    name: str
    type: str
    properties: Dict[str, Any] = {}
    relationships: List[str] = []

class CollectionInfo(BaseModel):
    name: str
    description: str
    documents: List[str]
    created_date: str
    last_modified: str

class SystemStats(BaseModel):
    total_documents: int
    total_entities: int
    total_relationships: int
    database_size: int
    collections_count: int
    last_updated: Optional[str] = None

class BackupRequest(BaseModel):
    backup_path: str = Field(..., description="Path to store backup")
    include_images: bool = Field(True, description="Include image files in backup")
    compress: bool = Field(True, description="Compress backup archive")

class RestoreRequest(BaseModel):
    backup_id: str = Field(..., description="ID of backup to restore")
    overwrite_existing: bool = Field(False, description="Overwrite existing data")
    verify_checksum: bool = Field(True, description="Verify backup integrity")

class BackupInfo(BaseModel):
    backup_id: str
    backup_type: str
    creation_time: str
    file_count: int
    total_size: int
    compression: str
    checksum: str
    description: Optional[str] = None

# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize API components on startup"""
    global config_manager, db_path
    
    try:
        logger.info("🚀 Starting Cortex Suite API...")
        
        # Initialize config manager
        config_manager = ConfigManager()
        config = config_manager.get_config()
        db_path = convert_windows_to_wsl_path(config.get('ai_database_path', '/mnt/f/ai_databases'))
        
        # Ensure database directory exists
        os.makedirs(db_path, exist_ok=True)
        
        logger.info(f"✅ API initialized with database path: {db_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on API shutdown"""
    logger.info("🛑 Shutting down Cortex Suite API...")

# Dependency injection

def get_config() -> ConfigManager:
    """Get config manager dependency"""
    if not config_manager:
        raise HTTPException(status_code=500, detail="Config manager not initialized")
    return config_manager

def get_db_path() -> str:
    """Get database path dependency"""
    if not db_path:
        raise HTTPException(status_code=500, detail="Database path not configured")
    return db_path

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (implement your authentication logic)"""
    # TODO: Implement proper authentication
    # For now, just log the request
    if credentials:
        logger.debug(f"API request with token: {credentials.credentials[:10]}...")
    return True

# Health and status endpoints

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/status", response_model=SystemStats, tags=["Health"])
async def get_system_status(db_path: str = Depends(get_db_path)):
    """Get comprehensive system status and statistics"""
    try:
        # Initialize engines for stats
        search_engine = AsyncSearchEngine(db_path)
        await search_engine.initialize()
        
        try:
            search_stats = await search_engine.get_search_stats()
            
            # Get additional stats
            stats = SystemStats(
                total_documents=search_stats.get('total_documents', 0),
                total_entities=search_stats.get('total_entities', 0),
                total_relationships=0,  # TODO: Get from graph manager
                database_size=0,  # TODO: Calculate database size
                collections_count=0,  # TODO: Get collections count
                last_updated=datetime.now().isoformat()
            )
            
            return stats
            
        finally:
            await search_engine.cleanup()
            
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

# Search endpoints

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_knowledge_base(
    request: SearchRequest,
    db_path: str = Depends(get_db_path),
    _: bool = Depends(verify_token)
):
    """Search the knowledge base with vector, graph, or hybrid search"""
    try:
        logger.info(f"🔍 API search request: '{request.query[:50]}...'")
        
        # Configure search engine
        config = AsyncQueryConfig(
            max_results_per_query=request.max_results,
            similarity_threshold=request.similarity_threshold
        )
        
        # Perform search
        search_engine = AsyncSearchEngine(db_path, config)
        await search_engine.initialize()
        
        try:
            result = await search_engine.search_async(
                request.query,
                request.search_type,
                request.include_synthesis
            )
            
            response = SearchResponse(
                query=result.query,
                results=result.results,
                synthesis=result.synthesis,
                total_results=result.total_results,
                processing_time=result.processing_time,
                status=result.status,
                error=result.error
            )
            
            return response
            
        finally:
            await search_engine.cleanup()
            
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/batch", response_model=List[SearchResponse], tags=["Search"])
async def batch_search_knowledge_base(
    request: BatchSearchRequest,
    db_path: str = Depends(get_db_path),
    _: bool = Depends(verify_token)
):
    """Perform multiple searches concurrently"""
    try:
        logger.info(f"🔍 API batch search: {len(request.queries)} queries")
        
        search_engine = AsyncSearchEngine(db_path)
        await search_engine.initialize()
        
        try:
            results = await search_engine.batch_search_async(
                request.queries,
                request.search_type,
                request.include_synthesis
            )
            
            responses = [
                SearchResponse(
                    query=result.query,
                    results=result.results,
                    synthesis=result.synthesis,
                    total_results=result.total_results,
                    processing_time=result.processing_time,
                    status=result.status,
                    error=result.error
                ) for result in results
            ]
            
            return responses
            
        finally:
            await search_engine.cleanup()
            
    except Exception as e:
        logger.error(f"Batch search error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")

# Ingestion endpoints

@app.post("/ingest", response_model=IngestionResponse, tags=["Ingestion"])
async def ingest_documents(
    background_tasks: BackgroundTasks,
    request: IngestionRequest,
    db_path: str = Depends(get_db_path),
    _: bool = Depends(verify_token)
):
    """Ingest documents into the knowledge base"""
    try:
        logger.info(f"📄 API ingestion request: {len(request.file_paths)} files")
        
        # Validate file paths
        valid_paths = []
        for path in request.file_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                logger.warning(f"File not found: {path}")
        
        if not valid_paths:
            raise HTTPException(status_code=400, detail="No valid file paths provided")
        
        # Configure ingestion engine
        config = AsyncIngestionConfig(
            max_concurrent_files=request.max_concurrent
        )
        
        # Perform ingestion
        ingestion_engine = AsyncIngestionEngine(db_path, config)
        await ingestion_engine.initialize()
        
        try:
            result = await ingestion_engine.process_documents_async(
                valid_paths,
                request.exclusion_patterns
            )
            
            response = IngestionResponse(
                success_count=result.success_count,
                error_count=result.error_count,
                skipped_count=result.skipped_count,
                processed_files=result.processed_files,
                errors=result.errors,
                total_entities=result.total_entities,
                total_relationships=result.total_relationships,
                processing_time=result.processing_time
            )
            
            return response
            
        finally:
            await ingestion_engine.cleanup()
            
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/ingest/upload", tags=["Ingestion"])
async def upload_and_ingest_file(
    file: UploadFile = File(...),
    db_path: str = Depends(get_db_path),
    _: bool = Depends(verify_token)
):
    """Upload and ingest a single file"""
    try:
        logger.info(f"📤 API file upload: {file.filename}")
        
        # Save uploaded file temporarily
        temp_dir = Path(db_path) / "temp_uploads"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
        
        async with aiofiles.open(temp_file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        try:
            # Ingest the uploaded file
            config = AsyncIngestionConfig(max_concurrent_files=1)
            ingestion_engine = AsyncIngestionEngine(db_path, config)
            await ingestion_engine.initialize()
            
            try:
                result = await ingestion_engine.process_documents_async([str(temp_file_path)])
                
                return {
                    "filename": file.filename,
                    "status": "success" if result.success_count > 0 else "failed",
                    "entities_extracted": result.total_entities,
                    "relationships_extracted": result.total_relationships,
                    "processing_time": result.processing_time
                }
                
            finally:
                await ingestion_engine.cleanup()
                
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
            
    except Exception as e:
        logger.error(f"Upload ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Collection management endpoints

@app.get("/collections", response_model=List[CollectionInfo], tags=["Collections"])
async def get_collections(
    config: ConfigManager = Depends(get_config),
    _: bool = Depends(verify_token)
):
    """Get all collections"""
    try:
        collection_manager = WorkingCollectionManager()
        collections = collection_manager.get_all_collections()
        
        collection_infos = []
        for name, data in collections.items():
            collection_infos.append(CollectionInfo(
                name=name,
                description=data.get('description', ''),
                documents=data.get('documents', []),
                created_date=data.get('created_date', ''),
                last_modified=data.get('last_modified', '')
            ))
        
        return collection_infos
        
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collections: {str(e)}")

@app.post("/collections/{collection_name}", tags=["Collections"])
async def create_collection(
    collection_name: str,
    description: str = Form(...),
    documents: List[str] = Form(default=[]),
    config: ConfigManager = Depends(get_config),
    _: bool = Depends(verify_token)
):
    """Create a new collection"""
    try:
        collection_manager = WorkingCollectionManager()
        
        collection_data = {
            'description': description,
            'documents': documents,
            'created_date': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat()
        }
        
        collection_manager.save_collection(collection_name, collection_data)
        
        return {"message": f"Collection '{collection_name}' created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")

# Entity endpoints

@app.get("/entities/{entity_name}", response_model=EntityInfo, tags=["Entities"])
async def get_entity_info(
    entity_name: str,
    db_path: str = Depends(get_db_path),
    _: bool = Depends(verify_token)
):
    """Get information about a specific entity"""
    try:
        from ..cortex_engine.graph_manager import EnhancedGraphManager
        
        graph_path = os.path.join(db_path, "knowledge_cortex.gpickle")
        if not os.path.exists(graph_path):
            raise HTTPException(status_code=404, detail="Knowledge graph not found")
        
        graph_manager = EnhancedGraphManager(graph_path)
        entity_info = graph_manager.get_entity_context(entity_name)
        
        if not entity_info:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")
        
        relationships = graph_manager.get_entity_relationships(entity_name)
        
        return EntityInfo(
            name=entity_name,
            type=entity_info.get('type', 'unknown'),
            properties=entity_info,
            relationships=relationships
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get entity info: {str(e)}")

# Backup and restore endpoints

@app.post("/backup/create", response_model=BackupInfo, tags=["Backup"])
async def create_backup(
    request: BackupRequest,
    db_path: str = Depends(get_db_path),
    _: bool = Depends(verify_token)
):
    """Create a backup of the knowledge base"""
    try:
        logger.info(f"📦 API backup request: {request.backup_path}")
        
        backup_manager = BackupManager(db_path)
        
        # Extract backup name from path
        backup_name = Path(request.backup_path).stem
        
        metadata = await backup_manager.create_backup_async(
            backup_name=backup_name,
            backup_type="full",
            include_images=request.include_images,
            compress=request.compress
        )
        
        return BackupInfo(
            backup_id=metadata.backup_id,
            backup_type=metadata.backup_type,
            creation_time=metadata.creation_time,
            file_count=metadata.file_count,
            total_size=metadata.total_size,
            compression=metadata.compression,
            checksum=metadata.checksum,
            description=metadata.description
        )
        
    except Exception as e:
        logger.error(f"Backup creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")

@app.get("/backup/list", response_model=List[BackupInfo], tags=["Backup"])
async def list_backups(
    db_path: str = Depends(get_db_path),
    _: bool = Depends(verify_token)
):
    """List all available backups"""
    try:
        backup_manager = BackupManager(db_path)
        metadata_list = await backup_manager.list_backups()
        
        return [
            BackupInfo(
                backup_id=meta.backup_id,
                backup_type=meta.backup_type,
                creation_time=meta.creation_time,
                file_count=meta.file_count,
                total_size=meta.total_size,
                compression=meta.compression,
                checksum=meta.checksum,
                description=meta.description
            ) for meta in metadata_list
        ]
        
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list backups: {str(e)}")

@app.post("/backup/restore", tags=["Backup"])
async def restore_backup(
    request: RestoreRequest,
    db_path: str = Depends(get_db_path),
    _: bool = Depends(verify_token)
):
    """Restore a backup to the knowledge base"""
    try:
        logger.info(f"📥 API restore request: {request.backup_id}")
        
        backup_manager = BackupManager(db_path)
        
        restore_metadata = await backup_manager.restore_backup_async(
            backup_id=request.backup_id,
            overwrite_existing=request.overwrite_existing,
            verify_checksum=request.verify_checksum
        )
        
        return {
            "restore_id": restore_metadata.restore_id,
            "backup_id": restore_metadata.backup_id,
            "files_restored": restore_metadata.files_restored,
            "success": restore_metadata.success,
            "errors": restore_metadata.errors,
            "restore_time": restore_metadata.restore_time
        }
        
    except Exception as e:
        logger.error(f"Restore error: {e}")
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")

@app.delete("/backup/{backup_id}", tags=["Backup"])
async def delete_backup(
    backup_id: str,
    db_path: str = Depends(get_db_path),
    _: bool = Depends(verify_token)
):
    """Delete a specific backup"""
    try:
        backup_manager = BackupManager(db_path)
        success = await backup_manager.delete_backup(backup_id)
        
        if success:
            return {"message": f"Backup '{backup_id}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Backup '{backup_id}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting backup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete backup: {str(e)}")

@app.get("/backup/{backup_id}/verify", tags=["Backup"])
async def verify_backup(
    backup_id: str,
    db_path: str = Depends(get_db_path),
    _: bool = Depends(verify_token)
):
    """Verify the integrity of a backup"""
    try:
        backup_manager = BackupManager(db_path)
        is_valid = await backup_manager.verify_backup_integrity(backup_id)
        
        return {
            "backup_id": backup_id,
            "valid": is_valid,
            "message": "Backup integrity verified" if is_valid else "Backup integrity check failed"
        }
        
    except Exception as e:
        logger.error(f"Error verifying backup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify backup: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)