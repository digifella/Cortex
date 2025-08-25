"""
Enhanced Document Ingestion with Docling Integration
Refactored ingestion pipeline using Docling for superior document processing with intelligent fallbacks.

Version: 14.0.0 (Docling Integration)
Date: 2025-08-22
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import json

# Existing imports (maintain compatibility)
from llama_index.core import Document
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import (
    DocxReader,
    PptxReader, 
    PyMuPDFReader,
    FlatReader,
    UnstructuredReader
)

# Cortex imports
from .config import INGESTION_LOG_PATH, STAGING_INGESTION_FILE, EMBED_MODEL
from .utils.logging_utils import get_logger
from .utils.smart_ollama_llm import create_smart_ollama_llm
from .docling_reader import DoclingDocumentReader, create_docling_reader
from .entity_extractor import EntityExtractor
from .graph_manager import EnhancedGraphManager

logger = get_logger(__name__)

# File type categories
# Enhanced Visual Processing - Comprehensive Image Format Support  
IMAGE_EXTENSIONS = {
    # Standard formats
    '.png', '.jpg', '.jpeg',
    # Additional raster formats
    '.gif', '.bmp', '.webp', '.tiff', '.tif',
    # Vector and specialized formats
    '.svg',  # Will be converted to raster for VLM processing
    '.ico'   # Icon files
}
DOCLING_PREFERRED_FORMATS = {'.pdf', '.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls'}
LEGACY_FORMATS = {'.txt', '.md', '.markdown', '.csv', '.json', '.xml', '.html', '.htm'}


class EnhancedDocumentProcessor:
    """
    Enhanced document processor with Docling integration and intelligent fallbacks.
    
    Features:
    - Docling-first processing for optimal document parsing
    - Intelligent fallback to LlamaIndex readers  
    - Comprehensive error handling and recovery
    - Enhanced metadata extraction
    - Performance monitoring and optimization
    """
    
    def __init__(self, enable_docling: bool = True, enable_ocr: bool = True):
        """
        Initialize enhanced document processor.
        
        Args:
            enable_docling: Use Docling for supported formats
            enable_ocr: Enable OCR for scanned documents
        """
        self.enable_docling = enable_docling
        self.enable_ocr = enable_ocr
        
        # Initialize Docling reader
        self.docling_reader = None
        if enable_docling:
            try:
                self.docling_reader = create_docling_reader(
                    ocr_enabled=enable_ocr,
                    table_structure_recognition=True
                )
                if self.docling_reader.is_available:
                    logger.info("✅ Docling reader initialized successfully")
                else:
                    logger.warning("⚠️ Docling reader failed to initialize")
                    self.docling_reader = None
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize Docling reader: {e}")
                self.docling_reader = None
        
        # Initialize legacy readers as fallbacks
        self.legacy_readers = {
            ".pdf": PyMuPDFReader(),
            ".docx": DocxReader(), 
            ".pptx": PptxReader(),
            ".doc": UnstructuredReader(),
            ".ppt": UnstructuredReader()
        }
        self.default_reader = FlatReader()
        
        # Processing statistics
        self.stats = {
            'docling_processed': 0,
            'legacy_processed': 0,
            'docling_fallbacks': 0,
            'errors': 0,
            'total_files': 0
        }
    
    def get_processing_strategy(self, file_path: Path) -> Tuple[str, str]:
        """
        Determine optimal processing strategy for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (strategy, reason) where strategy is 'docling', 'legacy', or 'image'
        """
        extension = file_path.suffix.lower()
        
        # Image files always use VLM processing
        if extension in IMAGE_EXTENSIONS:
            return 'image', f'Image file ({extension})'
        
        # Docling-preferred formats
        if extension in DOCLING_PREFERRED_FORMATS and self.docling_reader and self.docling_reader.is_available:
            return 'docling', f'Docling optimal for {extension}'
        
        # Legacy formats or Docling unavailable
        if extension in self.legacy_readers or extension in LEGACY_FORMATS:
            return 'legacy', f'Legacy reader for {extension}'
        
        # Unknown format - try legacy default
        return 'legacy', f'Unknown format {extension}, using legacy default'
    
    def process_document(self, file_path: Path, skip_image_processing: bool = False) -> List[Document]:
        """
        Process a single document using optimal strategy.
        
        Args:
            file_path: Path to the document
            skip_image_processing: Skip image processing if True
            
        Returns:
            List of processed Document objects
        """
        self.stats['total_files'] += 1
        
        strategy, reason = self.get_processing_strategy(file_path)
        logger.info(f"Processing '{file_path.name}' using {strategy} strategy: {reason}")
        
        try:
            if strategy == 'image':
                return self._process_image(file_path, skip_image_processing)
            elif strategy == 'docling':
                return self._process_with_docling(file_path)
            else:  # legacy
                return self._process_with_legacy(file_path)
                
        except Exception as e:
            logger.error(f"❌ Failed to process '{file_path.name}': {e}")
            self.stats['errors'] += 1
            
            # Try fallback if Docling failed
            if strategy == 'docling':
                logger.info(f"🔄 Attempting legacy fallback for '{file_path.name}'")
                try:
                    result = self._process_with_legacy(file_path)
                    self.stats['docling_fallbacks'] += 1
                    return result
                except Exception as fallback_error:
                    logger.error(f"❌ Legacy fallback also failed: {fallback_error}")
            
            # Return minimal document with error info
            return [self._create_error_document(file_path, str(e))]
    
    def _process_with_docling(self, file_path: Path) -> List[Document]:
        """Process document using Docling."""
        if not self.docling_reader or not self.docling_reader.is_available:
            raise RuntimeError("Docling reader not available")
        
        documents = self.docling_reader.load_data(file_path)
        self.stats['docling_processed'] += 1
        
        # Enhance documents with additional metadata
        for doc in documents:
            doc.metadata['processing_strategy'] = 'docling'
            doc.metadata['enhanced_processing'] = True
            doc.metadata['layout_preserved'] = True
            doc.metadata['table_structure_preserved'] = True
        
        return documents
    
    def _process_with_legacy(self, file_path: Path) -> List[Document]:
        """Process document using legacy LlamaIndex readers."""
        extension = file_path.suffix.lower()
        reader = self.legacy_readers.get(extension, self.default_reader)
        
        logger.info(f"Using legacy reader: {reader.__class__.__name__}")
        
        # Handle different reader types with enhanced error handling
        if isinstance(reader, PyMuPDFReader):
            # Suppress PyMuPDF warnings
            import fitz
            fitz.TOOLS.mupdf_display_errors(False)
            
            try:
                documents = reader.load_data(file_path=file_path)
            except Exception as pdf_error:
                if "cannot find loader for this" in str(pdf_error):
                    logger.warning(f"PDF image extraction issue in {file_path.name}: {pdf_error}")
                    documents = reader.load_data(file_path=file_path)
                else:
                    raise
        
        elif isinstance(reader, UnstructuredReader):
            try:
                documents = reader.load_data(file_path=file_path)
            except Exception as unstructured_error:
                if "cannot find loader for this WMF file" in str(unstructured_error):
                    logger.warning(f"WMF image error in {file_path.name}, skipping problematic elements")
                    # Create basic text extraction
                    text = f"Document: {file_path.name}\n(Some multimedia elements could not be processed)"
                    documents = [Document(text=text)]
                else:
                    raise
        
        else:
            # Standard reader processing
            documents = reader.load_data(file_path=file_path)
        
        self.stats['legacy_processed'] += 1
        
        # Enhance legacy documents with metadata
        for doc in documents:
            doc.metadata.update({
                'file_path': str(file_path.as_posix()),
                'file_name': file_path.name,
                'processing_strategy': 'legacy',
                'reader_type': reader.__class__.__name__,
                'enhanced_processing': False
            })
        
        return documents
    
    def _process_image(self, file_path: Path, skip_processing: bool = False) -> List[Document]:
        """Process image files using VLM."""
        if skip_processing:
            logger.info(f"Skipping image processing for '{file_path.name}'")
            doc = Document(text=f"Image file: {file_path.name} (processing skipped)")
            doc.metadata = {
                'file_path': str(file_path.as_posix()),
                'file_name': file_path.name,
                'source_type': 'image_skipped',
                'processing_strategy': 'image_skipped'
            }
            return [doc]
        
        # Import here to avoid circular imports
        from .query_cortex import describe_image_with_vlm_for_ingestion
        
        try:
            description = describe_image_with_vlm_for_ingestion(str(file_path))
            doc = Document(text=description)
            doc.metadata = {
                'file_path': str(file_path.as_posix()),
                'file_name': file_path.name,
                'source_type': 'image',
                'processing_strategy': 'vlm',
                'vlm_processed': True
            }
            return [doc]
        except Exception as e:
            logger.error(f"VLM processing failed for '{file_path.name}': {e}")
            raise
    
    def _create_error_document(self, file_path: Path, error_message: str) -> Document:
        """Create a document representing a processing error."""
        text = f"Error processing document: {file_path.name}\nError: {error_message}"
        doc = Document(text=text)
        doc.metadata = {
            'file_path': str(file_path.as_posix()),
            'file_name': file_path.name,
            'source_type': 'error',
            'processing_strategy': 'error',
            'error_message': error_message,
            'processing_failed': True
        }
        return doc
    
    def process_batch(self, file_paths: List[str], skip_image_processing: bool = False) -> List[Document]:
        """
        Process a batch of documents.
        
        Args:
            file_paths: List of file paths to process
            skip_image_processing: Skip image processing if True
            
        Returns:
            List of all processed documents
        """
        all_documents = []
        
        logger.info(f"Processing batch of {len(file_paths)} files")
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                documents = self.process_document(path, skip_image_processing)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Batch processing error for '{file_path}': {e}")
                # Continue with other files
                continue
        
        self._log_processing_stats()
        return all_documents
    
    def _log_processing_stats(self):
        """Log processing statistics."""
        total = self.stats['total_files']
        if total == 0:
            return
        
        logger.info("📊 Processing Statistics:")
        logger.info(f"  Total files: {total}")
        logger.info(f"  Docling processed: {self.stats['docling_processed']} ({self.stats['docling_processed']/total*100:.1f}%)")
        logger.info(f"  Legacy processed: {self.stats['legacy_processed']} ({self.stats['legacy_processed']/total*100:.1f}%)")
        logger.info(f"  Docling fallbacks: {self.stats['docling_fallbacks']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        
        if self.docling_reader:
            docling_stats = self.docling_reader.get_processing_stats()
            logger.info(f"  Docling available: {docling_stats['docling_available']}")
            logger.info(f"  OCR enabled: {docling_stats.get('ocr_enabled', False)}")
    
    def get_capabilities_report(self) -> Dict[str, Any]:
        """Get comprehensive capabilities report."""
        report = {
            'enhanced_processing_available': True,
            'docling_enabled': self.enable_docling,
            'ocr_enabled': self.enable_ocr,
            'processing_stats': self.stats.copy()
        }
        
        if self.docling_reader:
            report['docling_status'] = self.docling_reader.get_processing_stats()
        else:
            report['docling_status'] = {'available': False, 'reason': 'Not initialized'}
        
        report['supported_strategies'] = {
            'docling': list(DOCLING_PREFERRED_FORMATS),
            'legacy': list(self.legacy_readers.keys()) + list(LEGACY_FORMATS),
            'image': list(IMAGE_EXTENSIONS)
        }
        
        return report


# Factory function for easy integration
def create_enhanced_processor(enable_docling: bool = True, enable_ocr: bool = True) -> EnhancedDocumentProcessor:
    """
    Create enhanced document processor with optimal settings.
    
    Args:
        enable_docling: Enable Docling processing for supported formats
        enable_ocr: Enable OCR for scanned documents
        
    Returns:
        Configured EnhancedDocumentProcessor
    """
    return EnhancedDocumentProcessor(
        enable_docling=enable_docling,
        enable_ocr=enable_ocr
    )


# Compatibility function for existing code
def enhanced_load_documents(file_paths: List[str], skip_image_processing: bool = False) -> List[Document]:
    """
    Enhanced document loading with Docling integration.
    Drop-in replacement for existing manual_load_documents function.
    
    Args:
        file_paths: List of file paths to process
        skip_image_processing: Skip image processing if True
        
    Returns:
        List of processed documents with enhanced metadata
    """
    processor = create_enhanced_processor()
    return processor.process_batch(file_paths, skip_image_processing)