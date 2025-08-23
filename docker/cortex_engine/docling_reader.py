"""
Docling Document Reader for Cortex Suite
Advanced document processing using IBM Research's Docling toolkit.

Version: 1.0.0
Date: 2025-08-22
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json

try:
    import docling
    # Test basic import without deep initialization to avoid torch conflicts
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Import these only when actually needed to avoid version conflicts
DocumentConverter = None
InputFormat = None
PdfPipelineOptions = None
StandardPdfPipeline = None

from llama_index.core import Document
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


class DoclingDocumentReader:
    """
    Advanced document reader using Docling for superior document processing.
    
    Features:
    - Context-aware document parsing
    - Layout preservation (headers, tables, reading order)
    - OCR for scanned PDFs
    - Support for PDF, DOCX, PPTX, XLSX, images, HTML, AsciiDoc, Markdown
    - Structured metadata extraction
    - High-performance processing
    """
    
    def __init__(self, ocr_enabled: bool = True, table_structure_recognition: bool = True):
        """
        Initialize Docling reader.
        
        Args:
            ocr_enabled: Enable OCR for scanned PDFs
            table_structure_recognition: Enable advanced table structure recognition
        """
        self.ocr_enabled = ocr_enabled
        self.table_structure_recognition = table_structure_recognition
        self._converter = None
        
        if not DOCLING_AVAILABLE:
            logger.warning("Docling not available. Install with: pip install docling")
            return
            
        self._init_converter()
    
    def _init_converter(self):
        """Initialize Docling converter with optimized settings."""
        try:
            # Lazy import to avoid torch conflicts at startup
            global DocumentConverter, InputFormat, PdfPipelineOptions, StandardPdfPipeline
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
            
            # Configure PDF pipeline options for optimal processing
            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = self.ocr_enabled
            pdf_options.do_table_structure = self.table_structure_recognition
            
            # Create converter with custom pipeline
            pipeline = StandardPdfPipeline(pdf_options)
            self._converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX, 
                    InputFormat.PPTX,
                    InputFormat.XLSX,
                    InputFormat.IMAGE,
                    InputFormat.HTML,
                    InputFormat.MD,
                    InputFormat.ASCIIDOC
                ],
                pdf_pipeline=pipeline
            )
            
            logger.info("✅ Docling converter initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Docling dependency missing: {e}")
            logger.info("💡 To install Docling: pip install docling")
            logger.info("📋 Falling back to legacy document readers")
            self._converter = None
        except Exception as e:
            logger.warning(f"Docling converter failed to initialize: {e}")
            logger.info("🔧 This may be due to incompatible dependencies (torch/numpy versions)")
            logger.info("📋 Falling back to legacy document readers")
            self._converter = None
    
    @property
    def is_available(self) -> bool:
        """Check if Docling is available and properly initialized."""
        return DOCLING_AVAILABLE and self._converter is not None
    
    def can_process_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if Docling can process the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if Docling can process this file type
        """
        if not self.is_available:
            return False
            
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Docling supported formats
        supported_extensions = {
            '.pdf', '.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif',
            '.html', '.htm', '.md', '.markdown', '.adoc', '.asciidoc'
        }
        
        return extension in supported_extensions
    
    def load_data(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load and process document using Docling.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of LlamaIndex Document objects with enhanced metadata
        """
        if not self.is_available:
            raise RuntimeError("Docling not available. Cannot process document.")
        
        path = Path(file_path)
        logger.info(f"Processing '{path.name}' with Docling")
        
        try:
            # Convert document using Docling
            conv_result = self._converter.convert(str(path))
            
            # Extract the DoclingDocument
            doc_result = conv_result.document
            
            # Get Markdown representation (preserves structure)
            markdown_content = doc_result.export_to_markdown()
            
            # Get JSON representation for metadata
            json_content = doc_result.export_to_dict()
            
            # Create enhanced LlamaIndex Document
            document = self._create_enhanced_document(
                path=path,
                markdown_content=markdown_content,
                docling_metadata=json_content,
                conv_result=conv_result
            )
            
            logger.info(f"✅ Successfully processed '{path.name}' with Docling ({len(markdown_content)} chars)")
            return [document]
            
        except Exception as e:
            logger.error(f"Docling processing failed for '{path.name}': {e}")
            raise
    
    def _create_enhanced_document(
        self, 
        path: Path, 
        markdown_content: str, 
        docling_metadata: dict,
        conv_result: Any
    ) -> Document:
        """
        Create enhanced LlamaIndex Document with Docling metadata.
        
        Args:
            path: File path
            markdown_content: Processed Markdown content
            docling_metadata: Docling JSON metadata
            conv_result: Docling conversion result
            
        Returns:
            Enhanced LlamaIndex Document
        """
        
        # Base metadata
        metadata = {
            'file_path': str(path.as_posix()),
            'file_name': path.name,
            'source_type': 'docling_processed',
            'processor': 'docling',
            'docling_version': getattr(conv_result, 'version', 'unknown')
        }
        
        # Extract Docling-specific metadata
        try:
            # Document structure information
            if 'main-text' in docling_metadata:
                metadata['docling_structure'] = {
                    'has_tables': self._contains_tables(docling_metadata),
                    'has_images': self._contains_images(docling_metadata),
                    'page_count': self._get_page_count(docling_metadata),
                    'section_count': self._get_section_count(docling_metadata)
                }
            
            # Layout analysis results
            if hasattr(conv_result, 'pages'):
                metadata['docling_layout'] = {
                    'layout_detected': True,
                    'reading_order_preserved': True,
                    'ocr_applied': self.ocr_enabled
                }
            
            # Quality metrics
            metadata['docling_quality'] = {
                'content_length': len(markdown_content),
                'processing_successful': True,
                'structured_extraction': True
            }
            
        except Exception as e:
            logger.warning(f"Could not extract enhanced Docling metadata: {e}")
            metadata['docling_metadata_warning'] = str(e)
        
        # Create Document with enhanced content and metadata
        document = Document(
            text=markdown_content,
            metadata=metadata
        )
        
        return document
    
    def _contains_tables(self, docling_data: dict) -> bool:
        """Check if document contains tables."""
        try:
            main_text = docling_data.get('main-text', [])
            return any(item.get('prov', [{}])[0].get('type') == 'table' for item in main_text)
        except:
            return False
    
    def _contains_images(self, docling_data: dict) -> bool:
        """Check if document contains images."""
        try:
            main_text = docling_data.get('main-text', [])
            return any(item.get('prov', [{}])[0].get('type') == 'figure' for item in main_text)
        except:
            return False
    
    def _get_page_count(self, docling_data: dict) -> int:
        """Get number of pages processed."""
        try:
            # Look for page references in the document
            main_text = docling_data.get('main-text', [])
            pages = set()
            for item in main_text:
                for prov in item.get('prov', []):
                    if 'page' in prov:
                        pages.add(prov['page'])
            return len(pages)
        except:
            return 0
    
    def _get_section_count(self, docling_data: dict) -> int:
        """Get number of sections/headings."""
        try:
            main_text = docling_data.get('main-text', [])
            sections = 0
            for item in main_text:
                text = item.get('text', '')
                if text.startswith('#'):  # Markdown headers
                    sections += 1
            return sections
        except:
            return 0
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and capabilities."""
        return {
            'docling_available': DOCLING_AVAILABLE,
            'converter_initialized': self._converter is not None,
            'ocr_enabled': self.ocr_enabled,
            'table_structure_recognition': self.table_structure_recognition,
            'supported_formats': [
                'PDF', 'DOCX', 'PPTX', 'XLSX', 'DOC', 'PPT', 'XLS',
                'PNG', 'JPG', 'JPEG', 'GIF', 'BMP', 'TIFF', 'TIF',
                'HTML', 'HTM', 'MD', 'MARKDOWN', 'ADOC', 'ASCIIDOC'
            ] if self.is_available else []
        }


def create_docling_reader(ocr_enabled: bool = True, table_structure_recognition: bool = True) -> DoclingDocumentReader:
    """
    Factory function to create Docling reader.
    
    Args:
        ocr_enabled: Enable OCR for scanned documents
        table_structure_recognition: Enable advanced table recognition
        
    Returns:
        Configured DoclingDocumentReader instance
    """
    return DoclingDocumentReader(
        ocr_enabled=ocr_enabled,
        table_structure_recognition=table_structure_recognition
    )