"""
Docling Document Reader for Cortex Suite
Advanced document processing using IBM Research's Docling toolkit.

Version: 1.0.0
Date: 2025-08-22
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import base64
from io import BytesIO

try:
    import docling
    # Test basic import without deep initialization to avoid torch conflicts
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Import these only when actually needed to avoid version conflicts
# Docling 1.8.5 simplified API - removed InputFormat/allowed_formats
DocumentConverter = None
PipelineOptions = None

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
    
    def __init__(self, ocr_enabled: bool = True, table_structure_recognition: bool = True, skip_vlm_processing: bool = False):
        """
        Initialize Docling reader.

        Args:
            ocr_enabled: Enable OCR for scanned PDFs
            table_structure_recognition: Enable advanced table structure recognition
            skip_vlm_processing: Skip VLM processing for extracted figures (default: False)
        """
        self.ocr_enabled = ocr_enabled
        self.table_structure_recognition = table_structure_recognition
        self.skip_vlm_processing = skip_vlm_processing
        self._converter = None
        
        if not DOCLING_AVAILABLE:
            logger.warning("Docling not available. Install with: pip install docling")
            return
            
        self._init_converter()
    
    def _init_converter(self):
        """Initialize Docling converter with optimized settings."""
        try:
            # Lazy import to avoid torch conflicts at startup
            # Docling 1.8.5 simplified API - no InputFormat/allowed_formats
            global DocumentConverter, PipelineOptions
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import PipelineOptions

            # Configure pipeline options for optimal processing (Docling 1.8.5 API)
            # Note: Docling 1.8.5 removed format restrictions - converter handles all supported formats
            pipeline_options = PipelineOptions(
                do_ocr=self.ocr_enabled,
                do_table_structure=self.table_structure_recognition
            )

            # Create converter with simplified API
            # Docling 1.8.5 auto-detects format and handles: PDF, DOCX, PPTX, XLSX, Images, HTML, MD, etc.
            self._converter = DocumentConverter(
                pipeline_options=pipeline_options
            )

            logger.info("✅ Docling converter initialized successfully (v1.8.5 API)")
            
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

        # Phase 1 Enhancement: Extract provenance metadata
        provenance_data = self._extract_provenance_metadata(docling_metadata)
        if provenance_data.get('has_provenance'):
            metadata['docling_provenance'] = provenance_data
            logger.debug(f"Added provenance for {provenance_data['page_count']} pages")

        figures_summary, figure_payloads = self._extract_docling_figures(docling_metadata, conv_result)

        # Phase 1 Enhancement: Generate VLM descriptions for figures
        if figure_payloads and not self.skip_vlm_processing:
            vlm_descriptions = self._generate_vlm_descriptions_for_figures(figure_payloads)

            # Enrich figures_summary with VLM descriptions
            if vlm_descriptions:
                for figure_entry in figures_summary:
                    idx = figure_entry.get('index')
                    if idx in vlm_descriptions:
                        figure_entry['vlm_description'] = vlm_descriptions[idx]
                        logger.debug(f"Figure {idx}: Added VLM description to metadata")

        if figures_summary:
            metadata['docling_figures'] = figures_summary
        if figure_payloads:
            metadata['docling_figures_payload'] = figure_payloads

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
        except (KeyError, IndexError, TypeError) as e:
            logger.debug(f"Error checking for tables in document: {e}")
            return False
    
    def _contains_images(self, docling_data: dict) -> bool:
        """Check if document contains images."""
        try:
            main_text = docling_data.get('main-text', [])
            return any(item.get('prov', [{}])[0].get('type') == 'figure' for item in main_text)
        except (KeyError, IndexError, TypeError) as e:
            logger.debug(f"Error checking for images in document: {e}")
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
        except (KeyError, TypeError) as e:
            logger.debug(f"Error counting pages in document: {e}")
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
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Error counting sections in document: {e}")
            return 0

    def _extract_docling_figures(
        self,
        docling_metadata: Dict[str, Any],
        conv_result: Any
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Capture figure metadata and payloads for downstream VLM analysis."""
        figures_summary: List[Dict[str, Any]] = []
        figure_payloads: List[Dict[str, Any]] = []

        figures_data = docling_metadata.get('figures', []) or docling_metadata.get('figures'.replace('-', '_'), [])
        main_text = docling_metadata.get('main-text', []) or docling_metadata.get('main_text', [])
        caption_map = self._build_caption_map(main_text)

        try:
            rendered_figures = list(conv_result.render_element_images())
        except Exception as render_error:
            logger.warning(f"Docling figure rendering unavailable: {render_error}")
            rendered_figures = []

        for idx, rendered in enumerate(rendered_figures):
            try:
                element, pil_image = rendered
            except ValueError:
                # Older Docling versions only return the image
                element, pil_image = None, rendered

            figure_dict = figures_data[idx] if idx < len(figures_data) else {}
            prov = None
            if isinstance(figure_dict, dict):
                prov_list = figure_dict.get('prov') or figure_dict.get('provenance') or []
                if isinstance(prov_list, list) and prov_list:
                    prov = prov_list[0]

            figure_entry = {
                'index': idx,
                'page': (prov or {}).get('page'),
                'bbox': (prov or {}).get('bbox'),
                'object_type': figure_dict.get('type') or figure_dict.get('obj_type'),
                'caption': caption_map.get(idx) or figure_dict.get('text') or '',
            }

            try:
                buffer = BytesIO()
                pil_image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                payload = {
                    'index': idx,
                    'image_base64': base64.b64encode(image_bytes).decode('utf-8'),
                    'image_mime_type': 'image/png',
                    'width': getattr(pil_image, 'width', None),
                    'height': getattr(pil_image, 'height', None)
                }
                figure_payloads.append(payload)
                figure_entry['has_image_payload'] = True
            except Exception as encode_error:
                logger.warning(f"Unable to serialize Docling figure {idx}: {encode_error}")
                figure_entry['has_image_payload'] = False

            figures_summary.append(figure_entry)

        return figures_summary, figure_payloads

    def _build_caption_map(self, main_text: List[Dict[str, Any]]) -> Dict[int, str]:
        """Attempt to align captions with figure indices using Docling main text."""
        caption_map: Dict[int, str] = {}
        pending_index: Optional[int] = None
        fallback_counter = 0

        for item in main_text:
            if not isinstance(item, dict):
                continue

            ref_value = item.get('$ref') or item.get('ref')
            if ref_value and '/figures/' in ref_value:
                try:
                    pending_index = int(ref_value.split('/')[-1])
                    continue
                except ValueError:
                    pending_index = None
                    continue

            text_content = (item.get('text') or '').strip()
            if not text_content:
                continue

            obj_type = (item.get('type') or item.get('obj_type') or item.get('name') or '').lower()
            if obj_type != 'caption' and not text_content.lower().startswith('figure'):
                continue

            target_index: int
            if pending_index is not None:
                target_index = pending_index
                pending_index = None
            else:
                target_index = fallback_counter
                fallback_counter += 1

            caption_map[target_index] = text_content

        return caption_map

    def _generate_vlm_descriptions_for_figures(
        self,
        figure_payloads: List[Dict[str, Any]]
    ) -> Dict[int, str]:
        """
        Generate VLM descriptions for Docling-extracted figures.

        This method processes figure images extracted by Docling and generates
        rich AI descriptions using a Vision Language Model (VLM). Figures are
        processed in parallel for performance.

        Args:
            figure_payloads: List of figure payload dicts containing:
                - index: Figure index
                - image_base64: Base64-encoded PNG image
                - image_mime_type: MIME type (usually 'image/png')
                - width: Image width in pixels
                - height: Image height in pixels

        Returns:
            Dictionary mapping figure index to VLM description string

        Notes:
            - Respects DOCLING_VLM_ENABLED config flag
            - Uses ThreadPoolExecutor for parallel processing
            - Configurable workers via DOCLING_VLM_MAX_WORKERS
            - Configurable timeout via DOCLING_VLM_TIMEOUT
            - Creates temp files for VLM processing, cleans up after
        """
        # Import config and VLM function
        try:
            from .config import (
                DOCLING_VLM_ENABLED,
                DOCLING_VLM_MAX_WORKERS,
                DOCLING_VLM_TIMEOUT
            )
            from .query_cortex import describe_image_with_vlm_async
        except ImportError as e:
            logger.warning(f"Could not import VLM dependencies: {e}")
            return {}

        # Check if VLM processing is enabled
        if self.skip_vlm_processing or not DOCLING_VLM_ENABLED:
            logger.debug("VLM processing disabled, skipping figure descriptions")
            return {}

        if not figure_payloads:
            return {}

        logger.info(f"🎨 Processing {len(figure_payloads)} Docling figures with VLM (max_workers={DOCLING_VLM_MAX_WORKERS})")

        descriptions = {}
        temp_files = []  # Track temp files for cleanup

        try:
            # Create temp files from base64 payloads
            figure_temp_files = {}
            for payload in figure_payloads:
                idx = payload['index']
                image_base64 = payload.get('image_base64', '')

                if not image_base64:
                    logger.warning(f"Figure {idx}: No image payload, skipping VLM")
                    continue

                # Decode base64 to bytes
                try:
                    image_bytes = base64.b64decode(image_base64)
                except Exception as decode_error:
                    logger.warning(f"Figure {idx}: Failed to decode base64: {decode_error}")
                    continue

                # Create temp file
                try:
                    temp_fd, temp_path = tempfile.mkstemp(suffix='.png', prefix=f'docling_fig_{idx}_')
                    os.close(temp_fd)  # Close file descriptor

                    # Write image bytes to temp file
                    with open(temp_path, 'wb') as f:
                        f.write(image_bytes)

                    figure_temp_files[idx] = temp_path
                    temp_files.append(temp_path)
                    logger.debug(f"Figure {idx}: Created temp file {temp_path}")

                except Exception as temp_error:
                    logger.warning(f"Figure {idx}: Failed to create temp file: {temp_error}")
                    continue

            # Process figures in parallel with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=DOCLING_VLM_MAX_WORKERS) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(
                        describe_image_with_vlm_async,
                        temp_path,
                        timeout=DOCLING_VLM_TIMEOUT
                    ): idx
                    for idx, temp_path in figure_temp_files.items()
                }

                # Collect results as they complete
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        description = future.result()
                        if description:
                            descriptions[idx] = description
                            logger.debug(f"Figure {idx}: VLM description generated ({len(description)} chars)")
                        else:
                            logger.warning(f"Figure {idx}: VLM returned no description")
                    except Exception as vlm_error:
                        logger.warning(f"Figure {idx}: VLM processing failed: {vlm_error}")

            logger.info(f"✅ Generated {len(descriptions)}/{len(figure_payloads)} figure descriptions with VLM")

        except Exception as e:
            logger.error(f"Error during VLM figure processing: {e}")

        finally:
            # Clean up temp files
            for temp_path in temp_files:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        logger.debug(f"Cleaned up temp file: {temp_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")

        return descriptions

    def _extract_provenance_metadata(
        self,
        docling_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract provenance metadata from Docling output for precise source attribution.

        Provenance includes page numbers, bounding boxes, and element types
        which enable citation back to specific document locations.

        Args:
            docling_metadata: Docling's structured metadata dict

        Returns:
            Dictionary with provenance information:
                - elements: List of element provenances (page, bbox, type)
                - page_count: Total number of pages
                - has_provenance: Whether provenance data is available

        Notes:
            - Respects DOCLING_PROVENANCE_ENABLED config flag
            - Extracts from 'main-text' or 'main_text' keys
            - Handles missing/malformed provenance gracefully
        """
        try:
            from .config import DOCLING_PROVENANCE_ENABLED
        except ImportError:
            logger.warning("Could not import provenance config")
            DOCLING_PROVENANCE_ENABLED = True  # Default to enabled

        if not DOCLING_PROVENANCE_ENABLED:
            return {
                'has_provenance': False,
                'elements': [],
                'page_count': 0
            }

        elements_provenance = []
        pages = set()

        try:
            # Extract from main-text (or main_text for compatibility)
            main_text = docling_metadata.get('main-text', []) or docling_metadata.get('main_text', [])

            for idx, item in enumerate(main_text):
                # Get provenance list
                prov_list = item.get('prov', []) or item.get('provenance', [])

                if not isinstance(prov_list, list) or not prov_list:
                    continue

                # Usually first provenance entry
                prov = prov_list[0]

                if not isinstance(prov, dict):
                    continue

                # Extract element provenance
                element_prov = {
                    'element_index': idx,
                    'page': prov.get('page'),
                    'bbox': prov.get('bbox'),  # Bounding box [x1, y1, x2, y2]
                    'type': prov.get('type') or item.get('type') or item.get('obj_type'),
                    'text_sample': (item.get('text', '') or '')[:100]  # First 100 chars
                }

                # Track pages
                if element_prov['page'] is not None:
                    pages.add(element_prov['page'])

                elements_provenance.append(element_prov)

            logger.debug(f"Extracted provenance for {len(elements_provenance)} elements across {len(pages)} pages")

        except Exception as e:
            logger.warning(f"Error extracting provenance metadata: {e}")

        return {
            'has_provenance': len(elements_provenance) > 0,
            'elements': elements_provenance,
            'page_count': len(pages),
            'pages': sorted(list(pages)) if pages else []
        }

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


def create_docling_reader(ocr_enabled: bool = True, table_structure_recognition: bool = True, skip_vlm_processing: bool = False) -> DoclingDocumentReader:
    """
    Factory function to create Docling reader.

    Args:
        ocr_enabled: Enable OCR for scanned documents
        table_structure_recognition: Enable advanced table recognition
        skip_vlm_processing: Skip VLM processing for extracted figures (default: False)

    Returns:
        Configured DoclingDocumentReader instance
    """
    return DoclingDocumentReader(
        ocr_enabled=ocr_enabled,
        table_structure_recognition=table_structure_recognition,
        skip_vlm_processing=skip_vlm_processing
    )
