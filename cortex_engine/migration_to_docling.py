"""
Migration Script: Transition to Docling-Enhanced Ingestion
Gradual migration from current ingestion to Docling-enhanced pipeline.

Version: 1.0.0
Date: 2025-08-22
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from .enhanced_ingest_cortex import create_enhanced_processor, enhanced_load_documents
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


class IngestionMigrationManager:
    """
    Manages migration from legacy ingestion to Docling-enhanced pipeline.
    
    Features:
    - Gradual rollout with A/B testing capability
    - Performance comparison between old and new pipelines  
    - Fallback mechanisms for robustness
    - Migration progress tracking
    """
    
    def __init__(self, migration_mode: str = "hybrid"):
        """
        Initialize migration manager.
        
        Args:
            migration_mode: 
                - "legacy": Use only legacy pipeline
                - "docling": Use only Docling pipeline  
                - "hybrid": Use both with comparison
                - "gradual": Gradual transition based on file types
        """
        self.migration_mode = migration_mode
        self.enhanced_processor = None
        self.comparison_results = []
        
        # Initialize enhanced processor if needed
        if migration_mode in ["docling", "hybrid", "gradual"]:
            try:
                self.enhanced_processor = create_enhanced_processor()
                logger.info(f"✅ Enhanced processor initialized for {migration_mode} mode")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize enhanced processor: {e}")
                if migration_mode == "docling":
                    logger.error("❌ Docling-only mode requested but initialization failed")
                    raise
                self.enhanced_processor = None
    
    def process_documents(self, file_paths: List[str], skip_image_processing: bool = False) -> List[Any]:
        """
        Process documents using selected migration strategy.
        
        Args:
            file_paths: List of file paths to process
            skip_image_processing: Skip image processing if True
            
        Returns:
            List of processed documents
        """
        if self.migration_mode == "legacy":
            return self._process_legacy_only(file_paths, skip_image_processing)
        elif self.migration_mode == "docling":
            return self._process_docling_only(file_paths, skip_image_processing)
        elif self.migration_mode == "hybrid":
            return self._process_hybrid_comparison(file_paths, skip_image_processing)
        elif self.migration_mode == "gradual":
            return self._process_gradual_migration(file_paths, skip_image_processing)
        else:
            raise ValueError(f"Unknown migration mode: {self.migration_mode}")
    
    def _process_legacy_only(self, file_paths: List[str], skip_image_processing: bool) -> List[Any]:
        """Process using legacy pipeline only."""
        from .ingest_cortex import manual_load_documents
        
        logger.info("📋 Using legacy ingestion pipeline")
        return manual_load_documents(file_paths, skip_image_processing)
    
    def _process_docling_only(self, file_paths: List[str], skip_image_processing: bool) -> List[Any]:
        """Process using Docling pipeline only."""
        if not self.enhanced_processor:
            raise RuntimeError("Enhanced processor not available for Docling-only mode")
        
        logger.info("🚀 Using Docling-enhanced ingestion pipeline")
        return self.enhanced_processor.process_batch(file_paths, skip_image_processing)
    
    def _process_hybrid_comparison(self, file_paths: List[str], skip_image_processing: bool) -> List[Any]:
        """Process using both pipelines for comparison."""
        logger.info("🔄 Running hybrid comparison between legacy and Docling pipelines")
        
        # Process with both pipelines
        legacy_results = self._process_legacy_only(file_paths, skip_image_processing)
        
        if self.enhanced_processor:
            try:
                docling_results = self._process_docling_only(file_paths, skip_image_processing)
                
                # Compare results
                comparison = self._compare_pipeline_results(
                    legacy_results, docling_results, file_paths
                )
                self.comparison_results.append(comparison)
                
                # Log comparison summary
                self._log_comparison_summary(comparison)
                
                # Return Docling results if successful, otherwise legacy
                return docling_results if comparison['docling_success_rate'] > 0.8 else legacy_results
                
            except Exception as e:
                logger.warning(f"⚠️ Docling pipeline failed in hybrid mode: {e}")
                return legacy_results
        else:
            logger.warning("⚠️ Enhanced processor not available, using legacy only")
            return legacy_results
    
    def _process_gradual_migration(self, file_paths: List[str], skip_image_processing: bool) -> List[Any]:
        """Process using gradual migration strategy based on file types."""
        logger.info("📈 Using gradual migration strategy")
        
        if not self.enhanced_processor:
            logger.warning("⚠️ Enhanced processor not available, falling back to legacy")
            return self._process_legacy_only(file_paths, skip_image_processing)
        
        # Categorize files for gradual migration
        docling_files = []
        legacy_files = []
        
        for file_path in file_paths:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            # Start with PDFs and DOCX (Docling's strongest formats)
            if extension in {'.pdf', '.docx'}:
                docling_files.append(file_path)
            else:
                legacy_files.append(file_path)
        
        all_documents = []
        
        # Process Docling files
        if docling_files:
            logger.info(f"🚀 Processing {len(docling_files)} files with Docling")
            try:
                docling_docs = self.enhanced_processor.process_batch(docling_files, skip_image_processing)
                all_documents.extend(docling_docs)
            except Exception as e:
                logger.error(f"❌ Docling processing failed, falling back to legacy: {e}")
                # Fall back to legacy for these files
                legacy_docs = self._process_legacy_only(docling_files, skip_image_processing)
                all_documents.extend(legacy_docs)
        
        # Process legacy files
        if legacy_files:
            logger.info(f"📋 Processing {len(legacy_files)} files with legacy pipeline")
            legacy_docs = self._process_legacy_only(legacy_files, skip_image_processing)
            all_documents.extend(legacy_docs)
        
        return all_documents
    
    def _compare_pipeline_results(self, legacy_docs: List[Any], docling_docs: List[Any], file_paths: List[str]) -> Dict[str, Any]:
        """Compare results from legacy and Docling pipelines."""
        comparison = {
            'timestamp': Path(__file__).stat().st_mtime,
            'file_count': len(file_paths),
            'legacy_doc_count': len(legacy_docs),
            'docling_doc_count': len(docling_docs),
            'legacy_success_rate': len(legacy_docs) / len(file_paths) if file_paths else 0,
            'docling_success_rate': len(docling_docs) / len(file_paths) if file_paths else 0,
            'content_length_comparison': {},
            'metadata_enhancement': {},
            'processing_quality': {}
        }
        
        # Compare content lengths
        if legacy_docs and docling_docs:
            legacy_lengths = [len(doc.text) for doc in legacy_docs if hasattr(doc, 'text')]
            docling_lengths = [len(doc.text) for doc in docling_docs if hasattr(doc, 'text')]
            
            if legacy_lengths and docling_lengths:
                comparison['content_length_comparison'] = {
                    'legacy_avg': sum(legacy_lengths) / len(legacy_lengths),
                    'docling_avg': sum(docling_lengths) / len(docling_lengths),
                    'docling_improvement': (sum(docling_lengths) - sum(legacy_lengths)) / sum(legacy_lengths) if sum(legacy_lengths) > 0 else 0
                }
        
        # Compare metadata richness
        legacy_metadata_fields = sum(len(doc.metadata) for doc in legacy_docs if hasattr(doc, 'metadata'))
        docling_metadata_fields = sum(len(doc.metadata) for doc in docling_docs if hasattr(doc, 'metadata'))
        
        comparison['metadata_enhancement'] = {
            'legacy_total_fields': legacy_metadata_fields,
            'docling_total_fields': docling_metadata_fields,
            'enhancement_ratio': docling_metadata_fields / legacy_metadata_fields if legacy_metadata_fields > 0 else 0
        }
        
        return comparison
    
    def _log_comparison_summary(self, comparison: Dict[str, Any]):
        """Log comparison summary."""
        logger.info("📊 Pipeline Comparison Summary:")
        logger.info(f"  Files processed: {comparison['file_count']}")
        logger.info(f"  Legacy success rate: {comparison['legacy_success_rate']:.1%}")
        logger.info(f"  Docling success rate: {comparison['docling_success_rate']:.1%}")
        
        if 'content_length_comparison' in comparison and comparison['content_length_comparison']:
            improvement = comparison['content_length_comparison']['docling_improvement']
            logger.info(f"  Content extraction improvement: {improvement:.1%}")
        
        if 'metadata_enhancement' in comparison:
            enhancement = comparison['metadata_enhancement']['enhancement_ratio']
            logger.info(f"  Metadata enhancement ratio: {enhancement:.1f}x")
    
    def get_migration_report(self) -> Dict[str, Any]:
        """Get comprehensive migration report."""
        report = {
            'migration_mode': self.migration_mode,
            'enhanced_processor_available': self.enhanced_processor is not None,
            'comparison_results': self.comparison_results,
            'total_comparisons': len(self.comparison_results)
        }
        
        if self.enhanced_processor:
            report['enhanced_processor_capabilities'] = self.enhanced_processor.get_capabilities_report()
        
        # Calculate aggregate statistics
        if self.comparison_results:
            docling_rates = [r['docling_success_rate'] for r in self.comparison_results]
            legacy_rates = [r['legacy_success_rate'] for r in self.comparison_results]
            
            report['aggregate_statistics'] = {
                'avg_docling_success_rate': sum(docling_rates) / len(docling_rates),
                'avg_legacy_success_rate': sum(legacy_rates) / len(legacy_rates),
                'total_files_processed': sum(r['file_count'] for r in self.comparison_results)
            }
        
        return report
    
    def recommend_migration_strategy(self) -> Dict[str, Any]:
        """Recommend optimal migration strategy based on results."""
        if not self.comparison_results:
            return {
                'recommendation': 'hybrid',
                'reason': 'No comparison data available, hybrid mode recommended for safety',
                'confidence': 'low'
            }
        
        # Analyze comparison results
        avg_docling_success = sum(r['docling_success_rate'] for r in self.comparison_results) / len(self.comparison_results)
        avg_legacy_success = sum(r['legacy_success_rate'] for r in self.comparison_results) / len(self.comparison_results)
        
        if avg_docling_success > 0.95 and avg_docling_success >= avg_legacy_success:
            return {
                'recommendation': 'docling',
                'reason': f'Docling consistently outperforms (Success rate: {avg_docling_success:.1%})',
                'confidence': 'high'
            }
        elif avg_docling_success > 0.8:
            return {
                'recommendation': 'gradual',
                'reason': f'Docling shows good results (Success rate: {avg_docling_success:.1%}), gradual migration recommended',
                'confidence': 'medium'
            }
        else:
            return {
                'recommendation': 'hybrid',
                'reason': f'Docling success rate too low ({avg_docling_success:.1%}), continue hybrid approach',
                'confidence': 'medium'
            }


# Factory functions for easy integration
def create_migration_manager(mode: str = "gradual") -> IngestionMigrationManager:
    """Create migration manager with specified mode."""
    return IngestionMigrationManager(migration_mode=mode)


def migrate_document_processing(file_paths: List[str], skip_image_processing: bool = False, mode: str = "gradual") -> List[Any]:
    """
    Migrated document processing with intelligent strategy selection.
    
    Args:
        file_paths: List of file paths to process
        skip_image_processing: Skip image processing if True
        mode: Migration mode ('legacy', 'docling', 'hybrid', 'gradual')
        
    Returns:
        List of processed documents
    """
    manager = create_migration_manager(mode)
    return manager.process_documents(file_paths, skip_image_processing)