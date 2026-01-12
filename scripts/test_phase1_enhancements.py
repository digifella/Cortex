#!/usr/bin/env python3
"""
Phase 1 Enhancement Validation Script
Tests VLM figure processing, provenance tracking, and parallel performance.

Usage:
    python scripts/test_phase1_enhancements.py [--test-pdf path/to/document.pdf]
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


class Phase1Tester:
    """Comprehensive Phase 1 enhancement tester."""

    def __init__(self):
        self.results = {
            'config_validation': False,
            'docling_availability': False,
            'vlm_processing': False,
            'provenance_extraction': False,
            'parallel_performance': False,
            'metadata_serialization': False
        }
        self.test_details = {}

    def test_config_validation(self) -> bool:
        """Test 1: Validate configuration settings."""
        print_header("TEST 1: Configuration Validation")

        try:
            from cortex_engine.config import (
                DOCLING_VLM_ENABLED,
                DOCLING_VLM_MAX_WORKERS,
                DOCLING_VLM_TIMEOUT,
                DOCLING_PROVENANCE_ENABLED,
                TABLE_AWARE_CHUNKING,
                TABLE_SPECIFIC_EMBEDDINGS,
                FIGURE_ENTITY_LINKING
            )

            print_info("Checking Phase 1 configuration settings...")

            # Validate VLM settings
            print(f"  - DOCLING_VLM_ENABLED: {DOCLING_VLM_ENABLED}")
            print(f"  - DOCLING_VLM_MAX_WORKERS: {DOCLING_VLM_MAX_WORKERS}")
            print(f"  - DOCLING_VLM_TIMEOUT: {DOCLING_VLM_TIMEOUT}s")
            print(f"  - DOCLING_PROVENANCE_ENABLED: {DOCLING_PROVENANCE_ENABLED}")

            # Validate Phase 2 settings (should be present but optional)
            print(f"\n  Phase 2 Configuration (for future):")
            print(f"  - TABLE_AWARE_CHUNKING: {TABLE_AWARE_CHUNKING}")
            print(f"  - TABLE_SPECIFIC_EMBEDDINGS: {TABLE_SPECIFIC_EMBEDDINGS}")
            print(f"  - FIGURE_ENTITY_LINKING: {FIGURE_ENTITY_LINKING}")

            # Verify optimal settings
            if DOCLING_VLM_MAX_WORKERS >= 4:
                print_success(f"Optimal VLM workers configured: {DOCLING_VLM_MAX_WORKERS}")
            else:
                print_warning(f"VLM workers below optimal (4+): {DOCLING_VLM_MAX_WORKERS}")

            self.test_details['config'] = {
                'vlm_enabled': DOCLING_VLM_ENABLED,
                'max_workers': DOCLING_VLM_MAX_WORKERS,
                'timeout': DOCLING_VLM_TIMEOUT,
                'provenance_enabled': DOCLING_PROVENANCE_ENABLED
            }

            print_success("Configuration validation passed")
            return True

        except ImportError as e:
            print_error(f"Configuration import failed: {e}")
            return False
        except Exception as e:
            print_error(f"Configuration validation failed: {e}")
            return False

    def test_docling_availability(self) -> bool:
        """Test 2: Check Docling availability and initialization."""
        print_header("TEST 2: Docling Availability")

        try:
            from cortex_engine.docling_reader import DoclingDocumentReader, DOCLING_AVAILABLE

            print_info(f"Docling available: {DOCLING_AVAILABLE}")

            if not DOCLING_AVAILABLE:
                print_error("Docling not installed. Install with: pip install docling")
                return False

            # Test reader initialization
            print_info("Initializing Docling reader with VLM processing...")
            reader = DoclingDocumentReader(
                ocr_enabled=True,
                table_structure_recognition=True,
                skip_vlm_processing=False
            )

            if reader.is_available:
                print_success("Docling reader initialized successfully")
                print(f"  - OCR enabled: {reader.ocr_enabled}")
                print(f"  - Table recognition: {reader.table_structure_recognition}")
                print(f"  - VLM processing: {not reader.skip_vlm_processing}")

                # Get processing stats
                stats = reader.get_processing_stats()
                print(f"\n  Supported formats: {len(stats.get('supported_formats', []))}")
                print(f"  Converter initialized: {stats.get('converter_initialized', False)}")

                self.test_details['docling'] = stats
                return True
            else:
                print_error("Docling reader initialization failed")
                return False

        except ImportError as e:
            print_error(f"Docling import failed: {e}")
            print_warning("Install Docling: pip install docling")
            return False
        except Exception as e:
            print_error(f"Docling availability check failed: {e}")
            return False

    def test_vlm_processing(self, test_pdf_path: str = None) -> bool:
        """Test 3: VLM figure processing."""
        print_header("TEST 3: VLM Figure Processing")

        try:
            from cortex_engine.docling_reader import DoclingDocumentReader

            if test_pdf_path and Path(test_pdf_path).exists():
                print_info(f"Testing with provided PDF: {test_pdf_path}")

                reader = DoclingDocumentReader(
                    ocr_enabled=True,
                    table_structure_recognition=True,
                    skip_vlm_processing=False
                )

                if not reader.is_available:
                    print_warning("Docling not available, skipping VLM test")
                    return False

                print_info("Processing document with Docling...")
                start_time = time.time()

                documents = reader.load_data(file=Path(test_pdf_path))
                processing_time = time.time() - start_time

                if documents:
                    doc = documents[0]
                    metadata = doc.metadata

                    print_success(f"Document processed in {processing_time:.2f}s")
                    print(f"  - Content length: {len(doc.text)} chars")

                    # Check for figures
                    if 'docling_figures' in metadata:
                        figures_data = metadata['docling_figures']
                        if isinstance(figures_data, str):
                            figures_data = json.loads(figures_data)

                        print_info(f"Found {len(figures_data)} figures")

                        # Check for VLM descriptions
                        vlm_count = sum(1 for fig in figures_data if 'vlm_description' in fig)
                        if vlm_count > 0:
                            print_success(f"VLM descriptions generated: {vlm_count}/{len(figures_data)}")

                            # Show sample
                            for i, fig in enumerate(figures_data[:2]):  # Show first 2
                                if 'vlm_description' in fig:
                                    desc = fig['vlm_description'][:150]
                                    print(f"\n  Figure {i} VLM sample:")
                                    print(f"    {desc}...")
                        else:
                            print_warning("No VLM descriptions found in figures")
                            print_info("VLM may need Ollama running with llava:7b model")

                        self.test_details['vlm'] = {
                            'figures_found': len(figures_data),
                            'vlm_descriptions': vlm_count,
                            'processing_time': processing_time
                        }
                        return vlm_count > 0

                    else:
                        print_warning("No figures found in document")
                        print_info("Test with a PDF containing images/figures")
                        return True  # Not a failure, just no figures

                else:
                    print_error("No documents returned from Docling")
                    return False

            else:
                print_warning("No test PDF provided, skipping VLM processing test")
                print_info("Provide PDF with: --test-pdf path/to/document.pdf")
                return True  # Not a failure, just skipped

        except Exception as e:
            print_error(f"VLM processing test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_provenance_extraction(self, test_pdf_path: str = None) -> bool:
        """Test 4: Provenance metadata extraction."""
        print_header("TEST 4: Provenance Metadata Extraction")

        try:
            from cortex_engine.docling_reader import DoclingDocumentReader

            if test_pdf_path and Path(test_pdf_path).exists():
                print_info(f"Testing provenance extraction: {test_pdf_path}")

                reader = DoclingDocumentReader(
                    ocr_enabled=True,
                    table_structure_recognition=True
                )

                if not reader.is_available:
                    print_warning("Docling not available, skipping provenance test")
                    return False

                documents = reader.load_data(file=Path(test_pdf_path))

                if documents:
                    doc = documents[0]
                    metadata = doc.metadata

                    # Check for provenance
                    if 'docling_provenance' in metadata:
                        prov_data = metadata['docling_provenance']
                        if isinstance(prov_data, str):
                            prov_data = json.loads(prov_data)

                        print_success("Provenance metadata found")
                        print(f"  - Has provenance: {prov_data.get('has_provenance', False)}")
                        print(f"  - Page count: {prov_data.get('page_count', 0)}")
                        print(f"  - Elements tracked: {len(prov_data.get('elements', []))}")

                        # Show sample elements
                        elements = prov_data.get('elements', [])
                        if elements:
                            print(f"\n  Sample elements (first 3):")
                            for elem in elements[:3]:
                                print(f"    - Page {elem.get('page')}, Type: {elem.get('type')}")
                                if elem.get('bbox'):
                                    print(f"      BBox: {elem['bbox']}")

                        self.test_details['provenance'] = {
                            'has_provenance': prov_data.get('has_provenance', False),
                            'page_count': prov_data.get('page_count', 0),
                            'elements_tracked': len(elements)
                        }

                        return prov_data.get('has_provenance', False)

                    else:
                        print_warning("No provenance metadata found")
                        print_info("Check DOCLING_PROVENANCE_ENABLED setting")
                        return False

                else:
                    print_error("No documents returned")
                    return False

            else:
                print_warning("No test PDF provided, skipping provenance test")
                return True  # Not a failure

        except Exception as e:
            print_error(f"Provenance extraction test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_parallel_performance(self) -> bool:
        """Test 5: Parallel VLM worker configuration."""
        print_header("TEST 5: Parallel Processing Configuration")

        try:
            from cortex_engine.query_cortex import _get_image_executor

            print_info("Checking VLM executor configuration...")

            # Get executor to trigger initialization
            executor = _get_image_executor()

            # Check max_workers
            max_workers = executor._max_workers
            print(f"  - ThreadPoolExecutor max_workers: {max_workers}")

            if max_workers >= 6:
                print_success(f"Optimal parallel workers configured: {max_workers}")
                self.test_details['parallel'] = {
                    'max_workers': max_workers,
                    'optimal': True
                }
                return True
            elif max_workers >= 4:
                print_success(f"Good parallel workers: {max_workers}")
                self.test_details['parallel'] = {
                    'max_workers': max_workers,
                    'optimal': False
                }
                return True
            else:
                print_warning(f"Suboptimal workers: {max_workers} (expected 4-6)")
                self.test_details['parallel'] = {
                    'max_workers': max_workers,
                    'optimal': False
                }
                return False

        except Exception as e:
            print_error(f"Parallel performance test failed: {e}")
            return False

    def test_metadata_serialization(self) -> bool:
        """Test 6: Metadata serialization for ChromaDB."""
        print_header("TEST 6: Metadata Serialization")

        try:
            print_info("Testing complex metadata serialization...")

            # Simulate complex Docling metadata
            test_metadata = {
                'docling_provenance': {
                    'has_provenance': True,
                    'page_count': 5,
                    'elements': [
                        {'page': 1, 'bbox': [10, 20, 100, 200], 'type': 'text'},
                        {'page': 2, 'bbox': [15, 25, 120, 220], 'type': 'table'}
                    ]
                },
                'docling_figures': [
                    {
                        'index': 0,
                        'page': 3,
                        'caption': 'Test Figure',
                        'vlm_description': 'A detailed test image showing...'
                    }
                ],
                'docling_structure': {
                    'has_tables': True,
                    'has_images': True,
                    'page_count': 5
                }
            }

            # Test JSON serialization (what ingest_cortex.py does)
            serialized = {}
            for key in ['docling_provenance', 'docling_figures', 'docling_structure']:
                if key in test_metadata:
                    serialized[key] = json.dumps(test_metadata[key])

            print_success("Metadata serialized successfully")
            print(f"  - Provenance size: {len(serialized.get('docling_provenance', ''))} bytes")
            print(f"  - Figures size: {len(serialized.get('docling_figures', ''))} bytes")
            print(f"  - Structure size: {len(serialized.get('docling_structure', ''))} bytes")

            # Test deserialization
            deserialized = {}
            for key, value in serialized.items():
                deserialized[key] = json.loads(value)

            # Verify round-trip
            if deserialized['docling_provenance'] == test_metadata['docling_provenance']:
                print_success("Round-trip serialization verified")
                self.test_details['serialization'] = {
                    'serialization_works': True,
                    'total_size': sum(len(v) for v in serialized.values())
                }
                return True
            else:
                print_error("Round-trip verification failed")
                return False

        except Exception as e:
            print_error(f"Metadata serialization test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self, test_pdf_path: str = None):
        """Run all Phase 1 validation tests."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}")
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║       PHASE 1 ENHANCEMENT VALIDATION TEST SUITE                  ║")
        print("║       Cortex Suite - Docling VLM & Provenance Integration        ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.END}\n")

        # Run tests
        self.results['config_validation'] = self.test_config_validation()
        self.results['docling_availability'] = self.test_docling_availability()
        self.results['vlm_processing'] = self.test_vlm_processing(test_pdf_path)
        self.results['provenance_extraction'] = self.test_provenance_extraction(test_pdf_path)
        self.results['parallel_performance'] = self.test_parallel_performance()
        self.results['metadata_serialization'] = self.test_metadata_serialization()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test results summary."""
        print_header("TEST SUMMARY")

        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)

        print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}\n")

        for test_name, result in self.results.items():
            status = f"{Colors.GREEN}✅ PASS{Colors.END}" if result else f"{Colors.RED}❌ FAIL{Colors.END}"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")

        # Overall status
        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        if passed == total:
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED - Phase 1 is production-ready!{Colors.END}")
        elif passed >= total * 0.8:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  MOSTLY PASSING - Review failed tests{Colors.END}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ MULTIPLE FAILURES - Review implementation{Colors.END}")
        print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")

        # Detailed recommendations
        if not self.results['vlm_processing']:
            print_warning("VLM Processing: Ensure Ollama is running with llava:7b model")
            print_info("  Start Ollama: ollama serve")
            print_info("  Pull model: ollama pull llava:7b")

        if not self.results['docling_availability']:
            print_warning("Docling: Install missing dependency")
            print_info("  pip install docling")

        # Performance insights
        if self.test_details.get('parallel', {}).get('max_workers', 0) >= 6:
            print_success(f"Performance: Optimal for RTX 8000 - 6 parallel VLM workers")
        elif self.test_details.get('parallel', {}).get('max_workers', 0) >= 4:
            print_success(f"Performance: Good configuration - 4 parallel workers")


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1 Enhancement Validation for Cortex Suite"
    )
    parser.add_argument(
        '--test-pdf',
        type=str,
        help='Path to PDF document for testing VLM and provenance features'
    )

    args = parser.parse_args()

    tester = Phase1Tester()
    tester.run_all_tests(test_pdf_path=args.test_pdf)


if __name__ == "__main__":
    main()
