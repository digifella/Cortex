# Cortex Suite Changelog

All notable changes to the Cortex Suite project will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## v4.10.3 - 2025-11-16

### Docling Ingest & Recovery Hardening

Improved document ingestion with Docling, robust database cleanup, and safer WSL/Docker path handling for long-running batches.

#### Post-release updates (2025-11-17)
- Fixed Docker start scripts (CPU & GPU) by switching to heredoc generation to avoid shell quoting errors during image build.
- Knowledge Ingest “Install missing models now” button now re-checks availability after pulling and surfaces Ollama connectivity errors inline.

### ✨ New Features
- Enabled Docling-based ingestion pipeline for richer Office/PDF parsing in both host and Docker engines.
- Added a Clean Start flow in Maintenance with detailed debug output and step-by-step verification of what is deleted.
- Detects orphaned ingestion artifacts (staging, batch_state, progress) during Knowledge Search validation and guides users to Clean Start.
- Introduced shared path utilities to resolve user-provided database roots consistently across Windows, WSL, and Docker.
- New **Metadata & Tag Management** page to browse, filter, and bulk-edit document metadata (including thematic_tags).
- Knowledge Search supports filtering by thematic tags alongside document type and proposal outcome.
- Docker model-init now also pulls `llava:7b` so vision ingestion works out of the box.

### 🚀 Improvements
- Knowledge Ingest now centralizes database path resolution via runtime-safe helpers instead of relying on hardcoded defaults.
- Clean Start and Delete KB functions aggressively clear batch_state, staging_ingestion, recovery metadata, and ingestion logs even after failed runs.
- Knowledge Search validates the configured database path, suggests populated fallback roots when available, and surfaces stale state warnings clearly.
- Batch ingest routing now ensures that analysis and finalization use the live log view with automatic reruns, avoiding confusion with manual refresh-only views.

### 🔧 Bug Fixes
- Fixed Clean Start NameError and ensured it behaves safely when `knowledge_hub_db` is missing or partially created.
- Resolved cases where ingestion logs and UI state would remain in a “processing” state after the ingestion subprocess had terminated.
- Prevented the Active Batch Management panel from blocking automatic finalization and completion routing.
- Hardened WSL/Docker path conversion so ingestion never attempts to write under the project directory instead of the external database path.

## v4.10.2 - 2025-10-11

### Bug Fixes & UX Improvements

Critical bug fixes for ingestion errors, improved search result deduplication, and better user feedback during processing.

### 🔧 Bug Fixes
- Fixed LlamaIndex DocxReader API errors (changed `file_path=` keyword to positional argument)
- Removed calls to non-existent `get_recently_ingested_documents()` method
- Fixed BackupManager async coroutine warning (added asyncio.run wrapper)
- Removed auto-submit behavior in Knowledge Search (now requires explicit button press)
- Improved LLM timeout handling (reduced 600s → 180s with better error messages)

### ✨ New Features
- Search results now deduplicate by document (shows unique docs instead of all chunks)
- Display "X results (Y unique documents)" for better clarity
- Shows "Showing Y unique documents (from X chunks)" when deduplication occurs

### 🚀 Improvements
- Consistent UI behavior across all pages - Tab accepts input, button press initiates action
- Better progress feedback during long-running LLM metadata extraction
- Logs elapsed time for LLM calls taking >60 seconds
- Explicit TimeoutError handling with clear error messages
- Ingestion completion screen now queries ChromaDB for accurate document counts

### 📋 Technical Details
- Fixed 12+ DocxReader.load_data() calls across enhanced_ingest_cortex.py and ingest_cortex.py
- All fixes synchronized to Docker distribution
- Improved error handling prevents confusing timeout messages






## v4.10.1 - 2025-10-09

### GPU Acceleration & Docker Parity

GPU acceleration support for Docker with automatic detection and build optimization, plus UI completion bug fixes

### ✨ New Features
- GPU-enabled Docker build with CUDA 12.1 support (Dockerfile.gpu)
- Automatic GPU detection in run-cortex.bat - builds GPU or CPU image appropriately
- CUDA-enabled PyTorch wheels for NVIDIA GPU acceleration (torch==2.3.1+cu121)
- Comprehensive GPU setup documentation (GPU_SETUP.md)
- Performance benchmarks: 3-5x speedup with GPU acceleration

### 🚀 Improvements
- Docker build now automatically detects NVIDIA GPU via nvidia-smi
- Builds Dockerfile.gpu for NVIDIA systems, standard Dockerfile for CPU-only
- GPU setup guide includes Windows/Linux installation, verification, troubleshooting
- Requirements-gpu.txt for CUDA dependencies separate from base requirements

## v4.10.0 - 2025-10-09

### Performance Analytics & Adaptive Optimization

Comprehensive performance monitoring system with real-time analytics dashboard and GPU-adaptive batch sizing for optimal throughput

### ✨ New Features
- Performance monitoring system tracking all critical operations (image, embedding, query)
- Real-time performance dashboard in Maintenance page with GPU metrics
- GPU memory monitoring with adaptive batch size optimization (4-128)
- Automatic batch size tuning based on available GPU memory
- Cache hit/miss analytics with time savings calculations
- Performance metrics export to JSON for analysis

### 🚀 Improvements
- Embedding batch sizes now adapt to GPU memory (24GB GPU: 128, 16GB: 64, 8GB: 32, CPU: 4)
- Performance data collection with percentile statistics (P50, P95, P99)
- Query cache now tracks hit/miss rates with detailed analytics
- Image processing timing instrumentation for bottleneck identification
- Ingestion finalization UI now properly transitions without persistent messages
- GPU utilization monitoring for CUDA, MPS, and CPU devices

## v4.9.0 - 2025-10-08

### Critical Performance Optimization

Major performance improvements with async image processing, embedding batch optimization, and intelligent query caching delivering 3-5x faster ingestion

### ✨ New Features
- Async parallel image processing with 30s timeout and 3-image concurrency
- Embedding batch processing with GPU vectorization (batch size 32)
- LRU query result caching for instant repeated searches (100 query cache)
- Enhanced progress feedback during parallel image processing
- Automatic cache invalidation on new ingestion

### 🚀 Improvements
- Image processing enabled by default with optimized performance
- Reduced VLM timeout from 120s to 30s per image with graceful fallback
- Batch embedding generation (32 documents per batch) for GPU efficiency
- Query cache with thread-safe OrderedDict-based LRU eviction
- Better error handling and fallback for image processing timeouts
- Improved UI feedback for ingestion finalization completion

## v4.8.1 - 2025-10-06

### Docker Path Configuration & GPU Acceleration

Enhanced Docker installer with dual-path prompts, GPU auto-detection for embeddings, and improved progress visibility

### ✨ New Features
- Dual-path configuration prompts in Docker installer (AI database + Knowledge source)
- GPU auto-detection for embedding model (CUDA, MPS, CPU fallback)
- Per-file progress logging during knowledge ingestion
- Automatic directory creation with validation

### 🚀 Improvements
- Clear separation between database storage and source documents in Docker setup
- Embedding model now uses available GPU (5-10x speedup for NVIDIA/Apple Silicon)
- Better error messages with delayed expansion in batch files
- Drive detection without spurious error messages
- Enhanced visual feedback during Docker initialization

## v4.8.0 - 2025-09-21

### Module Harmonization & Path Standardization

Complete module synchronization between main and docker, standardized path handling, and GraphRAG integration fixes

### ✨ New Features
- Docker version now has complete module parity with main project
- Full EntityExtractor implementation in docker for GraphRAG functionality
- Enhanced directory processing with recursive file discovery
- Comprehensive module synchronization including missing utilities

### 🚀 Improvements
- Replaced all hardcoded paths with centralized get_default_ai_database_path() calls
- Direct script execution in subprocess to avoid module resolution confusion
- Synchronized API signatures between main and docker versions
- Added safety checks for batch status to prevent KeyError crashes
- Consistent Ollama service checking across all environments
- Complete docker distribution with all missing core modules

### 🚧 Known Issues (Docker)
- In some container runs, the ingest UI may briefly report “Analysis complete, but no documents were staged for review” while the staging file does contain documents. A new diagnostics block now shows the exact staging path, parsed count, and a Retry Finalization button to complete the flow.
- Knowledge Search now has a Docker‑local fallback implementation when the project root page is not present in the image; functionality is minimal (direct Chroma search) until the full page is bundled.
- Some .docx readers in Docker can log `DocxReader.load_data() got an unexpected keyword argument 'file_path'`. Analysis still completes; normalization of reader calls is planned.

### ✨ New (Unreleased)
- Host and Docker ingest pages: staging diagnostics (path, exists, parsed count) and JSON preview when no staged docs are found.
- “Retry Finalization” button appears when the staging file contains documents but the UI did not auto‑finalize.
- Docker Knowledge Search loader now falls back to a local implementation if the project page is unavailable in the image.

### 🚀 Improvements (Unreleased)
- Maintenance (host and Docker): clearer Clean Start UI with Windows vs resolved/container paths and pre/post verification of key files (DB folder, collections, staging, graph).


## v4.7.0 - 2025-09-02

### Search Stability & Batch Finalization

Stabilized search embeddings, improved Hybrid results, fixed batch finalization collections, Docker parity, and UX helpers.

### ✨ New Features
- Centralized embedding service powering search and async ingest
- Hybrid search now unions vector + graph results without underflow
- Retry Finalization button when staging is present
- Collections migration health check from project root to external DB

### 🚀 Improvements
- Direct Chroma queries use explicit `query_embeddings` for reliability
- GraphRAG adapter no longer imports LlamaIndex in search path
- Batch finalization writes collections to external DB via manager
- Docker ingest uses same collection manager for parity
- Finalization success toast with collection and counts
- Collections file quick preview and path display

### 🐞 Bug Fixes
- Fixed post-ingest search failures due to embedding mismatches
- Resolved batch finalization not creating collections in external path
- Clarified Chroma tenant/DB validation errors with guidance
- Prevented Hybrid search from returning fewer results than Traditional

















## v4.6.0 - 2025-09-01

### Enhanced Collection Management & Database Clarity

Major improvements to collection management, database deletion clarity, and error handling

### ✨ New Features
- Comprehensive collection clearing functions in Collection Management
- Enhanced log display with scrollable interface and line count selection
- Clear Empty Collections, Clear All Documents, and Reset Collections tools
- Improved error messaging for batch ingestion with success rate context

### 🚀 Improvements
- Collections now stored in KB database path for better organization
- Simplified collection management with single storage location
- Enhanced database deletion with granular error reporting
- Better log display with proper scroll controls and navigation
- Clearer database deletion terminology (Ingested Document Database vs Knowledge Base)
- More informative error messages showing success rates for batch operations

## v4.5.1 - 2025-09-01

### Enhanced Search & Docker Stability

Improved search functionality with timeout protection, better error handling, and Docker path stability fixes

### ✨ New Features
- GraphRAG retroactive extraction utility for existing knowledge bases
- Comprehensive timeout protection for GraphRAG and hybrid search operations
- Enhanced batch ingestion error recovery with user-friendly guidance
- Improved search fallback mechanisms when GraphRAG components fail

### 🚀 Improvements
- Added 45-second timeout for GraphRAG Enhanced search with automatic fallback
- Added 60-second timeout for Hybrid search with graceful degradation
- Enhanced Knowledge Ingest session state handling to preserve user-entered paths
- Better error messages and recovery options for corrupted batch states
- Improved hybrid search deduplication logic for optimal result combination
- Enhanced GraphRAG search debugging with detailed logging

## v4.5.0 - 2025-08-31

### System Stabilization & ChromaDB Consistency Fixes

Emergency stabilization after navigation redesign failure - fixed ChromaDB inconsistencies, Docker path handling, and database validation errors

### ✨ New Features
- Enhanced Clean Start debug logging with step-by-step operations display
- Docker environment detection with proper path fallbacks
- Advanced Database Recovery section with safer Clean Start placement
- Session state synchronization with configuration values

### 🚀 Improvements
- Standardized ChromaDB settings across all components for consistent connections
- Fixed working collections schema compatibility (doc_ids vs documents)
- Enhanced ingestion recovery logic to not flag empty collections as issues
- Improved Docker path handling with /.dockerenv detection
- Better session state initialization in Knowledge Ingest page
- Safer Clean Start button placement in Advanced section
- Enhanced debug information display with visual pauses

## v4.4.2 - 2025-08-31

### Enhanced Debug Logging & Clean Start Fixes

Improved Clean Start debug logging with comprehensive step-by-step information and Docker compatibility fixes

### ✨ New Features
- Step-by-step Clean Start debug logging with visual pause for review
- Comprehensive debug information display in expandable text areas
- Enhanced error logging with detailed troubleshooting information
- Docker-compatible debug logging that stays visible on screen

### 🚀 Improvements
- Clean Start operations now display step-by-step with clear visual feedback
- Debug information remains visible on screen for user review and copying
- Enhanced Docker environment path handling and debugging
- Comprehensive error reporting with detailed troubleshooting steps
- Visual pause after operations completion for thorough review
- Improved debug log format with structured step-by-step information

## v5.0.0 - 2025-08-31

### Navigation Redesign & Knowledge Management Hub

Major navigation restructure with centralized Knowledge Management hub, simplified 7-page menu, and sub-page organization for improved user experience

### 🔥 Breaking Changes
- Complete navigation menu restructure - pages now organized under logical hubs
- Knowledge Management functions consolidated into single hub page
- Sub-pages moved to hidden directory structure
- Page numbering sequence changed from 16 items to 7 clean items

### ✨ New Features
- Knowledge Management Hub - centralized access to all knowledge operations
- Simplified 7-page navigation menu (down from 16+ pages)
- Hidden sub-page organization for cleaner menu structure
- Quick-access buttons for sub-functions (Proposal Copilot, System Terminal)
- Comprehensive debug logging for Clean Start operations with downloadable logs

### 🚀 Improvements
- Dramatically simplified user interface - 56% reduction in visible menu items
- Logical workflow organization - related functions grouped under hubs
- Improved navigation patterns with direct access to commonly used functions
- Clean page numbering sequence (1-7) for better mental model
- Enhanced user experience through reduced cognitive load
- Better separation of core workflows from specialized tools
- Maintained full functionality while improving discoverability

## v4.4.1 - 2025-08-31

### Database Management & Clean Start System

Comprehensive database maintenance system with Clean Start functionality for ChromaDB schema conflict resolution and complete system reset capabilities

### ✨ New Features
- Clean Start function for complete system reset and database schema conflict resolution
- Enhanced ChromaDB schema error detection and user guidance in Collection Management
- Comprehensive database cleanup system addressing Docker vs non-Docker conflicts
- Smart error handling for 'collections.config_json_str' column missing errors
- Complete maintenance workflow with guided user experience for database issues

### 🚀 Improvements
- Enhanced Maintenance page (v4.4.0) with prominent Clean Start functionality
- Improved Collection Management (v4.3.0) with specific schema error detection
- Comprehensive system reset capability removing all databases, logs, and metadata
- Docker compatibility improvements for database schema consistency
- User-friendly error messages with actionable next steps for schema conflicts
- Technical documentation and educational content about ChromaDB version conflicts
- Streamlined database maintenance workflow with one-click solutions

## v4.4.0 - 2025-08-31

### Database Management & Clean Start System

Comprehensive database maintenance system with Clean Start functionality for ChromaDB schema conflict resolution and complete system reset capabilities

### ✨ New Features
- Clean Start function for complete system reset and database schema conflict resolution
- Enhanced ChromaDB schema error detection and user guidance in Collection Management
- Comprehensive database cleanup system addressing Docker vs non-Docker conflicts
- Smart error handling for 'collections.config_json_str' column missing errors
- Complete maintenance workflow with guided user experience for database issues

### 🚀 Improvements
- Enhanced Maintenance page (v4.4.0) with prominent Clean Start functionality
- Improved Collection Management (v4.3.0) with specific schema error detection
- Comprehensive system reset capability removing all databases, logs, and metadata
- Docker compatibility improvements for database schema consistency
- User-friendly error messages with actionable next steps for schema conflicts
- Technical documentation and educational content about ChromaDB version conflicts
- Streamlined database maintenance workflow with one-click solutions

## v4.3.0 - 2025-08-30

### Critical Search Functionality Restoration

Major fix for search functionality with embedding dimension mismatch resolution and robust text-based fallback search system

### ✨ New Features
- Intelligent search fallback system with text-based search when vector embeddings fail
- Enhanced search diagnostics with embedding dimension mismatch detection
- Robust ChromaDB error handling with graceful degradation to text search
- Multi-strategy search approach prioritizing result accuracy over search method
- Search reliability improvements ensuring results are always returned when documents exist

### 🚀 Improvements
- Fixed critical search functionality that was returning zero results due to embedding dimension conflicts
- Restored GraphRAG search capabilities with proper error handling and fallback mechanisms
- Enhanced search result accuracy by implementing text-based matching when vector search fails
- Improved search performance by detecting and avoiding incompatible embedding operations
- Added comprehensive search debugging and error reporting for better troubleshooting
- Strengthened ChromaDB telemetry error suppression for cleaner user experience
- Enhanced Ollama model service with synchronous fallback for event loop issues

## v4.2.1 - 2025-08-30

### GraphRAG Search Integration

Re-enabled GraphRAG enhanced search capabilities with radio button selection and entity relationship analysis in Knowledge Search interface

### ✨ New Features
- GraphRAG search mode selection with Traditional Vector Search, GraphRAG Enhanced, and Hybrid Search options
- Entity-based search feedback showing knowledge graph statistics and relationship counts
- Graph context enhancement for search results with entity relationship analysis
- Real-time GraphRAG health monitoring and status display
- Search source identification in results (Vector Search vs GraphRAG Enhanced)
- Hybrid search combining vector similarity with graph-based entity relationships

### 🚀 Improvements
- Re-connected existing GraphRAG infrastructure (EnhancedGraphManager, EntityExtractor, GraphRAGIntegration)
- Enhanced search result display with graph context information
- Graceful fallbacks from GraphRAG to traditional search when graph data unavailable
- Improved search strategy selection with comprehensive help text
- Real-time feedback on knowledge graph status and entity availability

## v4.2.0 - 2025-08-30

### Streamlined Document Anonymizer Interface

Complete redesign of Document Anonymizer with modern UI patterns matching Document Summarizer for consistent user experience

### 🔥 Breaking Changes
- Document Anonymizer interface completely redesigned (functionality unchanged)

### ✨ New Features
- Streamlined Document Anonymizer interface matching Document Summarizer patterns
- Radio button selection between Upload File and Browse Knowledge Base
- Smart knowledge base directory detection using ConfigManager
- Enhanced anonymization results display with entity metrics
- Interactive entity mapping table with type categorization
- Improved progress tracking with status messages during processing

### 🚀 Improvements
- Eliminated complex folder navigation and drag-drop complexity
- Cleaner two-column layout with focused workflow
- Better file selection with descriptive names and locations
- Enhanced results section with preview and mapping options
- More intuitive confidence threshold slider with better help text
- Consistent UI patterns across Document Summarizer and Anonymizer

## v4.1.3 - 2025-08-30

### Intelligent Model Selection & Resource Management

Smart model selection based on system resources prevents memory crashes and optimizes performance across all hardware configurations

### ✨ New Features
- Intelligent model selection based on available system memory and Docker resource limits
- Automatic Docker environment detection with memory limit handling
- Real-time resource monitoring with user-friendly Streamlit alerts
- Smart model tier selection (efficient vs powerful) based on system capabilities
- Resource compatibility checking before model loading

### 🚀 Improvements
- Default model changed from mistral-small3.2 (26GB) to mistral:7b-instruct-v0.3-q4_K_M (4.4GB)
- Increased document ingestion timeout from 2 minutes to 10 minutes for large documents
- Dynamic task-specific model mapping adapts to system resources
- Memory usage reduced by 80%+ through intelligent model selection
- Added psutil dependency for cross-platform system resource monitoring

## v4.1.2 - 2025-08-29

### ARM64 Compatibility & Multi-Architecture Support

Fixed numpy dependency conflicts on ARM64 processors (Windows Snapdragon, Apple Silicon) with universal CPU-first architecture

### ✨ New Features
- Universal ARM64 and Snapdragon processor support for Windows, Mac, and Linux
- Intelligent PyTorch installation strategy (CPU-first with optional GPU upgrade)
- Multi-architecture Docker builds supporting x86_64, ARM64, and aarch64
- Optional CUDA dependency handling with clear upgrade paths

### 🚀 Improvements
- Removed hardcoded NVIDIA CUDA dependencies from core requirements.txt
- Flexible PyTorch version ranges (>=2.3.1,<2.5.0) for better compatibility
- CPU-optimized installations work immediately on all architectures
- Clear documentation for architecture-specific GPU acceleration upgrades
- Follows dependency resolution best practices from DEPENDENCY_RESOLUTION_GUIDE.md

## v4.0.4 - 2025-08-28

### Simplified Anonymizer Interface

Completely simplified Document Anonymizer with auto-processing and download-only interface

### 🔥 Breaking Changes
- Removed complex multi-tab interface (Browse Files, Browse Directory, Manual Paths)
- Removed preview functionality - files now download only
- Removed manual processing button - files process automatically on upload

### ✨ New Features
- Single drag-and-drop interface with immediate processing
- Automatic file processing upon upload
- Streamlined download-only results interface

### 🚀 Improvements
- Simplified interface from complex multi-tab to single drag-drop area
- Removed unnecessary preview functionality that was causing navigation issues
- Auto-processing eliminates need for manual 'Start Anonymization' button
- Cleaner configuration with persistent session state storage
- Reduced file complexity from 499 lines to 436 lines (-13%)

## v4.0.3 - 2025-08-28

### Anonymizer Output Fix

Fixed Document Anonymizer output file handling and persistence issues

### ✨ New Features
- Enhanced anonymizer output with view and download buttons
- Automatic file saving to Downloads folder when no output directory specified

### 🚀 Improvements
- Fixed output files disappearing after anonymization process
- Added permanent file storage for drag-and-drop uploads
- Enhanced file management with proper temporary file cleanup
- Improved user experience with visual file access controls

## v4.0.2 - 2025-08-28

### Drag-Drop Fix & Utility Registry

Fixed drag-and-drop functionality in Document Anonymizer and created comprehensive utility function registry

### ✨ New Features
- Comprehensive utility function registry documented in CLAUDE.md
- True file drag-and-drop support in Document Anonymizer using st.file_uploader

### 🚀 Improvements
- Fixed Document Anonymizer drag-and-drop that was opening PDFs in browser instead of processing
- Added automatic temporary file cleanup after anonymization
- Enhanced drag-drop with fallback manual path entry
- Centralized utility functions to prevent code duplication

## v4.0.1 - 2025-08-28

### Knowledge Maintenance Consolidation

Moved knowledge maintenance functions from Collection Management to Maintenance page for better organization

### 🔥 Breaking Changes
- Knowledge maintenance functions moved from Collection Management page to Maintenance page

### ✨ New Features
- Enhanced Maintenance page with database deduplication functionality
- Comprehensive knowledge base management in single location

### 🚀 Improvements
- Collection Management page now focused exclusively on collections
- Better separation of concerns between pages
- Clearer user navigation with maintenance functions in dedicated page

## v4.0.0 - 2025-08-28

### Centralized Version Management

Major refactor with centralized versioning, comprehensive cleanup, and software engineering best practices

### 🔥 Breaking Changes
- Centralized version management system
- Consolidated documentation structure
- Removed legacy version inconsistencies

### ✨ New Features
- Single source of truth for version numbers
- Automated changelog generation
- Improved software engineering practices

### 🚀 Improvements
- Consistent versioning across all components
- Cleaner codebase with removed obsolete files
- Better documentation organization

## v3.1.1 - 2025-08-26

### Database Status Enhancement

- Added comprehensive database status display in sidebar
- Showing database path, directory existence, knowledge base status, and document count
- Enhanced user visibility into system state

## v3.1.0 - 2025-08-26

### Major Feature Updates

- **Administrative Function Consolidation**: All database management and administrative functions moved to Maintenance page (page 13)
- **Enhanced Maintenance Interface**: Tabbed interface with Database, Terminal, Setup, Backups, and Info tabs
- **Improved Security**: Consolidated administrative functions with proper access controls
- **Better Organization**: Single location for all maintenance operations

### 🗄️ Database Management
- Clear ingestion logs functionality
- Delete knowledge base operations
- Database health analysis
- Orphaned document recovery
- Collection consistency repairs
- Quick recovery collections from recent files

### 💻 System Terminal
- Safe command execution with whitelisted commands
- Quick actions for system status checks
- Secure environment with audit logging
- Real-time command output display

### ⚙️ Setup Management
- Reset installation state capabilities
- Clear setup progress options
- System component reset and troubleshooting

### 💾 Backup Management
- Timestamped knowledge base backups
- Restore from existing backup files
- Backup lifecycle management
- Backup validation and integrity checking

## Previous Versions (Pre-Centralized Changelog)

Prior to v4.0.0, version tracking was distributed across multiple files and not centrally managed. Key historical versions include:

- **v39.x.x Series**: Hybrid model architecture improvements, Docling integration, advanced document processing
- **v22.x.x Series**: Knowledge search enhancements, multi-strategy search
- **v3.x.x Series**: Service-first architecture, real-time setup monitoring
- **v2.x.x Series**: Major architectural refactoring
- **v1.x.x Series**: Initial stable releases

For detailed historical information, see archived documentation files and git commit history.

---

## Changelog Guidelines

### Version Format
- Follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`
- Use `v` prefix for version tags (e.g., `v4.0.0`)
- Include pre-release indicators when applicable (e.g., `v4.1.0-beta.1`)

### Change Categories
- **🔥 Breaking Changes**: Incompatible API changes, major refactors
- **✨ New Features**: New functionality added
- **🚀 Improvements**: Enhancements to existing functionality
- **🐛 Bug Fixes**: Bug fixes and error corrections
- **📚 Documentation**: Documentation changes
- **🔧 Internal**: Internal changes, refactoring, dependencies
- **🏗️ Infrastructure**: Build system, CI/CD, development tools

### Entry Format
```markdown
## vX.Y.Z - YYYY-MM-DD

### Release Name

Brief description of the release

### Category
- Change description
```

### Maintenance Notes
- This changelog is automatically updated when version_config.py is modified
- All version information should reference cortex_engine/version_config.py
- Breaking changes must be clearly documented
- Include migration guides for major version changes
## v4.7.0 - 2025-09-02

### Search Stability & Docker Parity (Patch Addendum)

This patch delivers end-to-end Docker parity for the Streamlit UI and fixes external-path issues for Chroma and collections when running inside containers. It also restores Proposal Copilot and auxiliary pages in Docker-only distributions via targeted shims.

### ✨ New (Docker-only) Shims
- Added minimal modules under `docker/cortex_engine/` so docker distributions can run without the full engine:
  - `session_state`, `batch_manager`, `document_type_manager`
  - `instruction_parser` (iter_block_items, CortexInstruction, parse_template_for_instructions)
  - `task_engine` (refine, generate/retrieve from KB, assemble document)
  - `proposal_manager` (persist proposals under AI_DATABASE_PATH)
  - `graph_manager` (EnhancedGraphManager), `entity_extractor`
  - `document_summarizer`, `knowledge_synthesizer`, `synthesise`
  - `visual_search`, `help_system`, `setup_manager`, `backup_manager`, `sync_backup_manager`
  - `utils` additions: `process_drag_drop_path`, `process_multiple_drag_drop_paths`

### 🚀 Improvements
- Docker-aware path resolution: consistently maps Windows paths to container mounts, preferring `AI_DATABASE_PATH`.
- Collections and staging stored next to the external DB path (not the repo) in both host and Docker.
- Knowledge Ingest, Collection Management, and Knowledge Search updated to resolve container-visible paths.
- Batch state persisted under `knowledge_hub_db/batch_state.json` in Docker shim.
- Proposal Copilot restored in Docker with instruction parsing, basic generation/refinement, and doc assembly.
- Visual Analysis page now resolves drag/drop paths via centralized utils.

### 🐛 Fixes
- Resolved Chroma/collections not writing to external DB path when running inside Docker.
- Eliminated ModuleNotFoundError across Docker pages by adding targeted shims.
- Fixed local NameError in Knowledge Search (path conversion import).

### 🧩 Installer
- Updated `docker/run-cortex.*` batch installers to show `v4.7.0` and current release name.
