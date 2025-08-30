# Cortex Suite Changelog

All notable changes to the Cortex Suite project will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]











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