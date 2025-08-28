# Cortex Suite Changelog

All notable changes to the Cortex Suite project will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]





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