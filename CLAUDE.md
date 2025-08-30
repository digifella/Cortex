# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Cortex Suite is a Streamlit-based AI-powered knowledge management and proposal generation system. It features integrated GraphRAG capabilities with entity extraction, relationship mapping, and hybrid vector + graph search. The system operates in a WSL2 environment and requires Python 3.11.

## Architecture

### Core Components
- **Streamlit Application**: `Cortex_Suite.py` - Main entry point with multi-page UI
- **Backend Engine**: `cortex_engine/` - Core business logic and data processing
- **Model Services**: `cortex_engine/model_services/` - Hybrid Docker/Ollama backend management
- **Document Processing**: Enhanced ingestion with Docling + LlamaIndex fallback
- **Page Components**: `pages/` - Individual UI pages for different workflows
- **Knowledge Storage**: ChromaDB vector store + NetworkX knowledge graph
- **Entity Extraction**: spaCy NER with custom pattern matching for consultants, clients, projects

### Key Workflows
1. **AI Assisted Research** → **Knowledge Ingest** → **Knowledge Search** → **Collection Management**
2. **Proposal Step 1 Prep** (Template Editor) → **Proposal Step 2 Make** → **Proposal Copilot**

### Data Flow
- Documents ingested through 3-stage process with entity/relationship extraction
- Entities (people, organizations, projects) stored in NetworkX graph
- Vector embeddings stored in ChromaDB
- Working collections curated from search results
- Proposals generated using knowledge base + graph context

## Development Commands

### ⚠️ CRITICAL REMINDERS FOR DEVELOPERS

**🚨 NEW: Centralized Version Management (v4.0.0+)**

All version information is now centralized in `cortex_engine/version_config.py`. This is the SINGLE SOURCE OF TRUTH for version numbers.

**Essential Version Management Workflow:**
1. 📝 **Update version_config.py** - Change CORTEX_VERSION and VERSION_METADATA  
2. 🔄 **Run sync command** - `python scripts/version_manager.py --sync-all`
3. 📋 **Update changelog** - `python scripts/version_manager.py --update-changelog`  
4. ✅ **Verify consistency** - `python scripts/version_manager.py --check`
5. 💾 **Commit all changes** together with proper version tags
6. 📤 **Push all changes** to remote repository

**FAILURE TO FOLLOW VERSION WORKFLOW = INCONSISTENT VERSION NUMBERS ACROSS COMPONENTS**

**🚨 MANDATORY: Before making ANY code changes, follow DISTRIBUTION_SYNC_CHECKLIST.md**

**Essential steps:**
1. 📋 **Follow Distribution Sync Checklist** - See `DISTRIBUTION_SYNC_CHECKLIST.md`
2. 📝 **Use centralized versioning** - Update only `cortex_engine/version_config.py`
3. 🔄 **Sync all components** - Run `python scripts/version_manager.py --sync-all`
4. 🐳 **Sync Docker directory** with updated files using rsync
5. ✅ **Test platform installers** on Windows/Mac/Linux
6. 💾 **Commit with proper version tags** 
7. 📤 **Push all changes** together

**FAILURE TO FOLLOW CHECKLIST = BROKEN INSTALLATIONS FOR USERS**

**See "Development Workflow & Synchronization" and `DISTRIBUTION_SYNC_CHECKLIST.md`**

### Environment Setup
```bash
# Create Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Install Mistral Small 3.2 for proposals (REQUIRED)
ollama pull mistral-small3.2
```

### Running the Application
```bash
# Start Ollama service (for local models)
ollama serve

# Start the Streamlit application
streamlit run Cortex_Suite.py
```

### System Dependencies
```bash
# Install Graphviz for mind map generation (Ubuntu/Debian)
sudo apt-get install graphviz

# Install Ollama for local LLM support
curl -fsSL https://ollama.ai/install.sh | sh

# Download required NLTK data for document processing
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# For other systems, see README.md section 6
```

### Administrative Functions

**IMPORTANT**: As of v4.0.4, all database management and administrative functions have been consolidated into the **Maintenance page (page 13)** for better organization and security.

#### Available in Maintenance Page:
- **Database Maintenance**: Clear ingestion logs, delete knowledge base, analyze database state
- **Recovery Tools**: Recover orphaned documents, repair collections, create recovery collections
- **System Terminal**: Safe command execution with quick actions
- **Setup Management**: Reset installation state, reconfigure system components  
- **Backup Management**: Create, restore, and manage knowledge base backups
- **Changelog Viewer**: **NEW (v4.0.4)**: Browse project version history and development changes

#### Command Line Tools (Alternative):
```bash
# Inspect knowledge graph and database contents
python scripts/cortex_inspector.py --db-path /mnt/f/ai_databases --stats

# Clean up Windows metadata files in WSL
find . -type f -name "*:Zone.Identifier" -delete

# Fix Windows batch file line endings after editing in WSL
sed -i 's/$/\r/' docker/run-cortex.bat

# Fix Windows Docker build context issues
# Ensure .dockerignore excludes Windows system folders
echo '$RECYCLE.BIN/' >> docker/.dockerignore
echo 'System Volume Information/' >> docker/.dockerignore
```

## Configuration

### Required Environment Variables (.env file)
```bash
LLM_PROVIDER="gemini"  # or "ollama" or "openai"
OLLAMA_MODEL="mistral:7b-instruct-v0.3-q4_K_M"
OPENAI_API_KEY="your_openai_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
YOUTUBE_API_KEY="your_google_api_key_for_youtube_search"
GRAPHVIZ_DOT_EXECUTABLE="/usr/bin/dot"
```

### Core Configuration Files
- `cortex_config.json` - Persistent user settings (database paths, last used directories)
- `working_collections.json` - User-curated document collections
- `boilerplate.json` - Reusable text snippets
- `staging_ingestion.json` - Temporary file for AI-suggested metadata review

## Key Technical Details

### Entity Extraction System
- **Location**: `cortex_engine/entity_extractor.py`
- **Dependencies**: spaCy NER + custom patterns
- **Entities**: People (consultants, authors), Organizations (clients, partners), Projects, Documents
- **Relationships**: authored, client_of, collaborated_with, mentioned_in, documented_in

### Database Structure
- **Vector Store**: ChromaDB at `<db_path>/knowledge_hub_db/`
- **Knowledge Graph**: NetworkX pickle at `<db_path>/knowledge_cortex.gpickle`
- **Image Store**: `<db_path>/knowledge_hub_db/images/`

### Default Paths (WSL2 Environment)
- **Base Data Path**: `/mnt/f/ai_databases` (fallback, overrideable)
- **Project Root**: Repository root directory
- **Logs**: `logs/` directory (ingestion.log, query.log)

### Critical Version Requirements
- **Python**: 3.11 (required for stability)
- **NumPy**: <2.0.0 (compatibility with spaCy/ChromaDB)
- **spaCy**: 3.5.0-3.8.0 range

## File Organization

### Backend (`cortex_engine/`)
- `ingest_cortex.py` - Document ingestion with entity extraction
- `graph_manager.py` - Knowledge graph operations
- `entity_extractor.py` - Entity and relationship extraction
- `query_cortex.py` - Knowledge base querying
- `task_engine.py` - AI task execution for proposals
- `config.py` - Central configuration
- `exceptions.py` - Standardized exception hierarchy
- `utils/` - **NEW**: Centralized utility modules
  - `path_utils.py` - Cross-platform path handling
  - `logging_utils.py` - Standardized logging configuration
  - `config_utils.py` - Configuration validation utilities
  - `file_utils.py` - File operations and hashing
  - `ollama_utils.py` - Ollama service connection and error handling
  - `model_checker.py` - Model availability validation and user notifications

### UI Pages (`pages/`)
- `1_AI_Assisted_Research.py` - Multi-agent research system
- `2_Knowledge_Ingest.py` - Document ingestion UI (maintenance functions moved to page 13)
- `3_Knowledge_Search.py` - Vector + graph search
- `4_Collection_Management.py` - Working collections CRUD
- `5_Proposal_Step_1_Prep.py` - Template editor
- `6_Proposal_Step_2_Make.py` - Proposal lifecycle
- `Proposal_Copilot.py` - AI-assisted proposal drafting
- `10_Idea_Generator.py` - Double Diamond innovation methodology
- `12_System_Terminal.py` - Redirects to Maintenance page (consolidated)
- `13_Maintenance.py` - **NEW**: Consolidated administrative functions

#### Maintenance Page Details (`13_Maintenance.py`)
The Maintenance page provides a centralized, tabbed interface for all administrative functions:

**🗄️ Database Tab:**
- Clear ingestion logs to re-scan all files
- Delete entire knowledge base (destructive operation)  
- Analyze ingestion state and database health
- Recover orphaned documents from failed ingestions
- Auto-repair collection inconsistencies
- Quick recovery: create collections from recent files

**💻 Terminal Tab:**
- Safe command execution with whitelisted commands
- Quick actions: check models, system status, disk usage
- Secure environment with audit logging
- Real-time command output display

**⚙️ Setup Tab:**
- Reset installation state if setup gets stuck
- Clear setup progress and allow reconfiguration
- System component reset and troubleshooting

**💾 Backups Tab:**
- Create timestamped knowledge base backups
- Restore from existing backup files
- Manage backup lifecycle (view, verify, delete)
- Backup validation and integrity checking

**ℹ️ Info Tab:**
- Documentation and help information
- Security notes and best practices
- Feature descriptions and usage guidance

## Development Notes

### Distribution & Versioning Standards

#### Docker Distribution Rules
- **Single .dockerignore**: Keep ONLY in `/docker/.dockerignore` (remove any from project root)
- **Minimal Documentation**: Only `/docker/README.md` needed (remove FRIEND_INSTALL_GUIDE.md, DISTRIBUTION_PACKAGE.md, etc.)
- **Clean Package**: Exclude ALL user data, databases, proposals, external_research, logs, media files
- **Fresh Installation**: Every Docker deployment creates completely fresh databases and configurations

#### Version Numbering Standards (v2.0.0+)

**IMPORTANT**: All version numbers follow semantic versioning with "v" prefix (e.g., v1.0.0, v2.1.3)

##### Main Application Versioning
- **v1.x.x**: Initial stable release
- **v2.x.x**: Service-First Architecture (major refactor)
- **v3.x.x+**: Future major architectural changes

##### Page Versioning Standard
All Streamlit pages must follow this format:
```python
# [Page Name] Page
# Version: v1.0.0  
# Date: 2025-08-16
# [Brief description]

import streamlit as st
# ... other imports ...

st.set_page_config(page_title="Page Title", layout="wide")

# Page configuration
PAGE_VERSION = "v1.0.0"

# ... page logic ...

st.title("📊 Page Title")
st.caption(f"Version: {PAGE_VERSION}")
```

**Page Version Numbering Rules:**
- **v1.0.0**: Baseline functionality or new pages
- **v1.0.1**: Bug fixes, minor improvements, code organization
- **v1.1.0**: New features or significant enhancements  
- **v1.2.0**: Major feature additions or multiple enhancements
- **v2.0.0+**: Major architectural changes (rare - usually follows main app versioning)

#### Distribution File Structure
```
docker/
├── README.md                 # Single consolidated guide
├── .dockerignore            # Excludes user data/Windows folders  
├── run-cortex.bat          # Windows installer
├── run-cortex.sh           # Unix installer
├── Dockerfile              # Build instructions
├── docker-compose.yml      # Container orchestration
├── .env.example            # Configuration template
└── requirements.txt        # Python dependencies (REQUIRED for Docker build)
```

**CRITICAL**: When creating distribution packages, ensure these core files are copied to the distribution directory:

#### Required Files
- `requirements.txt` - Python dependencies needed by Dockerfile
- `Cortex_Suite.py` - Main application entry point (includes setup progress page)
- `Dockerfile` - Container build instructions (service-first architecture)
- `.env.example` - Configuration template
- `run-cortex-FIXED.bat` - Windows launcher script (with persistent volumes)

#### Required Directories (copy entire directories)
- `api/` - API server module (FastAPI endpoints with graceful model handling)
- `cortex_engine/` - Core business logic and data processing
  - **INCLUDES**: `system_status.py` for real-time setup monitoring
- `pages/` - Streamlit page components
- `scripts/` - Utility scripts (optional but recommended)

#### Docker Distribution Checklist
When preparing a Docker distribution for Windows:

1. **Create target directory** (e.g., `E:\Docker_Cortex`)

2. **Copy root files:**
   ```
   requirements.txt
   Cortex_Suite.py  
   Dockerfile
   .env.example
   run-cortex-FIXED.bat
   ```

3. **Copy complete directories:**
   ```
   api/           (entire directory with all subdirectories)
   cortex_engine/ (entire directory with all subdirectories)  
   pages/         (entire directory with all subdirectories)
   scripts/       (optional - utility scripts)
   ```

4. **Verify structure:**
   ```
   E:\Docker_Cortex\
   ├── Cortex_Suite.py
   ├── Dockerfile  
   ├── requirements.txt
   ├── .env.example
   ├── run-cortex-FIXED.bat
   ├── api\
   │   ├── __init__.py
   │   ├── main.py
   │   └── README.md
   ├── cortex_engine\
   │   ├── __init__.py
   │   ├── config.py
   │   ├── [all other .py files]
   │   └── utils\
   └── pages\
       ├── 1_AI_Assisted_Research.py
       ├── 2_Knowledge_Ingest.py
       └── [all other page files]
   ```

5. **Missing any of these files/directories will cause:**
   - `requirements.txt` missing → Docker build fails
   - `api/` missing → `ModuleNotFoundError: No module named 'api'`
   - `cortex_engine/` missing → Import errors for core functionality
   - `cortex_engine/system_status.py` missing → Setup progress page fails
   - `pages/` missing → Streamlit navigation fails
   - `Cortex_Suite.py` missing → Application won't start

6. **New User Experience (Service-First Architecture):**
   - **10 seconds**: Web interface becomes accessible
   - **Setup Progress Page**: Shows real-time download status and system health
   - **Background Downloads**: 20GB of AI models download while interface is usable
   - **Progressive Activation**: Features enable automatically as models become available
   - **No Command Line Waiting**: Users interact with professional web interface instead of terminal logs

## 🐳 Critical Docker vs WSL Path Handling Guidelines

**CRITICAL**: This is a recurring issue that causes functionality to break. Must be understood and followed.

### **Path Handling Rules by Environment**

**1. Docker Container Environment (`os.path.exists('/.dockerenv')`):**
- ✅ **Use paths EXACTLY as configured** by the user
- ❌ **DO NOT** use `convert_windows_to_wsl_path()` - Docker handles volume mapping
- **User path**: `/data/ai_databases` → **Use**: `/data/ai_databases` 
- **User path**: `C:\ai_databases` → **Use**: `C:\ai_databases` (mapped by Docker volume)
- **WHY**: Docker volumes automatically map host paths to container paths

**2. WSL Environment (Windows Subsystem for Linux):**
- ✅ **Use** `convert_windows_to_wsl_path()` for Windows-style paths
- **User path**: `C:\ai_databases` → **Convert to**: `/mnt/c/ai_databases`
- **User path**: `/mnt/e/data` → **Use directly**: `/mnt/e/data`

**3. Native Linux/Host Environment:**
- ✅ **Use paths directly** without conversion
- **User path**: `/home/user/data` → **Use**: `/home/user/data`

### **Detection Pattern**
```python
if os.path.exists('/.dockerenv'):
    # Docker container - use path as-is, Docker volumes handle mapping
    final_path = user_configured_path
else:
    # WSL/Host - use conversion if needed  
    final_path = convert_windows_to_wsl_path(user_configured_path)
```

### **❌ Common Mistakes to AVOID**
1. **Hardcoding Docker paths** like `/data`, `/app/data` - always use user's configured path
2. **Using WSL conversion in Docker** - breaks because `/mnt/c/` doesn't exist in containers
3. **Assuming path structure** - different users have different mount points and database locations
4. **Overriding user configuration** - always respect whatever database path user has configured

### **✅ Correct Implementation**
```python
def get_database_path(configured_path: str) -> str:
    """Get the correct database path for current environment"""
    if os.path.exists('/.dockerenv'):
        # Docker: Trust user's configured path, Docker volumes handle host mapping
        return configured_path
    else:
        # WSL/Host: Convert Windows paths to proper format if needed
        return convert_windows_to_wsl_path(configured_path)
```

### **🔍 Why This Issue Keeps Recurring**
- **Volume Mapping Confusion**: Docker `-v C:\data:/container_data` maps host to container automatically
- **WSL Path Similarity**: WSL `/mnt/c/` looks like container paths but operates differently
- **Environment Detection**: Must correctly detect Docker vs WSL vs Native Linux
- **User Path Variety**: Users configure different database locations (`C:\ai_databases`, `/data`, etc.)

**GOLDEN RULE**: NEVER hardcode paths. Always use user's configured database path and let Docker/WSL handle the underlying mapping.

### Critical Windows Batch File Lessons Learned
1. **Multi-line If Statements**: Windows batch files cannot reliably parse multi-line if statements with parentheses. Use `goto` labels instead.
2. **ErrorLevel Overwriting**: Commands like `del`, `copy`, etc. overwrite the `%errorlevel%` variable. Always check `errorlevel` immediately after the command you care about.
3. **Docker Build Context on Windows**: When building from Windows paths, Docker may include system folders like `$RECYCLE.BIN`. Use `.dockerignore` in the build context directory (not just the docker subdirectory).
4. **Container Detection Logic**: `findstr` returns errorlevel 1 when NO match is found, errorlevel 0 when match IS found. Logic should be `if not errorlevel 1` for "found" conditions.

### Recent Architectural Improvements (v4.0.0+)

#### Centralized Version Management (v4.0.0+)
- **Single Source of Truth**: `cortex_engine/version_config.py` manages all version information
- **Automated Syncing**: `scripts/version_manager.py` synchronizes versions across 50+ files
- **Semantic Versioning**: Proper MAJOR.MINOR.PATCH with metadata tracking
- **Changelog Management**: Automated changelog generation and docker directory syncing
- **Version Consistency**: Automated checking prevents version drift across components

#### Document Anonymizer Simplification (v4.0.4)
- **Interface Streamlined**: Eliminated complex multi-tab UI (Browse Files, Browse Directory, Manual Paths)
- **Auto-Processing**: Files process immediately upon drag-and-drop upload
- **Download-Only Results**: Removed problematic preview functionality causing navigation issues
- **Fixed Navigation Bug**: Eye icon redirect to Idea Generator completely resolved
- **Code Reduction**: 499 lines → 436 lines (-13% complexity)

#### Maintenance Page Enhancement (v4.0.4)
- **Changelog Viewer**: Browse complete project version history with recent/full view options
- **Download Integration**: Export changelog as markdown file
- **Auto-Sync**: Changelog automatically synced between project and docker directories
- **Consolidated Administration**: All maintenance functions in single tabbed interface

#### Previous Major Enhancements
- **Hybrid Model Architecture**: Docker Model Runner + Ollama integration with intelligent selection
- **Docling Integration**: IBM Research document processing with layout preservation and OCR
- **Centralized Utilities**: Code deduplication via `cortex_engine/utils/` modules
- **Standardized Logging**: Consistent logging patterns across all components
- **Exception Hierarchy**: Structured error handling with `cortex_engine/exceptions.py`

#### Service-First Architecture (v71.3.0+)
- **Immediate Web Access**: Streamlit and API start within 10 seconds, regardless of model download status
- **Background Model Downloads**: AI models (20GB total) download in background while interface is accessible
- **Real-time Setup Progress**: Live progress tracking with service status, model download progress, and error reporting
- **Progressive Feature Enablement**: Basic features work immediately, AI features activate as models become available
- **Professional Setup Experience**: Users see branded setup progress page instead of waiting at command line
- **System Status Monitoring**: `cortex_engine/system_status.py` provides real-time health checks for all components
- **Docker Volume Persistence**: AI models persist between container rebuilds via `cortex_ollama` volume
- **Enhanced User Experience**: No more 30-minute waits - users can explore interface immediately while setup completes

#### Model Availability & Error Handling (v39.2.0+)
- **Pre-flight Model Checking**: Validates required models before ingestion starts
- **System Status Dashboard**: Real-time model availability in main page sidebar
- **User-Friendly Error Messages**: Clear instructions for missing models with installation commands
- **Graceful Fallbacks**: Option to skip image processing if VLM unavailable
- **Warning Suppression**: Cleaned up logs by filtering harmless library warnings
- **Encoding Error Handling**: Robust handling of binary files and unsupported formats
- **Enhanced File Filtering**: Updated exclusions (added .key, .swf; removed .ppt/.pptx)
- **Progress Monitoring**: Real-time file scanning progress with time estimates

#### Idea Generator Feature (v39.1.0+)
- **Complete Double Diamond Implementation**: All four phases (Discover, Define, Develop, Deliver) fully operational
- **Multi-Agent Ideation**: Solution brainstormer, analogy finder, and feasibility analyzer for diverse idea generation
- **Structured Output**: Detailed reports with implementation roadmaps, risk analysis, and resource requirements
- **Save Functionality**: Export results as organized Markdown and JSON files with timestamps
- **Phase Navigation**: Seamless flow through the ideation process with context preservation
- **Collection Integration**: Works with existing working collections for knowledge-based idea generation

#### Windows Distribution & Docker Issues (v39.3.0+)
- **Batch File Syntax Fixes**: Resolved multi-line if statement parsing errors by using goto labels instead of parenthetical blocks
- **Error Level Handling**: Fixed container detection logic where `del` commands were overwriting `errorlevel` values from `findstr`
- **Docker Build Context**: Resolved Windows `$RECYCLE.BIN` access denied errors by dynamically copying `.dockerignore` to parent build context
- **Line Ending Issues**: Standardized CRLF handling for Windows batch file compatibility
- **Container Detection**: Fixed backwards logic in Docker container existence checking

#### Hybrid Model Services Architecture (Latest - Aug 2025)
- **Multi-Backend Support**: Intelligent orchestration between Docker Model Runner and Ollama
- **Automatic Fallbacks**: Graceful degradation when preferred backends unavailable
- **Performance Optimization**: 15% faster inference through Docker Model Runner's native execution
- **Migration Tools**: Seamless model transitions with compatibility validation
- **Distribution Strategies**: Configurable backend preferences (hybrid_docker_preferred, hybrid_ollama_preferred, auto_optimal)
- **Enterprise Ready**: OCI-compliant model distribution with centralized registry support

#### Advanced Document Processing (Latest - Aug 2025)  
- **Docling Pipeline**: IBM Research's state-of-the-art document conversion technology
- **Layout Preservation**: Maintains document structure, headers, tables, and reading order
- **Multi-Format Support**: PDF, DOCX, PPTX, XLSX, Images, HTML, Markdown, AsciiDoc
- **OCR Integration**: Advanced optical character recognition for scanned documents
- **Smart Migration**: Gradual transition from legacy readers with A/B testing capability
- **Fallback Resilience**: Automatic fallback to LlamaIndex readers for maximum compatibility

#### Docker Environment Fixes (Aug 24, 2025)
- **Automatic Legacy Mode**: Docker environments automatically default to legacy mode (detects `/.dockerenv`)
- **Infinite Recursion Fix**: Fixed migration manager calling itself recursively in legacy mode
- **Torch Compatibility**: Graceful PowerPoint reader initialization with fallback for torch < 2.6
- **Stable Docker Ingestion**: Zero dependency conflicts, reliable document processing in containers
- **Error Recovery**: Comprehensive error handling prevents crashes from library incompatibilities

#### Planned Enhancements (Sprints 4-7):
- **Smart Filtering**: Metadata-based collection filtering (document types, clients, outcomes)
- **Theme Visualization**: Interactive network graphs showing theme relationships and connections
- **Visual Sparks**: Image upload and VLM integration for visual concept inspiration
- **Advanced Analytics**: Entity relationship analysis and knowledge cluster detection

## Version Management System

### 🏷️ Centralized Version Control (v4.0.0+)

The Cortex Suite uses a centralized version management system to maintain consistency across all components.

#### Core Components
- **`cortex_engine/version_config.py`** - Single source of truth for all version information
- **`scripts/version_manager.py`** - Automated version syncing and management utility
- **`CHANGELOG.md`** - Structured version history (auto-synced to docker directory)

#### Version Manager Commands
```bash
# Check version consistency across all files
python scripts/version_manager.py --check

# Sync version numbers across all components (50+ files)
python scripts/version_manager.py --sync-all

# Update CHANGELOG.md with current version information
python scripts/version_manager.py --update-changelog

# Display current version details
python scripts/version_manager.py --info
```

#### Development Workflow with Versions
1. **Update version in `cortex_engine/version_config.py`** - Change CORTEX_VERSION and VERSION_METADATA
2. **Run sync command** - `python scripts/version_manager.py --sync-all`
3. **Update changelog** - `python scripts/version_manager.py --update-changelog`
4. **Verify consistency** - `python scripts/version_manager.py --check`
5. **Commit changes** - All version updates are synchronized automatically

## 🚀 Major Release & Feature Update Workflow

### **MANDATORY: Complete Release Process for Major Updates**

When implementing major features, bug fixes, or significant functionality changes (like Clean Start system), follow this comprehensive workflow to ensure full synchronization:

#### **Phase 1: Development & Testing**
1. **Implement core functionality** in relevant files
2. **Update page versions** where changes were made (increment appropriately)
3. **Test functionality thoroughly** in development environment
4. **Document changes** with clear comments and purpose statements

#### **Phase 2: Version Management (CRITICAL)**
1. **Update Central Version Config**: `cortex_engine/version_config.py`
   ```python
   CORTEX_VERSION = "X.Y.Z"  # Increment major/minor/patch appropriately
   VERSION_METADATA = {
       "release_date": "YYYY-MM-DD",
       "release_name": "Feature Name/Description",
       "description": "Brief description of changes",
       "new_features": [...],
       "improvements": [...],
       "bug_fixes": [...]
   }
   ```

2. **Run Complete Version Sync**:
   ```bash
   # Sync all components to new version
   python scripts/version_manager.py --sync-all
   
   # Update changelog with version info
   python scripts/version_manager.py --update-changelog
   
   # Verify consistency across all files
   python scripts/version_manager.py --check
   ```

3. **Manual Version Fixes** (if needed):
   - Check main README.md version references
   - Verify Docker batch file version strings
   - Update splash page version displays

#### **Phase 3: Docker Distribution Sync**
1. **Copy updated version config** to docker directory:
   ```bash
   cp cortex_engine/version_config.py docker/cortex_engine/
   ```

2. **Sync all modified pages** to docker:
   ```bash
   cp pages/*.py docker/pages/  # Copy all updated pages
   ```

3. **Verify docker distribution completeness**:
   - Main application files (Cortex_Suite.py)
   - All page files with latest versions
   - Updated batch install files (.bat, .sh)
   - Synchronized documentation (README.md)
   - Changelog files (CHANGELOG.md)

#### **Phase 4: Commit & Release**
1. **Stage all changes**:
   ```bash
   git add -A
   ```

2. **Create comprehensive commit message**:
   ```bash
   git commit -m "release: Version X.Y.Z - Feature Name
   
   ## 🚀 Major Release: vX.Y.Z
   **Release Name:** Feature Description
   **Release Date:** YYYY-MM-DD
   
   ### ✨ New Features
   - Feature 1 description
   - Feature 2 description
   
   ### 🚀 Improvements  
   - Improvement 1
   - Improvement 2
   
   ### 🔧 Bug Fixes
   - Bug fix 1
   - Bug fix 2
   
   ## 📋 Synchronized Components
   - ✅ Version consistency verified
   - ✅ Docker distribution updated
   - ✅ Batch installers updated
   - ✅ Documentation synchronized
   
   🤖 Generated with [Claude Code](https://claude.ai/code)
   
   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

3. **Push to remote**:
   ```bash
   git push origin main
   ```

#### **Phase 5: Release Verification**
1. **Confirm version consistency** across all files
2. **Verify Docker distribution** has all required files
3. **Test batch installers** show correct version
4. **Validate changelog** is updated and synced
5. **Check documentation** reflects new version

### **🔄 When to Use This Workflow**
- **Major new features** (Clean Start, new pages, significant functionality)
- **Critical bug fixes** (database schema errors, search failures)
- **Architectural changes** (new utilities, refactoring)
- **Docker distribution updates** (installation improvements)
- **User-facing improvements** (UI enhancements, error handling)

### **📋 Release Checklist Template**
```markdown
- [ ] Core functionality implemented and tested
- [ ] Version config updated with appropriate increment
- [ ] Version sync run across all components  
- [ ] Changelog updated with release information
- [ ] Version consistency verified (check command passes)
- [ ] Docker distribution synchronized
- [ ] Batch install files updated with new version
- [ ] Documentation updated (README files)
- [ ] Comprehensive commit message created
- [ ] Changes pushed to remote repository
- [ ] Release verified in target environment
```

### **📋 Automatic "What's New" System**

**SOLVED**: The "What's New" section now automatically reads from CHANGELOG.md, eliminating manual synchronization issues.

**How it works:**
- `Cortex_Suite.py` contains `load_recent_changelog_entries()` function
- Automatically parses and displays the 3 most recent versions from CHANGELOG.md  
- Provides fallback to version config if changelog unavailable
- Updates automatically when changelog is updated via version manager
- **No manual maintenance required** - one less sync issue to worry about

**Benefits:**
- ✅ **Always current** - Shows latest changelog entries automatically
- ✅ **Single source of truth** - CHANGELOG.md drives both documentation and UI
- ✅ **Eliminates sync issues** - No more outdated "What's New" content
- ✅ **Zero maintenance** - Works without manual intervention during releases

**FAILURE TO FOLLOW THIS WORKFLOW RESULTS IN:**
- Inconsistent version numbers across components
- Broken Docker distributions for users
- Missing functionality in deployed versions  
- Incomplete documentation and changelogs
- User confusion about feature availability

**NOTE**: The "What's New" sync issue has been permanently solved via automatic changelog reading.

#### Automatic Synchronization
The version manager automatically updates:
- Main application files (Cortex_Suite.py)
- All page files (pages/*.py) 
- Docker distribution files (docker/*)
- Installation scripts (.bat, .sh)
- Documentation (README.md, docker/README.md)
- Changelog syncing between project and docker directories

## Utility Functions Registry

### 🔧 Available Utility Functions

**IMPORTANT**: Always use these centralized utilities instead of duplicating functionality. This prevents code duplication and ensures consistent behavior across the codebase.

#### Path Handling (`cortex_engine.utils.path_utils`)
- **`convert_windows_to_wsl_path(path)`** - Convert Windows paths to WSL format
- **`normalize_path(path)`** - Normalize paths across platforms
- **`ensure_directory(path)`** - Create directory if it doesn't exist
- **`validate_path_exists(path, must_be_dir=False)`** - Check if path exists
- **`process_drag_drop_path(raw_path)`** - Handle single drag-drop path from any platform
- **`process_multiple_drag_drop_paths(raw_paths)`** - Handle multiple drag-drop paths
- **`get_file_size_display(path)`** - Get human-readable file size
- **`is_safe_path(path, base_path=None)`** - Check for directory traversal attacks

#### File Operations (`cortex_engine.utils.file_utils`)
- **`get_file_hash(filepath)`** - Generate SHA256 hash for file
- **`get_project_root()`** - Get project root directory
- **`get_home_directory()`** - Get user's home directory

#### Logging (`cortex_engine.utils.logging_utils`)
- **`get_logger(name, log_file=None)`** - Get standardized logger instance
- **`setup_logging(name, level=INFO)`** - Configure logging with standard format
- **`LoggerMixin`** - Mixin class for adding logging to classes

#### Input Validation (`cortex_engine.utils.validation_utils`)
- **`InputValidator.validate_file_path(path, must_exist=True)`** - Validate file paths
- **`InputValidator.validate_directory_path(path, must_exist=True)`** - Validate directories
- **`InputValidator.validate_search_query(query, max_length=1000)`** - Sanitize search queries
- **`InputValidator.validate_filename(filename)`** - Validate filenames
- **`InputValidator.validate_collection_name(name)`** - Validate collection names
- **`InputValidator.validate_file_extensions(paths, allowed_exts)`** - Check file extensions

#### Model Management (`cortex_engine.utils.model_checker`)
- **`model_checker.check_ollama_service()`** - Check if Ollama is running
- **`model_checker.get_available_models()`** - List available models
- **`model_checker.check_ingestion_requirements(include_images=True)`** - Check models for ingestion
- **`model_checker.check_research_requirements()`** - Check models for research
- **`model_checker.format_status_message(results)`** - Format user-friendly status messages

#### Safe Command Execution (`cortex_engine.utils.command_executor`)
- **`SafeCommandExecutor.execute_command(cmd, timeout=60)`** - Execute commands safely
- **`display_command_executor_widget(title, suggested_cmds)`** - Streamlit command widget
- **`display_model_installer_widget()`** - Streamlit model installer widget

#### Platform Detection (`cortex_engine.utils.default_paths`)
- **`get_default_ai_database_path()`** - Get platform-aware database path
- **`get_default_knowledge_source_path()`** - Get platform-aware knowledge source path  
- **`get_platform_info()`** - Get platform information for debugging

#### Configuration (`cortex_engine.utils.config_utils`)
- **`get_env_var(key, default=None, required=False)`** - Get environment variables safely
- **`validate_model_config(config)`** - Validate model configuration
- **`validate_database_config(config)`** - Validate database configuration
- **`merge_configs(*configs)`** - Merge multiple configuration dictionaries

#### Ollama Integration (`cortex_engine.utils.ollama_utils`)
- **`check_ollama_service(host, port)`** - Check Ollama service status
- **`get_ollama_status_message(is_running, error)`** - Get user-friendly status message
- **`format_ollama_error_for_user(operation, error_details)`** - Format user-friendly error messages

### 📋 Common Import Patterns

```python
# Most common utilities
from cortex_engine.utils import (
    convert_windows_to_wsl_path,
    normalize_path, 
    ensure_directory,
    get_logger,
    get_file_hash,
    InputValidator
)

# Specialized utilities
from cortex_engine.utils.model_checker import model_checker
from cortex_engine.utils.command_executor import SafeCommandExecutor
from cortex_engine.utils.default_paths import get_default_ai_database_path
```

### 🚨 Development Guidelines

1. **Always check this registry** before writing path handling, validation, or file operations
2. **Use `get_logger(__name__)`** instead of `logging.getLogger(__name__)`
3. **Use `InputValidator`** for all user input sanitization  
4. **Use `model_checker`** before AI operations to verify model availability
5. **Use path utilities** instead of manual path manipulation
6. **Use `ensure_directory()`** instead of `os.makedirs()` or `Path.mkdir()`
7. **Use centralized config utilities** instead of direct environment variable access

### System Status Monitoring

The `cortex_engine/system_status.py` module provides comprehensive real-time monitoring:

#### Features
- **Service Health Checks**: Monitors Ollama, API server, and Streamlit status
- **Model Availability**: Tracks which AI models are installed vs. downloading
- **Setup Progress**: Calculates overall completion percentage
- **Error Detection**: Reports connection issues, missing dependencies, etc.
- **Auto-Refresh**: Updates every 30 seconds during setup

#### Usage in Components
```python
from cortex_engine.system_status import system_status

# Get overall setup progress
progress_info = system_status.get_setup_progress()
setup_complete = progress_info["setup_complete"]
progress_percent = progress_info["progress_percent"]

# Get detailed system health
health = system_status.get_system_health()
```

#### Integration Points
- **Main App**: `Cortex_Suite.py` uses this for the setup progress page
- **All Pages**: Can check if AI features are available before showing AI-dependent UI
- **API Endpoints**: Can provide system status via REST API for external monitoring

## Development Workflow & Synchronization

### Code Changes Workflow

**CRITICAL**: Follow this workflow for ALL significant code changes:

#### 1. Update Footer Date
When making code changes, update the main app footer in `Cortex_Suite.py`:
```python
# Latest code changes footer
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.85em; margin: 1em 0;'>
        <strong>🕒 Latest Code Changes:</strong> YYYY-MM-DD<br>
        <em>Brief description of changes</em>
    </div>
    """, 
    unsafe_allow_html=True
)
```

#### 2. Update Version Numbers
- **Pages with changes**: Increment version appropriately (v1.0.1 → v1.0.2 for bugs, v1.0.0 → v1.1.0 for features)
- **Main app**: Increment for architectural changes (v2.0.0 → v2.1.0 for features, v2.0.0 → v3.0.0 for major refactor)
- **Update date**: Change date in all modified files to current date

#### 3. Git Synchronization
```bash
# Stage all changes
git add .

# Commit with descriptive message
git commit -m "feat: Brief description of changes

Detailed description of what was changed and why

🎯 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to remote
git push origin main
```

#### 4. Docker Distribution Sync
After committing to git, sync the Docker distribution:
```bash
# Copy updated files to Docker directory
cp Cortex_Suite.py docker/
cp -r pages/* docker/pages/
cp -r api/* docker/api/
cp -r cortex_engine/* docker/cortex_engine/

# Commit Docker updates
git add docker/
git commit -m "sync: Update Docker distribution with latest changes"
git push origin main
```

#### 5. Verification Checklist
- [ ] Footer date updated to current date
- [ ] Version numbers incremented appropriately  
- [ ] All modified files have current date
- [ ] Changes committed to git with descriptive message
- [ ] Docker subdirectory synchronized
- [ ] Docker sync committed to git

### Database Migration
If upgrading from versions prior to 70.0.0, delete the entire `knowledge_hub_db` directory to trigger recreation with the stable format.

### Common Issues
- **MuPDF color profile warnings**: Harmless, text extraction continues normally
- **spaCy model errors**: Ensure `en_core_web_sm` model is downloaded
- **Path issues in WSL**: All paths support both Linux and Windows formats
- **Import errors**: Use `from cortex_engine.utils import <function>` for utility functions
- **Windows batch file line ending errors**: When editing `.bat` files in WSL, ensure proper CRLF line endings with `sed -i 's/$/\r/' filename.bat`
- **Windows batch file multi-line if statements**: Use `goto` labels instead of multi-line `if (...) { }` syntax, which causes "unexpected" parser errors
- **Windows batch file errorlevel checking**: Check `errorlevel` immediately after the command you care about - other commands like `del` will overwrite the errorlevel value
- **Windows Docker build context issues**: When building from Windows directories (e.g., `E:\Docker_Cortex`), Docker may try to include Windows system folders like `$RECYCLE.BIN`, causing "Access is denied" errors. Solution: Copy `.dockerignore` to the parent build context directory before building

### Dependency Management & Resolution Guidelines

#### ⚠️ Critical Lessons Learned (August 2025)
**See `DEPENDENCY_RESOLUTION_GUIDE.md` for complete details.**

During Docling integration, dependency hell was encountered and resolved. Key lessons:

**❌ What NOT to Do:**
- **Over-pin flexible version ranges** to exact versions
- **Force incompatible dependencies** into requirements.txt
- **Remove pip's flexibility** to resolve compatible versions

**✅ Best Practices:**
- **Make enhancements optional** with graceful fallbacks
- **Keep version ranges flexible** unless specific versions required  
- **Test dependency changes** in isolated environments first
- **Document integration rationale** clearly

#### Current Stable Dependency Strategy
```txt
# Core Dependencies (Flexible Ranges)
chromadb>=0.5.15,<0.6.0    # Allows pip to resolve compatible versions
pydantic>=2.7.0            # Flexible for ecosystem compatibility
pydantic_core>=2.18.0      # Matches pydantic requirements

# Optional Enhancements (Manual Installation)
# pip install "docling>=1.0.0,<1.9.0"  # Enhanced document processing
# System automatically detects and uses when available
```

#### Dependency Conflict Resolution Process
1. **Identify root cause** - which dependency introduced conflicts
2. **Consider optional approach** - can it be manually installed?
3. **Implement graceful fallbacks** - system works without enhancement  
4. **Only pin as last resort** - document why exact versions needed
5. **Test thoroughly** - verify builds and functionality

#### Red Flags
- **Pip backtracking warnings**: "This is taking longer than usual"
- **ResolutionImpossible errors**: Fundamental incompatibility detected
- **Cascading conflicts**: One dependency change affecting many others

### Logging Locations
- **Ingestion**: `logs/ingestion.log`
- **Query**: `logs/query.log`  
- **Processed Files**: `<db_path>/knowledge_hub_db/ingested_files.log`
- **Module Logging**: Centralized through `cortex_engine.utils.get_logger(__name__)`