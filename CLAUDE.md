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

### ⚠️ IMPORTANT REMINDERS FOR DEVELOPERS

**Before making ANY code changes, remember:**
1. 🗓️ **Update footer date** in `Cortex_Suite.py` with current date  
2. 📝 **Increment version numbers** appropriately in changed files
3. 💾 **Commit to git** with descriptive message after changes
4. 🐳 **Sync Docker directory** with updated files  
5. 📤 **Push Docker sync** to git as separate commit

**See "Development Workflow & Synchronization" section below for detailed steps.**

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

### Database Management
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
- `2_Knowledge_Ingest.py` - Document ingestion UI
- `3_Knowledge_Search.py` - Vector + graph search
- `4_Collection_Management.py` - Working collections CRUD
- `5_Proposal_Step_1_Prep.py` - Template editor
- `6_Proposal_Step_2_Make.py` - Proposal lifecycle
- `Proposal_Copilot.py` - AI-assisted proposal drafting
- `10_Idea_Generator.py` - **NEW**: Double Diamond innovation methodology

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

### Critical Windows Batch File Lessons Learned
1. **Multi-line If Statements**: Windows batch files cannot reliably parse multi-line if statements with parentheses. Use `goto` labels instead.
2. **ErrorLevel Overwriting**: Commands like `del`, `copy`, etc. overwrite the `%errorlevel%` variable. Always check `errorlevel` immediately after the command you care about.
3. **Docker Build Context on Windows**: When building from Windows paths, Docker may include system folders like `$RECYCLE.BIN`. Use `.dockerignore` in the build context directory (not just the docker subdirectory).
4. **Container Detection Logic**: `findstr` returns errorlevel 1 when NO match is found, errorlevel 0 when match IS found. Logic should be `if not errorlevel 1` for "found" conditions.

### Recent Architectural Improvements (v39.0.0+)
- **Hybrid Model Architecture (v3.0.0)**: Advanced backend management system
  - **Docker Model Runner**: Enterprise-grade OCI distribution with 15% faster inference
  - **Ollama Integration**: Traditional reliable model management with automatic fallback
  - **Intelligent Selection**: Optimal backend choice per model and use case
  - **Migration Utilities**: Seamless model transitions between backends
- **Docling Integration (v14.0.0)**: Enhanced document processing pipeline
  - **IBM Research Technology**: State-of-the-art document parsing with layout preservation
  - **OCR Support**: Advanced optical character recognition for scanned documents
  - **Structured Extraction**: Table recognition, headers, and reading order preservation
  - **Graceful Fallback**: Automatic fallback to legacy readers for compatibility
- **Centralized Utilities**: Eliminated code duplication by extracting common functionality to `cortex_engine/utils/`
- **Standardized Logging**: All modules now use consistent logging instead of mixed print statements  
- **Exception Hierarchy**: Implemented structured exception handling with `cortex_engine/exceptions.py`
- **Path Handling**: Unified cross-platform path conversion logic

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