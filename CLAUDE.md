# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Cortex Suite is a Streamlit-based AI-powered knowledge management and proposal generation system. It features integrated GraphRAG capabilities with entity extraction, relationship mapping, and hybrid vector + graph search. The system operates in a WSL2 environment and requires Python 3.11.

## Architecture

### Core Components
- **Streamlit Application**: `Cortex_Suite.py` - Main entry point with multi-page UI
- **Backend Engine**: `cortex_engine/` - Core business logic and data processing
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

#### Page Versioning Standard
All Streamlit pages must follow this format:
```python
# [Page Name] Page
# Version: v1.0.0  
# [Brief description]

import streamlit as st
# ... other imports ...

st.set_page_config(page_title="Page Title", layout="wide")

# Page configuration
PAGE_VERSION = "v1.0.0"

# ... page logic ...

st.title("📊 Page Title")
st.caption(f"Page Version: {PAGE_VERSION}")
```

**Version Numbering Rules:**
- Major version (v1.x.x): Synchronized across all pages for major releases
- Minor version (vx.1.x): Individual page increments for features
- Patch version (vx.x.1): Individual page increments for bug fixes

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
- **Hybrid Model Architecture**: Optimal model selection per task type
  - **Local Only**: Proposals (`mistral-small3.2`), KB operations, embeddings  
  - **Flexible Research**: User choice between Gemini (cloud) or Local Mistral
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

### Logging Locations
- **Ingestion**: `logs/ingestion.log`
- **Query**: `logs/query.log`  
- **Processed Files**: `<db_path>/knowledge_hub_db/ingested_files.log`
- **Module Logging**: Centralized through `cortex_engine.utils.get_logger(__name__)`