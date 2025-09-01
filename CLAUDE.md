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

### Key Workflows
1. **AI Assisted Research** → **Knowledge Ingest** → **Knowledge Search** → **Collection Management**
2. **Proposal Step 1 Prep** (Template Editor) → **Proposal Step 2 Make** → **Proposal Copilot**

## ⚠️ CRITICAL: Version Management Workflow

**🚨 Centralized Version Management (v4.0.0+)**

All version information is centralized in `cortex_engine/version_config.py`. This is the SINGLE SOURCE OF TRUTH for version numbers.

### Essential Version Management Steps
1. 📝 **Update version_config.py** - Change CORTEX_VERSION and VERSION_METADATA  
2. 🔄 **Run sync command** - `python scripts/version_manager.py --sync-all`
3. 📋 **Update changelog** - `python scripts/version_manager.py --update-changelog`  
4. ✅ **Verify consistency** - `python scripts/version_manager.py --check`
5. 💾 **Commit all changes** together with proper version tags
6. 📤 **Push all changes** to remote repository

**FAILURE TO FOLLOW VERSION WORKFLOW = INCONSISTENT VERSION NUMBERS ACROSS COMPONENTS**

### Version Manager Commands
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

## 🚀 Git Synchronization Workflow

### Code Changes Process
**CRITICAL**: Follow this workflow for ALL significant code changes:

#### 1. Update Footer Date
Update the main app footer in `Cortex_Suite.py`:
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
- **Pages with changes**: Increment appropriately (v1.0.1 → v1.0.2 for bugs, v1.0.0 → v1.1.0 for features)
- **Main app**: Increment for architectural changes (v2.0.0 → v2.1.0 for features)
- **Update date**: Change date in all modified files to current date

#### 3. Git Commit Process
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

## 🐳 Docker Distribution & Packaging

### Docker Distribution Rules
- **Single .dockerignore**: Keep ONLY in `/docker/.dockerignore` (remove any from project root)
- **Minimal Documentation**: Only `/docker/README.md` needed
- **Clean Package**: Exclude ALL user data, databases, proposals, external_research, logs, media files
- **Fresh Installation**: Every Docker deployment creates completely fresh databases and configurations

### Required Docker Files Structure
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

### Docker Distribution Checklist
When preparing a Docker distribution:

#### Required Files
- `requirements.txt` - Python dependencies needed by Dockerfile
- `Cortex_Suite.py` - Main application entry point
- `Dockerfile` - Container build instructions
- `.env.example` - Configuration template
- `run-cortex-FIXED.bat` - Windows launcher script

#### Required Directories (copy entire directories)
- `api/` - API server module
- `cortex_engine/` - Core business logic and data processing
- `pages/` - Streamlit page components
- `scripts/` - Utility scripts (optional but recommended)

### Critical Docker Path Handling

**CRITICAL**: This is a recurring issue that causes functionality to break.

#### Path Handling Rules by Environment

**1. Docker Container Environment (`os.path.exists('/.dockerenv')`):**
- ✅ **Use paths EXACTLY as configured** by the user
- ❌ **DO NOT** use `convert_windows_to_wsl_path()` - Docker handles volume mapping
- **User path**: `/data/ai_databases` → **Use**: `/data/ai_databases` 
- **User path**: `C:\ai_databases` → **Use**: `C:\ai_databases` (mapped by Docker volume)
- **WHY**: Docker volumes automatically map host paths to container paths

**2. WSL Environment (Windows Subsystem for Linux):**
- ✅ **Use** `convert_windows_to_wsl_path()` for Windows-style paths
- **User path**: `C:\ai_databases` → **Convert to**: `/mnt/c/ai_databases`

#### Detection Pattern
```python
if os.path.exists('/.dockerenv'):
    # Docker container - use path as-is, Docker volumes handle mapping
    final_path = user_configured_path
else:
    # WSL/Host - use conversion if needed  
    final_path = convert_windows_to_wsl_path(user_configured_path)
```

**GOLDEN RULE**: NEVER hardcode paths. Always use user's configured database path and let Docker/WSL handle the underlying mapping.

## 📋 Major Release Workflow

### Complete Release Process for Major Updates

#### Phase 1: Development & Testing
1. **Implement core functionality** in relevant files
2. **Update page versions** where changes were made
3. **Test functionality thoroughly** in development environment

#### Phase 2: Version Management (CRITICAL)
1. **Update Central Version Config**: `cortex_engine/version_config.py`
   ```python
   CORTEX_VERSION = "X.Y.Z"  # Increment appropriately
   VERSION_METADATA = {
       "release_date": "YYYY-MM-DD",
       "release_name": "Feature Name",
       "description": "Brief description",
       "new_features": [...],
       "improvements": [...],
       "bug_fixes": [...]
   }
   ```

2. **Run Complete Version Sync**:
   ```bash
   python scripts/version_manager.py --sync-all
   python scripts/version_manager.py --update-changelog
   python scripts/version_manager.py --check
   ```

#### Phase 3: Docker Distribution Sync
1. **Copy updated files** to docker directory:
   ```bash
   cp cortex_engine/version_config.py docker/cortex_engine/
   cp pages/*.py docker/pages/
   ```

2. **Verify docker distribution completeness**:
   - Main application files (Cortex_Suite.py)
   - All page files with latest versions
   - Updated batch install files (.bat, .sh)
   - Synchronized documentation (README.md)

#### Phase 4: Commit & Release
```bash
git add -A
git commit -m "release: Version X.Y.Z - Feature Name

## 🚀 Major Release: vX.Y.Z
**Release Name:** Feature Description
**Release Date:** YYYY-MM-DD

### ✨ New Features
- Feature 1 description

### 🚀 Improvements  
- Improvement 1

### 🔧 Bug Fixes
- Bug fix 1

## 📋 Synchronized Components
- ✅ Version consistency verified
- ✅ Docker distribution updated
- ✅ Documentation synchronized

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin main
```

#### Release Verification Checklist
- [ ] Core functionality implemented and tested
- [ ] Version config updated with appropriate increment
- [ ] Version sync run across all components  
- [ ] Changelog updated with release information
- [ ] Version consistency verified (check command passes)
- [ ] Docker distribution synchronized
- [ ] Comprehensive commit message created
- [ ] Changes pushed to remote repository

## Environment Setup

### Quick Start
```bash
# Create Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Install required model for proposals
ollama pull mistral-small3.2

# Start the application
streamlit run Cortex_Suite.py
```

### Critical Version Requirements
- **Python**: 3.11 (required for stability)
- **NumPy**: <2.0.0 (compatibility with spaCy/ChromaDB)
- **spaCy**: 3.5.0-3.8.0 range

## Key Technical Details

### Database Structure
- **Vector Store**: ChromaDB at `<db_path>/knowledge_hub_db/`
- **Knowledge Graph**: NetworkX pickle at `<db_path>/knowledge_cortex.gpickle`
- **Default Path**: `/mnt/f/ai_databases` (fallback, overrideable)

### Important Utility Functions
Always use these centralized utilities:

```python
# Most common utilities
from cortex_engine.utils import (
    convert_windows_to_wsl_path,
    normalize_path, 
    ensure_directory,
    get_logger,
    InputValidator
)

# Model checking before AI operations
from cortex_engine.utils.model_checker import model_checker
```

## Development Guidelines

### 🚨 Development Rules
1. **Always follow version management workflow** before making changes
2. **Use centralized utilities** instead of duplicating functionality
3. **Test Docker distribution** after major changes
4. **Keep version numbers consistent** across all components
5. **Document changes** with clear commit messages

### Common Issues
- **Path issues in WSL**: All paths support both Linux and Windows formats
- **Windows batch file errors**: Ensure proper CRLF line endings with `sed -i 's/$/\r/' filename.bat`
- **Docker build context issues**: Copy `.dockerignore` to parent build context directory
- **Version inconsistencies**: Always run version sync commands after updates

**FAILURE TO FOLLOW GUIDELINES RESULTS IN:**
- Inconsistent version numbers across components
- Broken Docker distributions for users
- Missing functionality in deployed versions
- User confusion about feature availability