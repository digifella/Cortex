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

## 🐳 Docker (removed 2026-08-01)

The Docker distribution was **deleted** in v6.5.1 — it was no longer used and its
copies of `cortex_engine/`, `pages/` and `api/` had to be hand-synced on every
release, which repeatedly went stale. Recover it from git history before that
commit if it is ever needed.

`scripts/version_manager.py` no longer syncs anything to `docker/`, and the
release workflow has no Docker step.

### Legacy `/.dockerenv` branches in the code

44 files still branch on `os.path.exists('/.dockerenv')` to decide whether to
apply `convert_windows_to_wsl_path()`. Those branches are now **dead code** — the
condition can never be true without the distribution. They were left in place
because removing them touches path handling across the whole codebase, which is a
separate change with real regression risk.

Do not add new `/.dockerenv` branches. In WSL and on the host, always use
`convert_windows_to_wsl_path()` for Windows-style paths.

**GOLDEN RULE (unchanged)**: never hardcode paths — always use the user's
configured database path.

## 👁️ Vision (image descriptions)

`DocumentTextifier.describe_image` tries providers in this order:

1. **Claude Haiku** — when `ANTHROPIC_API_KEY` is set and `prefer_local_vision` is False.
2. **LM Studio** — a VLM **already loaded** there. Added v6.6.0.
3. **Ollama** — the VRAM-adaptive local selection.

**LM Studio only ever uses an already-loaded model.** It never asks LM Studio to
load one: the entire point is to reuse a resident VLM rather than making Ollama
evict it to load a smaller model into leftover VRAM. If nothing is loaded, or the
endpoint is unreachable, it returns `""` and the chain falls through to Ollama —
so this is a silent no-op on machines without LM Studio.

```bash
CORTEX_LMSTUDIO_BASE_URL=http://192.168.0.118:1234/v1   # in .env — see below
CORTEX_LMSTUDIO_VISION_MODEL=...                        # optional: pin a model, skips probing
```

Gotchas:

- **LM Studio runs on the Windows host, not in WSL**, so the `http://localhost:1234/v1`
  default does NOT reach it. `.env` sets the LAN address. If that IP changes,
  vision silently falls back to Ollama — check `curl http://<host>:1234/api/v0/models`.
- **The loaded/type fields are only on the native `/api/v0/models` API**, not `/v1`.
- **`reasoning_effort: "none"` is mandatory.** Without it qwen3.6 spends the whole
  token budget on reasoning and returns empty content — the same trap the wiki loop
  documents. Verified: a 30-token cap returned `""`; reasoning off returned the
  answer in 5 tokens.
- First call costs ~30s of warmup; steady state is ~1.2-1.9s per image.

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

#### Phase 3: Commit & Release

⚠️ **Never `git add -A` or `git add .`** — this working tree routinely carries
unrelated user files (loose PDFs, `.claude/` edits, untracked scratch), and a
blanket add sweeps them into the release commit. Stage tracked changes with those
excluded, then *verify before committing*:

```bash
git add -u -- ':!*.pdf' ':!*Zone.Identifier' ':!.claude'
git status --short          # confirm nothing unrelated is staged
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
- [ ] Comprehensive commit message created
- [ ] Changes pushed to remote repository

## Environment Setup

### Quick Start
```bash
# System dependencies (graphviz = mind maps, exiftool = photo metadata tools)
sudo apt-get install graphviz libimage-exiftool-perl

# Create Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Install required model for proposals
ollama pull mistral-small3.2

# Install required vision model for photo/image description (6.1GB)
ollama pull qwen3-vl:8b

# Start the application
streamlit run Cortex_Suite.py
```

### Critical Version Requirements
- **Python**: 3.11 (required for stability)
- **NumPy**: <2.0.0 (compatibility with spaCy/ChromaDB)
- **spaCy**: 3.5.0-3.8.0 range

### Required System Binaries
These are **not** installed by `pip install -r requirements.txt` — they must be present on `PATH`:
- **`exiftool`** — required by the Photo & Metadata Tools page (both tabs). Invoked as a subprocess; never reimplement EXIF/XMP parsing in Python. Detected via `shutil.which("exiftool")`; when absent the UI disables Scan/Apply.
- **`dot`** (graphviz) — required for mind map generation in AI Assisted Research.

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
3. **Keep version numbers consistent** across all components
4. **Document changes** with clear commit messages

### Common Issues
- **Path issues in WSL**: All paths support both Linux and Windows formats
- **Windows batch file errors**: Ensure proper CRLF line endings with `sed -i 's/$/\r/' filename.bat`
- **Version inconsistencies**: Always run version sync commands after updates

**FAILURE TO FOLLOW GUIDELINES RESULTS IN:**
- Inconsistent version numbers across components
- Missing functionality in deployed versions
- User confusion about feature availability

## Coding Behaviour Guidelines
*(via [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills))*

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
