# **`README.md`**

**Version:** v4.8.0 (Module Harmonization & Path Standardization)
**Date:** 2025-08-31

**Prelim:**
Please be aware that the system operates in a WSL2 environment, so all paths should support linux and windows.

### 1. System Status: UI Organization Enhanced & Administrative Functions Consolidated

This document reflects recent major enhancements to the **Project Cortex Suite**, now featuring a consolidated Maintenance page for all administrative functions, improved UI organization, and enhanced user experience.

*   **[NEW] ARM64 Compatibility & Multi-Architecture Support (v4.1.2)**
    *   **Status:** Complete. Universal compatibility across x86_64, ARM64, Apple Silicon, and Windows Snapdragon processors.
    *   **Capabilities:** CPU-first installation strategy with optional GPU acceleration upgrades. Zero architecture detection required.
    *   **Benefits:** Immediate Docker compatibility on all platforms, eliminated hardcoded CUDA dependencies, flexible PyTorch ranges.

*   **[ENHANCED] Knowledge Search: Direct ChromaDB + Progress Indicators (v4.0.2)**
    *   **Status:** Completely rebuilt. Knowledge Search now bypasses LlamaIndex entirely, using direct ChromaDB queries with comprehensive timeout protection and real-time progress indicators.
    *   **Capabilities:** Fast, reliable search with visual feedback during operations. Multi-strategy search approach for better results.
    *   **Technical:** Docker environment compatibility, UnboundLocalError fixes, and enhanced fallback mechanisms.

*   **[RESOLVED] Cross-Platform Path Compatibility**
    *   **Status:** Complete. Eliminated all hardcoded WSL paths with intelligent default path detection for Windows/Mac/Linux environments.
    *   **Implementation:** New `cortex_engine/utils/default_paths.py` provides platform-aware path resolution with graceful fallbacks.

*   **[RESOLVED] GraphRAG Integration: Entity and Relationship Extraction**
    *   **Status:** The ingestion pipeline now automatically extracts consultants, clients, projects, and their relationships using spaCy NER and pattern matching. This data is stored in a NetworkX graph alongside the vector embeddings.
    *   **Capabilities:** The system can now answer queries like "What projects did consultant X work on?", "Who collaborated with person Y?", and "What types of work has client Z requested?"

*   **[RESOLVED] AI Research: Sources Found During Foundational Search**
    *   **Status:** Fixed. The foundational query functions work correctly.

*   **[RESOLVED] AI Research: Correct UI Step Numbering**
    *   **Status:** Fixed. UI uses centralized dictionary for step headers.

*   **[RESOLVED] AI Research: Citations in Deep Research Report**
    *   **Status:** Fixed. Enhanced prompts ensure proper citation formatting.

### 2. MANDATORY: Database Migration (v4.0.1+)

**To resolve critical stability issues and pathing errors, the database structure was updated. If you are upgrading from a version prior to 70.0.0, you MUST perform the following one-time action before first use:**

1.  Navigate to your primary AI databases folder (e.g., `/mnt/f/ai_databases/`).
2.  **Permanently delete the entire `knowledge_hub_db` directory.**
3.  The system will automatically recreate this directory with the correct, stable format the next time you run the ingestion process.

### 3. Usage & Licensing Disclaimer

**This system is developed for private, non-commercial use only.** All components are integrated under the assumption of personal, fair-use for research and development on a local machine.

### 4. Introduction

This document provides a comprehensive overview of the **Project Cortex Suite**. The suite is an integrated workbench designed to build a high-quality, human-verified knowledge base with graph-based relationship tracking and use it to produce AI-assisted proposals.

### 5. System Vision: The Knowledge Graph Workflow

The Cortex Suite now incorporates a knowledge graph that captures relationships between people, organizations, projects, and documents. This enables relationship-based queries and enhanced context understanding.

```mermaid
graph TD
    subgraph "Phase 1: Knowledge Base Curation with Graph"
        Z[<b>AI Assisted Research</b><br/>Multi-agent 'Double Diamond' research] --> Y[External Research<br>Folder];
        Y --> A;
        A[<b>Knowledge Ingest UI</b><br/>3-stage ingestion with<br/>entity & relationship extraction] --> H((Main Knowledge Base<br/>ChromaDB + Graph));
        A --> G((Knowledge Graph<br/>NetworkX));
        I[<b>Knowledge Search UI</b><br/>Vector + Graph queries] -->|Searches & Prunes| H;
        I -->|Leverages relationships| G;
        I -->|Curates search results into| J[<b>Working Collections</b>];
        J --> K[<b>Collection Management UI</b>];
    end

    subgraph "Phase 2: Proposal Lifecycle Management"
        subgraph "2a: Template Preparation"
        T_IN[Raw Document<br>e.g., Client RFP .docx] --> T[<b>Proposal Step 1 Prep UI</b>]
        T --> T_OUT[<b>Tagged Template .docx</b>]
        end

        subgraph "2b: Content Generation"
        L[<b>Proposal Step 2 Make UI</b>] -->|Creates/Loads| M[<b>Proposal State</b>];
        T_OUT --> N[<b>Proposal Co-pilot UI</b><br/>Uses knowledge + graph context]
        M --> N
        N -->|Selects & Uses| J;
        N -->|Generates from| H;
        N -->|Context from| G;
        N --> O[✅ Final Proposal Document];
        end
    end

### 6. Installation & Operation

**1. System-Level Dependencies (for Mind Maps):**
The AI Research Assistant requires the Graphviz system command `dot` to be installed and accessible in the system's PATH.
-   **Debian/Ubuntu:** `sudo apt-get install graphviz`
-   **MacOS (Homebrew):** `brew install graphviz`
-   **Windows:** Download from the official site and add the `bin` directory to your system's `PATH`.

**2. Python Environment (Python 3.11 Required):**
This system is stabilized on Python 3.11. If you are using a different version, you must set up a 3.11 environment.
```bash
# For Ubuntu users, if 3.11 is not available:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv

# Create and activate the virtual environment
python3.11 -m venv venv
source venv/bin/activate
```

**3. Install Python Dependencies:**
```bash
pip install -r requirements.txt
# Download spaCy language model for entity extraction
python -m spacy download en_core_web_sm
```

**3a. Optional: Enhanced Document Processing (Docling)**

For superior document layout recognition and structure extraction:
```bash
# Install Docling for enhanced document processing
pip install "docling>=1.0.0,<1.9.0"
```

**Benefits of Docling (when installed):**
- Better table structure recognition
- Improved layout preservation  
- Enhanced OCR for scanned documents
- Superior handling of complex document formats

**Note:** System works perfectly without Docling using proven legacy document readers (PyMuPDF, DocxReader, etc.). Docling is automatically detected and used when available.

**4. Set Up Environment Variables:**
Create a `.env` file in the project root with your API keys.```# .env file
# For the AI Research module, choose 'openai', 'ollama', or 'gemini'.
# Other modules use the settings defined in their respective files.
# .env file
LLM_PROVIDER="gemini"  # or "ollama" or "openai"
OLLAMA_MODEL="mistral:7b-instruct-v0.3-q4_K_M"
OPENAI_API_KEY="your_openai_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
YOUTUBE_API_KEY="your_google_api_key_for_youtube_search"
GRAPHVIZ_DOT_EXECUTABLE="/usr/bin/dot"

# (Optional) Explicit path to Graphviz 'dot' executable to help resolve mind map issues.
# Find this path by running `which dot` in your Ubuntu/WSL terminal.
GRAPHVIZ_DOT_EXECUTABLE="/usr/bin/dot"
```

**5. Run the Streamlit Application:**```bash
streamlit run Cortex_Suite.py```

### 6.5. ARM64 & Multi-Architecture Support (v4.1.2+)

The Cortex Suite now provides **universal compatibility** across all major processor architectures:

#### **✅ Supported Architectures**
- **Intel x86_64**: Windows, Linux, macOS (with optional NVIDIA GPU acceleration)
- **ARM64 Snapdragon**: Windows on ARM (Surface Pro X, Dev Kits, etc.)
- **Apple Silicon**: M1, M2, M3 MacBooks and iMacs (with automatic MPS acceleration) 
- **Linux ARM64**: Various ARM64 Linux distributions and servers
- **Docker Multi-Arch**: Universal container support across all platforms

#### **🔧 Installation Strategy**
The system uses a **CPU-first approach** that works immediately on all architectures:

```bash
# Works universally on ANY architecture:
pip install -r requirements.txt
```

#### **⚡ GPU Acceleration (Optional Upgrade)**
For performance enhancement on supported systems:

**Intel x86_64 with NVIDIA GPU:**
```bash
# After base installation:
pip uninstall torch torchvision
pip install torch==2.3.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon (Automatic):**
```bash
# MPS acceleration is automatically enabled when available
# No additional steps required
```

#### **🐳 Docker Support**
Docker builds now work seamlessly across architectures:

```bash
# Works on ANY system:
docker build -t cortex-suite .

# Explicit multi-architecture build:
docker buildx build --platform linux/amd64,linux/arm64 -t cortex-suite .
```

#### **🔍 Architecture Detection**

### Docker Status & Known Issues

- Rebuild after updates: when docker/ files change, rebuild the image to pick them up: `docker compose down && docker compose build --no-cache && docker compose up -d`.
- Ingest staging diagnostics: if the UI shows “Analysis complete, but no documents were staged for review”, the ingest page now displays the container path to `staging_ingestion.json`, whether it exists, and the parsed document count. If the file contains documents, click “Retry Finalization” to complete the flow.
- Knowledge Search fallback: if the project root page is not present in the image, Docker falls back to a minimal search implementation with direct Chroma queries.
- .docx reader warnings: Docker may log `DocxReader.load_data() got an unexpected keyword argument 'file_path'`. Analysis proceeds, but a reader‑normalization pass is planned.
- Logs inside container: use `docker exec -it <container> bash` then `tail -n 120 -f /home/cortex/logs/ingestion.log` (or `/app/logs/ingestion.log` depending on image path).

Check your system architecture:
```bash
# Check processor type
uname -m

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

#### **🛠️ Troubleshooting**
If you encounter architecture-related issues:

1. **"No matching distribution" errors**: Update to v4.1.2+ requirements.txt
2. **Docker build failures**: Ensure using latest Docker distribution files
3. **Performance concerns**: Follow GPU acceleration upgrade path for your architecture
4. **Import errors**: Check that all dependencies installed correctly

For comprehensive troubleshooting, see `ARM64_COMPATIBILITY_GUIDE.md`.

### 7. Version History & Future Roadmap

Entity Extraction
During ingestion, the system automatically identifies:

People: Consultants, authors, team members
Organizations: Clients, partners, vendors
Projects: From document content and thematic tags
Documents: All ingested files with metadata

Relationship Mapping
The system tracks relationships such as:

authored: Person → Document
client_of: Organization → Document
collaborated_with: Person ↔ Person
mentioned_in: Entity → Document
documented_in: Project → Document

Graph Queries (In Development)

Find all projects a consultant worked on
Identify collaborators on documents
Track work done for specific clients
Analyze document type preferences by client

8. Version History & Future Roadmap
Completed Sprints

Sprint 1-20.5: Core system development through search bugfix. ✅
Sprint 21: Multi-Modal Knowledge Ingestion (Images) - VLM integration for image description. ✅
Sprint 22: GraphRAG Foundation - Entity extraction and knowledge graph building during ingestion. ✅ Done

In Progress

Sprint 23: Graph-Enhanced Retrieval - Implement hybrid vector + graph search queries. 📝 Next Up

Planned Sprints

Sprint 24: Graph Visualization - Interactive graph exploration UI
Sprint 25: Advanced Graph Queries - Complex relationship traversal
Sprint 26: Agentic Tool Use - [GENERATE_TABLE_FROM_KB] with graph context
Sprint 27: Full Co-pilot Upgrade with graph-aware generation

9. Troubleshooting
spaCy Model Download Issues
If you encounter errors with spaCy, ensure the model is downloaded:
bashpython -m spacy download en_core_web_sm
MuPDF Color Profile Warnings
You may see warnings like MuPDF error: format error: cmsOpenProfileFromMem failed. These are harmless and relate to PDF color profiles. Text extraction continues normally.
Inspecting the Knowledge Graph
Use the cortex_inspector tool to examine graph contents:
bashpython scripts/cortex_inspector.py --db-path /mnt/f/ai_databases --stats

10. Logs and Data Locations

Ingestion Log: logs/ingestion.log - Detailed processing information
Ingested Files Log: <db_path>/knowledge_hub_db/ingested_files.log - List of processed files
Knowledge Graph: <db_path>/knowledge_cortex.gpickle - NetworkX graph with entities and relationships
Vector Store: <db_path>/knowledge_hub_db/ - ChromaDB embeddings

11. Core Components Update

cortex_engine/entity_extractor.py: NEW - Extracts people, organizations, and projects using spaCy NER and pattern matching
cortex_engine/graph_manager.py: ENHANCED - Now EnhancedGraphManager with relationship queries
cortex_engine/ingest_cortex.py: v13.0.0 - Integrated entity extraction during analysis phase
cortex_engine/graph_query.py: NEW (Coming in Sprint 23) - Hybrid vector + graph search

12. Dependencies Update
Key additions to requirements.txt:

spacy>=3.5.0,<3.8.0 - For entity extraction
numpy<2.0.0,>=1.22.5 - Pinned for compatibility

The system now builds a comprehensive knowledge graph during ingestion, laying the foundation for powerful relationship-based queries and enhanced context understanding in proposal generation.

#### **Handling `:Zone.Identifier` Files in WSL**

If you see files ending with `:Zone.Identifier`, these are harmless metadata artifacts from Windows. You can safely remove all of them from your project by running the following command from your `cortex_suite` root directory in your WSL terminal:

```bash
find . -type f -name "*:Zone.Identifier" -delete
```

This command finds all files (`-type f`) whose names (`-name`) end with `":Zone.Identifier"` and deletes them.

### 8.1. CRITICAL: Windows Batch File Requirements (Docker Distribution)

**⚠️ IMPORTANT:** The `docker/run-cortex.bat` file has specific encoding and formatting requirements that MUST be maintained to prevent distribution failures. These issues have been encountered multiple times and must not be repeated.

#### **Mandatory Requirements:**

1. **File Encoding: ANSI (Windows-1252) ONLY**
   - ❌ **NOT** UTF-8, UTF-16, or any Unicode encoding
   - ❌ **NOT** "Unicode text, UTF-8 text" (as shown by `file` command)
   - ✅ **MUST** be pure ANSI/Windows-1252 encoding
   - **Validation**: `file docker/run-cortex.bat` should show "ASCII text" or "ISO-8859 text"

2. **Line Endings: CRLF (Windows) ONLY**
   - ❌ **NOT** LF (Unix/Linux line endings `\n`)
   - ✅ **MUST** be CRLF (Windows line endings `\r\n`)
   - **Validation**: `hexdump -C docker/run-cortex.bat | head -5` should show `0d 0a` (CRLF) not just `0a` (LF)

3. **Character Set: Standard ASCII Characters ONLY**
   - ❌ **NO** Unicode emoji (🚀, ❌, ✅, etc.)
   - ❌ **NO** Special Unicode characters
   - ✅ **ONLY** standard ASCII characters (codes 32-126)
   - **Replace**: Unicode symbols with ASCII equivalents:
     - `🚀` → `>>>`
     - `❌` → `ERROR:`
     - `✅` → `OK:`
     - `⏳` → `WAIT:`
     - `📦` → `PACK:`

4. **Logic Complexity: Keep Simple**
   - ❌ **AVOID** complex nested conditionals
   - ❌ **AVOID** advanced batch scripting features
   - ✅ **USE** simple `if %errorlevel% neq 0` patterns
   - ✅ **USE** basic `echo`, `pause`, `timeout` commands
   - ✅ **TEST** on actual Windows systems, not WSL

#### **Previous Issues Encountered:**
- **UTF-8 Encoding**: Caused batch file execution failures on some Windows systems
- **LF Line Endings**: Prevented proper command parsing on Windows
- **Unicode Characters**: Displayed as `?` or caused parsing errors in Windows Command Prompt
- **Complex Logic**: Created unpredictable behavior across different Windows versions

#### **Validation Commands:**
```bash
# Check encoding (should show ASCII text, not UTF-8)
file docker/run-cortex.bat

# Check line endings (should show 0d 0a patterns)
hexdump -C docker/run-cortex.bat | head -10

# Check for non-ASCII characters
grep -P '[^\x20-\x7E]' docker/run-cortex.bat || echo "Clean ASCII"
```

#### **How to Fix When Issues Occur:**
1. **Convert to ANSI**: Open in Notepad++, Encoding → Convert to ANSI
2. **Fix Line Endings**: Notepad++, Edit → EOL Conversion → Windows (CR LF)
3. **Replace Unicode**: Find/replace all Unicode symbols with ASCII equivalents
4. **Test on Windows**: Always test the actual .bat file on a real Windows system

**📋 Remember**: These requirements apply to ALL Windows batch files in the distribution, not just `run-cortex.bat`.

### 9. Final Code Manifest

-   `Cortex_Suite.py`: Main entrypoint for the unified Streamlit application.
-   `requirements.txt`: Frozen Python dependencies for a stable Python 3.11 environment.
-   `.env`: For storing API keys and optional executable paths.
-   `.gitignore`: Specifies which files and directories to ignore for version control.
-   `boilerplate.json`: Stores reusable boilerplate text snippets.
-   `cortex_config.json`: File to store last-used paths and other persistent settings.
-   `working_collections.json`: Stores user-curated document collections.
-   `staging_ingestion.json`: A temporary file holding AI-suggested metadata for user review.

-   **`cortex_engine/`**: The core backend logic of the application.
    -   `__init__.py`: Makes the engine a Python package.
    -   `boilerplate_manager.py`: Manages boilerplate text snippets.
    -   `collection_manager.py`: Handles CRUD operations for Working Collections.
    -   `config.py`: Central configuration for paths, models, and default settings.
    -   `config_manager.py`: Manages persistent user settings file (`cortex_config.json`).
    -   `graph_extraction_worker.py`: Subprocess worker for knowledge graph extraction.
    -   `graph_manager.py`: Manages the knowledge graph file.
    -   `ingest_cortex.py`: Core ingestion script with validation and progress reporting.
    -   `instruction_parser.py`: Parses `.docx` files for Cortex instructions.
    -   `proposal_manager.py`: Manages the lifecycle of proposals.
    -   `query_cortex.py`: Provides models and prompts for querying the knowledge base.
    -   `session_state.py`: Manages Streamlit session state.
    -   `synthesise.py`: Backend for the AI Assisted Research agent.
    -   `task_engine.py`: Backend for AI task execution in the Proposal Co-pilot.
    -   `utils.py`: Shared, low-dependency helper functions.

-   **`pages/`**: The individual Streamlit UI pages.
    -   `1_AI_Assisted_Research.py`: UI for the multi-agent research engine.
    -   `2_Knowledge_Ingest.py`: UI for the three-stage document ingestion process.
    -   `3_Knowledge_Search.py`: UI for searching the knowledge base. Includes a robust fix for complex filter combinations.
    -   `4_Collection_Management.py`: UI for managing Working Collections.
    -   `5_Proposal_Step_1_Prep.py`: UI for creating proposal templates.
    -   `6_Proposal_Step_2_Make.py`: UI for creating and loading proposals.
    -   `Proposal_Copilot.py`: The core UI for drafting proposals.

-   **`scripts/`**: Standalone utility and diagnostic scripts.
    -   `__init__.py`: Makes scripts a Python package.
    -   `cortex_inspector.py`: A command-line tool to inspect the knowledge base.

-   **`proposals/`**: Default directory to store all saved proposal data.
-   **`external_research/`**: Default location for synthesized research notes from the AI Research agent.
-   **`template_maps/`**: Default directory to store saved progress from the Template Editor.
-   **`logs/`**: Directory containing log files like `ingestion.log` and `query.log`.
    -   `ingested_files.log`: Note: This specific log is stored inside the `knowledge_hub_db` directory, not in the main `logs` folder.
```
