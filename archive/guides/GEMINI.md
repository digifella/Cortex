# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

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
1.  **AI Assisted Research** → **Knowledge Ingest** → **Knowledge Search** → **Collection Management**
2.  **Proposal Step 1 Prep** (Template Editor) → **Proposal Step 2 Make** → **Proposal Copilot**

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
```

### Running the Application
```bash
# Start the Streamlit application
streamlit run Cortex_Suite.py
```

### System Dependencies
```bash
# Install Graphviz for mind map generation (Ubuntu/Debian)
sudo apt-get install graphviz
```

### Database Management
```bash
# Inspect knowledge graph and database contents
python scripts/cortex_inspector.py --db-path /mnt/f/ai_databases --stats

# Clean up Windows metadata files in WSL
find . -type f -name "*:Zone.Identifier" -delete
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

### UI Pages (`pages/`)
- `1_AI_Assisted_Research.py` - Multi-agent research system
- `2_Knowledge_Ingest.py` - Document ingestion UI
- `3_Knowledge_Search.py` - Vector + graph search
- `4_Collection_Management.py` - Working collections CRUD
- `5_Proposal_Step_1_Prep.py` - Template editor
- `6_Proposal_Step_2_Make.py` - Proposal lifecycle
- `Proposal_Copilot.py` - AI-assisted proposal drafting

## Development Notes

### Recent Architectural Improvements (v39.0.0+)
- **Hybrid Model Architecture**: Optimal model selection per task type
- **Centralized Utilities**: Eliminated code duplication by extracting common functionality to `cortex_engine/utils/`
- **Standardized Logging**: All modules now use consistent logging instead of mixed print statements
- **Exception Hierarchy**: Implemented structured exception handling with `cortex_engine/exceptions.py`
- **Path Handling**: Unified cross-platform path conversion logic

### Database Migration
If upgrading from versions prior to 70.0.0, delete the entire `knowledge_hub_db` directory to trigger recreation with the stable format.

### Common Issues
- **MuPDF color profile warnings**: Harmless, text extraction continues normally
- **spaCy model errors**: Ensure `en_core_web_sm` model is downloaded
- **Path issues in WSL**: All paths support both Linux and Windows formats
- **Import errors**: Use `from cortex_engine.utils import <function>` for utility functions

### Logging Locations
- **Ingestion**: `logs/ingestion.log`
- **Query**: `logs/query.log`
- **Processed Files**: `<db_path>/knowledge_hub_db/ingested_files.log`
- **Module Logging**: Centralized through `cortex_engine.utils.get_logger(__name__)`
