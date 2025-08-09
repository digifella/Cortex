# ## File: config.py
# Version: 2.7.0 (Enable Image Ingestion)
# Date: 2025-07-22
# Purpose: Central configuration file for Project Cortex.
#          - CHANGE (v2.7.0): Removed image extensions from the default
#            exclusion patterns to allow them to be discovered by the
#            ingestion scanner.

import os
from pathlib import Path

# --- Core Paths ---
# This is now a FALLBACK. The scripts will accept a path argument to override this.
BASE_DATA_PATH = "/mnt/f/ai_databases"

# The following paths are placeholders; they will be dynamically set by scripts.
# They are derived from the BASE_DATA_PATH by default.
CHROMA_DB_PATH = os.path.join(BASE_DATA_PATH, "knowledge_hub_db")
GRAPH_FILE_PATH = os.path.join(BASE_DATA_PATH, "knowledge_cortex.gpickle")
IMAGE_STORE_PATH = os.path.join(CHROMA_DB_PATH, "images")

# --- Log Files & Staging ---
# Go up one level from this file's directory (cortex_engine) to the project root.
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True) # Ensure the logs directory exists

# UNIFIED LOG: Tracks all processed or excluded files. Stored inside the DB dir.
INGESTED_FILES_LOG = "ingested_files.log"
STAGING_INGESTION_FILE = str(PROJECT_ROOT / "staging_ingestion.json")

# Centralize logs into the 'logs' directory
INGESTION_LOG_PATH = str(LOGS_DIR / "ingestion.log")
QUERY_LOG_PATH = str(LOGS_DIR / "query.log")


# --- ChromaDB and VectorStore Settings ---
COLLECTION_NAME = "knowledge_hub_collection"

# --- Model Configuration Strategy ---
# LOCAL ONLY: Proposals, embedding, retrieval, indexing - MUST be local
# RESEARCH: Choice between local (speed) or cloud (capability)

# --- Core Local Models (Required) ---
EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # Embedding model for vector storage
VLM_MODEL = "llava"  # Vision language model for image processing

# --- Task-Specific Model Configuration ---
# Proposal Generation: MUST be local, optimized for instruction following
PROPOSAL_LLM_MODEL = "mistral-small3.2"  # Mistral Small 3.2 for better proposals

# Knowledge Base Operations: Local, optimized for retrieval and indexing
KB_LLM_MODEL = "mistral-small3.2"  # Same as proposals for consistency

# Research Assistant Models: Flexible (user choice in UI)
RESEARCH_LOCAL_MODEL = "mistral:7b-instruct-v0.3-q4_K_M"  # Fast local option
RESEARCH_CLOUD_MODEL = "gemini-1.5-flash"  # Powerful cloud option

# Legacy/Fallback
LLM_MODEL = "mistral-small:3.2"  # Default fallback

# --- UI Defaults ---
# SPRINT 21 CHANGE: Removed image files from default exclusions.
DEFAULT_EXCLUSION_PATTERNS_STR = (
    # Office temp files
    "~$*.docx\n~$*.xlsx\n~$*.pptx\n"
    # Common junk
    "*.tmp\n*.lnk\n"
    # Web & code files
    "*.css\n*.html\n*.js\n*.py\n*.json\n"
    # Data & archives
    "*.xls\n*.xlsx\n*.csv\n*.zip\n*.ppt\n*.pptx\n"
    # Multimedia files (excluding images)
    "*.mp4\n*.mov\n*.avi\n*.mp3\n*.wav\n*.raf\n"
    # Business document types to ignore by default
    "*invoice*\n*timesheet*\n*contract*\n*receipt*"
)