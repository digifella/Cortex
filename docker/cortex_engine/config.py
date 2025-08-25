# ## File: config.py
# Version: 3.0.0 (Hybrid Model Architecture)
# Date: 2025-08-20
# Purpose: Central configuration file for Project Cortex with hybrid model support.
#          - CHANGE (v3.0.0): Added hybrid model distribution configuration
#          - Support for both Docker Model Runner and Ollama backends
#          - Environment-aware model selection strategy

import os
from pathlib import Path
from typing import Dict, Any, Optional

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

# --- Hybrid Model Configuration Strategy ---
# Model distribution strategy configuration
MODEL_DISTRIBUTION_STRATEGY = os.getenv("MODEL_DISTRIBUTION_STRATEGY", "hybrid_docker_preferred")
DEPLOYMENT_ENVIRONMENT = os.getenv("DEPLOYMENT_ENV", "production")

# Model Backend Configuration
DOCKER_MODEL_REGISTRY = os.getenv("DOCKER_MODEL_REGISTRY", "docker.io/ai")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Core Local Models (Required) ---
EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # Embedding model for vector storage
# Vision Language Model Configuration
# Options: "llava:7b", "llava:13b", "llava:34b" (newer, more capable models)
# or "moondream" (smaller, faster alternative)
VLM_MODEL = "llava:7b"  # Vision language model for image processing - upgraded to 7B parameter model

# --- Task-Specific Model Configuration ---
# Proposal Generation: MUST be local, optimized for instruction following
PROPOSAL_LLM_MODEL = "mistral-small3.2"  # Mistral Small 3.2 for better proposals

# Knowledge Base Operations: Local, optimized for retrieval and indexing  
KB_LLM_MODEL = "mistral-small3.2"  # Same as proposals for consistency

# Research Assistant Models: Flexible (user choice in UI)
RESEARCH_LOCAL_MODEL = "mistral:7b-instruct-v0.3-q4_K_M"  # Fast local option
RESEARCH_CLOUD_MODEL = "gemini-1.5-flash"  # Powerful cloud option

# Model Registry Configuration
MODEL_REGISTRY_FILE = os.path.join(BASE_DATA_PATH, "model_registry.json")

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
    "*.xls\n*.xlsx\n*.csv\n*.zip\n"
    # Multimedia files (excluding images)
    "*.mp4\n*.mov\n*.avi\n*.mp3\n*.wav\n*.raf\n"
    # Business document types to ignore by default
    "*invoice*\n*timesheet*\n*contract*\n*receipt*"
)

# --- Configuration Helper Functions ---

def get_cortex_config() -> Dict[str, Any]:
    """Get complete Cortex configuration dictionary."""
    return {
        # Paths
        "base_data_path": BASE_DATA_PATH,
        "chroma_db_path": CHROMA_DB_PATH,
        "graph_file_path": GRAPH_FILE_PATH,
        "image_store_path": IMAGE_STORE_PATH,
        "model_registry_file": MODEL_REGISTRY_FILE,
        
        # Model Configuration
        "model_distribution_strategy": MODEL_DISTRIBUTION_STRATEGY,
        "deployment_environment": DEPLOYMENT_ENVIRONMENT,
        "docker_model_registry": DOCKER_MODEL_REGISTRY,
        "ollama_base_url": OLLAMA_BASE_URL,
        
        # Model Assignments
        "embed_model": EMBED_MODEL,
        "vlm_model": VLM_MODEL,
        "proposal_llm_model": PROPOSAL_LLM_MODEL,
        "kb_llm_model": KB_LLM_MODEL,
        "research_local_model": RESEARCH_LOCAL_MODEL,
        "research_cloud_model": RESEARCH_CLOUD_MODEL,
        
        # Environment
        "environment": DEPLOYMENT_ENVIRONMENT,
        "project_root": str(PROJECT_ROOT),
        "logs_dir": str(LOGS_DIR),
    }

def get_model_config_for_task(task_type: str) -> Dict[str, str]:
    """Get model configuration for a specific task type."""
    task_models = {
        "proposals": {
            "model": PROPOSAL_LLM_MODEL,
            "backend_preference": "local_only",
            "performance_tier": "premium"
        },
        "knowledge": {
            "model": KB_LLM_MODEL,
            "backend_preference": "local_only", 
            "performance_tier": "premium"
        },
        "research": {
            "local_model": RESEARCH_LOCAL_MODEL,
            "cloud_model": RESEARCH_CLOUD_MODEL,
            "backend_preference": "user_choice",
            "performance_tier": "standard"
        },
        "embeddings": {
            "model": EMBED_MODEL,
            "backend_preference": "local_only",
            "performance_tier": "standard"
        },
        "vision": {
            "model": VLM_MODEL,
            "backend_preference": "local_preferred",
            "performance_tier": "standard"
        }
    }
    
    return task_models.get(task_type, {
        "model": LLM_MODEL,
        "backend_preference": "auto",
        "performance_tier": "standard"
    })

def update_model_config(updates: Dict[str, Any]):
    """Update model configuration with new values."""
    global MODEL_DISTRIBUTION_STRATEGY, DOCKER_MODEL_REGISTRY, OLLAMA_BASE_URL
    global PROPOSAL_LLM_MODEL, KB_LLM_MODEL, RESEARCH_LOCAL_MODEL, RESEARCH_CLOUD_MODEL
    
    if "model_distribution_strategy" in updates:
        MODEL_DISTRIBUTION_STRATEGY = updates["model_distribution_strategy"]
    
    if "docker_model_registry" in updates:
        DOCKER_MODEL_REGISTRY = updates["docker_model_registry"]
    
    if "ollama_base_url" in updates:
        OLLAMA_BASE_URL = updates["ollama_base_url"]
    
    if "proposal_llm_model" in updates:
        PROPOSAL_LLM_MODEL = updates["proposal_llm_model"]
    
    if "kb_llm_model" in updates:
        KB_LLM_MODEL = updates["kb_llm_model"]
    
    if "research_local_model" in updates:
        RESEARCH_LOCAL_MODEL = updates["research_local_model"]
    
    if "research_cloud_model" in updates:
        RESEARCH_CLOUD_MODEL = updates["research_cloud_model"]

def validate_model_config() -> Dict[str, Any]:
    """Validate the current model configuration."""
    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required models are specified
    required_models = [EMBED_MODEL, VLM_MODEL, PROPOSAL_LLM_MODEL, KB_LLM_MODEL]
    for model in required_models:
        if not model or model.strip() == "":
            validation["errors"].append(f"Required model not specified: {model}")
            validation["valid"] = False
    
    # Check strategy is valid
    valid_strategies = ["hybrid_docker_preferred", "hybrid_ollama_preferred", "docker_only", "ollama_only", "auto_optimal"]
    if MODEL_DISTRIBUTION_STRATEGY not in valid_strategies:
        validation["warnings"].append(f"Unknown distribution strategy: {MODEL_DISTRIBUTION_STRATEGY}")
    
    # Check environment settings
    if DEPLOYMENT_ENVIRONMENT not in ["development", "staging", "production"]:
        validation["warnings"].append(f"Unknown deployment environment: {DEPLOYMENT_ENVIRONMENT}")
    
    return validation