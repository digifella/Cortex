# ## File: config.py
# Version: 3.2.0 (Qwen3-VL Integration)
# Date: 2026-01-17
# Purpose: Central configuration file for Project Cortex with hybrid model support.
#          - CHANGE (v3.2.0): Added Qwen3-VL multimodal embedding and reranking configuration
#          - CHANGE (v3.0.0): Added hybrid model distribution configuration
#          - Support for both Docker Model Runner and Ollama backends
#          - Environment-aware model selection strategy

import os
from pathlib import Path
from typing import Dict, Any, Optional
from .utils.default_paths import get_default_ai_database_path

# --- Core Paths ---
# This is now a FALLBACK. The scripts will accept a path argument to override this.
# Uses cross-platform path detection instead of hardcoded paths
BASE_DATA_PATH = get_default_ai_database_path()

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
STAGING_INGESTION_FILE = os.path.join(BASE_DATA_PATH, "staging_ingestion.json")

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
# UNIFIED ADAPTIVE EMBEDDING SELECTION
# =====================================
# Automatically selects optimal embedding approach based on hardware:
# - Qwen3-VL (multimodal) for GPUs with 6GB+ VRAM + qwen-vl-utils installed
# - NV-Embed-v2 (GPU-optimized) for NVIDIA GPUs with 2-6GB VRAM
# - BGE-base (CPU-friendly) for systems without GPU
#
# Environment variable CORTEX_EMBED_MODEL can override auto-detection
# Environment variable QWEN3_VL_ENABLED can force Qwen3-VL on/off

_EMBED_STRATEGY_CACHE = None

def get_embedding_strategy() -> dict:
    """
    Get the optimal embedding strategy for current hardware (lazy-loaded).

    Returns a dict with full strategy details including model, dimensions,
    multimodal support, and reasoning.
    """
    global _EMBED_STRATEGY_CACHE
    if _EMBED_STRATEGY_CACHE is not None:
        return _EMBED_STRATEGY_CACHE

    # Check for Qwen3-VL manual override
    qwen3vl_override = os.getenv("QWEN3_VL_ENABLED", "").lower()

    try:
        from cortex_engine.utils.smart_model_selector import get_adaptive_embedding_strategy

        strategy = get_adaptive_embedding_strategy()

        # Apply Qwen3-VL override if explicitly set
        if qwen3vl_override == "true":
            # Force Qwen3-VL even if auto-selection chose something else
            from cortex_engine.utils.smart_model_selector import get_optimal_qwen3_vl_config
            qwen_config = get_optimal_qwen3_vl_config()
            if qwen_config["embedding_model"]:
                strategy = {
                    "approach": "qwen3vl",
                    "model": qwen_config["embedding_model"],
                    "dimensions": qwen_config["embedding_dim"],
                    "multimodal": True,
                    "reranker": qwen_config["reranker_model"],
                    "vram_required_gb": 5.0 if "2B" in qwen_config["embedding_model"] else 16.0,
                    "reason": "Manual override via QWEN3_VL_ENABLED=true",
                    "config": qwen_config
                }
        elif qwen3vl_override == "false":
            # Force disable Qwen3-VL even if auto-selection chose it
            if strategy["approach"] == "qwen3vl":
                from cortex_engine.utils.smart_model_selector import detect_nvidia_gpu
                has_nvidia, _ = detect_nvidia_gpu()
                strategy = {
                    "approach": "nv-embed" if has_nvidia else "bge",
                    "model": "nvidia/NV-Embed-v2" if has_nvidia else "BAAI/bge-base-en-v1.5",
                    "dimensions": 4096 if has_nvidia else 768,
                    "multimodal": False,
                    "reranker": None,
                    "vram_required_gb": 1.2 if has_nvidia else 0,
                    "reason": "Manual override via QWEN3_VL_ENABLED=false",
                    "config": {}
                }

        _EMBED_STRATEGY_CACHE = strategy
    except Exception as e:
        # Fallback to BGE on any error
        _EMBED_STRATEGY_CACHE = {
            "approach": "bge",
            "model": "BAAI/bge-base-en-v1.5",
            "dimensions": 768,
            "multimodal": False,
            "reranker": None,
            "vram_required_gb": 0,
            "reason": f"Error in auto-detection: {e}",
            "config": {}
        }

    return _EMBED_STRATEGY_CACHE

def get_embed_model() -> str:
    """Get the optimal embedding model identifier (lazy-loaded)."""
    strategy = get_embedding_strategy()
    return strategy["model"]


def invalidate_embedding_cache():
    """
    Invalidate all embedding-related caches.
    Call this after making hardware changes or removing env overrides.
    """
    global _EMBED_STRATEGY_CACHE
    _EMBED_STRATEGY_CACHE = None

    # Also invalidate smart_model_selector cache if it exists
    try:
        from cortex_engine.utils import smart_model_selector
        if hasattr(smart_model_selector, '_STRATEGY_CACHE'):
            smart_model_selector._STRATEGY_CACHE = None
    except ImportError:
        pass

# For backwards compatibility - but prefer get_embed_model() for lazy loading
EMBED_MODEL = os.getenv("CORTEX_EMBED_MODEL", "BAAI/bge-base-en-v1.5")  # Fast default, actual detection is lazy

# Vision Language Model Configuration
# Options: "llava:7b", "llava:13b", "llava:34b" (newer, more capable models)
# or "moondream" (smaller, faster alternative)
VLM_MODEL = "llava:7b"  # Vision language model for image processing - upgraded to 7B parameter model

# --- Docling VLM Processing Configuration ---
# Phase 1: Enhanced figure processing with VLM descriptions
DOCLING_VLM_ENABLED = os.getenv("DOCLING_VLM_ENABLED", "true").lower() == "true"
DOCLING_VLM_MAX_WORKERS = int(os.getenv("DOCLING_VLM_MAX_WORKERS", "4"))  # Optimized for RTX 8000
DOCLING_VLM_TIMEOUT = int(os.getenv("DOCLING_VLM_TIMEOUT", "30"))  # Seconds per figure
DOCLING_PROVENANCE_ENABLED = os.getenv("DOCLING_PROVENANCE_ENABLED", "true").lower() == "true"

# Phase 2: Table-aware processing configuration
TABLE_AWARE_CHUNKING = os.getenv("TABLE_AWARE_CHUNKING", "true").lower() == "true"
TABLE_SPECIFIC_EMBEDDINGS = os.getenv("TABLE_SPECIFIC_EMBEDDINGS", "true").lower() == "true"
FIGURE_ENTITY_LINKING = os.getenv("FIGURE_ENTITY_LINKING", "true").lower() == "true"

# Phase 3: Multi-modal and advanced features
MULTIMODAL_ENABLED = os.getenv("MULTIMODAL_ENABLED", "false").lower() == "true"  # Off by default until Phase 3
MULTIMODAL_WEIGHT_TEXT = float(os.getenv("MULTIMODAL_WEIGHT_TEXT", "0.5"))
MULTIMODAL_WEIGHT_TABLES = float(os.getenv("MULTIMODAL_WEIGHT_TABLES", "0.3"))
MULTIMODAL_WEIGHT_IMAGES = float(os.getenv("MULTIMODAL_WEIGHT_IMAGES", "0.2"))
MCP_SERVER_ENABLED = os.getenv("MCP_SERVER_ENABLED", "false").lower() == "true"
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8001"))
MCP_SERVER_API_KEY = os.getenv("MCP_SERVER_API_KEY", "")
SCHEMA_EXTRACTION_ENABLED = os.getenv("SCHEMA_EXTRACTION_ENABLED", "false").lower() == "true"  # Off by default

# --- Qwen3-VL Multimodal Embedding Configuration ---
# Unified multimodal embeddings for text, images, and video in same vector space
# Enables cross-modal search (e.g., text query finding relevant images/charts)
# NOW AUTO-DETECTED: Will use Qwen3-VL if 6GB+ VRAM available and qwen-vl-utils installed
# Set QWEN3_VL_ENABLED=true to force enable, or =false to force disable
def _should_use_qwen3vl() -> bool:
    """Auto-detect if Qwen3-VL should be used based on hardware and dependencies."""
    override = os.getenv("QWEN3_VL_ENABLED", "").lower()
    if override == "true":
        return True
    elif override == "false":
        return False
    # Auto-detect based on embedding strategy
    try:
        strategy = get_embedding_strategy()
        return strategy["approach"] == "qwen3vl"
    except Exception:
        return False

QWEN3_VL_ENABLED = _should_use_qwen3vl()

# Model size selection: "auto", "2B", "8B"
# - auto: Selects based on available VRAM (8B if >=20GB free, else 2B)
# - 2B: Qwen3-VL-Embedding-2B (~5GB VRAM, 2048 dimensions)
# - 8B: Qwen3-VL-Embedding-8B (~16GB VRAM, 4096 dimensions)
QWEN3_VL_MODEL_SIZE = os.getenv("QWEN3_VL_MODEL_SIZE", "auto")

# Matryoshka Representation Learning (MRL) dimension reduction
# Set to reduce embedding dimensions for storage efficiency while maintaining quality
# Options: None (full dims), 64, 128, 256, 512, 1024, 2048
# e.g., QWEN3_VL_MRL_DIM=1024 reduces 4096D to 1024D (75% storage savings, ~2% quality loss)
QWEN3_VL_MRL_DIM = os.getenv("QWEN3_VL_MRL_DIM")
if QWEN3_VL_MRL_DIM:
    QWEN3_VL_MRL_DIM = int(QWEN3_VL_MRL_DIM)
else:
    QWEN3_VL_MRL_DIM = None

# Reranker configuration
QWEN3_VL_RERANKER_ENABLED = os.getenv("QWEN3_VL_RERANKER_ENABLED", "false").lower() == "true"
QWEN3_VL_RERANKER_SIZE = os.getenv("QWEN3_VL_RERANKER_SIZE", "auto")  # auto, 2B, 8B
QWEN3_VL_RERANKER_TOP_K = int(os.getenv("QWEN3_VL_RERANKER_TOP_K", "20"))  # Results after reranking
QWEN3_VL_RERANKER_CANDIDATES = int(os.getenv("QWEN3_VL_RERANKER_CANDIDATES", "50"))  # Candidates before reranking

# Flash Attention 2 (recommended for memory efficiency)
QWEN3_VL_USE_FLASH_ATTENTION = os.getenv("QWEN3_VL_USE_FLASH_ATTENTION", "true").lower() == "true"

# Batch processing limits (adjust based on your GPU memory)
QWEN3_VL_EMBED_BATCH_SIZE = int(os.getenv("QWEN3_VL_EMBED_BATCH_SIZE", "8"))
QWEN3_VL_RERANK_BATCH_SIZE = int(os.getenv("QWEN3_VL_RERANK_BATCH_SIZE", "4"))

# --- Task-Specific Model Configuration ---
# Dynamic Model Selection Based on System Resources
# Import smart model selector for intelligent model selection
try:
    from cortex_engine.utils.smart_model_selector import get_recommended_text_model
    SMART_MODEL_SELECTION = get_recommended_text_model()
except Exception:
    SMART_MODEL_SELECTION = "mistral:latest"  # Fallback to efficient model

# Proposal Generation: MUST be local, optimized for instruction following
PROPOSAL_LLM_MODEL = SMART_MODEL_SELECTION  # Intelligent selection based on system resources

# Knowledge Base Operations: Local, optimized for retrieval and indexing  
KB_LLM_MODEL = SMART_MODEL_SELECTION  # Same as proposals for consistency

# Research Assistant Models: Flexible (user choice in UI)
RESEARCH_LOCAL_MODEL = "mistral:latest"  # Fast local option
RESEARCH_CLOUD_MODEL = "gemini-1.5-flash"  # Powerful cloud option

# Model Registry Configuration
MODEL_REGISTRY_FILE = os.path.join(BASE_DATA_PATH, "model_registry.json")

# Legacy/Fallback
LLM_MODEL = SMART_MODEL_SELECTION  # Intelligent default based on system resources

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

def get_db_path(custom_path: Optional[str] = None) -> str:
    """Get database path - supports custom path override or falls back to BASE_DATA_PATH."""
    if custom_path:
        return custom_path.strip()
    return BASE_DATA_PATH

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

        # Docling Enhancements
        "docling_vlm_enabled": DOCLING_VLM_ENABLED,
        "docling_vlm_max_workers": DOCLING_VLM_MAX_WORKERS,
        "docling_vlm_timeout": DOCLING_VLM_TIMEOUT,
        "docling_provenance_enabled": DOCLING_PROVENANCE_ENABLED,
        "table_aware_chunking": TABLE_AWARE_CHUNKING,
        "table_specific_embeddings": TABLE_SPECIFIC_EMBEDDINGS,
        "figure_entity_linking": FIGURE_ENTITY_LINKING,
        "multimodal_enabled": MULTIMODAL_ENABLED,
        "multimodal_weights": {
            "text": MULTIMODAL_WEIGHT_TEXT,
            "tables": MULTIMODAL_WEIGHT_TABLES,
            "images": MULTIMODAL_WEIGHT_IMAGES
        },
        "mcp_server_enabled": MCP_SERVER_ENABLED,
        "mcp_server_host": MCP_SERVER_HOST,
        "mcp_server_port": MCP_SERVER_PORT,
        "schema_extraction_enabled": SCHEMA_EXTRACTION_ENABLED,

        # Qwen3-VL Configuration
        "qwen3_vl_enabled": QWEN3_VL_ENABLED,
        "qwen3_vl_model_size": QWEN3_VL_MODEL_SIZE,
        "qwen3_vl_mrl_dim": QWEN3_VL_MRL_DIM,
        "qwen3_vl_reranker_enabled": QWEN3_VL_RERANKER_ENABLED,
        "qwen3_vl_reranker_size": QWEN3_VL_RERANKER_SIZE,
        "qwen3_vl_reranker_top_k": QWEN3_VL_RERANKER_TOP_K,
        "qwen3_vl_reranker_candidates": QWEN3_VL_RERANKER_CANDIDATES,
        "qwen3_vl_use_flash_attention": QWEN3_VL_USE_FLASH_ATTENTION,
        "qwen3_vl_embed_batch_size": QWEN3_VL_EMBED_BATCH_SIZE,
        "qwen3_vl_rerank_batch_size": QWEN3_VL_RERANK_BATCH_SIZE,

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
        },
        "multimodal_embeddings": {
            "enabled": QWEN3_VL_ENABLED,
            "model_size": QWEN3_VL_MODEL_SIZE,
            "mrl_dim": QWEN3_VL_MRL_DIM,
            "backend_preference": "local_only",
            "performance_tier": "premium"
        },
        "reranking": {
            "enabled": QWEN3_VL_RERANKER_ENABLED,
            "model_size": QWEN3_VL_RERANKER_SIZE,
            "top_k": QWEN3_VL_RERANKER_TOP_K,
            "candidates": QWEN3_VL_RERANKER_CANDIDATES,
            "backend_preference": "local_only",
            "performance_tier": "premium"
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