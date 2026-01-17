# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "5.1.0"

# Version details
VERSION_INFO = {
    "major": 5,
    "minor": 1,
    "patch": 0,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2026-01-17",
    "release_name": "Qwen3-VL Multimodal Intelligence",
    "description": "Major retrieval upgrade with Qwen3-VL multimodal embeddings and neural reranking. Enables cross-modal search (text queries finding images/charts), visual document search, and two-stage retrieval with ~95% precision neural reranking.",
    "breaking_changes": [],
    "new_features": [
        "Qwen3-VL Multimodal Embeddings: Unified vector space for text, images, and video",
        "Neural Reranking: Two-stage retrieval (fast recall + precision reranking) using Qwen3-VL-Reranker",
        "Cross-Modal Search: Use text queries to find relevant images, charts, and diagrams",
        "Matryoshka Representation Learning (MRL): Dimension reduction for storage efficiency",
        "Database Embedding Inspector: Analyze stored embeddings and check model compatibility",
        "Auto-selection: Automatically selects optimal Qwen3-VL model size based on VRAM",
        "LlamaIndex Integration: Drop-in Qwen3VLEmbedding and Qwen3VLReranker adapters"
    ],
    "improvements": [
        "Switchable embedding backend: Seamlessly switch between BGE/NV-Embed and Qwen3-VL",
        "Three-stage retrieval pipeline: Vector search → Graph enhancement → Neural reranking",
        "UI status displays: Qwen3-VL status shown in Ingest and Search sidebars",
        "Compatibility matrix: Shows which embedding models work with your database",
        "Backward compatible: Existing databases continue working with default settings",
        "Reranker works with any embedding: No re-ingest needed to use neural reranking"
    ],
    "bug_fixes": [
        "Fixed embedding dimension validation for cross-model compatibility checking"
    ],
    "performance": [
        "Neural reranking improves search precision from ~85% to ~95%+",
        "Qwen3-VL-2B: 5GB VRAM, 2048 dimensions - efficient multimodal",
        "Qwen3-VL-8B: 16GB VRAM, 4096 dimensions - premium quality",
        "Flash Attention 2 support for memory optimization",
        "Batch processing with GPU optimization for large document sets"
    ]
}

# ============================================================================
# VERSION FORMATTING FUNCTIONS
# ============================================================================

def get_version_string() -> str:
    """Get the full version string (e.g., 'v4.0.0')"""
    return f"v{CORTEX_VERSION}"

def get_version_display() -> str:
    """Get version for UI display with release name"""
    return f"{get_version_string()} - {VERSION_METADATA['release_name']}"

def get_full_version_info() -> Dict[str, Any]:
    """Get complete version information"""
    return {
        **VERSION_INFO,
        **VERSION_METADATA,
        "formatted_version": get_version_string(),
        "display_version": get_version_display(),
    }

def get_version_footer() -> str:
    """Get version footer for pages"""
    return f"Version: {get_version_string()} • {VERSION_METADATA['description']}"

def get_changelog_entry() -> str:
    """Generate changelog entry for this version"""
    entry = f"""## {get_version_string()} - {VERSION_METADATA['release_date']}

### {VERSION_METADATA['release_name']}

{VERSION_METADATA['description']}

"""
    
    if VERSION_METADATA.get('breaking_changes'):
        entry += "### 🔥 Breaking Changes\n"
        for change in VERSION_METADATA['breaking_changes']:
            entry += f"- {change}\n"
        entry += "\n"
    
    if VERSION_METADATA.get('new_features'):
        entry += "### ✨ New Features\n"
        for feature in VERSION_METADATA['new_features']:
            entry += f"- {feature}\n"
        entry += "\n"
    
    if VERSION_METADATA.get('improvements'):
        entry += "### 🚀 Improvements\n"
        for improvement in VERSION_METADATA['improvements']:
            entry += f"- {improvement}\n"
        entry += "\n"
    
    return entry

# ============================================================================
# VERSION VALIDATION
# ============================================================================

def validate_version_format(version: str) -> bool:
    """Validate that a version string follows semantic versioning"""
    import re
    pattern = r'^v?\d+\.\d+\.\d+(-[a-zA-Z0-9-]+)?(\+[a-zA-Z0-9-]+)?$'
    return bool(re.match(pattern, version))

# Validate our own version
if not validate_version_format(CORTEX_VERSION):
    raise ValueError(f"Invalid version format: {CORTEX_VERSION}")

# ============================================================================
# EXPORT CONSTANTS FOR EASY IMPORTING
# ============================================================================

# Most commonly used exports
VERSION = CORTEX_VERSION
VERSION_STRING = get_version_string()
VERSION_DISPLAY = get_version_display()
RELEASE_DATE = VERSION_METADATA['release_date']

# Update timestamp for tracking when version file was last modified
LAST_UPDATED = datetime.now().isoformat()
