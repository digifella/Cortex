# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "5.0.0"

# Version details
VERSION_INFO = {
    "major": 5,
    "minor": 0,
    "patch": 0,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2026-01-02",
    "release_name": "Mixture of Experts (MoE) & Adaptive Intelligence",
    "description": "Major intelligence upgrade with Mixture of Experts synthesis, adaptive model selection, creativity controls, and enhanced source attribution. Universal Knowledge Assistant now runs multiple 70B models in parallel for superior analysis quality.",
    "breaking_changes": [],
    "new_features": [
        "Mixture of Experts (MoE) mode: Runs 2-3 expert models (qwen2.5:72b, llama3.3:70b) in parallel with meta-synthesis",
        "Adaptive model selection with reasoning display: System explains WHY each model was chosen for the task",
        "Creativity slider (0-3 scale): Adjust from factual/conservative to highly experimental outputs",
        "Enhanced source citations: Shows collection names, file names, and relevance scores with proper fallbacks",
        "Model selection override: Manual selection of power models when auto-select isn't desired",
        "Real-time streaming synthesis with async architecture for responsive UI",
        "Intent classification: Automatic detection of ideation, synthesis, research, or exploration tasks"
    ],
    "improvements": [
        "Extended timeout support (600s) for large 70B models to prevent premature timeouts",
        "Improved ModernOllamaLLM with proper async/await streaming using aiohttp",
        "Better source metadata extraction: Tries multiple fields (title, file_name, source) with content fallback",
        "Debug logging for timeout diagnostics in production environments",
        "Collection-aware search with 'All Collections' global search option",
        "Temperature mapping: Creativity slider intelligently maps to LLM temperature (0.2-2.0)",
        "MoE meta-synthesis: Combines expert analyses into superior insights beyond individual outputs",
        "Collapsible expert outputs: Clean UI with expandable sections for each expert analysis"
    ],
    "bug_fixes": [
        "Fixed async completion callback validation errors in LlamaIndex integration",
        "Fixed aiohttp streaming timeout issues by using sock_read instead of total timeout",
        "Fixed tuple unpacking errors when retrieving model selection with reasoning",
        "Fixed ChromaVectorStore import path for compatibility with current LlamaIndex version",
        "Fixed missing list_collections() method in WorkingCollectionManager",
        "Cleared Python bytecode cache to ensure fresh module loading after updates"
    ],
    "performance": [
        "MoE synthesis delivers higher quality analysis by leveraging multiple expert perspectives",
        "Non-blocking async streaming ensures responsive UI during long synthesis operations",
        "Proper timeout handling prevents resource waste on stalled requests",
        "GPU-accelerated embeddings with optimal batch sizing for 48GB Quadro RTX 8000"
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
