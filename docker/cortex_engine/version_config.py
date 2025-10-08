# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "4.9.0"

# Version details
VERSION_INFO = {
    "major": 4,
    "minor": 9,
    "patch": 0,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2025-10-08",
    "release_name": "Critical Performance Optimization",
    "description": "Major performance improvements with async image processing, embedding batch optimization, and intelligent query caching delivering 3-5x faster ingestion",
    "breaking_changes": [],
    "new_features": [
        "Async parallel image processing with 30s timeout and 3-image concurrency",
        "Embedding batch processing with GPU vectorization (batch size 32)",
        "LRU query result caching for instant repeated searches (100 query cache)",
        "Enhanced progress feedback during parallel image processing",
        "Automatic cache invalidation on new ingestion"
    ],
    "improvements": [
        "Image processing enabled by default with optimized performance",
        "Reduced VLM timeout from 120s to 30s per image with graceful fallback",
        "Batch embedding generation (32 documents per batch) for GPU efficiency",
        "Query cache with thread-safe OrderedDict-based LRU eviction",
        "Better error handling and fallback for image processing timeouts",
        "Improved UI feedback for ingestion finalization completion"
    ],
    "bug_fixes": [
        "Fixed check_ollama_service() unpacking errors across 7 files (3-value return)",
        "Fixed missing UI feedback for finalization completion",
        "Fixed Ollama status check unpacking 3 values instead of 2"
    ],
    "performance": [
        "Image ingestion: 3-5x faster with parallel processing",
        "Embedding generation: 2-5x faster with GPU batching",
        "Repeated queries: Instant response from LRU cache",
        "Overall ingestion: 40-60% faster for mixed content"
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
