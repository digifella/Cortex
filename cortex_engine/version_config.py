# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "4.3.0"

# Version details
VERSION_INFO = {
    "major": 4,
    "minor": 3,
    "patch": 0,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2025-08-30",
    "release_name": "Critical Search Functionality Restoration",
    "description": "Major fix for search functionality with embedding dimension mismatch resolution and robust text-based fallback search system",
    "breaking_changes": [],
    "new_features": [
        "Intelligent search fallback system with text-based search when vector embeddings fail",
        "Enhanced search diagnostics with embedding dimension mismatch detection",
        "Robust ChromaDB error handling with graceful degradation to text search",
        "Multi-strategy search approach prioritizing result accuracy over search method",
        "Search reliability improvements ensuring results are always returned when documents exist"
    ],
    "improvements": [
        "Fixed critical search functionality that was returning zero results due to embedding dimension conflicts",
        "Restored GraphRAG search capabilities with proper error handling and fallback mechanisms",
        "Enhanced search result accuracy by implementing text-based matching when vector search fails",
        "Improved search performance by detecting and avoiding incompatible embedding operations",
        "Added comprehensive search debugging and error reporting for better troubleshooting",
        "Strengthened ChromaDB telemetry error suppression for cleaner user experience",
        "Enhanced Ollama model service with synchronous fallback for event loop issues"
    ],
    "bug_fixes": [
        "CRITICAL: Fixed search returning zero results due to embedding dimension mismatch (384 vs 768)",
        "Fixed GraphRAG health check failing with missing get_db_path import",
        "Fixed Ollama model service 'Event loop is closed' errors with synchronous fallback",
        "Fixed ChromaDB telemetry spam with proper warning suppression",
        "Restored ability to find documents containing search terms (e.g., 61+ documents with 'strategy')",
        "Fixed search interface showing 'No results found' when documents actually existed in database"
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