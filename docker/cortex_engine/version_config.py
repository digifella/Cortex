# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "4.10.3"

# Version details
VERSION_INFO = {
    "major": 4,
    "minor": 10,
    "patch": 3,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2025-12-02",
    "release_name": "Docker Path Configuration & Offline Embedding Support",
    "description": "Enhanced Docker setup workflow with interactive path configuration and full offline operation support for embeddings.",
    "breaking_changes": [],
    "new_features": [
        "Docker batch file now prompts for both AI database and knowledge source paths with pre-filled defaults",
        "Pre-downloads embedding model (BAAI/bge-base-en-v1.5) during Docker build for offline operation",
        "Automatic path reconfiguration workflow - press ENTER to keep existing paths or type new ones",
        "Knowledge Ingest UI now properly pre-fills configured source paths from environment variables"
    ],
    "improvements": [
        "Docker containers now work fully offline for document ingestion and search after initial build",
        "HuggingFace offline mode enforcement at both system and application levels",
        "Graceful fallback from offline to online mode for embedding model loading",
        "Enhanced session state initialization to respect environment-configured paths in Docker",
        "Clear error messages when embedding model cache is missing and internet unavailable"
    ],
    "bug_fixes": [
        "Fixed Docker batch file only prompting for AI database path (now prompts for both paths)",
        "Fixed blank 'Root Source Documents Path' field in Knowledge Ingest UI when running in Docker",
        "Resolved embedding service attempting HuggingFace connections in offline Docker environments",
        "Fixed session state not respecting KNOWLEDGE_SOURCE_PATH environment variable on first load"
    ],
    "performance": [
        "Eliminated unnecessary HuggingFace connection attempts in offline mode (faster embedding loading)",
        "Embedding model cached in Docker image (~420MB) eliminates runtime download overhead",
        "Reduced Docker startup time by preventing network timeouts when offline"
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
