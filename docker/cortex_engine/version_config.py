# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "4.5.0"

# Version details
VERSION_INFO = {
    "major": 4,
    "minor": 5,
    "patch": 0,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2025-08-31",
    "release_name": "System Stabilization & ChromaDB Consistency Fixes",
    "description": "Emergency stabilization after navigation redesign failure - fixed ChromaDB inconsistencies, Docker path handling, and database validation errors",
    "breaking_changes": [],
    "new_features": [
        "Enhanced Clean Start debug logging with step-by-step operations display",
        "Docker environment detection with proper path fallbacks",
        "Advanced Database Recovery section with safer Clean Start placement",
        "Session state synchronization with configuration values"
    ],
    "improvements": [
        "Standardized ChromaDB settings across all components for consistent connections",
        "Fixed working collections schema compatibility (doc_ids vs documents)",
        "Enhanced ingestion recovery logic to not flag empty collections as issues",
        "Improved Docker path handling with /.dockerenv detection",
        "Better session state initialization in Knowledge Ingest page",
        "Safer Clean Start button placement in Advanced section",
        "Enhanced debug information display with visual pauses"
    ],
    "bug_fixes": [
        "Fixed ChromaDB 'different settings' instance conflicts across Knowledge Search and Collection Management",
        "Resolved nested expander error in Maintenance page UI",
        "Fixed Docker path hardcoding issue where Clean Start used /data/ai_databases instead of user configuration",
        "Corrected Knowledge Ingest session state showing wrong database paths",
        "Fixed persistent 'Fix collection inconsistencies' warnings for empty collections",
        "Resolved navigation chaos from failed hub page implementation (rolled back to v4.4.2)",
        "Fixed hardcoded version references in Knowledge Ingest to use centralized VERSION_STRING"
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