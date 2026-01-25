# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "5.4.1"

# Version details
VERSION_INFO = {
    "major": 5,
    "minor": 4,
    "patch": 1,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2026-01-26",
    "release_name": "Database Health & Search Model Selection",
    "description": "Database Health Check with orphan detection fix, Maintenance tab reorganization, and runtime model selection for Knowledge Search.",
    "breaking_changes": [],
    "new_features": [
        "Database Health Check: Scan and fix orphaned log entries, collection issues",
        "Knowledge Search Model Selector: Choose Qwen3-VL 2B/8B based on database compatibility",
        "Auto-scan option on database import for clean portable transfers",
        "Database dimension detection with compatibility warnings"
    ],
    "improvements": [
        "Maintenance tab reorganized: 'Backups' renamed to 'Transfer', Health Check moved to Database tab",
        "Removed duplicate Clean Start description and redundant recovery sections",
        "Health Check results persist in session state with auto-rescan after fixes",
        "Maintenance page shows actual embedding model from get_embedding_strategy()",
        "Simplified Danger Zone section with helpful tip pointing to Health Check"
    ],
    "bug_fixes": [
        "Fixed orphaned document detection: Now compares file paths (not hash vs UUID)",
        "Fixed nested expander error in Database Health Check",
        "Fixed Health Check not updating after Remove Orphaned Entries",
        "Fixed Maintenance showing BAAI/bge-base instead of actual Qwen3-VL model",
        "Fixed Knowledge Search model selector causing infinite loop",
        "Fixed ChromaSettings scoping error in dimension detection"
    ],
    "performance": [
        "Model size changes take effect without full page reload",
        "Embedding info derived from config rather than stale strategy cache"
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
