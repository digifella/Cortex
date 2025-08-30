# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "4.4.0"

# Version details
VERSION_INFO = {
    "major": 4,
    "minor": 4,
    "patch": 0,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2025-08-31",
    "release_name": "Database Management & Clean Start System",
    "description": "Comprehensive database maintenance system with Clean Start functionality for ChromaDB schema conflict resolution and complete system reset capabilities",
    "breaking_changes": [],
    "new_features": [
        "Clean Start function for complete system reset and database schema conflict resolution",
        "Enhanced ChromaDB schema error detection and user guidance in Collection Management",
        "Comprehensive database cleanup system addressing Docker vs non-Docker conflicts",
        "Smart error handling for 'collections.config_json_str' column missing errors",
        "Complete maintenance workflow with guided user experience for database issues"
    ],
    "improvements": [
        "Enhanced Maintenance page (v4.4.0) with prominent Clean Start functionality",
        "Improved Collection Management (v4.3.0) with specific schema error detection",
        "Comprehensive system reset capability removing all databases, logs, and metadata",
        "Docker compatibility improvements for database schema consistency",
        "User-friendly error messages with actionable next steps for schema conflicts",
        "Technical documentation and educational content about ChromaDB version conflicts",
        "Streamlined database maintenance workflow with one-click solutions"
    ],
    "bug_fixes": [
        "CRITICAL: Fixed ChromaDB 'collections.config_json_str' column missing schema errors",
        "Resolved Docker vs non-Docker database compatibility conflicts",
        "Fixed Collection Management page failing to load due to schema mismatches",
        "Eliminated database corruption issues through comprehensive Clean Start reset",
        "Fixed inconsistent database states between different deployment environments",
        "Resolved ChromaDB version incompatibility issues with complete schema refresh"
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