# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "4.10.2"

# Version details
VERSION_INFO = {
    "major": 4,
    "minor": 10,
    "patch": 2,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2025-11-12",
    "release_name": "Docker Path Auto-Detection Hotfix",
    "description": "Improves Docker path detection for Knowledge Search and batch ingestion, eliminating false setup errors and aligning UI headers with the global version.",
    "breaking_changes": [],
    "new_features": [
        "Knowledge Search now auto-detects knowledge_hub_db across configured paths, Windows mounts, and container defaults.",
        "WorkingCollectionManager and sidebar path tooling surface actionable guidance when a fallback mount is used.",
        "Documentation updated with Docker-specific troubleshooting for bind-mount verification."
    ],
    "improvements": [
        "Batch ingestion UI now sources its version header directly from the central version configuration.",
        "App headers and Docker launch scripts display the same semantic version as the engine.",
        "Clearer user messaging when the configured database path is missing but a fallback path succeeds."
    ],
    "bug_fixes": [
        "Resolved false 'Docker Setup Required' warnings when the knowledge base is available via /mnt/<drive> mounts.",
        "Prevented Knowledge Search from hard-failing when the configured path is unavailable but a mounted fallback exists.",
        "Ensured documentation and runtime messaging consistently reference the updated Docker path guidance."
    ],
    "performance": [
        "Reduced redundant ChromaDB initialization attempts by caching the successful path selection within a validation cycle.",
        "Faster troubleshooting thanks to explicit path audit logs in the Knowledge Search UI.",
        "Improved ingestion status accuracy by aligning version headers across interfaces."
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
