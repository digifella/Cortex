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
    "release_date": "2025-11-16",
    "release_name": "Docling Ingest & Recovery Hardening",
    "description": "Improved document ingestion with Docling, robust database cleanup, and safer WSL/Docker path handling for long-running batches.",
    "breaking_changes": [],
    "new_features": [
        "Docling-based ingestion pipeline enabled for richer Office/PDF parsing inside Docker engine.",
        "Clean Start flow with detailed debug output and step-by-step verification for mounted volumes.",
        "Detection of orphaned ingestion artifacts (staging, batch_state, progress) during Knowledge Search validation.",
        "Shared path utilities for resolving user-specified database roots across Windows hosts and Docker containers."
    ],
    "improvements": [
        "Knowledge Ingest now centralizes DB path resolution via runtime-safe helpers, avoiding hardcoded container paths.",
        "Clean Start and Delete KB flows clear batch_state, staging_ingestion, recovery metadata, and logs even after failed runs.",
        "Knowledge Search validates the configured database path in-container, suggests existing populated roots, and reports stale state clearly.",
        "Batch ingest UI uses stage-based routing so analysis and finalization auto-refresh correctly in Docker without manual refresh confusion."
    ],
    "bug_fixes": [
        "Fixed Clean Start NameError and made it resilient when knowledge_hub_db is missing or partial on mounted volumes.",
        "Resolved cases where ingestion logs and UI state could stay in a 'processing' state after subprocess termination.",
        "Prevented active batch management view from blocking automatic finalization and completion routing.",
        "Ensured Docker/host path conversions do not attempt to write under the project directory inside the container."
    ],
    "performance": [
        "Non-blocking ingestion log reader keeps GPU/CPU throttle metrics live while processing batches in Docker.",
        "Reduced likelihood of stalls in long-running ingests by enforcing unbuffered subprocess output.",
        "Cleaner recovery of interrupted ingests reduces the need for manual directory cleanup in mounted volumes."
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
