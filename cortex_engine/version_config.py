# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "6.0.0"

# Version details
VERSION_INFO = {
    "major": 6,
    "minor": 0,
    "patch": 0,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2026-02-07",
    "release_name": "Reset Reliability, ID Integrity, and Smart Extract Preface",
    "description": "Major reliability update for maintenance reset and ingestion ID consistency, plus machine-readable metadata preface generation in Document Extract.",
    "breaking_changes": [
        "Document Extract now prepends YAML front matter metadata preface to converted Markdown files (all supported document types).",
        "Knowledge Ingest UI no longer performs post-finalization collection assignment; assignment is now finalized only inside the ingestion engine."
    ],
    "new_features": [
        "Document Extract metadata preface extraction with source classification (Academic, Consulting Company, AI Generated Report, Other)",
        "YAML preface includes preface_schema, title, source_type, publisher, publishing_date, authors, keywords, and abstract",
        "LLM-assisted metadata extraction with robust heuristic fallback for missing fields",
        "Collection ZIP export now includes export_manifest.json with included and skipped files",
        "Maintenance reset now terminates active ingest processes targeting the same DB path before cleanup"
    ],
    "improvements": [
        "Clean Start flow simplified to deterministic reset + verification of critical KB artifacts",
        "Maintenance reset logic consolidated into shared reusable reset routine",
        "Maintenance and Document Extract pages now use dynamic central version string",
        "Recovery collection creation now validates IDs against vector store to avoid reintroducing orphans",
        "Preface keywords capped to top 8 for consistent downstream machine use"
    ],
    "bug_fixes": [
        "Fixed ingestion identity persistence by setting LlamaIndex document id via doc.id_ during finalization",
        "Fixed orphan collection references caused by stale UI-side assignment IDs",
        "Fixed reset behavior where lingering ingestion processes could recreate data during/after cleanup"
    ],
    "performance": []
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
