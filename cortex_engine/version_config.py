# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "5.2.0"

# Version details
VERSION_INFO = {
    "major": 5,
    "minor": 2,
    "patch": 0,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2026-01-20",
    "release_name": "Intelligent Proposal Completion Overhaul",
    "description": "Complete redesign of Proposal Intelligent Completion with simplified UX, per-question settings, human-in-the-loop regeneration, and significant performance improvements.",
    "breaking_changes": [],
    "new_features": [
        "Simplified 3-button workflow: Skip | Edit | Auto-Generate",
        "Per-question Evidence Source: Each question can search a different collection",
        "Per-question Creativity slider: Set Factual/Balanced/Creative per question",
        "Human-in-the-loop Regeneration: Refine responses with guidance hints",
        "Per-field Export: Download individual responses for external editing",
        "Entire Knowledge Base search option for evidence retrieval",
        "Simplified Proposal Workspace with clear navigation"
    ],
    "improvements": [
        "Ollama status check cached (60 seconds) - faster UI",
        "ConfigManager cached - eliminates file reads on interactions",
        "Recovery analysis cached (120 seconds) - faster Knowledge Ingest page",
        "Collection manager reloads fresh - fixes stale collection counts",
        "Response text area enlarged with word count indicator",
        "Session state persists during page navigation"
    ],
    "bug_fixes": [
        "Fixed '0 documents in collection' bug - stale cache issue",
        "Fixed Generate button infinite loop - enum serialization fix",
        "Fixed GraphQueryEngine initialization error",
        "Fixed st.rerun() causing premature page restart"
    ],
    "performance": [
        "Knowledge Ingest page now responds instantly to button clicks",
        "Removed 3 expensive operations from every page interaction",
        "Cached network calls, file reads, and database analysis"
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
