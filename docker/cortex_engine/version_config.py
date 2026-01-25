# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "5.4.0"

# Version details
VERSION_INFO = {
    "major": 5,
    "minor": 4,
    "patch": 0,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2026-01-25",
    "release_name": "Database Portability & Model Size Selection",
    "description": "Portable database transfers between machines with different GPU configurations, interactive Qwen3-VL model size selector, and streamlined backup UI.",
    "breaking_changes": [],
    "new_features": [
        "Database Portability: Export/import databases with embedding model auto-configuration",
        "Qwen3-VL Model Size Selector: Interactive dropdown to choose 2B/8B/Auto in Knowledge Ingest sidebar",
        "Export Manifest: Packages include hardware requirements, model config, and MRL compatibility info",
        "Hardware Compatibility Check: Validates GPU VRAM before import"
    ],
    "improvements": [
        "Unified Embedding Model UI: Single section in sidebar replaces confusing dual-section layout",
        "Status bar shows clean model display (e.g., 'Qwen3-VL (2B, 2048D)')",
        "Sidebar dimensions update immediately when selecting different model size",
        "Streamlined Maintenance tab: Removed redundant backup sections, kept unified Backup & Transfer",
        "Export summary correctly shows dimensions and VRAM based on actual model size config"
    ],
    "bug_fixes": [
        "Fixed dimensions showing 4096 when 2B model selected (stale cache issue)",
        "Fixed status bar showing wrong model size after selection change",
        "Fixed duplicate 'Ingest New Documents' header",
        "Fixed export VRAM requirements showing 16GB for 2B model (should be 5GB)"
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
