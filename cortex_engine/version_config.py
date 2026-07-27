# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "6.2.0"

# Version details
VERSION_INFO = {
    "major": 6,
    "minor": 2,
    "patch": 0,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2026-07-28",
    "release_name": "Offline Photo Enrichment",
    "description": "Photo Processor can run with no network access: local vision model, offline reverse geocoding from a local GeoNames dataset, and post-hoc person-name substitution. Combined with in-place folder enrichment, a Lightroom catalog can be tagged end to end while travelling.",
    "breaking_changes": [],
    "new_features": [
        "Photo Processor: 'Folder on disk' source mode — enriches all supported images in a folder (recursively) in place, using real file paths instead of temp copies",
        "Photo Processor: offline reverse geocoding (cortex_engine/offline_geocoder.py) with Auto / Online only / Offline only modes — no network and no Nominatim rate limit",
        "Photo Processor: 'Use local vision model only' skips Claude even when ANTHROPIC_API_KEY is set, for fully offline runs",
        "Photo Processor: person-name substitution (cortex_engine/photo_name_tags.py) rewrites 'A man smiles' to 'Paul smiles' from configurable Tag=Name keywords, applied after the model rather than via the prompt",
    ],
    "improvements": [
        "Photo Processor: folder mode skips exiftool *_original backup files when collecting images",
        "Offline geocoding degrades loudly with install instructions rather than silently returning empty location fields",
        "Docs: added docs/photo_processor_spec.md covering the enrichment pipeline, offline mode, name substitution, UI options, and known sharp edges",
        "Docs: exiftool declared as a required system binary in README, CLAUDE.md, and the Docker image",
        "Tests: 28 unit tests across offline geocoding modes and name substitution edge cases",
    ],
    "bug_fixes": [
        "Photo Processor: removed the dead 'Write to original files' toggle, which was never wired into run settings and had no effect",
        "Dependencies: anthropic SDK pinned in requirements.txt — Claude Haiku vision silently fell back to Ollama when the package was absent",
        "Dependencies: reverse_geocoder and pycountry pinned for offline geocoding",
    ],
    "performance": [
        "Offline geocoding removes the ~1 req/sec Nominatim pacing, making offline batches faster than online ones",
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
