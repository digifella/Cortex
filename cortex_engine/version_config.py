# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "4.11.0"

# Version details
VERSION_INFO = {
    "major": 4,
    "minor": 11,
    "patch": 0,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2025-12-28",
    "release_name": "Embedding Model Safeguards & BGE Stability",
    "description": "Comprehensive embedding model safeguards system to prevent data corruption from mixed embeddings. Includes environment variable override for forcing stable BGE model instead of auto-detected NVIDIA models with compatibility issues.",
    "breaking_changes": [],
    "new_features": [
        "Environment variable CORTEX_EMBED_MODEL to override auto-detection and force specific embedding models",
        "Embedding model metadata tracking in collection manager (model name and dimension)",
        "Embedding compatibility validation at ingestion and query time",
        "Embedding inspector tool (scripts/embedding_inspector.py) for database diagnostics",
        "Embedding migration tool (scripts/embedding_migrator.py) for safe model switching",
        "BGE model setup script (setup_bge_model.sh) for one-click stable configuration",
        "Startup script (start_cortex_bge.sh) with automatic BGE model environment setup"
    ],
    "improvements": [
        "Knowledge Ingest page now validates embedding compatibility before ingestion",
        "Knowledge Search page displays warnings when model mismatch detected",
        "Maintenance page shows embedding model status with compatibility checks",
        "Comprehensive documentation in RECOVERY_GUIDE.md and QUICK_START_BGE.md",
        "Embedding model safeguards prevent silent data corruption from mixed embeddings",
        "Added trust_remote_code=True parameter to all SentenceTransformer instantiations",
        "Added datasets>=2.14.0 and einops>=0.7.0 dependencies for advanced models"
    ],
    "bug_fixes": [
        "Fixed NVIDIA NV-Embed-v2 model compatibility issues with transformers library",
        "Fixed ingestion failures when embedding model cache corrupted mid-process",
        "Fixed confusing UI status display during finalization phase of ingestion",
        "Prevented auto-detection from selecting unstable NVIDIA models in production"
    ],
    "performance": [
        "BGE model (768D) provides stable production performance without compatibility issues",
        "Eliminated overnight ingestion failures by avoiding problematic NVIDIA model API changes",
        "Proper batch processing with optimal GPU memory utilization"
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
