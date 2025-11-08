# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "4.10.1"

# Version details
VERSION_INFO = {
    "major": 4,
    "minor": 10,
    "patch": 1,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2025-10-09",
    "release_name": "GPU Acceleration & Docker Parity",
    "description": "GPU acceleration support for Docker with automatic detection and build optimization, plus UI completion bug fixes",
    "breaking_changes": [],
    "new_features": [
        "GPU-enabled Docker build with CUDA 12.1 support (Dockerfile.gpu)",
        "Automatic GPU detection in run-cortex.bat - builds GPU or CPU image appropriately",
        "CUDA-enabled PyTorch wheels for NVIDIA GPU acceleration (torch==2.3.1+cu121)",
        "Comprehensive GPU setup documentation (GPU_SETUP.md)",
        "Performance benchmarks: 3-5x speedup with GPU acceleration"
    ],
    "improvements": [
        "Docker build now automatically detects NVIDIA GPU via nvidia-smi",
        "Builds Dockerfile.gpu for NVIDIA systems, standard Dockerfile for CPU-only",
        "GPU setup guide includes Windows/Linux installation, verification, troubleshooting",
        "Requirements-gpu.txt for CUDA dependencies separate from base requirements"
    ],
    "bug_fixes": [
        "Fixed persistent 'Starting automatic finalization...' message in Knowledge Ingest",
        "Added finalize_done_detected flag for proper completion state tracking",
        "Removed lingering st.info() widgets that persisted across streamlit reruns",
        "GPU now properly utilized in Docker containers (was CPU-only before)"
    ],
    "performance": [
        "GPU acceleration in Docker: 3-5x faster embedding generation",
        "Batch sizing up to 128 for high-memory GPUs (24GB+)",
        "CUDA 12.1 compatibility with compute capability 6.0+ (Pascal+)",
        "Automatic fallback to CPU-only build when no NVIDIA GPU detected"
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
