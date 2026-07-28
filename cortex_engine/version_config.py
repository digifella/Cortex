# Cortex Suite Version Configuration
# Central source of truth for all version information
# This file should be the ONLY place where version numbers are defined

from datetime import datetime
from typing import Dict, Any

# ============================================================================
# CENTRAL VERSION CONFIGURATION - SINGLE SOURCE OF TRUTH
# ============================================================================

# Main application version - increment this for any significant changes
CORTEX_VERSION = "6.3.1"

# Version details
VERSION_INFO = {
    "major": 6,
    "minor": 3,
    "patch": 1,
    "pre_release": None,  # e.g., "alpha", "beta", "rc1"
    "build": None,        # e.g., build number for CI/CD
}

# Version metadata
VERSION_METADATA = {
    "version": CORTEX_VERSION,
    "release_date": "2026-07-29",
    "release_name": "Caption Provenance + Two-Pass Retry",
    "description": "Captions record which model wrote them, and photos the fast model cannot describe are retried automatically with a stronger one. Local vision model selection adapts to the free VRAM on the machine, so one install runs well on an 8GB laptop and a 48GB workstation. Reasoning-model output can no longer leak into photo metadata. Photo Processor can run with no network access: local vision model, offline reverse geocoding from a local GeoNames dataset, and post-hoc person-name substitution. Combined with in-place folder enrichment, a Lightroom catalog can be tagged end to end while travelling.",
    "breaking_changes": [],
    "new_features": [
        "Photo Processor: 'Folder on disk' source mode — enriches all supported images in a folder (recursively) in place, using real file paths instead of temp copies",
        "Photo Processor: offline reverse geocoding (cortex_engine/offline_geocoder.py) with Auto / Online only / Offline only modes — no network and no Nominatim rate limit",
        "Photo Processor: 'Use local vision model only' skips Claude even when ANTHROPIC_API_KEY is set, for fully offline runs",
        "Photo Processor: person-name substitution (cortex_engine/photo_name_tags.py) rewrites 'A man smiles' to 'Paul smiles' from configurable Tag=Name keywords, applied after the model rather than via the prompt",
        "VRAM-adaptive vision model selection (cortex_engine/vision_model_selector.py): picks the highest-quality installed model that fits current free VRAM, so an 8GB laptop and a 48GB workstation each get an appropriate model with no configuration",
        "Caption provenance: every generated description records its author in IPTC:Writer-Editor and XMP-photoshop:CaptionWriter as 'Cortex <version> / <model>', visible in Lightroom's metadata panel",
        "Two-pass batch enrichment (scripts/photo_enrich_batch.py): a fast VRAM-appropriate model first, then an automatic retry of undescribed photos with the strongest installed model — resumable, and --only-empty treats a placeholder as unprocessed",
    ],
    "improvements": [
        "Photo Processor: folder mode skips exiftool *_original backup files when collecting images",
        "Offline geocoding degrades loudly with install instructions rather than silently returning empty location fields",
        "Docs: added docs/photo_processor_spec.md covering the enrichment pipeline, offline mode, name substitution, UI options, and known sharp edges",
        "Docs: exiftool declared as a required system binary in README, CLAUDE.md, and the Docker image",
        "Tests: 79 unit tests across VRAM-adaptive selection, caption provenance, placeholder detection, offline geocoding, name substitution, and VLM text normalization",
        "Setup: gemma4:e2b-it-qat declared as the baseline vision model — 1.6GB resident, so it runs alongside Lightroom on an 8GB laptop",
        "Photo Processor shows which local model will be used and why, including a warning when nothing fits available VRAM",
        "Benchmarked four local vision models on identical photos: gemma4:e2b-it-qat 5/6 clean at 80s/photo and 1.6GB; qwen3-vl:8b best content accuracy but 7.4GB and 127s; llava:7b 2/6; qwen3-vl:4b 3/6 with empty outputs",
    ],
    "bug_fixes": [
        "Vision output: a reasoning model that exhausted its token budget returned only chain-of-thought, which the pipeline wrote into photo metadata as the caption — this silently corrupted 44 captions before being caught. The fallback is now off by default and logs the remedy",
        "Token budget is per model family: reasoning models (qwen3-vl, gemma4) need 640 to reach an answer; instruct models (llava, gemma3) need 160 or they overrun the word limit",
        "Prompt structure: rules and reference facts moved to the system turn. Concatenated into the user turn, small models paraphrased the facts back instead of describing the image",
        "Vision model: config.VLM_MODEL was 'llava:7b' while textifier.VISION_MODELS listed 'qwen3-vl:8b' first — model_checker therefore never prompted users to install the model the pipeline actually preferred",
        "Text normalizer: strip reasoning leakage from local vision models ('The main thing is X. So: Y' now yields only Y), with guards so ordinary prose containing 'so,' or 'in the foreground' survives",
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
