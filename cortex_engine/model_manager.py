# ## File: cortex_engine/model_manager.py
# Version: v1.0.0
# Date: 2025-12-24
# Purpose: Model management utilities for embedding and LLM model selection

"""
Model Manager
=============
Provides utilities for managing embedding models, checking availability,
and allowing user selection.
"""

import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .utils.logging_utils import get_logger
from .utils.smart_model_selector import detect_nvidia_gpu

logger = get_logger(__name__)

# Available embedding models
EMBEDDING_MODELS = {
    "nvidia/NV-Embed-v2": {
        "name": "NVIDIA Nemotron NV-Embed-v2",
        "description": "Latest NVIDIA Nemotron embedding model - Best for NVIDIA GPUs",
        "size_gb": 1.2,
        "requires_gpu": False,
        "optimized_for_gpu": True,
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "⚡⚡⚡⚡⚡ (on GPU)",
        "recommended_for": "NVIDIA GPU systems"
    },
    "BAAI/bge-base-en-v1.5": {
        "name": "BGE Base English v1.5",
        "description": "Standard high-quality embedding model - Balanced performance",
        "size_gb": 0.5,
        "requires_gpu": False,
        "optimized_for_gpu": False,
        "quality": "⭐⭐⭐⭐",
        "speed": "⚡⚡⚡⚡",
        "recommended_for": "All systems"
    },
    "BAAI/bge-large-en-v1.5": {
        "name": "BGE Large English v1.5",
        "description": "Large high-quality embedding model - Best quality for CPU",
        "size_gb": 1.3,
        "requires_gpu": False,
        "optimized_for_gpu": False,
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "⚡⚡⚡",
        "recommended_for": "High-end CPU systems"
    },
    "BAAI/bge-small-en-v1.5": {
        "name": "BGE Small English v1.5",
        "description": "Lightweight embedding model - Fast and efficient",
        "size_gb": 0.13,
        "requires_gpu": False,
        "optimized_for_gpu": False,
        "quality": "⭐⭐⭐",
        "speed": "⚡⚡⚡⚡⚡",
        "recommended_for": "Low-resource systems"
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "name": "MiniLM L6 v2",
        "description": "Very lightweight model - Fastest option",
        "size_gb": 0.09,
        "requires_gpu": False,
        "optimized_for_gpu": False,
        "quality": "⭐⭐⭐",
        "speed": "⚡⚡⚡⚡⚡",
        "recommended_for": "Resource-constrained environments"
    }
}


def get_available_embedding_models() -> Dict[str, Dict]:
    """Get dictionary of available embedding models with metadata."""
    return EMBEDDING_MODELS


def get_recommended_embedding_model() -> str:
    """
    Get recommended embedding model based on system hardware.

    Returns:
        Model identifier string
    """
    has_nvidia, gpu_info = detect_nvidia_gpu()

    if has_nvidia:
        return "nvidia/NV-Embed-v2"
    else:
        return "BAAI/bge-base-en-v1.5"


def check_model_cached(model_id: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a model is already downloaded/cached locally.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Tuple of (is_cached, cache_path)
    """
    try:
        # sentence-transformers uses ~/.cache/torch/sentence_transformers/
        cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers"

        # Model path format: organization_model-name
        model_path = cache_dir / model_id.replace("/", "_")

        if model_path.exists():
            return True, str(model_path)

        # Also check huggingface cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        model_name = f"models--{model_id.replace('/', '--')}"
        hf_path = hf_cache / model_name

        if hf_path.exists():
            return True, str(hf_path)

        return False, None

    except Exception as e:
        logger.debug(f"Error checking model cache: {e}")
        return False, None


def validate_model_available(model_id: str) -> Tuple[bool, str]:
    """
    Validate that a model can be loaded (cached or downloadable).

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Tuple of (is_available, status_message)
    """
    # Check if cached first
    is_cached, cache_path = check_model_cached(model_id)

    if is_cached:
        return True, f"✅ Model cached locally at {cache_path}"

    # Check if we have internet connectivity to download
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=5)
        return True, f"⬇️ Model will be downloaded on first use (requires internet)"
    except Exception:
        return False, f"❌ Model not cached and no internet connection available"


def download_model(model_id: str, device: str = "cpu") -> Tuple[bool, str]:
    """
    Pre-download an embedding model.

    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on (cpu, cuda, mps)

    Returns:
        Tuple of (success, message)
    """
    try:
        logger.info(f"Downloading embedding model: {model_id}")

        # Temporarily enable online mode
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")

        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

        # Download model
        model = SentenceTransformer(model_id, device=device, trust_remote_code=True)

        # Restore offline settings
        if old_offline:
            os.environ["HF_HUB_OFFLINE"] = old_offline
        if old_transformers_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline

        logger.info(f"✅ Successfully downloaded model: {model_id}")
        return True, f"✅ Successfully downloaded {model_id}"

    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        return False, f"❌ Failed to download: {str(e)[:100]}"


def get_model_info_summary() -> Dict:
    """
    Get summary of current model configuration and system capabilities.

    Uses adaptive embedding selection to automatically choose the best model
    for available hardware (Qwen3-VL > NV-Embed > BGE).

    Returns:
        Dictionary with model and system information
    """
    from .config import get_embed_model, get_embedding_strategy, KB_LLM_MODEL, VLM_MODEL

    has_nvidia, gpu_info = detect_nvidia_gpu()

    # Get adaptive embedding selection
    embed_strategy = get_embedding_strategy()
    embed_model = embed_strategy["model"]

    embed_cached, embed_path = check_model_cached(embed_model)

    return {
        "embedding_model": embed_model,
        "embedding_strategy": embed_strategy,  # Full strategy details
        "embedding_cached": embed_cached,
        "embedding_path": embed_path,
        "llm_model": KB_LLM_MODEL,
        "vlm_model": VLM_MODEL,
        "has_nvidia_gpu": has_nvidia,
        "gpu_info": gpu_info or {},
        "recommended_model": get_recommended_embedding_model(),
        "available_models": list(EMBEDDING_MODELS.keys())
    }


def get_pytorch_cuda_install_command() -> str:
    """
    Get the command to install PyTorch with CUDA support.

    Returns:
        pip install command string
    """
    # CUDA 12.1 is widely compatible with recent NVIDIA drivers
    return "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
