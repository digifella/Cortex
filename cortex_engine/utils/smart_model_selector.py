# Smart Model Selector - Intelligent Model Selection Based on System Resources
# Version: v3.0.0 - Fully Adaptive Embedding Selection
# Date: 2026-01-21
# Purpose: Automatically select appropriate models based on available system resources
#          - NEW (v3.0.0): Unified adaptive embedding selection (Qwen3-VL vs NV-Embed vs BGE)
#          - NEW (v2.0.0): Added NVIDIA GPU detection and Nematron model selection

import psutil
import subprocess
import logging
import os
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_nvidia_gpu() -> Tuple[bool, Optional[Dict]]:
    """
    Detect NVIDIA GPU presence and capabilities with WSL support.

    Returns:
        Tuple of (has_nvidia_gpu, gpu_info_dict)
    """
    gpu_info = {
        "detected": False,
        "method": None,
        "device_name": None,
        "issues": []
    }

    # Method 1: PyTorch CUDA detection
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info.update({
                "detected": True,
                "method": "pytorch",
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown",
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.device_count() > 0 else 0,
                "cuda_version": torch.version.cuda,
                "compute_capability": torch.cuda.get_device_capability(0) if torch.cuda.device_count() > 0 else (0, 0)
            })
            logger.info(f"🎮 NVIDIA GPU Detected (PyTorch): {gpu_info['device_name']} ({gpu_info['memory_total_gb']:.1f}GB)")
            return True, gpu_info
        else:
            # PyTorch available but no CUDA
            gpu_info["issues"].append("PyTorch installed without CUDA support")
            logger.debug("PyTorch is available but CUDA is not enabled")
    except ImportError:
        gpu_info["issues"].append("PyTorch not installed")
        logger.debug("PyTorch not available for GPU detection")
    except Exception as e:
        gpu_info["issues"].append(f"PyTorch error: {str(e)}")
        logger.debug(f"PyTorch GPU detection failed: {e}")

    # Method 2: nvidia-smi (works on Windows host, may fail in WSL)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            stderr=subprocess.PIPE
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            gpu_name = parts[0].strip()
            memory_str = parts[1].strip() if len(parts) > 1 else "Unknown"

            gpu_info.update({
                "detected": True,
                "method": "nvidia-smi",
                "device_name": gpu_name,
                "memory_info": memory_str
            })
            logger.info(f"🎮 NVIDIA GPU Detected (nvidia-smi): {gpu_name}")
            return True, gpu_info
        else:
            if "GPU access blocked" in result.stderr or "Failed to initialize NVML" in result.stderr:
                gpu_info["issues"].append("NVIDIA GPU present but access blocked (WSL/permissions issue)")
            else:
                gpu_info["issues"].append("nvidia-smi failed to detect GPU")
    except FileNotFoundError:
        gpu_info["issues"].append("nvidia-smi not found (NVIDIA drivers not installed)")
    except subprocess.TimeoutExpired:
        gpu_info["issues"].append("nvidia-smi timed out")
    except Exception as e:
        gpu_info["issues"].append(f"nvidia-smi error: {str(e)}")

    # Method 3: Check for CUDA toolkit installation (indicates GPU likely present)
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda/bin/nvcc",
        Path.home() / ".local/cuda",
    ]

    cuda_found = any(Path(p).exists() for p in cuda_paths)
    if cuda_found:
        gpu_info["issues"].append("CUDA toolkit found but GPU not accessible - reinstall PyTorch with CUDA support")

    logger.debug(f"GPU detection complete: {gpu_info}")
    return False, gpu_info


# Approximate VRAM requirements for models (in GB)
MODEL_VRAM_REQUIREMENTS = {
    # Large language models
    "llama3.3:70b": 40.0,
    "qwen2.5:72b": 42.0,
    "mistral-small3.2": 20.0,
    "llama3.2:11b": 8.0,
    "mistral:latest": 4.0,
    "llama3.2:3b": 2.0,

    # Vision models
    "llava:34b": 20.0,
    "llava:13b": 8.0,
    "llava:7b": 4.5,

    # Embedding models (lightweight)
    "nomic-embed-text": 0.7,
    "BAAI/bge-base-en-v1.5": 0.5,  # Standard BGE model
    "nvidia/NV-Embed-v2": 1.2,  # Latest NVIDIA Nemotron embedding model (optimized for NVIDIA GPUs)

    # Qwen3-VL Multimodal Embedding models
    "Qwen/Qwen3-VL-Embedding-2B": 5.0,  # 2048 dimensions
    "Qwen/Qwen3-VL-Embedding-8B": 16.0,  # 4096 dimensions

    # Qwen3-VL Reranker models
    "Qwen/Qwen3-VL-Reranker-2B": 5.0,
    "Qwen/Qwen3-VL-Reranker-8B": 16.0,

    # Qwen3 Text Embedding models
    "Qwen/Qwen3-Embedding-0.6B": 1.5,
    "Qwen/Qwen3-Embedding-4B": 8.0,
    "Qwen/Qwen3-Embedding-8B": 16.0,
}

# Model capability tiers
MODEL_TIERS = {
    "efficient": {
        "text_model": "mistral:latest",
        "vision_model": "llava:7b",
        "memory_requirement": 6.0,  # Total for both models
        "description": "Efficient models suitable for systems with 16-32GB RAM"
    },
    "powerful": {
        "text_model": "mistral-small3.2",
        "vision_model": "llava:7b",
        "memory_requirement": 20.0,  # Total for both models
        "description": "High-performance models requiring 32GB+ RAM and preferably dedicated GPU"
    }
}

class SmartModelSelector:
    def __init__(self):
        self.system_memory_gb = self._get_system_memory_gb()
        self.available_memory_gb = self._get_available_memory_gb()
        self.is_docker_environment = self._detect_docker_environment()
        self.docker_memory_limit = self._get_docker_memory_limit() if self.is_docker_environment else None

        logger.info(f"System Memory: {self.system_memory_gb:.1f}GB, Available: {self.available_memory_gb:.1f}GB")
        if self.is_docker_environment:
            logger.info(f"Docker Environment Detected - Memory Limit: {self.docker_memory_limit:.1f}GB" if self.docker_memory_limit else "Docker Environment Detected - No memory limit")

    def _get_system_memory_gb(self) -> float:
        """Get total system memory in GB"""
        return psutil.virtual_memory().total / (1024**3)

    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / (1024**3)

    def _detect_docker_environment(self) -> bool:
        """Detect if running inside Docker container"""
        try:
            # Check for Docker-specific files
            docker_indicators = [
                Path("/.dockerenv").exists(),
                Path("/proc/self/cgroup").exists() and "docker" in Path("/proc/self/cgroup").read_text(),
                os.environ.get("DOCKER_CONTAINER") is not None
            ]
            return any(docker_indicators)
        except Exception:
            return False

    def _get_docker_memory_limit(self) -> Optional[float]:
        """Get Docker container memory limit in GB"""
        try:
            # Try to get memory limit from cgroup
            cgroup_paths = [
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "/sys/fs/cgroup/memory.max"
            ]

            for path in cgroup_paths:
                if Path(path).exists():
                    limit_bytes = int(Path(path).read_text().strip())
                    # If limit is very large (> 100GB), assume no limit
                    if limit_bytes > 100 * 1024**3:
                        return None
                    return limit_bytes / (1024**3)

            # Fallback: use docker stats if available
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemLimit}}", os.environ.get("HOSTNAME", "")],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                limit_str = result.stdout.strip()
                if "GiB" in limit_str:
                    return float(limit_str.replace("GiB", ""))
                elif "GB" in limit_str:
                    return float(limit_str.replace("GB", ""))

        except Exception as e:
            logger.debug(f"Could not determine Docker memory limit: {e}")

        return None

    def select_model_tier(self) -> str:
        """
        Select model tier based on available system memory.

        Returns:
            "efficient" or "powerful"
        """
        # Use Docker limit if available, otherwise use system memory
        effective_memory = self.docker_memory_limit if self.docker_memory_limit else self.system_memory_gb

        if effective_memory >= MODEL_TIERS["powerful"]["memory_requirement"]:
            logger.info(f"💪 Selected 'powerful' model tier ({effective_memory:.1f}GB available)")
            return "powerful"
        else:
            logger.info(f"⚡ Selected 'efficient' model tier ({effective_memory:.1f}GB available)")
            return "efficient"

    def can_run_model(self, model_name: str) -> bool:
        """
        Check if a specific model can run in current environment.

        Args:
            model_name: Name of the model (e.g., "llama3.3:70b")

        Returns:
            True if model can run, False otherwise
        """
        if model_name not in MODEL_VRAM_REQUIREMENTS:
            logger.warning(f"Unknown model {model_name}, cannot determine requirements")
            return False

        required_memory = MODEL_VRAM_REQUIREMENTS[model_name]
        effective_memory = self.docker_memory_limit if self.docker_memory_limit else self.available_memory_gb

        can_run = effective_memory >= required_memory
        if can_run:
            logger.debug(f"✅ Model {model_name} can run ({required_memory:.1f}GB required, {effective_memory:.1f}GB available)")
        else:
            logger.debug(f"❌ Model {model_name} cannot run ({required_memory:.1f}GB required, {effective_memory:.1f}GB available)")

        return can_run

    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system resource summary"""
        has_nvidia, gpu_info = detect_nvidia_gpu()

        return {
            "system_memory_gb": self.system_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "is_docker": self.is_docker_environment,
            "docker_memory_limit_gb": self.docker_memory_limit,
            "selected_tier": self.select_model_tier(),
            "has_nvidia_gpu": has_nvidia,
            "gpu_info": gpu_info
        }


# Global selector instance (lazy-loaded)
_selector = None

def get_smart_selector() -> SmartModelSelector:
    """Get or create global SmartModelSelector instance"""
    global _selector
    if _selector is None:
        _selector = SmartModelSelector()
    return _selector


def get_system_summary() -> Dict[str, Any]:
    """Get comprehensive system resource summary"""
    return get_smart_selector().get_system_summary()


def get_recommended_text_model() -> str:
    """
    Get recommended local text model based on effective memory tier.

    Returns:
        Ollama model name suited to current environment.
    """
    selector = get_smart_selector()
    tier = selector.select_model_tier()
    return MODEL_TIERS.get(tier, MODEL_TIERS["efficient"])["text_model"]


def get_recommended_vision_model() -> str:
    """
    Get recommended local vision model based on effective memory tier.

    Returns:
        Ollama vision model name suited to current environment.
    """
    selector = get_smart_selector()
    tier = selector.select_model_tier()
    return MODEL_TIERS.get(tier, MODEL_TIERS["efficient"])["vision_model"]


def get_optimal_embedding_model() -> str:
    """
    Get the optimal embedding model based on available hardware.

    Returns NVIDIA Nemotron models when NVIDIA GPUs are detected,
    otherwise returns standard BGE model.

    Returns:
        Model identifier string for sentence-transformers
    """
    has_nvidia, gpu_info = detect_nvidia_gpu()

    if has_nvidia:
        logger.info("🚀 NVIDIA GPU detected - using latest Nemotron NV-Embed-v2 model for superior embedding quality")
        return "nvidia/NV-Embed-v2"  # Latest NVIDIA Nemotron embedding model
    else:
        logger.info("💻 No NVIDIA GPU detected - using standard BGE embedding model")
        return "BAAI/bge-base-en-v1.5"  # Standard high-quality embedding model


def get_optimal_qwen3_vl_config() -> Dict[str, Any]:
    """
    Get optimal Qwen3-VL model configuration.

    PRIORITY ORDER:
    1. QWEN3_VL_MODEL_SIZE env/config (which checks database dimensions first)
    2. VRAM-based auto-selection (for new databases)

    Returns configuration for both embedding and reranker models that will
    fit comfortably in available VRAM.

    Returns:
        Dict with optimal model configuration:
        {
            "embedding_model": str,  # Model name
            "embedding_dim": int,    # Full embedding dimension
            "reranker_model": str,   # Model name
            "can_run_both": bool,    # Whether both can run simultaneously
            "recommended_mrl_dim": int or None,  # Suggested MRL dimension for storage efficiency
            "notes": str
        }
    """
    # Check if model size is already determined (e.g., from database dimensions)
    # Import here to avoid circular import
    import os
    forced_size = os.getenv("QWEN3_VL_MODEL_SIZE", "auto").upper()

    # Also check if config module already computed the size based on database
    try:
        from cortex_engine.config import QWEN3_VL_MODEL_SIZE
        if QWEN3_VL_MODEL_SIZE in ("2B", "8B"):
            forced_size = QWEN3_VL_MODEL_SIZE
    except ImportError:
        pass

    has_nvidia, gpu_info = detect_nvidia_gpu()

    result = {
        "embedding_model": None,
        "embedding_dim": None,
        "reranker_model": None,
        "can_run_both": False,
        "recommended_mrl_dim": None,
        "notes": ""
    }

    if not has_nvidia:
        result["notes"] = "No NVIDIA GPU detected - Qwen3-VL requires CUDA GPU"
        logger.warning("⚠️ Qwen3-VL requires NVIDIA GPU with CUDA support")
        return result

    # Get available VRAM
    available_gb = gpu_info.get("memory_total_gb", 0)
    if available_gb == 0:
        # Try to parse from memory_info string
        memory_info = gpu_info.get("memory_info", "")
        if "MiB" in memory_info:
            try:
                available_gb = float(memory_info.replace("MiB", "").strip()) / 1024
            except ValueError:
                available_gb = 0

    logger.info(f"🎮 GPU: {gpu_info.get('device_name', 'Unknown')} with {available_gb:.1f}GB VRAM")

    # Reserve ~20% for other operations (LLM, etc.)
    usable_gb = available_gb * 0.8

    # PRIORITY: If model size is explicitly set (e.g., from database dimensions), use that
    if forced_size == "2B":
        logger.info("📊 Using Qwen3-VL-2B (forced by QWEN3_VL_MODEL_SIZE or database compatibility)")
        result.update({
            "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
            "embedding_dim": 2048,
            "reranker_model": "Qwen/Qwen3-VL-Reranker-2B",
            "can_run_both": True,
            "recommended_mrl_dim": None,
            "notes": f"2B model forced (database compatibility or explicit setting)"
        })
        return result
    elif forced_size == "8B":
        logger.info("📊 Using Qwen3-VL-8B (forced by QWEN3_VL_MODEL_SIZE or database compatibility)")
        result.update({
            "embedding_model": "Qwen/Qwen3-VL-Embedding-8B",
            "embedding_dim": 4096,
            "reranker_model": "Qwen/Qwen3-VL-Reranker-8B" if usable_gb >= 40 else "Qwen/Qwen3-VL-Reranker-2B",
            "can_run_both": usable_gb >= 24,
            "recommended_mrl_dim": None if usable_gb >= 24 else 2048,
            "notes": f"8B model forced (database compatibility or explicit setting)"
        })
        return result

    # AUTO mode: Configuration matrix based on available memory
    if usable_gb >= 40:
        # RTX 8000 / A100 class - run both 8B models with room to spare
        result.update({
            "embedding_model": "Qwen/Qwen3-VL-Embedding-8B",
            "embedding_dim": 4096,
            "reranker_model": "Qwen/Qwen3-VL-Reranker-8B",
            "can_run_both": True,
            "recommended_mrl_dim": None,  # Use full dimensions
            "notes": f"Premium config: 8B embedding + 8B reranker ({usable_gb:.0f}GB available)"
        })
        logger.info("🚀 Premium Qwen3-VL config: 8B embedding + 8B reranker")

    elif usable_gb >= 24:
        # RTX 4090 / A6000 class - run 8B embedding, 2B reranker
        result.update({
            "embedding_model": "Qwen/Qwen3-VL-Embedding-8B",
            "embedding_dim": 4096,
            "reranker_model": "Qwen/Qwen3-VL-Reranker-2B",
            "can_run_both": True,
            "recommended_mrl_dim": 2048,  # Reduce for efficiency
            "notes": f"High config: 8B embedding + 2B reranker ({usable_gb:.0f}GB available)"
        })
        logger.info("📦 High Qwen3-VL config: 8B embedding + 2B reranker")

    elif usable_gb >= 16:
        # RTX 3090/4080 class - 8B embedding OR 2B both, load reranker on-demand
        result.update({
            "embedding_model": "Qwen/Qwen3-VL-Embedding-8B",
            "embedding_dim": 4096,
            "reranker_model": "Qwen/Qwen3-VL-Reranker-2B",
            "can_run_both": False,  # Load reranker on-demand
            "recommended_mrl_dim": 1024,  # Reduce for storage
            "notes": f"Standard config: 8B embedding, 2B reranker on-demand ({usable_gb:.0f}GB available)"
        })
        logger.info("📦 Standard Qwen3-VL config: 8B embedding, reranker on-demand")

    elif usable_gb >= 10:
        # RTX 3080/4070 class - 2B models
        result.update({
            "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
            "embedding_dim": 2048,
            "reranker_model": "Qwen/Qwen3-VL-Reranker-2B",
            "can_run_both": True,
            "recommended_mrl_dim": None,  # 2B is already efficient
            "notes": f"Efficient config: 2B embedding + 2B reranker ({usable_gb:.0f}GB available)"
        })
        logger.info("💡 Efficient Qwen3-VL config: 2B models")

    elif usable_gb >= 6:
        # RTX 3060/4060 class - 2B embedding only
        result.update({
            "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
            "embedding_dim": 2048,
            "reranker_model": None,
            "can_run_both": False,
            "recommended_mrl_dim": 512,  # Aggressive reduction
            "notes": f"Minimal config: 2B embedding only ({usable_gb:.0f}GB available)"
        })
        logger.info("⚡ Minimal Qwen3-VL config: 2B embedding only")

    else:
        result["notes"] = f"Insufficient VRAM ({usable_gb:.1f}GB) for Qwen3-VL models"
        logger.warning(f"⚠️ Qwen3-VL requires at least 6GB VRAM, have {usable_gb:.1f}GB")

    return result


def get_optimal_qwen3_vl_embedding_model() -> str:
    """
    Get the optimal Qwen3-VL embedding model for current hardware.

    Returns:
        Model name string, or None if hardware is insufficient
    """
    config = get_optimal_qwen3_vl_config()
    return config.get("embedding_model")


def get_optimal_qwen3_vl_reranker_model() -> str:
    """
    Get the optimal Qwen3-VL reranker model for current hardware.

    Returns:
        Model name string, or None if hardware is insufficient
    """
    config = get_optimal_qwen3_vl_config()
    return config.get("reranker_model")


def get_adaptive_embedding_strategy() -> Dict[str, Any]:
    """
    UNIFIED ADAPTIVE EMBEDDING SELECTION
    ======================================

    Automatically determines the best embedding approach for the current hardware:
    - Qwen3-VL (multimodal) for GPUs with 6GB+ VRAM
    - NV-Embed-v2 (text-only, GPU-optimized) for NVIDIA GPUs with <6GB VRAM
    - BGE-base (CPU-friendly) for systems without GPU

    This is the SINGLE SOURCE OF TRUTH for embedding selection.
    Environment variables can override for manual control.

    Returns:
        Dict with embedding strategy:
        {
            "approach": str,  # "qwen3vl", "nv-embed", or "bge"
            "model": str,  # Model identifier
            "dimensions": int,  # Embedding dimensions
            "multimodal": bool,  # Supports images/video
            "reranker": str or None,  # Reranker model if available
            "vram_required_gb": float,  # VRAM requirement
            "reason": str,  # Why this was selected
            "config": dict  # Additional configuration
        }
    """
    # Check for manual override
    override_model = os.getenv("CORTEX_EMBED_MODEL")
    if override_model:
        logger.info(f"🔧 Manual embedding model override: {override_model}")
        return {
            "approach": "manual",
            "model": override_model,
            "dimensions": 768 if "bge" in override_model.lower() else 2048,
            "multimodal": False,
            "reranker": None,
            "vram_required_gb": 0.5,
            "reason": "Manual override via CORTEX_EMBED_MODEL environment variable",
            "config": {}
        }

    # Detect hardware
    has_nvidia, gpu_info = detect_nvidia_gpu()

    if not has_nvidia:
        # No GPU: Use CPU-friendly BGE
        logger.info("💻 No NVIDIA GPU detected - using CPU-friendly BGE embedding")
        return {
            "approach": "bge",
            "model": "BAAI/bge-base-en-v1.5",
            "dimensions": 768,
            "multimodal": False,
            "reranker": None,
            "vram_required_gb": 0,
            "reason": "No NVIDIA GPU detected, using CPU-optimized model",
            "config": {}
        }

    # Get available VRAM
    vram_gb = gpu_info.get("memory_total_gb", 0)
    gpu_name = gpu_info.get("device_name", "Unknown")

    # Check if Qwen3-VL dependencies are available
    qwen3vl_available = False
    try:
        import qwen_vl_utils
        qwen3vl_available = True
    except ImportError:
        logger.debug("qwen-vl-utils not installed, Qwen3-VL unavailable")

    # Decision logic based on VRAM and dependencies
    if vram_gb >= 6 and qwen3vl_available:
        # Qwen3-VL for multimodal (6GB+ VRAM)
        qwen_config = get_optimal_qwen3_vl_config()
        logger.info(f"🎨 {gpu_name} ({vram_gb:.1f}GB) - using Qwen3-VL multimodal embedding")
        return {
            "approach": "qwen3vl",
            "model": qwen_config["embedding_model"],
            "dimensions": qwen_config["embedding_dim"],
            "multimodal": True,
            "reranker": qwen_config["reranker_model"],
            "vram_required_gb": 5.0 if "2B" in qwen_config["embedding_model"] else 16.0,
            "reason": f"NVIDIA GPU with {vram_gb:.1f}GB VRAM, Qwen3-VL provides multimodal capabilities",
            "config": qwen_config
        }

    elif vram_gb >= 2:
        # NV-Embed-v2 for text-only (2GB+ VRAM)
        logger.info(f"🚀 {gpu_name} ({vram_gb:.1f}GB) - using NV-Embed-v2 GPU-optimized embedding")
        return {
            "approach": "nv-embed",
            "model": "nvidia/NV-Embed-v2",
            "dimensions": 4096,
            "multimodal": False,
            "reranker": None,
            "vram_required_gb": 1.2,
            "reason": f"NVIDIA GPU with {vram_gb:.1f}GB VRAM, NV-Embed-v2 provides superior text embedding",
            "config": {}
        }

    else:
        # Fallback to BGE (low VRAM or no Qwen3-VL deps)
        logger.info(f"💡 {gpu_name} ({vram_gb:.1f}GB) - using BGE-base embedding (fallback)")
        return {
            "approach": "bge",
            "model": "BAAI/bge-base-en-v1.5",
            "dimensions": 768,
            "multimodal": False,
            "reranker": None,
            "vram_required_gb": 0.5,
            "reason": f"NVIDIA GPU but insufficient VRAM ({vram_gb:.1f}GB) or Qwen3-VL not installed",
            "config": {}
        }


def should_use_qwen3vl() -> bool:
    """
    Determine if Qwen3-VL should be used based on adaptive selection.

    Returns:
        True if Qwen3-VL should be used, False otherwise
    """
    strategy = get_adaptive_embedding_strategy()
    return strategy["approach"] == "qwen3vl"
