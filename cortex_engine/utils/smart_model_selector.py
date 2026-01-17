# Smart Model Selector - Intelligent Model Selection Based on System Resources
# Version: v2.0.0
# Date: 2025-12-24
# Purpose: Automatically select appropriate models based on available system resources
#          - NEW (v2.0.0): Added NVIDIA GPU detection and Nematron model selection

import psutil
import subprocess
import logging
import os
from typing import Dict, List, Tuple, Optional
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
        logger.info("💡 CUDA toolkit detected but GPU not accessible via PyTorch")

    # Method 4: Check Windows nvidia-smi.exe (WSL-specific detection)
    # Only check in WSL environment (not Docker, Mac, or native Linux)
    is_wsl = "microsoft" in os.uname().release.lower() and not os.path.exists("/.dockerenv")

    if is_wsl:
        try:
            # Try Windows nvidia-smi from WSL
            result = subprocess.run(
                ["/mnt/c/Windows/System32/nvidia-smi.exe", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                gpu_name = parts[0].strip()
                memory_str = parts[1].strip() if len(parts) > 1 else "Unknown"

                gpu_info.update({
                    "detected": True,
                    "method": "wsl-windows-nvidia-smi",
                    "device_name": gpu_name,
                    "memory_info": memory_str,
                    "wsl_note": "GPU detected on Windows host - install PyTorch with CUDA to use it"
                })
                logger.info(f"🎮 NVIDIA GPU Detected on Windows host (WSL): {gpu_name}")
                logger.warning("⚠️ Install PyTorch with CUDA support to enable GPU acceleration in WSL")
                return True, gpu_info
        except Exception as e:
            logger.debug(f"Windows nvidia-smi check failed: {e}")

    # No GPU detected by any method
    logger.info("💻 No NVIDIA GPU detected")
    if gpu_info["issues"]:
        logger.debug(f"GPU detection issues: {', '.join(gpu_info['issues'])}")

    return False, gpu_info

# Model resource requirements (in GB)
MODEL_MEMORY_REQUIREMENTS = {
    # Efficient models (recommended for most systems)
    "mistral:latest": 4.4,
    "llava:7b": 4.7,

    # Memory-intensive models (require 32GB+ systems)
    "mistral-small3.2": 15.0,  # Actually uses ~26GB when loaded
    "codellama": 8.0,

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

    def get_effective_memory_limit(self) -> float:
        """Get the effective memory limit for model selection"""
        if self.is_docker_environment and self.docker_memory_limit:
            return min(self.docker_memory_limit, self.system_memory_gb)
        return self.system_memory_gb

    def recommend_model_tier(self) -> str:
        """Recommend model tier based on available resources"""
        effective_memory = self.get_effective_memory_limit()
        
        # Conservative approach: leave 40% memory free for OS and other processes
        usable_memory = effective_memory * 0.6
        
        if usable_memory >= MODEL_TIERS["powerful"]["memory_requirement"]:
            if effective_memory >= 32.0:  # Only recommend powerful tier for 32GB+ systems
                return "powerful"
        
        return "efficient"

    def get_recommended_models(self) -> Dict[str, str]:
        """Get recommended models based on system resources"""
        tier = self.recommend_model_tier()
        models = MODEL_TIERS[tier]
        
        return {
            "text_model": models["text_model"],
            "vision_model": models["vision_model"],
            "tier": tier,
            "description": models["description"],
            "memory_requirement": models["memory_requirement"]
        }

    def can_run_model(self, model_name: str) -> Tuple[bool, str]:
        """Check if a specific model can run on this system"""
        if model_name not in MODEL_MEMORY_REQUIREMENTS:
            return True, f"Unknown model '{model_name}' - cannot estimate requirements"
            
        required_memory = MODEL_MEMORY_REQUIREMENTS[model_name]
        effective_memory = self.get_effective_memory_limit()
        
        # Leave 40% memory free
        usable_memory = effective_memory * 0.6
        
        if required_memory <= usable_memory:
            return True, f"Model '{model_name}' can run (requires {required_memory}GB, {usable_memory:.1f}GB available)"
        else:
            return False, f"Model '{model_name}' requires {required_memory}GB but only {usable_memory:.1f}GB available"

    def get_system_summary(self) -> Dict:
        """Get comprehensive system resource summary"""
        recommendations = self.get_recommended_models()
        
        return {
            "system_memory_gb": self.system_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "effective_memory_gb": self.get_effective_memory_limit(),
            "is_docker": self.is_docker_environment,
            "docker_memory_limit_gb": self.docker_memory_limit,
            "recommended_tier": recommendations["tier"],
            "recommended_text_model": recommendations["text_model"],
            "recommended_vision_model": recommendations["vision_model"],
            "memory_requirement_gb": recommendations["memory_requirement"],
            "description": recommendations["description"]
        }

# Global instance for easy importing
smart_selector = SmartModelSelector()

def get_smart_model_recommendations() -> Dict[str, str]:
    """Convenience function to get model recommendations"""
    return smart_selector.get_recommended_models()

def can_system_run_model(model_name: str) -> Tuple[bool, str]:
    """Convenience function to check if system can run a model"""
    return smart_selector.can_run_model(model_name)

def get_recommended_text_model() -> str:
    """Get the recommended text model for this system"""
    return smart_selector.get_recommended_models()["text_model"]

def get_system_resource_summary() -> Dict:
    """Get comprehensive system resource summary"""
    return smart_selector.get_system_summary()


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
    Get optimal Qwen3-VL model configuration based on available GPU memory.

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

    # Configuration matrix based on available memory
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