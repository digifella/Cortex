# Smart Model Selector - Intelligent Model Selection Based on System Resources
# Version: v1.0.0
# Date: 2025-08-30
# Purpose: Automatically select appropriate models based on available system resources

import psutil
import subprocess
import logging
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Model resource requirements (in GB)
MODEL_MEMORY_REQUIREMENTS = {
    # Efficient models (recommended for most systems)
    "mistral:7b-instruct-v0.3-q4_K_M": 4.4,
    "llava:7b": 4.7,
    
    # Memory-intensive models (require 32GB+ systems)
    "mistral-small3.2": 15.0,  # Actually uses ~26GB when loaded
    "codellama": 8.0,
    
    # Embedding models (lightweight)
    "nomic-embed-text": 0.7,
}

# Model capability tiers
MODEL_TIERS = {
    "efficient": {
        "text_model": "mistral:7b-instruct-v0.3-q4_K_M",
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