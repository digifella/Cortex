"""
System Status Checker for Cortex Suite
Provides real-time status of AI models, services, and system health
"""

import subprocess
import json
import requests
import platform
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

class ServiceStatus(Enum):
    RUNNING = "running"
    STARTING = "starting"
    STOPPED = "stopped"
    ERROR = "error"

class ModelStatus(Enum):
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    MISSING = "missing"
    ERROR = "error"

@dataclass
class ModelInfo:
    name: str
    status: ModelStatus
    size_gb: float
    download_progress: Optional[float] = None  # 0.0 to 1.0
    error_message: Optional[str] = None

@dataclass
class PlatformInfo:
    platform_name: str  # "Windows", "macOS", "Linux" 
    architecture: str    # "x86_64", "arm64", "aarch64"
    gpu_type: str       # "NVIDIA GPU", "Apple Silicon", "Intel GPU", "CPU Only"
    optimization: str   # "CUDA Acceleration", "Metal Acceleration", "CPU Optimized"
    docker_env: bool    # True if running in Docker

@dataclass
class SystemHealth:
    ollama_status: ServiceStatus
    api_status: ServiceStatus
    models: List[ModelInfo]
    platform_info: PlatformInfo
    setup_complete: bool
    error_messages: List[str]
    last_updated: float

class SystemStatusChecker:
    """Checks the status of all Cortex Suite components"""
    
    REQUIRED_MODELS = [
        ("mistral:7b-instruct-v0.3-q4_K_M", 4.4),
        ("mistral-small3.2", 15.0)
    ]
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.api_url = "http://localhost:8000"
    
    def detect_platform_info(self) -> PlatformInfo:
        """Detect platform, architecture, and hardware acceleration capabilities"""
        # Detect if running in Docker first
        docker_env = os.path.exists("/.dockerenv") or os.path.exists("/proc/1/cgroup")
        
        # Detect basic platform info
        system = platform.system()
        machine = platform.machine().lower()
        
        # When running in Docker, detect the host OS from environment or kernel info
        if docker_env:
            # Try to detect host OS from kernel version or other indicators
            if "microsoft" in platform.release().lower():
                # Running in WSL2 on Windows (common Docker Desktop scenario)
                platform_name = "Windows"
            else:
                # Could be Linux host or Mac with Docker Desktop
                # For simplicity, we'll indicate it's containerized
                platform_name = "Container Host"
        else:
            # Running directly on host OS
            if system == "Darwin":
                platform_name = "macOS"
            elif system == "Windows":
                platform_name = "Windows"
            elif system == "Linux":
                platform_name = "Linux"
            else:
                platform_name = system
        
        # Normalize architecture
        if machine in ["x86_64", "amd64"]:
            architecture = "x86_64"
        elif machine in ["arm64", "aarch64"]:
            architecture = "ARM64"
        else:
            architecture = machine.upper()
        
        # Detect GPU and determine optimization
        gpu_type, optimization = self._detect_gpu_and_optimization(platform_name, architecture)
        
        return PlatformInfo(
            platform_name=platform_name,
            architecture=architecture,
            gpu_type=gpu_type,
            optimization=optimization,
            docker_env=docker_env
        )
    
    def _detect_gpu_and_optimization(self, platform_name: str, architecture: str) -> Tuple[str, str]:
        """Detect GPU type and determine optimization strategy"""
        
        # Check for NVIDIA GPU
        nvidia_detected = False
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
            if result.returncode == 0:
                nvidia_detected = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Determine GPU type and optimization based on platform and detection
        if nvidia_detected:
            return "NVIDIA GPU", "CUDA Acceleration"
        
        elif platform_name == "macOS":
            if architecture == "ARM64":
                return "Apple Silicon GPU", "Metal Acceleration"
            else:
                return "Intel GPU", "CPU Optimized"
        
        elif platform_name == "Windows":
            # Could add AMD/Intel GPU detection here if needed
            return "Integrated GPU", "CPU Optimized"
        
        elif platform_name == "Linux":
            # Could add AMD GPU detection here if needed
            return "Integrated GPU", "CPU Optimized"
        
        else:
            return "CPU Only", "CPU Optimized"
    
    def check_ollama_status(self) -> ServiceStatus:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return ServiceStatus.RUNNING if response.status_code == 200 else ServiceStatus.ERROR
        except requests.exceptions.ConnectionError:
            return ServiceStatus.STOPPED
        except Exception:
            return ServiceStatus.ERROR
    
    def check_api_status(self) -> ServiceStatus:
        """Check if the API server is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            return ServiceStatus.RUNNING if response.status_code == 200 else ServiceStatus.ERROR
        except requests.exceptions.ConnectionError:
            return ServiceStatus.STOPPED
        except Exception:
            return ServiceStatus.ERROR
    
    def get_installed_models(self) -> List[Dict]:
        """Get list of installed models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
        except Exception:
            pass
        return []
    
    def check_model_status(self, model_name: str) -> Tuple[ModelStatus, Optional[str]]:
        """Check the status of a specific model"""
        installed_models = self.get_installed_models()
        
        # Check if model is installed
        for model in installed_models:
            if model.get('name', '').startswith(model_name):
                return ModelStatus.AVAILABLE, None
        
        # Check if model is currently downloading
        # This is a simplified check - in a real implementation you'd monitor
        # the Ollama logs or use a more sophisticated progress tracking
        try:
            # Try to get model info - this will trigger download if not present
            response = requests.get(f"{self.ollama_url}/api/show", 
                                  json={"name": model_name}, timeout=2)
            if response.status_code == 404:
                return ModelStatus.MISSING, None
            elif response.status_code == 200:
                return ModelStatus.AVAILABLE, None
            else:
                return ModelStatus.ERROR, f"HTTP {response.status_code}"
        except requests.exceptions.ConnectionError:
            return ModelStatus.ERROR, "Cannot connect to Ollama"
        except Exception as e:
            return ModelStatus.ERROR, str(e)
    
    def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        error_messages = []
        
        # Detect platform info
        platform_info = self.detect_platform_info()
        
        # Check service statuses
        ollama_status = self.check_ollama_status()
        api_status = self.check_api_status()
        
        if ollama_status != ServiceStatus.RUNNING:
            error_messages.append("Ollama service is not running")
        
        # Check model statuses
        models = []
        for model_name, size_gb in self.REQUIRED_MODELS:
            status, error = self.check_model_status(model_name)
            models.append(ModelInfo(
                name=model_name,
                status=status,
                size_gb=size_gb,
                error_message=error
            ))
            
            if status == ModelStatus.ERROR and error:
                error_messages.append(f"Model {model_name}: {error}")
        
        # Determine if setup is complete
        setup_complete = (
            ollama_status == ServiceStatus.RUNNING and
            all(model.status == ModelStatus.AVAILABLE for model in models)
        )
        
        return SystemHealth(
            ollama_status=ollama_status,
            api_status=api_status,
            models=models,
            platform_info=platform_info,
            setup_complete=setup_complete,
            error_messages=error_messages,
            last_updated=time.time()
        )
    
    def get_setup_progress(self) -> Dict:
        """Get setup progress as a percentage and status message"""
        health = self.get_system_health()
        
        total_steps = 2 + len(self.REQUIRED_MODELS)  # Ollama + API + models
        completed_steps = 0
        
        if health.ollama_status == ServiceStatus.RUNNING:
            completed_steps += 1
        
        if health.api_status == ServiceStatus.RUNNING:
            completed_steps += 1
        
        completed_steps += sum(1 for model in health.models 
                             if model.status == ModelStatus.AVAILABLE)
        
        progress_percent = (completed_steps / total_steps) * 100
        
        # Generate status message
        if health.setup_complete:
            status_message = "✅ Setup complete! All features are available."
        elif health.ollama_status != ServiceStatus.RUNNING:
            status_message = "🤖 Starting Ollama service..."
        elif any(model.status == ModelStatus.MISSING for model in health.models):
            downloading_models = [model.name for model in health.models 
                                if model.status == ModelStatus.MISSING]
            status_message = f"⬇️ Downloading AI models: {', '.join(downloading_models)}"
        else:
            status_message = "🔄 Finalizing setup..."
        
        # Generate platform configuration message
        platform_info = health.platform_info
        if platform_info.docker_env:
            if platform_info.platform_name == "Windows":
                platform_config = f"🐳 Docker on Windows {platform_info.architecture} - {platform_info.optimization}"
            else:
                platform_config = f"🐳 Docker Container {platform_info.architecture} - {platform_info.optimization}"
        else:
            platform_config = f"💻 {platform_info.platform_name} {platform_info.architecture} - {platform_info.optimization}"
        
        return {
            "progress_percent": progress_percent,
            "status_message": status_message,
            "platform_config": platform_config,
            "setup_complete": health.setup_complete,
            "ollama_running": health.ollama_status == ServiceStatus.RUNNING,
            "api_running": health.api_status == ServiceStatus.RUNNING,
            "platform_info": {
                "platform": platform_info.platform_name,
                "architecture": platform_info.architecture,
                "gpu_type": platform_info.gpu_type,
                "optimization": platform_info.optimization,
                "docker_env": platform_info.docker_env
            },
            "models": [
                {
                    "name": model.name,
                    "status": model.status.value,
                    "size_gb": model.size_gb,
                    "available": model.status == ModelStatus.AVAILABLE
                }
                for model in health.models
            ],
            "errors": health.error_messages
        }

# Global instance for easy access
system_status = SystemStatusChecker()