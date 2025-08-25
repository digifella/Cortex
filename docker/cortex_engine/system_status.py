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
import asyncio

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
class BackendInfo:
    name: str
    status: ServiceStatus
    model_count: int
    performance_tier: str = "standard"  # standard, premium, enterprise
    
@dataclass
class SystemHealth:
    ollama_status: ServiceStatus
    api_status: ServiceStatus
    models: List[ModelInfo]
    backends: List[BackendInfo]
    platform_info: PlatformInfo
    setup_complete: bool
    error_messages: List[str]
    last_updated: float
    hybrid_strategy: Optional[str] = None

class SystemStatusChecker:
    """Checks the status of all Cortex Suite components"""
    
    REQUIRED_MODELS = [
        ("mistral:7b-instruct-v0.3-q4_K_M", 4.4),
        ("mistral-small3.2", 15.0)
    ]
    
    # Optional Visual Processing Models (for enhanced image analysis)
    VISUAL_MODELS = [
        ("llava:7b", 4.7),          # Standard visual model
        ("llava:13b", 7.8),         # Premium visual model
        ("moondream", 1.6)          # Lightweight visual model
    ]
    
    def __init__(self, model_distribution_strategy: str = "hybrid_ollama_preferred"):
        self.ollama_url = "http://localhost:11434"
        self.api_url = "http://localhost:8000"
        self.hybrid_strategy = model_distribution_strategy
        self.hybrid_manager = None  # Will be initialized async
    
    def detect_platform_info(self) -> PlatformInfo:
        """Detect platform, architecture, and hardware acceleration capabilities"""
        # Detect if running in Docker first - more accurate detection
        docker_env = self._is_running_in_docker()
        
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
    
    def _is_running_in_docker(self) -> bool:
        """More accurate Docker detection that distinguishes WSL2 from Docker"""
        # Check for Docker-specific files first
        if os.path.exists("/.dockerenv"):
            return True
        
        # Check container-specific environment variables
        if os.environ.get("container") or os.environ.get("DOCKER_CONTAINER"):
            return True
            
        # For Linux systems, check cgroup more carefully
        if os.path.exists("/proc/1/cgroup"):
            try:
                with open("/proc/1/cgroup", "r") as f:
                    cgroup_content = f.read()
                    # Look for Docker-specific cgroup patterns
                    if "docker" in cgroup_content.lower() or "containerd" in cgroup_content.lower():
                        return True
                    # WSL2 has different cgroup patterns - don't treat as Docker
                    if "wsl" in cgroup_content.lower() or "init.scope" in cgroup_content:
                        return False
            except (IOError, PermissionError):
                pass
        
        # Check for WSL environment (definitely not Docker)
        if os.path.exists("/proc/version"):
            try:
                with open("/proc/version", "r") as f:
                    version_content = f.read().lower()
                    if "microsoft" in version_content or "wsl" in version_content:
                        return False  # WSL2, not Docker
            except (IOError, PermissionError):
                pass
        
        return False
    
    def _detect_gpu_and_optimization(self, platform_name: str, architecture: str) -> Tuple[str, str]:
        """Detect GPU type and determine optimization strategy"""
        
        # Check for NVIDIA GPU with enhanced detection
        nvidia_detected = False
        nvidia_gpu_name = None
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"], 
                                  capture_output=True, timeout=3, text=True)
            if result.returncode == 0 and result.stdout.strip():
                nvidia_detected = True
                nvidia_gpu_name = result.stdout.strip().split('\n')[0]  # Get first GPU
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Try alternative detection method
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
                if result.returncode == 0:
                    nvidia_detected = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        # Determine GPU type and optimization based on platform and detection
        if nvidia_detected:
            gpu_name = nvidia_gpu_name if nvidia_gpu_name else "NVIDIA GPU"
            return gpu_name, "CUDA Acceleration"
        
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
    
    async def _initialize_hybrid_manager(self):
        """Initialize hybrid model manager if not already done."""
        if self.hybrid_manager is None:
            try:
                from .model_services import HybridModelManager, DistributionStrategy
                self.hybrid_manager = HybridModelManager(strategy=self.hybrid_strategy)
            except ImportError:
                # Hybrid model services not available, fallback to traditional approach
                pass
        return self.hybrid_manager
    
    async def _check_backend_availability(self) -> List[BackendInfo]:
        """Check availability of all model backends."""
        backends = []
        
        # Try to get hybrid manager
        hybrid_manager = await self._initialize_hybrid_manager()
        
        if hybrid_manager:
            try:
                # Get backend availability from hybrid manager
                backend_availability = await hybrid_manager._check_backend_availability()
                
                for backend_name, is_available in backend_availability.items():
                    status = ServiceStatus.RUNNING if is_available else ServiceStatus.STOPPED
                    model_count = 0
                    
                    # Get model count for available backends
                    if is_available:
                        try:
                            if backend_name == "ollama":
                                models = await hybrid_manager.ollama_service.list_available_models()
                                model_count = len(models)
                            elif backend_name == "docker_model_runner":
                                # For Docker, check if Model Runner extension is available
                                docker_service = hybrid_manager.docker_service
                                if await docker_service._check_docker_model_runner_available():
                                    models = await docker_service.list_available_models()
                                    model_count = len(models)
                                else:
                                    # Docker is available but Model Runner extension is not
                                    model_count = 0
                                    # Update status to indicate Docker is available but without Model Runner
                                    status = ServiceStatus.RUNNING  # Docker daemon is running
                        except Exception:
                            pass
                    
                    # Determine performance tier and display name based on backend
                    if backend_name == "docker_model_runner":
                        performance_tier = "premium"
                        # Check if this is just Docker available or Docker + Model Runner
                        if status == ServiceStatus.RUNNING and model_count == 0:
                            try:
                                docker_service = hybrid_manager.docker_service
                                if not await docker_service._check_docker_model_runner_available():
                                    display_name = "Docker (Model Runner extension not available)"
                                    performance_tier = "standard"
                                else:
                                    display_name = "Docker Model Runner"
                            except:
                                display_name = "Docker Model Runner"
                        else:
                            display_name = "Docker Model Runner"
                    else:
                        performance_tier = "standard"
                        display_name = backend_name.replace("_", " ").title()
                    
                    backends.append(BackendInfo(
                        name=display_name,
                        status=status,
                        model_count=model_count,
                        performance_tier=performance_tier
                    ))
                    
            except Exception as e:
                # If hybrid manager fails, add fallback info
                backends.append(BackendInfo(
                    name="ollama",
                    status=ServiceStatus.STOPPED,
                    model_count=0,
                    performance_tier="standard"
                ))
        else:
            # Fallback: just check Ollama directly
            ollama_status = self.check_ollama_status()
            model_count = len(self.get_installed_models()) if ollama_status == ServiceStatus.RUNNING else 0
            
            backends.append(BackendInfo(
                name="ollama", 
                status=ollama_status,
                model_count=model_count,
                performance_tier="standard"
            ))
        
        return backends
    
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
        
        # Check backend availability (sync wrapper for async method)
        backends = []
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            backends = loop.run_until_complete(self._check_backend_availability())
            loop.close()
        except Exception as e:
            # Fallback to basic ollama info
            backends = [BackendInfo(
                name="ollama",
                status=ollama_status,
                model_count=len(self.get_installed_models()) if ollama_status == ServiceStatus.RUNNING else 0,
                performance_tier="standard"
            )]
        
        # Determine if setup is complete
        setup_complete = (
            ollama_status == ServiceStatus.RUNNING and
            all(model.status == ModelStatus.AVAILABLE for model in models)
        )
        
        return SystemHealth(
            ollama_status=ollama_status,
            api_status=api_status,
            models=models,
            backends=backends,
            platform_info=platform_info,
            setup_complete=setup_complete,
            error_messages=error_messages,
            last_updated=time.time(),
            hybrid_strategy=self.hybrid_strategy
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
            "hybrid_strategy": health.hybrid_strategy,
            "backends": [
                {
                    "name": backend.name,
                    "status": backend.status.value,
                    "model_count": backend.model_count,
                    "performance_tier": backend.performance_tier,
                    "available": backend.status == ServiceStatus.RUNNING
                }
                for backend in health.backends
            ],
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

def get_system_status_with_strategy(strategy: str = "hybrid_ollama_preferred") -> SystemStatusChecker:
    """Get system status checker with specific hybrid strategy."""
    return SystemStatusChecker(model_distribution_strategy=strategy)