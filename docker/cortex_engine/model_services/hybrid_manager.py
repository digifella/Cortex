"""
Hybrid Model Manager
Orchestrates multiple model service backends with intelligent selection.
"""

import asyncio
from typing import List, Optional, Dict, Any, AsyncIterator, Union
from enum import Enum
from pathlib import Path

from .interfaces import ModelServiceInterface, ModelInfo, ModelStatus, ModelDownloadProgress
from .docker_model_service import DockerModelService
from .ollama_model_service import OllamaModelService
from .model_registry import ModelRegistry, ModelRegistryEntry, ModelCapability
from ..utils.logging_utils import get_logger
from ..config import get_cortex_config

logger = get_logger(__name__)


class DistributionStrategy(Enum):
    """Model distribution strategy options."""
    DOCKER_ONLY = "docker_only"
    OLLAMA_ONLY = "ollama_only"
    HYBRID_DOCKER_PREFERRED = "hybrid_docker_preferred"
    HYBRID_OLLAMA_PREFERRED = "hybrid_ollama_preferred"
    AUTO_OPTIMAL = "auto_optimal"


class HybridModelManager:
    """
    Manages multiple model service backends with intelligent selection.
    
    Features:
    - Automatic backend selection based on model availability and preferences
    - Fallback mechanisms for reliability
    - Performance-optimized distribution choices
    - Environment-aware configuration
    """
    
    def __init__(self, 
                 strategy: Union[DistributionStrategy, str] = DistributionStrategy.AUTO_OPTIMAL,
                 ollama_host: str = "localhost",
                 ollama_port: int = 11434):
        
        # Parse strategy if string
        if isinstance(strategy, str):
            try:
                strategy = DistributionStrategy(strategy)
            except ValueError:
                logger.warning(f"Unknown strategy '{strategy}', using AUTO_OPTIMAL")
                strategy = DistributionStrategy.AUTO_OPTIMAL
        
        self.strategy = strategy
        self.registry = ModelRegistry()
        
        # Initialize backends
        self.docker_service = DockerModelService()
        self.ollama_service = OllamaModelService(ollama_host, ollama_port)
        
        # Cache for backend availability
        self._backend_availability = {}
        self._last_availability_check = 0
        self._availability_check_interval = 60  # seconds
    
    async def close(self):
        """Clean up resources."""
        await self.ollama_service.close()
    
    async def _check_backend_availability(self, force: bool = False) -> Dict[str, bool]:
        """Check which backends are available."""
        current_time = asyncio.get_event_loop().time()
        
        if (not force and 
            current_time - self._last_availability_check < self._availability_check_interval and
            self._backend_availability):
            return self._backend_availability
        
        # Check both backends concurrently
        docker_available, ollama_available = await asyncio.gather(
            self.docker_service.is_available(),
            self.ollama_service.is_available(),
            return_exceptions=True
        )
        
        self._backend_availability = {
            "docker_model_runner": docker_available if not isinstance(docker_available, Exception) else False,
            "ollama": ollama_available if not isinstance(ollama_available, Exception) else False
        }
        
        self._last_availability_check = current_time
        return self._backend_availability
    
    def _get_optimal_backend(self, model_name: str, environment: str = "production") -> str:
        """Determine the optimal backend for a model based on strategy and environment."""
        registry_entry = self.registry.get_model_entry(model_name)
        preferred_backend = registry_entry.preferred_backend if registry_entry else None
        
        # Strategy-based selection
        if self.strategy == DistributionStrategy.DOCKER_ONLY:
            return "docker_model_runner"
        elif self.strategy == DistributionStrategy.OLLAMA_ONLY:
            return "ollama"
        elif self.strategy == DistributionStrategy.HYBRID_DOCKER_PREFERRED:
            return preferred_backend or "docker_model_runner"
        elif self.strategy == DistributionStrategy.HYBRID_OLLAMA_PREFERRED:
            return preferred_backend or "ollama"
        elif self.strategy == DistributionStrategy.AUTO_OPTIMAL:
            # Auto-optimal: consider environment and model characteristics
            if environment in ["production", "enterprise", "staging"]:
                # Prefer Docker Model Runner for production environments
                return preferred_backend or "docker_model_runner"
            else:
                # Prefer Ollama for development (easier setup)
                return preferred_backend or "ollama"
        
        # Default fallback
        return "ollama"
    
    async def _get_backend_service(self, backend_name: str) -> Optional[ModelServiceInterface]:
        """Get the service instance for a backend."""
        if backend_name == "docker_model_runner":
            return self.docker_service
        elif backend_name == "ollama":
            return self.ollama_service
        else:
            logger.error(f"Unknown backend: {backend_name}")
            return None
    
    async def get_available_backends(self) -> List[str]:
        """Get list of currently available backends."""
        availability = await self._check_backend_availability()
        return [backend for backend, available in availability.items() if available]
    
    async def get_optimal_service_for_model(self, model_name: str) -> Optional[ModelServiceInterface]:
        """Get the optimal service for a specific model."""
        environment = get_cortex_config().get("environment", "production")
        optimal_backend = self._get_optimal_backend(model_name, environment)
        
        # Check if optimal backend is available
        availability = await self._check_backend_availability()
        
        if availability.get(optimal_backend, False):
            return await self._get_backend_service(optimal_backend)
        
        # Fallback to any available backend
        for backend in availability:
            if availability[backend]:
                service = await self._get_backend_service(backend)
                if service and await service.is_model_available(model_name):
                    logger.info(f"Using fallback backend {backend} for model {model_name}")
                    return service
        
        logger.warning(f"No available backend found for model {model_name}")
        return None
    
    async def list_all_available_models(self) -> List[ModelInfo]:
        """List all models available across all backends."""
        models = []
        seen_models = set()
        
        # Get models from all available backends
        for backend_name in await self.get_available_backends():
            service = await self._get_backend_service(backend_name)
            if service:
                try:
                    backend_models = await service.list_available_models()
                    for model in backend_models:
                        # Avoid duplicates (same model available in multiple backends)
                        model_key = (model.name, model.tag)
                        if model_key not in seen_models:
                            models.append(model)
                            seen_models.add(model_key)
                except Exception as e:
                    logger.error(f"Failed to list models from {backend_name}: {e}")
        
        return models
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        service = await self.get_optimal_service_for_model(model_name)
        if service:
            return await service.get_model_info(model_name)
        
        # Try all available backends as fallback
        for backend_name in await self.get_available_backends():
            service = await self._get_backend_service(backend_name)
            if service:
                model_info = await service.get_model_info(model_name)
                if model_info:
                    return model_info
        
        return None
    
    async def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available in any backend."""
        service = await self.get_optimal_service_for_model(model_name)
        if service:
            return await service.is_model_available(model_name)
        
        # Check all backends as fallback
        for backend_name in await self.get_available_backends():
            service = await self._get_backend_service(backend_name)
            if service and await service.is_model_available(model_name):
                return True
        
        return False
    
    async def pull_model(self, 
                        model_name: str, 
                        preferred_backend: Optional[str] = None) -> AsyncIterator[ModelDownloadProgress]:
        """Pull a model using the optimal backend."""
        
        # Use specified backend or determine optimal one
        if preferred_backend:
            service = await self._get_backend_service(preferred_backend)
            if not service or not await service.is_available():
                yield ModelDownloadProgress(
                    model_name=model_name,
                    total_size=0,
                    downloaded_size=0,
                    status=f"Preferred backend {preferred_backend} is not available"
                )
                return
        else:
            service = await self.get_optimal_service_for_model(model_name)
            if not service:
                # Try to determine a good backend for downloading
                environment = get_cortex_config().get("environment", "production")
                backend_name = self._get_optimal_backend(model_name, environment)
                service = await self._get_backend_service(backend_name)
        
        if not service:
            yield ModelDownloadProgress(
                model_name=model_name,
                total_size=0,
                downloaded_size=0,
                status="No available backend for model download"
            )
            return
        
        # Stream progress from the service
        async for progress in service.pull_model(model_name):
            yield progress
    
    async def remove_model(self, model_name: str) -> bool:
        """Remove a model from all backends where it exists."""
        removed_any = False
        
        for backend_name in await self.get_available_backends():
            service = await self._get_backend_service(backend_name)
            if service and await service.is_model_available(model_name):
                try:
                    if await service.remove_model(model_name):
                        logger.info(f"Removed model {model_name} from {backend_name}")
                        removed_any = True
                except Exception as e:
                    logger.error(f"Failed to remove model {model_name} from {backend_name}: {e}")
        
        return removed_any
    
    async def get_model_endpoint(self, model_name: str) -> Optional[str]:
        """Get the API endpoint for a model."""
        service = await self.get_optimal_service_for_model(model_name)
        if service:
            return await service.get_model_endpoint(model_name)
        return None
    
    async def test_model_inference(self, model_name: str, test_prompt: str = "Hello") -> bool:
        """Test if a model can perform inference."""
        service = await self.get_optimal_service_for_model(model_name)
        if service:
            return await service.test_model_inference(model_name, test_prompt)
        return False
    
    async def get_performance_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a model."""
        service = await self.get_optimal_service_for_model(model_name)
        if service:
            metrics = await service.get_performance_metrics(model_name)
            metrics["hybrid_manager"] = {
                "strategy": self.strategy.value,
                "backend_availability": await self._check_backend_availability()
            }
            return metrics
        return {}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status for all backends."""
        availability = await self._check_backend_availability(force=True)
        
        status = {
            "strategy": self.strategy.value,
            "backends": {},
            "total_models": 0,
            "available_backends": []
        }
        
        for backend_name, available in availability.items():
            backend_info = {
                "available": available,
                "models": []
            }
            
            if available:
                service = await self._get_backend_service(backend_name)
                if service:
                    try:
                        models = await service.list_available_models()
                        backend_info["models"] = [m.full_name for m in models]
                        backend_info["model_count"] = len(models)
                        status["total_models"] += len(models)
                        status["available_backends"].append(backend_name)
                    except Exception as e:
                        backend_info["error"] = str(e)
            
            status["backends"][backend_name] = backend_info
        
        return status
    
    def get_model_recommendations(self, 
                                task_type: str,
                                performance_requirements: Optional[str] = None) -> List[ModelRegistryEntry]:
        """Get model recommendations based on task requirements."""
        return self.registry.get_model_recommendations(
            task_type=task_type,
            performance_tier=performance_requirements
        )