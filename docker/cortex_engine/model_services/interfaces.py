"""
Model Service Interfaces
Defines common interfaces for different model distribution backends.
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, AsyncIterator
import asyncio


class ModelStatus(Enum):
    """Model availability status."""
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    NOT_AVAILABLE = "not_available"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    tag: str
    size_gb: float
    status: ModelStatus
    backend: str
    description: Optional[str] = None
    capabilities: List[str] = None
    performance_tier: str = "standard"  # standard, premium, enterprise
    
    @property
    def full_name(self) -> str:
        """Get the full model name with tag."""
        return f"{self.name}:{self.tag}" if self.tag else self.name


@dataclass
class ModelDownloadProgress:
    """Progress information for model downloads."""
    model_name: str
    total_size: int
    downloaded_size: int
    status: str
    eta_seconds: Optional[int] = None
    
    @property
    def progress_percent(self) -> float:
        """Calculate download progress percentage."""
        if self.total_size <= 0:
            return 0.0
        return min(100.0, (self.downloaded_size / self.total_size) * 100.0)


class ModelServiceInterface(ABC):
    """Abstract interface for model service backends."""
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the backend service."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the backend service is available."""
        pass
    
    @abstractmethod
    async def list_available_models(self) -> List[ModelInfo]:
        """List all models available through this backend."""
        pass
    
    @abstractmethod
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        pass
    
    @abstractmethod
    async def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is locally available."""
        pass
    
    @abstractmethod
    async def pull_model(self, model_name: str) -> AsyncIterator[ModelDownloadProgress]:
        """
        Download/pull a model.
        
        Yields progress updates during download.
        """
        pass
    
    @abstractmethod
    async def remove_model(self, model_name: str) -> bool:
        """Remove a model from local storage."""
        pass
    
    @abstractmethod
    async def get_model_endpoint(self, model_name: str) -> Optional[str]:
        """Get the API endpoint URL for a model."""
        pass
    
    @abstractmethod
    async def test_model_inference(self, model_name: str, test_prompt: str = "Hello") -> bool:
        """Test if a model can perform inference."""
        pass
    
    @abstractmethod
    async def get_performance_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a model (load time, memory usage, etc.)."""
        pass


class ModelServiceFactory:
    """Factory for creating model service instances."""
    
    _services: Dict[str, type] = {}
    
    @classmethod
    def register_service(cls, backend_name: str, service_class: type):
        """Register a model service backend."""
        cls._services[backend_name] = service_class
    
    @classmethod
    def create_service(cls, backend_name: str, **kwargs) -> ModelServiceInterface:
        """Create a model service instance."""
        if backend_name not in cls._services:
            raise ValueError(f"Unknown model service backend: {backend_name}")
        
        service_class = cls._services[backend_name]
        return service_class(**kwargs)
    
    @classmethod
    def get_available_backends(cls) -> List[str]:
        """Get list of registered backend names."""
        return list(cls._services.keys())