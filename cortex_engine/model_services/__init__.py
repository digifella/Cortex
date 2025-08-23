"""
Model Services Package
Provides unified interfaces for different AI model distribution methods.

This package supports multiple model backends:
- Docker Model Runner (enterprise-grade OCI distribution)
- Ollama (traditional local model management)
- Future backends can be added by implementing ModelServiceInterface
"""

from .interfaces import ModelServiceInterface, ModelInfo, ModelStatus
from .hybrid_manager import HybridModelManager, DistributionStrategy
from .docker_model_service import DockerModelService
from .ollama_model_service import OllamaModelService
from .model_registry import ModelRegistry

__all__ = [
    'ModelServiceInterface',
    'ModelInfo', 
    'ModelStatus',
    'HybridModelManager',
    'DistributionStrategy',
    'DockerModelService',
    'OllamaModelService',
    'ModelRegistry'
]