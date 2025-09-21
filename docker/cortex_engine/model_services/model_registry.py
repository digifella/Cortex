"""
Model Registry
Maintains information about available models across different backends.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import json
from pathlib import Path

from .interfaces import ModelInfo, ModelStatus
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelCapability(Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    IMAGE_ANALYSIS = "image_analysis"
    CODE_GENERATION = "code_generation"
    EMBEDDINGS = "embeddings"
    CHAT = "chat"
    INSTRUCT = "instruct"


@dataclass
class ModelRegistryEntry:
    """Registry entry for a model with metadata."""
    name: str
    aliases: List[str] = field(default_factory=list)
    capabilities: Set[ModelCapability] = field(default_factory=set)
    preferred_backend: Optional[str] = None
    docker_name: Optional[str] = None
    ollama_name: Optional[str] = None
    description: str = ""
    performance_tier: str = "standard"
    size_estimate_gb: float = 0.0
    use_cases: List[str] = field(default_factory=list)
    
    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if model supports a specific capability."""
        return capability in self.capabilities


class ModelRegistry:
    """Central registry for model information and mappings."""
    
    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = registry_file or Path("model_registry.json")
        self._entries: Dict[str, ModelRegistryEntry] = {}
        self._load_default_registry()
        
        if self.registry_file.exists():
            self._load_registry()
    
    def _load_default_registry(self):
        """Load the default model registry with common models."""
        default_models = [
            ModelRegistryEntry(
                name="mistral-7b-instruct",
                aliases=["mistral", "mistral:7b-instruct-v0.3-q4_K_M"],
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.INSTRUCT
                },
                preferred_backend="docker_model_runner",
                docker_name="ai/mistral:7b-instruct-v0.3-q4_K_M",
                ollama_name="mistral:7b-instruct-v0.3-q4_K_M",
                description="Mistral 7B instruction-following model",
                performance_tier="standard",
                size_estimate_gb=4.1,
                use_cases=["general_chat", "knowledge_base_queries", "research"]
            ),
            ModelRegistryEntry(
                name="mistral-small",
                aliases=["mistral-small3.2"],
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.INSTRUCT
                },
                preferred_backend="docker_model_runner",
                docker_name="ai/mistral:small-3.2",
                ollama_name="mistral-small3.2",
                description="Mistral Small 3.2 - Enhanced model for complex tasks",
                performance_tier="premium",
                size_estimate_gb=7.2,
                use_cases=["proposal_generation", "complex_analysis", "professional_writing"]
            ),
            # LLaVA Model Family - Advanced Vision Language Models
            ModelRegistryEntry(
                name="llava:7b",
                aliases=["llava", "llava:latest"],
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.IMAGE_ANALYSIS,
                    ModelCapability.CHAT
                },
                preferred_backend="ollama",  # Ollama has excellent LLaVA support
                docker_name="ai/llava:7b",
                ollama_name="llava:7b",
                description="Large Language and Vision Assistant (7B parameters)",
                performance_tier="standard",
                size_estimate_gb=4.7,
                use_cases=["image_description", "document_analysis", "visual_qa", "chart_analysis"]
            ),
            ModelRegistryEntry(
                name="llava:13b",
                aliases=["llava:13b-chat"],
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.IMAGE_ANALYSIS,
                    ModelCapability.CHAT
                },
                preferred_backend="ollama",
                docker_name="ai/llava:13b",
                ollama_name="llava:13b",
                description="Large Language and Vision Assistant (13B parameters - higher accuracy)",
                performance_tier="premium",
                size_estimate_gb=7.8,
                use_cases=["detailed_image_analysis", "technical_diagram_analysis", "complex_visual_qa"]
            ),
            ModelRegistryEntry(
                name="moondream",
                aliases=["moondream:latest"],
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.IMAGE_ANALYSIS,
                    ModelCapability.CHAT
                },
                preferred_backend="ollama",
                docker_name="ai/moondream:latest", 
                ollama_name="moondream",
                description="Moondream - Compact Vision Language Model (fast inference)",
                performance_tier="standard",
                size_estimate_gb=1.6,
                use_cases=["fast_image_description", "basic_visual_qa", "mobile_deployment"]
            ),
            ModelRegistryEntry(
                name="codellama",
                aliases=["codellama:latest"],
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.INSTRUCT
                },
                preferred_backend="docker_model_runner",
                docker_name="ai/codellama:latest",
                ollama_name="codellama",
                description="Code Llama - Specialized for code generation",
                performance_tier="standard",
                size_estimate_gb=3.8,
                use_cases=["code_generation", "code_explanation", "programming_assistance"]
            ),
            ModelRegistryEntry(
                name="phi",
                aliases=["phi:latest"],
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CHAT,
                    ModelCapability.INSTRUCT
                },
                preferred_backend="docker_model_runner",
                docker_name="ai/phi:latest",
                ollama_name="phi",
                description="Microsoft Phi - Small but capable model",
                performance_tier="efficient",
                size_estimate_gb=1.6,
                use_cases=["quick_responses", "lightweight_chat", "resource_constrained"]
            )
        ]
        
        for entry in default_models:
            self._entries[entry.name] = entry
            # Add aliases
            for alias in entry.aliases:
                self._entries[alias] = entry
    
    def _load_registry(self):
        """Load model registry from file."""
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            
            for name, entry_data in data.items():
                # Convert capabilities back to enum set
                capabilities = {
                    ModelCapability(cap) for cap in entry_data.get("capabilities", [])
                }
                
                entry = ModelRegistryEntry(
                    name=entry_data["name"],
                    aliases=entry_data.get("aliases", []),
                    capabilities=capabilities,
                    preferred_backend=entry_data.get("preferred_backend"),
                    docker_name=entry_data.get("docker_name"),
                    ollama_name=entry_data.get("ollama_name"),
                    description=entry_data.get("description", ""),
                    performance_tier=entry_data.get("performance_tier", "standard"),
                    size_estimate_gb=entry_data.get("size_estimate_gb", 0.0),
                    use_cases=entry_data.get("use_cases", [])
                )
                
                self._entries[name] = entry
                
        except Exception as e:
            logger.warning(f"Failed to load model registry: {e}")
    
    def save_registry(self):
        """Save model registry to file."""
        try:
            # Convert to serializable format
            data = {}
            seen_entries = set()
            
            for name, entry in self._entries.items():
                if entry.name in seen_entries:
                    continue
                seen_entries.add(entry.name)
                
                data[entry.name] = {
                    "name": entry.name,
                    "aliases": entry.aliases,
                    "capabilities": [cap.value for cap in entry.capabilities],
                    "preferred_backend": entry.preferred_backend,
                    "docker_name": entry.docker_name,
                    "ollama_name": entry.ollama_name,
                    "description": entry.description,
                    "performance_tier": entry.performance_tier,
                    "size_estimate_gb": entry.size_estimate_gb,
                    "use_cases": entry.use_cases
                }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def get_model_entry(self, model_name: str) -> Optional[ModelRegistryEntry]:
        """Get registry entry for a model."""
        return self._entries.get(model_name)
    
    def get_preferred_backend(self, model_name: str) -> Optional[str]:
        """Get the preferred backend for a model."""
        entry = self.get_model_entry(model_name)
        return entry.preferred_backend if entry else None
    
    def get_docker_name(self, model_name: str) -> Optional[str]:
        """Get the Docker name for a model."""
        entry = self.get_model_entry(model_name)
        return entry.docker_name if entry else None
    
    def get_ollama_name(self, model_name: str) -> Optional[str]:
        """Get the Ollama name for a model."""
        entry = self.get_model_entry(model_name)
        return entry.ollama_name if entry else None
    
    def find_models_by_capability(self, capability: ModelCapability) -> List[ModelRegistryEntry]:
        """Find models that support a specific capability."""
        models = []
        seen_names = set()
        
        for entry in self._entries.values():
            if entry.name in seen_names:
                continue
            seen_names.add(entry.name)
            
            if entry.supports_capability(capability):
                models.append(entry)
        
        return models
    
    def find_models_by_use_case(self, use_case: str) -> List[ModelRegistryEntry]:
        """Find models suitable for a specific use case."""
        models = []
        seen_names = set()
        
        for entry in self._entries.values():
            if entry.name in seen_names:
                continue
            seen_names.add(entry.name)
            
            if use_case in entry.use_cases:
                models.append(entry)
        
        return models
    
    def get_all_models(self) -> List[ModelRegistryEntry]:
        """Get all unique models in the registry."""
        models = []
        seen_names = set()
        
        for entry in self._entries.values():
            if entry.name in seen_names:
                continue
            seen_names.add(entry.name)
            models.append(entry)
        
        return models
    
    def register_model(self, entry: ModelRegistryEntry):
        """Register a new model or update an existing one."""
        self._entries[entry.name] = entry
        
        # Register aliases
        for alias in entry.aliases:
            self._entries[alias] = entry
    
    def update_model_info(self, model_name: str, **kwargs):
        """Update information for an existing model."""
        entry = self.get_model_entry(model_name)
        if entry:
            for key, value in kwargs.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
    
    def get_model_recommendations(self, 
                                task_type: str, 
                                performance_tier: Optional[str] = None,
                                max_size_gb: Optional[float] = None) -> List[ModelRegistryEntry]:
        """Get model recommendations based on criteria."""
        models = []
        
        # Map task types to capabilities
        task_capability_map = {
            "research": ModelCapability.TEXT_GENERATION,
            "proposals": ModelCapability.INSTRUCT,
            "knowledge": ModelCapability.TEXT_GENERATION,
            "ideation": ModelCapability.CHAT,
            "image_analysis": ModelCapability.IMAGE_ANALYSIS,
            "code": ModelCapability.CODE_GENERATION
        }
        
        capability = task_capability_map.get(task_type)
        if capability:
            models = self.find_models_by_capability(capability)
        else:
            models = self.get_all_models()
        
        # Filter by performance tier
        if performance_tier:
            models = [m for m in models if m.performance_tier == performance_tier]
        
        # Filter by size
        if max_size_gb:
            models = [m for m in models if m.size_estimate_gb <= max_size_gb]
        
        # Sort by performance tier priority
        tier_priority = {"enterprise": 0, "premium": 1, "standard": 2, "efficient": 3}
        models.sort(key=lambda m: tier_priority.get(m.performance_tier, 999))
        
        return models