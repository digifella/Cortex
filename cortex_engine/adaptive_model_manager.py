"""
Adaptive Model Manager - Intelligent Ollama Model Discovery and Selection
Version: 1.0.0
Date: 2026-01-01

Purpose: Dynamically discover, categorize, and recommend Ollama models based on:
- Task requirements (research, ideation, synthesis, routing)
- System capabilities (GPU, memory)
- Model capabilities (size, quantization, architecture)

This enables seamless integration of new models (Nemotron, Qwen 2.5, etc.) without code changes.
"""

import asyncio
import re
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from enum import Enum

from .model_services.ollama_model_service import OllamaModelService
from .model_services.interfaces import ModelInfo
from .utils.smart_model_selector import detect_nvidia_gpu, get_smart_selector
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


class TaskType(str, Enum):
    """Task types with different model requirements."""
    ROUTER = "router"              # Fast classification/routing (3-8B models)
    RESEARCH = "research"          # Deep analysis, synthesis (32-72B models)
    IDEATION = "ideation"         # Creative idea generation (32-72B models)
    SYNTHESIS = "synthesis"       # Knowledge synthesis (32-72B models)
    ANALYSIS = "analysis"         # General analysis (14-32B models)
    CHAT = "chat"                 # Conversational (7-14B models)


class ModelTier(str, Enum):
    """Model capability tiers based on parameter count and quantization."""
    FAST = "fast"          # < 5GB - Router, classification, quick tasks
    MID = "mid"            # 5-20GB - Balanced performance
    POWER = "power"        # > 20GB - Heavy lifting, complex reasoning


@dataclass
class ModelCapabilities:
    """Comprehensive model capability information."""
    name: str                    # Model name (e.g., "llama3.3")
    tag: str                     # Model tag (e.g., "70b-instruct-q4_K_M")
    full_name: str              # Full identifier (e.g., "llama3.3:70b-instruct-q4_K_M")
    size_gb: float              # Model size in GB
    tier: ModelTier             # Capability tier
    family: str                 # Model family (llama, qwen, mistral, etc.)
    params_estimate: Optional[int]  # Estimated parameter count
    quantization: Optional[str]     # Quantization level (q4_K_M, q8_0, etc.)
    is_instruct: bool              # Whether it's an instruct-tuned model
    is_vision: bool                # Whether it supports vision
    nvidia_optimized: bool         # Whether it's NVIDIA-optimized (Nemotron)
    recommended_tasks: List[TaskType]  # Tasks this model excels at
    description: str               # Human-readable description


class AdaptiveModelManager:
    """
    Intelligent model manager that automatically discovers and categorizes
    available Ollama models, then recommends optimal models for specific tasks.
    """

    # Known model families and their characteristics
    MODEL_FAMILIES = {
        "llama": {
            "description": "Meta's Llama family - excellent general-purpose models",
            "strengths": ["reasoning", "instruction-following", "general-purpose"]
        },
        "qwen": {
            "description": "Alibaba's Qwen family - superior reasoning and coding",
            "strengths": ["reasoning", "coding", "multilingual", "structured-output"]
        },
        "mistral": {
            "description": "Mistral AI models - efficient and capable",
            "strengths": ["efficiency", "instruction-following", "speed"]
        },
        "nemotron": {
            "description": "NVIDIA Nemotron - optimized for NVIDIA GPUs",
            "strengths": ["gpu-optimization", "reasoning", "research"],
            "nvidia_optimized": True
        },
        "llava": {
            "description": "Vision-language models",
            "strengths": ["vision", "image-understanding"],
            "vision": True
        }
    }

    # Parameter count heuristics from model names
    PARAM_PATTERNS = {
        r'3b': 3,
        r'7b': 7,
        r'8b': 8,
        r'14b': 14,
        r'24b': 24,
        r'32b': 32,
        r'70b': 70,
        r'72b': 72,
    }

    # Quantization quality hierarchy (higher is better quality but larger)
    QUANT_QUALITY = {
        'q2': 1,
        'q3': 2,
        'q4_0': 3,
        'q4_K_M': 4,
        'q4_K_S': 3,
        'q5': 5,
        'q6': 6,
        'q8_0': 7,
        'fp16': 8,
        'fp32': 9
    }

    def __init__(self, host: str = "localhost", port: int = 11434):
        """
        Initialize the adaptive model manager.

        Args:
            host: Ollama service host
            port: Ollama service port
        """
        self.ollama_service = OllamaModelService(host, port)
        self._model_cache: Optional[List[ModelCapabilities]] = None
        self._has_nvidia_gpu, self._gpu_info = detect_nvidia_gpu()

        logger.info(f"AdaptiveModelManager initialized - NVIDIA GPU: {self._has_nvidia_gpu}")

    async def discover_models(self, force_refresh: bool = False) -> List[ModelCapabilities]:
        """
        Discover and categorize all available Ollama models.

        Args:
            force_refresh: Force refresh from Ollama API (ignore cache)

        Returns:
            List of models with comprehensive capability information
        """
        if self._model_cache and not force_refresh:
            return self._model_cache

        logger.info("Discovering available Ollama models...")

        raw_models = await self.ollama_service.list_available_models()

        if not raw_models:
            logger.warning("No Ollama models found - is Ollama running?")
            return []

        categorized_models = []
        for model in raw_models:
            capabilities = self._analyze_model(model)
            categorized_models.append(capabilities)

        # Cache results
        self._model_cache = categorized_models

        # Log summary
        tier_counts = {}
        for model in categorized_models:
            tier_counts[model.tier] = tier_counts.get(model.tier, 0) + 1

        logger.info(f"Discovered {len(categorized_models)} models: "
                   f"{tier_counts.get(ModelTier.FAST, 0)} fast, "
                   f"{tier_counts.get(ModelTier.MID, 0)} mid, "
                   f"{tier_counts.get(ModelTier.POWER, 0)} power")

        return categorized_models

    def _analyze_model(self, model: ModelInfo) -> ModelCapabilities:
        """
        Analyze a model and determine its capabilities.

        Args:
            model: Raw model information from Ollama

        Returns:
            Comprehensive capability information
        """
        full_name = f"{model.name}:{model.tag}"
        name_lower = full_name.lower()

        # Determine model family
        family = "unknown"
        for family_name in self.MODEL_FAMILIES.keys():
            if family_name in model.name.lower():
                family = family_name
                break

        # Extract parameter count
        params_estimate = None
        for pattern, params in self.PARAM_PATTERNS.items():
            if re.search(pattern, name_lower):
                params_estimate = params
                break

        # Extract quantization level
        quantization = None
        for quant_pattern in self.QUANT_QUALITY.keys():
            if quant_pattern in name_lower.replace('_', '').replace('.', ''):
                quantization = quant_pattern
                break

        # Determine tier based on size
        if model.size_gb < 5:
            tier = ModelTier.FAST
        elif model.size_gb < 20:
            tier = ModelTier.MID
        else:
            tier = ModelTier.POWER

        # Check for special characteristics
        is_instruct = 'instruct' in name_lower or 'chat' in name_lower
        is_vision = family == "llava" or "vision" in name_lower or "llava" in name_lower
        nvidia_optimized = family == "nemotron" or "nemotron" in name_lower

        # Recommend tasks based on capabilities
        recommended_tasks = self._recommend_tasks(
            tier=tier,
            params=params_estimate,
            is_instruct=is_instruct,
            is_vision=is_vision,
            family=family
        )

        # Generate description
        description = self._generate_description(
            family=family,
            params=params_estimate,
            tier=tier,
            quantization=quantization,
            is_vision=is_vision,
            nvidia_optimized=nvidia_optimized
        )

        return ModelCapabilities(
            name=model.name,
            tag=model.tag,
            full_name=full_name,
            size_gb=model.size_gb,
            tier=tier,
            family=family,
            params_estimate=params_estimate,
            quantization=quantization,
            is_instruct=is_instruct,
            is_vision=is_vision,
            nvidia_optimized=nvidia_optimized,
            recommended_tasks=recommended_tasks,
            description=description
        )

    def _recommend_tasks(
        self,
        tier: ModelTier,
        params: Optional[int],
        is_instruct: bool,
        is_vision: bool,
        family: str
    ) -> List[TaskType]:
        """Recommend suitable tasks for a model based on its characteristics."""
        tasks = []

        if not is_instruct:
            # Non-instruct models not recommended for any task
            return tasks

        if is_vision:
            # Vision models handled separately
            return tasks

        # Fast models (< 5GB) - routing and classification
        if tier == ModelTier.FAST or (params and params <= 8):
            tasks.append(TaskType.ROUTER)
            if params and params >= 7:
                tasks.append(TaskType.CHAT)

        # Mid-range models (5-20GB) - general purpose
        elif tier == ModelTier.MID or (params and 8 < params <= 32):
            tasks.extend([TaskType.CHAT, TaskType.ANALYSIS])
            if params and params >= 14:
                tasks.append(TaskType.SYNTHESIS)

        # Power models (> 20GB) - complex reasoning
        elif tier == ModelTier.POWER or (params and params > 32):
            tasks.extend([TaskType.RESEARCH, TaskType.IDEATION, TaskType.SYNTHESIS, TaskType.ANALYSIS])

            # Qwen models excel at research and structured reasoning
            if family == "qwen":
                tasks.insert(0, TaskType.RESEARCH)  # Prioritize research

        return tasks

    def _generate_description(
        self,
        family: str,
        params: Optional[int],
        tier: ModelTier,
        quantization: Optional[str],
        is_vision: bool,
        nvidia_optimized: bool
    ) -> str:
        """Generate human-readable model description."""
        parts = []

        # Family description
        if family in self.MODEL_FAMILIES:
            parts.append(self.MODEL_FAMILIES[family]["description"])

        # Parameter count and tier
        if params:
            parts.append(f"{params}B parameters")
        parts.append(f"{tier.value} tier")

        # Quantization
        if quantization:
            quant_quality = self.QUANT_QUALITY.get(quantization, 0)
            if quant_quality >= 7:
                parts.append("high-quality quantization")
            elif quant_quality >= 4:
                parts.append("balanced quantization")
            else:
                parts.append("efficient quantization")

        # Special features
        if nvidia_optimized:
            parts.append("NVIDIA-optimized")
        if is_vision:
            parts.append("vision-capable")

        return " - ".join(parts)

    async def recommend_model(
        self,
        task_type: TaskType,
        preference: Literal["fastest", "balanced", "best"] = "balanced"
    ) -> Optional[str]:
        """
        Recommend the best available model for a specific task.

        Args:
            task_type: The type of task to perform
            preference: Model selection preference
                - "fastest": Prioritize speed (smallest capable model)
                - "balanced": Balance speed and quality (medium model)
                - "best": Prioritize quality (largest capable model)

        Returns:
            Full model name (e.g., "llama3.3:70b-instruct-q4_K_M") or None if no suitable model
        """
        models = await self.discover_models()

        # Filter models suitable for this task
        suitable = [m for m in models if task_type in m.recommended_tasks]

        if not suitable:
            logger.warning(f"No suitable models found for task: {task_type}")
            return None

        # Boost NVIDIA-optimized models if we have NVIDIA GPU
        if self._has_nvidia_gpu:
            nvidia_models = [m for m in suitable if m.nvidia_optimized]
            if nvidia_models:
                logger.info("Prioritizing NVIDIA-optimized models for GPU acceleration")
                suitable = nvidia_models + [m for m in suitable if not m.nvidia_optimized]

        # Sort by preference
        if preference == "fastest":
            # Smallest model first (fastest inference)
            suitable.sort(key=lambda m: m.size_gb)
        elif preference == "best":
            # Largest model first (best quality)
            suitable.sort(key=lambda m: m.size_gb, reverse=True)
        else:  # balanced
            # Mid-sized models preferred, then large, then small
            suitable.sort(key=lambda m: (
                abs(m.size_gb - 15),  # Prefer ~15GB models
                -m.size_gb  # Break ties with larger models
            ))

        recommended = suitable[0]
        logger.info(f"Recommended model for {task_type} ({preference}): {recommended.full_name} "
                   f"({recommended.size_gb:.1f}GB)")

        return recommended.full_name

    async def get_model_info(self, model_name: str) -> Optional[ModelCapabilities]:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Model name or full name with tag

        Returns:
            Model capabilities or None if not found
        """
        models = await self.discover_models()

        for model in models:
            if model.full_name == model_name or model.name == model_name:
                return model

        return None

    async def categorize_models(self) -> Dict[ModelTier, List[ModelCapabilities]]:
        """
        Categorize all available models by tier.

        Returns:
            Dictionary mapping tiers to lists of models
        """
        models = await self.discover_models()

        categorized = {
            ModelTier.FAST: [],
            ModelTier.MID: [],
            ModelTier.POWER: []
        }

        for model in models:
            categorized[model.tier].append(model)

        return categorized

    async def get_task_recommendations(self) -> Dict[TaskType, List[str]]:
        """
        Get recommended models for all task types.

        Returns:
            Dictionary mapping task types to recommended model names
        """
        recommendations = {}

        for task_type in TaskType:
            models = []
            for pref in ["best", "balanced", "fastest"]:
                model = await self.recommend_model(task_type, pref)
                if model and model not in models:
                    models.append(model)

            if models:
                recommendations[task_type] = models

        return recommendations

    async def check_model_availability(
        self,
        required_models: Dict[str, TaskType]
    ) -> Tuple[bool, List[str]]:
        """
        Check if all required models are available for specific tasks.

        Args:
            required_models: Dictionary mapping model names to task types

        Returns:
            Tuple of (all_available, missing_models)
        """
        models = await self.discover_models()
        available_names = {m.full_name for m in models} | {m.name for m in models}

        missing = []
        for model_name, task_type in required_models.items():
            if model_name not in available_names:
                # Try to find alternative
                alternative = await self.recommend_model(task_type)
                if alternative:
                    logger.info(f"Model '{model_name}' not found - recommended alternative: {alternative}")
                else:
                    missing.append(model_name)

        return len(missing) == 0, missing

    def get_system_summary(self) -> Dict:
        """
        Get comprehensive system and model availability summary.

        Returns:
            Dictionary with system capabilities and available models
        """
        system_info = get_smart_selector().get_system_summary()

        summary = {
            **system_info,
            "has_nvidia_gpu": self._has_nvidia_gpu,
            "gpu_info": self._gpu_info if self._has_nvidia_gpu else None,
            "total_models_available": len(self._model_cache) if self._model_cache else 0,
        }

        if self._model_cache:
            tier_breakdown = {}
            for model in self._model_cache:
                tier_breakdown[model.tier.value] = tier_breakdown.get(model.tier.value, 0) + 1
            summary["models_by_tier"] = tier_breakdown

        return summary


# Global instance for easy importing
_global_manager: Optional[AdaptiveModelManager] = None


def get_model_manager() -> AdaptiveModelManager:
    """Get or create the global model manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = AdaptiveModelManager()
    return _global_manager


async def quick_recommend(task_type: TaskType, preference: str = "balanced") -> Optional[str]:
    """Convenience function for quick model recommendations."""
    manager = get_model_manager()
    return await manager.recommend_model(task_type, preference)
