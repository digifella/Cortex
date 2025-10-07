"""
Unified LLM Service Manager
Provides consistent LLM provider selection and availability checking across all modules.

Version: 1.2.0
Date: 2025-08-22
"""

import os
import logging
from typing import Any, Optional, Dict, Tuple
from enum import Enum

from .utils.ollama_utils import check_ollama_service, format_ollama_error_for_user
from .utils.logging_utils import get_logger
from .exceptions import ModelError, ConfigurationError

logger = get_logger(__name__)

class LLMProvider(Enum):
    """Standard LLM provider enumeration."""
    LOCAL_OLLAMA = "Local (Ollama)"
    CLOUD_GEMINI = "Cloud (Gemini)"
    CLOUD_OPENAI = "Cloud (OpenAI)"

class TaskType(Enum):
    """Task types with different LLM requirements."""
    RESEARCH = "research"          # User choice: Local or Cloud
    PROPOSALS = "proposals"        # Local only (privacy)
    KNOWLEDGE_OPS = "knowledge"    # Local only (performance) 
    IDEATION = "ideation"         # User choice: Local or Cloud

class LLMServiceManager:
    """
    Centralized LLM service management with consistent provider selection,
    availability checking, and error handling.
    """
    
    # Task-specific model mappings with intelligent selection
    @staticmethod
    def _get_smart_model():
        """Get recommended model based on system resources"""
        try:
            from cortex_engine.utils.smart_model_selector import get_recommended_text_model
            return get_recommended_text_model()
        except Exception:
            return "mistral:latest"  # Safe fallback to available model
    
    @classmethod
    def get_task_models(cls):
        """Get task-specific models with smart selection"""
        smart_model = cls._get_smart_model()
        return {
            TaskType.PROPOSALS: smart_model,
            TaskType.KNOWLEDGE_OPS: smart_model, 
            TaskType.RESEARCH: "mistral:latest",  # Always efficient for research
            TaskType.IDEATION: smart_model
        }
    
    # Provider constraints by task
    TASK_PROVIDER_CONSTRAINTS = {
        TaskType.PROPOSALS: [LLMProvider.LOCAL_OLLAMA],        # Privacy required
        TaskType.KNOWLEDGE_OPS: [LLMProvider.LOCAL_OLLAMA],    # Performance required
        TaskType.RESEARCH: [LLMProvider.LOCAL_OLLAMA, LLMProvider.CLOUD_GEMINI, LLMProvider.CLOUD_OPENAI],
        TaskType.IDEATION: [LLMProvider.LOCAL_OLLAMA, LLMProvider.CLOUD_GEMINI, LLMProvider.CLOUD_OPENAI]
    }
    
    def __init__(self, task_type: TaskType, user_provider: Optional[str] = None):
        """
        Initialize LLM service manager for a specific task.
        
        Args:
            task_type: The type of task requiring LLM services
            user_provider: User-selected provider (if applicable)
        """
        self.task_type = task_type
        self.user_provider = self._parse_provider(user_provider) if user_provider else None
        self._llm_instance = None
        self._current_provider = None
        
    def _parse_provider(self, provider_str: str) -> LLMProvider:
        """Parse provider string to enum, handling various formats."""
        provider_str = provider_str.strip()
        
        # Handle common variations
        provider_map = {
            "Local (Ollama)": LLMProvider.LOCAL_OLLAMA,
            "Cloud (Gemini)": LLMProvider.CLOUD_GEMINI,
            "Cloud (OpenAI)": LLMProvider.CLOUD_OPENAI,
            "ollama": LLMProvider.LOCAL_OLLAMA,
            "gemini": LLMProvider.CLOUD_GEMINI,
            "openai": LLMProvider.CLOUD_OPENAI,
            "local": LLMProvider.LOCAL_OLLAMA
        }
        
        if provider_str in provider_map:
            return provider_map[provider_str]
            
        # Fuzzy matching for common cases
        provider_lower = provider_str.lower()
        if "ollama" in provider_lower or "local" in provider_lower:
            return LLMProvider.LOCAL_OLLAMA
        elif "gemini" in provider_lower:
            return LLMProvider.CLOUD_GEMINI
        elif "openai" in provider_lower:
            return LLMProvider.CLOUD_OPENAI
            
        raise ConfigurationError(f"Unknown LLM provider: {provider_str}")
    
    def get_available_providers(self) -> list[LLMProvider]:
        """Get list of providers available for this task type."""
        return self.TASK_PROVIDER_CONSTRAINTS.get(self.task_type, [])
    
    def get_provider_display_names(self) -> list[str]:
        """Get user-friendly provider names for UI selection."""
        return [provider.value for provider in self.get_available_providers()]
    
    def select_provider(self) -> Tuple[LLMProvider, Optional[str]]:
        """
        Select the best available provider based on task type and user preference.
        
        Returns:
            Tuple of (selected_provider, error_message)
        """
        available_providers = self.get_available_providers()
        
        if not available_providers:
            return None, f"No providers configured for task type: {self.task_type.value}"
        
        # For tasks with only one provider option, use it
        if len(available_providers) == 1:
            provider = available_providers[0]
            if provider == LLMProvider.LOCAL_OLLAMA:
                is_running, error_msg, resolved_url = check_ollama_service()
                if not is_running:
                    return None, f"Required local LLM service not available: {error_msg}"
            return provider, None
        
        # For tasks with multiple options, respect user choice
        if self.user_provider and self.user_provider in available_providers:
            provider = self.user_provider
        else:
            # Default to first available provider
            provider = available_providers[0]

        # Validate provider availability
        if provider == LLMProvider.LOCAL_OLLAMA:
            is_running, error_msg, resolved_url = check_ollama_service()
            if not is_running:
                # Try fallback to cloud if available
                cloud_providers = [p for p in available_providers if p != LLMProvider.LOCAL_OLLAMA]
                if cloud_providers:
                    logger.warning(f"Local LLM unavailable, falling back to {cloud_providers[0].value}")
                    return cloud_providers[0], None
                else:
                    return None, f"Local LLM required but not available: {error_msg}"
                    
        return provider, None
    
    def get_llm(self, force_refresh: bool = False) -> Any:
        """
        Get configured LLM instance with availability checking.
        
        Args:
            force_refresh: Force recreation of LLM instance
            
        Returns:
            Configured LLM instance
            
        Raises:
            ModelError: If LLM cannot be initialized
        """
        # Refresh if requested or provider changed
        if force_refresh or self._llm_instance is None:
            provider, error_msg = self.select_provider()
            if error_msg:
                raise ModelError(f"LLM initialization failed: {error_msg}")
                
            self._llm_instance = self._create_llm_instance(provider)
            self._current_provider = provider
            
        return self._llm_instance
    
    def _create_llm_instance(self, provider: LLMProvider) -> Any:
        """Create LLM instance for the specified provider."""
        task_models = self.get_task_models()
        model = task_models.get(self.task_type, "mistral")
        
        if provider == LLMProvider.LOCAL_OLLAMA:
            from .utils.smart_ollama_llm import create_smart_ollama_llm
            from .exceptions import ModelError
            llm_instance = create_smart_ollama_llm(model=model, request_timeout=120.0)
            if llm_instance is None:
                raise ModelError("Failed to initialize Ollama LLM - service may not be available or model not found")
            return llm_instance
            
        elif provider == LLMProvider.CLOUD_GEMINI:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ConfigurationError("GEMINI_API_KEY not found in environment")
            from llama_index.llms.gemini import Gemini  
            return Gemini(api_key=api_key, model="gemini-1.5-flash")
            
        elif provider == LLMProvider.CLOUD_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ConfigurationError("OPENAI_API_KEY not found in environment")
            from llama_index.llms.openai import OpenAI
            return OpenAI(api_key=api_key, model="gpt-4o-mini")
            
        else:
            raise ConfigurationError(f"Unsupported provider: {provider}")
    
    def get_status_info(self) -> Dict[str, str]:
        """Get current service status information for UI display."""
        provider, error_msg = self.select_provider()
        
        if error_msg:
            return {
                "status": "error",
                "message": error_msg,
                "provider": "None",
                "model": "N/A"
            }
            
        return {
            "status": "ready",
            "message": f"✅ {provider.value} ready for {self.task_type.value}",
            "provider": provider.value,
            "model": self.get_task_models().get(self.task_type, "Default")
        }

def create_llm_service(task_type: str, user_provider: str = None) -> LLMServiceManager:
    """
    Convenience function to create LLM service manager.
    
    Args:
        task_type: String task type ("research", "proposals", "knowledge", "ideation")
        user_provider: User-selected provider string
        
    Returns:
        Configured LLMServiceManager instance
    """
    # Convert string to enum
    task_enum_map = {
        "research": TaskType.RESEARCH,
        "proposals": TaskType.PROPOSALS, 
        "knowledge": TaskType.KNOWLEDGE_OPS,
        "ideation": TaskType.IDEATION
    }
    
    task_enum = task_enum_map.get(task_type.lower())
    if not task_enum:
        raise ValueError(f"Unknown task type: {task_type}")
        
    return LLMServiceManager(task_enum, user_provider)