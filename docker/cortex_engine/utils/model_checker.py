#!/usr/bin/env python3
"""
Model Availability Checker
Validates that required Ollama models are available before processing
"""

import logging
from typing import Dict, List, Tuple, Optional
import ollama
from ollama import Client

from cortex_engine.config import VLM_MODEL, KB_LLM_MODEL, PROPOSAL_LLM_MODEL, RESEARCH_LOCAL_MODEL
from .logging_utils import get_logger

logger = get_logger(__name__)

class ModelAvailabilityChecker:
    """Check if required Ollama models are available for different operations"""
    
    def __init__(self):
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Ollama client with error handling"""
        try:
            self.client = ollama.Client()
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama client: {e}")
            self.client = None
    
    def check_ollama_service(self) -> Tuple[bool, str]:
        """Check if Ollama service is running"""
        if not self.client:
            return False, "Ollama client could not be initialized"
        
        try:
            # Try to list models as a health check
            self.client.list()
            return True, "Ollama service is running"
        except Exception as e:
            return False, f"Ollama service not available: {e}"
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        if not self.client:
            return []
        
        try:
            models = self.client.list()
            return [model['name'] for model in models.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to get model list: {e}")
            return []
    
    def check_model_availability(self, model_name: str) -> Tuple[bool, str]:
        """Check if a specific model is available"""
        available_models = self.get_available_models()
        
        if not available_models:
            return False, "No models available or Ollama not running"
        
        # Check for exact match or base model match (e.g., "mistral" matches "mistral:latest")
        model_base = model_name.split(':')[0]
        for available in available_models:
            available_base = available.split(':')[0]
            if model_name == available or model_base == available_base:
                return True, f"Model '{model_name}' is available"
        
        return False, f"Model '{model_name}' not found. Available: {', '.join(available_models)}"
    
    def check_ingestion_requirements(self, include_images: bool = True) -> Dict[str, any]:
        """
        Check all model requirements for ingestion process
        
        Args:
            include_images: Whether image processing is enabled
            
        Returns:
            Dict with status and recommendations
        """
        results = {
            "ollama_running": False,
            "kb_model_available": False,
            "vlm_model_available": False,
            "missing_models": [],
            "warnings": [],
            "can_proceed": False,
            "recommended_actions": []
        }
        
        # Check Ollama service
        service_ok, service_msg = self.check_ollama_service()
        results["ollama_running"] = service_ok
        
        if not service_ok:
            results["warnings"].append(f"⚠️ Ollama service issue: {service_msg}")
            results["recommended_actions"].append("Start Ollama service: `ollama serve`")
            return results
        
        # Check KB model (required for metadata analysis)
        kb_ok, kb_msg = self.check_model_availability(KB_LLM_MODEL)
        results["kb_model_available"] = kb_ok
        
        if not kb_ok:
            results["missing_models"].append(KB_LLM_MODEL)
            results["warnings"].append(f"⚠️ Knowledge base model missing: {kb_msg}")
            results["recommended_actions"].append(f"Install KB model: `ollama pull {KB_LLM_MODEL}`")
        
        # Check VLM model (required only if image processing enabled)
        if include_images:
            vlm_ok, vlm_msg = self.check_model_availability(VLM_MODEL)
            results["vlm_model_available"] = vlm_ok
            
            if not vlm_ok:
                results["missing_models"].append(VLM_MODEL)
                results["warnings"].append(f"⚠️ Vision model missing: {vlm_msg}")
                results["recommended_actions"].extend([
                    f"Install vision model: `ollama pull {VLM_MODEL}`",
                    "OR disable image processing to continue without image analysis"
                ])
        else:
            results["vlm_model_available"] = True  # Not required
        
        # Determine if processing can proceed
        results["can_proceed"] = results["ollama_running"] and results["kb_model_available"]
        if include_images:
            results["can_proceed"] = results["can_proceed"] and results["vlm_model_available"]
        
        return results
    
    def check_research_requirements(self) -> Dict[str, any]:
        """Check model requirements for research operations"""
        results = {
            "ollama_running": False,
            "local_research_available": False,
            "missing_models": [],
            "warnings": [],
            "recommended_actions": []
        }
        
        # Check Ollama service
        service_ok, service_msg = self.check_ollama_service()
        results["ollama_running"] = service_ok
        
        if not service_ok:
            results["warnings"].append(f"⚠️ Ollama service issue: {service_msg}")
            results["recommended_actions"].append("Start Ollama service: `ollama serve`")
            return results
        
        # Check local research model
        research_ok, research_msg = self.check_model_availability(RESEARCH_LOCAL_MODEL)
        results["local_research_available"] = research_ok
        
        if not research_ok:
            results["missing_models"].append(RESEARCH_LOCAL_MODEL)
            results["warnings"].append(f"ℹ️ Local research model not available: {research_msg}")
            results["recommended_actions"].append(f"Install research model: `ollama pull {RESEARCH_LOCAL_MODEL}`")
        
        return results
    
    def get_model_installation_commands(self, missing_models: List[str]) -> List[str]:
        """Generate installation commands for missing models"""
        commands = []
        for model in missing_models:
            commands.append(f"ollama pull {model}")
        return commands
    
    def format_status_message(self, check_results: Dict) -> str:
        """Format check results into a user-friendly message"""
        if check_results["can_proceed"]:
            return "✅ All required models are available"
        
        message_parts = []
        
        if check_results["warnings"]:
            message_parts.extend(check_results["warnings"])
        
        if check_results["recommended_actions"]:
            message_parts.append("\n**Recommended Actions:**")
            for action in check_results["recommended_actions"]:
                message_parts.append(f"• {action}")
        
        return "\n".join(message_parts)


# Global instance for easy access
model_checker = ModelAvailabilityChecker()