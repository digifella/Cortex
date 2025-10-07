"""
Smart Ollama LLM Selector
Automatically chooses between original LlamaIndex Ollama and modern wrapper based on environment.
"""

import os
import logging
from typing import Any, Optional

from .logging_utils import get_logger
from .ollama_utils import check_ollama_service

logger = get_logger(__name__)


def create_smart_ollama_llm(model: str = "mistral:latest", request_timeout: float = 120.0):
    """
    Create an Ollama LLM instance that works in both Docker and non-Docker environments.
    
    Strategy:
    1. Try original LlamaIndex Ollama first (works in most environments)
    2. Fall back to ModernOllama wrapper if original fails (needed in some Docker environments)
    
    Args:
        model: The Ollama model name to use
        request_timeout: Request timeout in seconds
        
    Returns:
        LLM instance that works in current environment
    """
    
    # Check if Ollama service is available
    is_running, error_msg, resolved_url = check_ollama_service()
    if not is_running:
        logger.error(f"Ollama service not available: {error_msg}")
        return None
    
    # First, try the original LlamaIndex Ollama (works in most cases)
    try:
        from llama_index.llms.ollama import Ollama
        
        logger.info(f"Attempting to create original LlamaIndex Ollama for model: {model}")
        llm = Ollama(model=model, request_timeout=request_timeout)
        
        # Test it with a simple completion
        test_response = llm.complete("Hello")
        if test_response and test_response.text:
            logger.info("✅ Original LlamaIndex Ollama working correctly")
            return llm
            
    except Exception as e:
        logger.warning(f"Original LlamaIndex Ollama failed: {e}")
    
    # Fallback to ModernOllama wrapper (for environments with API compatibility issues)
    try:
        from .modern_ollama_llm import create_modern_ollama_llm
        
        logger.info(f"Falling back to ModernOllama wrapper for model: {model}")
        llm = create_modern_ollama_llm(model=model, request_timeout=request_timeout)
        
        # Test it with a simple completion
        test_response = llm.complete("Hello")
        if test_response and test_response.text:
            logger.info("✅ ModernOllama wrapper working correctly")
            return llm
            
    except Exception as e:
        logger.error(f"ModernOllama wrapper also failed: {e}")
    
    # Both approaches failed
    logger.error("❌ Failed to create working Ollama LLM with both original and modern approaches")
    return None


def is_docker_environment() -> bool:
    """
    Detect if we're running inside a Docker container.
    
    Returns:
        True if running in Docker, False otherwise
    """
    try:
        # Check for .dockerenv file (most reliable indicator)
        if os.path.exists('/.dockerenv'):
            return True
            
        # Check cgroup for docker indicators
        with open('/proc/1/cgroup', 'r') as f:
            content = f.read()
            if 'docker' in content or 'containerd' in content:
                return True
                
    except (FileNotFoundError, PermissionError):
        pass
    
    # Check environment variables that Docker typically sets
    docker_env_vars = ['DOCKER_CONTAINER', 'CONTAINER_ID']
    if any(var in os.environ for var in docker_env_vars):
        return True
    
    return False


def get_environment_info() -> dict:
    """
    Get information about the current environment for debugging.
    
    Returns:
        Dictionary with environment information
    """
    return {
        "is_docker": is_docker_environment(),
        "platform": os.name,
        "working_directory": os.getcwd(),
        "ollama_service_available": check_ollama_service()[0],
        "python_executable": os.sys.executable
    }