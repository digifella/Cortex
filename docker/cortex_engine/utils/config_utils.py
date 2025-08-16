# ## File: cortex_engine/utils/config_utils.py
# Version: 1.0.0
# Date: 2025-07-23
# Purpose: Configuration validation and utility functions.
#          Centralizes configuration-related operations.

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from .path_utils import normalize_path, validate_path_exists


def validate_path_exists(path: Union[str, Path], must_be_dir: bool = False) -> bool:
    """
    Validate that a path exists and optionally is a directory.
    
    Args:
        path: Path to validate
        must_be_dir: If True, path must be a directory
        
    Returns:
        True if path exists and meets requirements
    """
    path_obj = normalize_path(path)
    if not path_obj or not path_obj.exists():
        return False
    
    if must_be_dir:
        return path_obj.is_dir()
    
    return True


def get_env_var(
    key: str, 
    default: Optional[str] = None, 
    required: bool = False
) -> Optional[str]:
    """
    Get environment variable with validation.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        required: If True, raises error if not found and no default
        
    Returns:
        Environment variable value or default
        
    Raises:
        ValueError: If required and not found
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' not found")
    
    return value


def validate_model_config(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate model configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration with error messages
    """
    errors = {}
    
    # Check required fields
    required_fields = ['embed_model', 'llm_model']
    for field in required_fields:
        if field not in config or not config[field]:
            errors[field] = f"{field} is required"
    
    # Validate provider if specified
    if 'llm_provider' in config:
        valid_providers = ['openai', 'ollama', 'gemini']
        if config['llm_provider'] not in valid_providers:
            errors['llm_provider'] = f"Must be one of: {', '.join(valid_providers)}"
    
    return errors


def validate_database_config(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate database configuration.
    
    Args:
        config: Configuration dictionary with database paths
        
    Returns:
        Dictionary of validation errors (empty if valid)
    """
    errors = {}
    
    # Check base data path
    if 'base_data_path' in config:
        if not validate_path_exists(config['base_data_path'], must_be_dir=True):
            errors['base_data_path'] = "Database path does not exist or is not a directory"
    
    return errors


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    for config in configs:
        if config:
            result.update(config)
    return result