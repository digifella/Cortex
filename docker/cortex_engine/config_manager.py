# ## File: cortex_engine/config_manager.py
# Version: 3.0.0 (Utilities Refactor)
# Date: 2025-07-23
# Purpose: Manages persistent user configuration settings using a class structure.
#          - REFACTOR (v3.0.0): Updated to use centralized utilities for logging
#            and error handling. Improved error handling consistency.

import json
from pathlib import Path

# Import centralized utilities
from .utils import get_logger, get_project_root
from .exceptions import ConfigurationError

# Set up logging
logger = get_logger(__name__)

# The config file is stored in the project's root directory.
CONFIG_FILE_PATH = get_project_root() / "cortex_config.json"

class ConfigManager:
    """
    Handles loading and saving of persistent user settings to a JSON file.
    This provides a simple way to remember user inputs across sessions,
    such as last-used directory paths.
    """
    def __init__(self):
        """Initializes the manager and loads the current config."""
        self.config_path = CONFIG_FILE_PATH
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Loads the configuration from the JSON file, with environment variable fallbacks."""
        import os
        
        # Start with environment variable defaults
        default_config = {
            "ai_database_path": os.getenv("AI_DATABASE_PATH", "/mnt/f/ai_databases"),
            "knowledge_source_path": os.getenv("KNOWLEDGE_SOURCE_PATH", "")
        }
        
        if not self.config_path.exists():
            logger.info(f"Configuration file not found at {self.config_path}, using environment defaults")
            return default_config
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                logger.debug(f"Loaded configuration from {self.config_path}")
                
                # Merge with defaults, prioritizing environment variables if config values are empty
                for key, default_value in default_config.items():
                    if not config.get(key) and default_value:
                        config[key] = default_value
                        
                return config
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load configuration from {self.config_path}: {e}")
            return default_config  # Return defaults on error

    def get_config(self) -> dict:
        """Returns the current configuration dictionary."""
        return self.config

    def update_config(self, new_settings: dict):
        """
        Updates the configuration with new settings and saves it to the file.
        
        Args:
            new_settings: Dictionary of settings to update
            
        Raises:
            ConfigurationError: If saving fails critically
        """
        self.config.update(new_settings)
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.debug(f"Successfully saved configuration to {self.config_path}")
        except IOError as e:
            logger.error(f"Failed to save configuration to {self.config_path}: {e}")
            # For non-critical operations, we don't raise an exception
            # but we do log the error properly