# ## File: cortex_engine/boilerplate_manager.py
# Version: 2.0.0 (Utilities Refactor)
# Date: 2025-07-23
# Purpose: Manages loading and accessing boilerplate text snippets from a JSON file.
#          - REFACTOR (v2.0.0): Updated to use centralized utilities for logging
#            and error handling. Improved error handling consistency.

import json
from pathlib import Path
import os

# Import centralized utilities
from .utils import get_logger, get_project_root

# Set up logging
logger = get_logger(__name__)

# Define the path to the boilerplate file in the project root
BOILERPLATE_FILE = get_project_root() / "boilerplate.json"

class BoilerplateManager:
    """
    Handles loading and providing access to boilerplate text blocks
    stored in a central JSON file.
    """
    def __init__(self):
        """Initializes the manager and loads the boilerplate data."""
        self.boilerplate_data = self._load()

    def _load(self) -> dict:
        """Loads the boilerplate content from the JSON file."""
        if not BOILERPLATE_FILE.exists():
            logger.info(f"Boilerplate file not found at {BOILERPLATE_FILE}, creating default")
            # Create a default file if it doesn't exist to guide the user
            default_content = {
                "about_us": "This is the default 'About Us' section. Please edit this in boilerplate.json.",
                "legals": "This is the default 'Legal Terms' section. Please edit this in boilerplate.json."
            }
            self._save(default_content)
            return default_content

        try:
            with open(BOILERPLATE_FILE, 'r', encoding='utf-8') as f:
                content = json.load(f)
                logger.debug(f"Loaded {len(content)} boilerplate entries from {BOILERPLATE_FILE}")
                return content
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load boilerplate file {BOILERPLATE_FILE}: {e}")
            # Return a default error message if the file is corrupt
            return {"error": "Could not load boilerplate.json. Please check the file format."}

    def _save(self, data: dict):
        """Saves data to the boilerplate JSON file."""
        try:
            with open(BOILERPLATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            logger.debug(f"Successfully saved boilerplate data to {BOILERPLATE_FILE}")
        except IOError as e:
            logger.error(f"Failed to save boilerplate file {BOILERPLATE_FILE}: {e}")
            raise

    def get_boilerplate(self, name: str) -> str:
        """
        Retrieves a specific boilerplate text by its name (key).

        Args:
            name: The key of the boilerplate text to retrieve.

        Returns:
            The boilerplate text as a string, or an error message if not found.
        """
        return self.boilerplate_data.get(name, f"[BOILERPLATE ERROR: Key '{name}' not found in boilerplate.json]")

    def get_boilerplate_names(self) -> list:
        """
        Returns a list of all available boilerplate names (keys).
        """
        return list(self.boilerplate_data.keys())