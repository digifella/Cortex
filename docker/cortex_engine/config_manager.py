# ## File: docker/cortex_engine/config_manager.py
# Minimal ConfigManager for Docker distribution

import json
from pathlib import Path

from .utils import get_logger, get_project_root
from .utils.default_paths import (
    get_default_ai_database_path,
    get_default_knowledge_source_path,
)

logger = get_logger(__name__)


CONFIG_FILE_PATH = get_project_root() / "cortex_config.json"


class ConfigManager:
    """Docker-friendly configuration manager with env/default fallbacks."""

    def __init__(self):
        self.config_path = CONFIG_FILE_PATH
        self.config = self._load_config()

    def _load_config(self) -> dict:
        default_config = {
            "ai_database_path": get_default_ai_database_path(),
            "knowledge_source_path": get_default_knowledge_source_path(),
        }

        if not self.config_path.exists():
            logger.info(f"Config not found at {self.config_path}; using defaults")
            return default_config

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # Merge defaults for empty or missing fields
            for k, v in default_config.items():
                if not cfg.get(k):
                    cfg[k] = v
            return cfg
        except Exception as e:
            logger.warning(f"Failed to load config: {e}; using defaults")
            return default_config

    def get_config(self) -> dict:
        return self.config

    def update_config(self, new_settings: dict):
        self.config.update(new_settings or {})
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
            logger.debug(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {self.config_path}: {e}")

