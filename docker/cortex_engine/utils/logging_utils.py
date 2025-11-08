# ## File: cortex_engine/utils/logging_utils.py
# Version: 1.0.0
# Date: 2025-07-23
# Purpose: Centralized logging configuration and utilities.
#          Standardizes logging across all modules.

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up standardized logging configuration.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)
        log_file: Optional log file path
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Default format
    if not format_string:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not create file handler for {log_file}: {e}")
    
    return logger


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a standardized logger instance.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    return setup_logging(name, log_file=log_file)


class LoggerMixin:
    """
    Mixin class to add standardized logging to any class.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger