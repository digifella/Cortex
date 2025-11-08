"""
Default path utilities that work across all platforms
Version: v1.0.0
Date: 2025-08-26

Provides cross-platform default paths without hardcoded assumptions
"""

import os
from pathlib import Path
from .path_utils import convert_windows_to_wsl_path


def get_default_ai_database_path() -> str:
    """
    Get the default AI database path, prioritizing environment variables
    and avoiding hardcoded platform assumptions.
    
    Priority order:
    1. AI_DATABASE_PATH environment variable
    2. Platform-aware defaults
    3. Current directory fallback
    
    Returns:
        Cross-platform compatible database path
    """
    # Check environment variable first
    env_path = os.getenv("AI_DATABASE_PATH")
    if env_path:
        return convert_windows_to_wsl_path(env_path)
    
    # Platform-aware defaults
    home = Path.home()
    
    # Check common locations in order of preference
    candidate_paths = [
        # User home directory
        home / "ai_databases",
        home / "Documents" / "ai_databases", 
        
        # WSL/Linux specific (if we're in WSL) - prioritize C drive since F drive doesn't exist
        Path("/mnt/c/ai_databases") if Path("/mnt/c").exists() else None,
        Path("/mnt/d/ai_databases") if Path("/mnt/d").exists() else None,
        
        # Fallback to project directory
        Path(__file__).parent.parent.parent / "data" / "ai_databases"
    ]
    
    # Filter out None values and find first accessible parent
    for path in candidate_paths:
        if path is not None:
            try:
                # Check if parent exists or can be created
                parent = path.parent
                if parent.exists() or parent == path:  # Handle root paths
                    return str(path)
            except (OSError, PermissionError):
                continue
    
    # Ultimate fallback - current directory
    return str(Path.cwd() / "ai_databases")


def get_default_knowledge_source_path() -> str:
    """
    Get the default knowledge source path.
    
    Returns:
        Cross-platform compatible knowledge source path
    """
    # Check environment variable first
    env_path = os.getenv("KNOWLEDGE_SOURCE_PATH")
    if env_path:
        return convert_windows_to_wsl_path(env_path)
    
    # Default to home directory
    home = Path.home()
    return str(home / "Documents" / "knowledge_base")


def get_platform_info() -> dict:
    """
    Get information about the current platform for debugging.
    
    Returns:
        Dictionary with platform information
    """
    import platform
    
    info = {
        "system": platform.system(),
        "platform": platform.platform(),
        "is_wsl": Path("/mnt/c").exists(),
        "is_docker": Path("/.dockerenv").exists(),
        "available_mounts": []
    }
    
    # Check available WSL mounts
    for drive in ['c', 'd', 'e', 'f', 'g', 'h']:
        mount_path = Path(f"/mnt/{drive}")
        if mount_path.exists():
            info["available_mounts"].append(mount_path)
    
    return info