# ## File: cortex_engine/utils/path_utils.py
# Version: 1.0.0
# Date: 2025-07-23
# Purpose: Centralized path handling utilities for cross-platform compatibility.
#          Eliminates code duplication across multiple modules.

import re
import os
from pathlib import Path
from typing import Union, Optional, List


def convert_windows_to_wsl_path(path_str: Union[str, Path, None]) -> str:
    """
    Convert Windows-style paths to WSL-compatible paths.
    
    Args:
        path_str: Windows path string or Path object
        
    Returns:
        WSL-compatible path string
        
    Examples:
        'C:/Users/docs' -> '/mnt/c/Users/docs'
        'F:\\ai_databases' -> '/mnt/f/ai_databases'
        '/mnt/c/existing' -> '/mnt/c/existing' (unchanged)
    """
    if not path_str:
        return ""
    
    path_str = str(path_str).strip()
    
    # Already WSL format, return as-is
    if path_str.startswith('/mnt/'):
        return path_str
    
    # Normalize backslashes to forward slashes
    normalized_path = path_str.replace('\\', '/')
    
    # Match Windows drive pattern (C:/, D:/, etc.)
    match = re.match(r'^([a-zA-Z]):/(.*)', normalized_path)
    if match:
        drive_letter, rest = match.groups()
        return f"/mnt/{drive_letter.lower()}/{rest}"
    
    return normalized_path


def normalize_path(path_str: Union[str, Path, None]) -> Optional[Path]:
    """
    Normalize a path string to a Path object, handling platform differences.
    
    Args:
        path_str: Path string or Path object
        
    Returns:
        Normalized Path object or None if invalid
    """
    if not path_str:
        return None
    
    # Convert Windows to WSL if needed
    normalized_str = convert_windows_to_wsl_path(path_str)
    
    try:
        return Path(normalized_str).resolve()
    except (OSError, ValueError):
        return None


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object of the ensured directory
        
    Raises:
        OSError: If directory cannot be created
    """
    path_obj = normalize_path(path)
    if not path_obj:
        raise ValueError(f"Invalid path provided: {path}")
    
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path object of the project root
    """
    # Go up two levels from this file (utils/path_utils.py -> utils -> cortex_engine -> project_root)
    return Path(__file__).parent.parent.parent


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


def process_drag_drop_path(raw_path: str) -> Optional[Path]:
    """
    Process a single drag-and-drop path from any platform.
    
    Handles:
    - Mac: file:// URLs, escaped spaces, quoted paths
    - Windows: file:\\ URLs, UNC paths, drive letters
    - Linux: standard paths
    - All: URL decoding, whitespace cleanup
    
    Args:
        raw_path: Raw path string from drag-and-drop operation
        
    Returns:
        Normalized Path object or None if invalid
    """
    if not raw_path or not raw_path.strip():
        return None
    
    path_str = raw_path.strip()
    
    # Remove file:// protocol (Mac/Linux)
    if path_str.startswith('file://'):
        path_str = path_str[7:]
    
    # Remove file:\\ protocol (Windows)
    if path_str.startswith('file:\\\\'):
        path_str = path_str[7:]
    
    # Handle quoted paths (common on Mac)
    if path_str.startswith('"') and path_str.endswith('"'):
        path_str = path_str[1:-1]
    
    if path_str.startswith("'") and path_str.endswith("'"):
        path_str = path_str[1:-1]
    
    # URL decode (handle %20 for spaces, etc.)
    try:
        import urllib.parse
        path_str = urllib.parse.unquote(path_str)
    except Exception:
        pass  # Continue with original if URL decode fails
    
    # Handle Mac escaped spaces and special characters
    path_str = path_str.replace('\\ ', ' ')  # Escaped spaces
    path_str = path_str.replace('\\&', '&')  # Escaped ampersands
    path_str = path_str.replace('\\(', '(')  # Escaped parentheses
    path_str = path_str.replace('\\)', ')')
    
    # Clean up any remaining backslash escapes (be careful not to break Windows paths)
    if not re.match(r'^[A-Za-z]:\\', path_str):  # Not a Windows drive path
        path_str = re.sub(r'\\(.)', r'\1', path_str)  # Remove escape backslashes
    
    # Normalize the path
    return normalize_path(path_str)


def process_multiple_drag_drop_paths(raw_paths: str) -> List[Path]:
    """
    Process multiple drag-and-drop paths (typically newline-separated).
    
    Args:
        raw_paths: Raw string containing multiple paths
        
    Returns:
        List of valid Path objects
    """
    if not raw_paths or not raw_paths.strip():
        return []
    
    paths = []
    
    # Split on newlines and process each path
    for line in raw_paths.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        processed_path = process_drag_drop_path(line)
        if processed_path and processed_path.exists():
            paths.append(processed_path)
    
    # Remove duplicates while preserving order
    unique_paths = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    
    return unique_paths


def get_home_directory() -> Path:
    """
    Get the user's home directory, handling cross-platform differences.
    
    Returns:
        Path object of the user's home directory
    """
    return Path.home()


def is_safe_path(path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Check if a path is safe (no directory traversal attacks).
    
    Args:
        path: Path to check
        base_path: Optional base path to restrict to
        
    Returns:
        True if path is safe
    """
    try:
        path_obj = normalize_path(path)
        if not path_obj:
            return False
        
        # Check for directory traversal patterns
        path_str = str(path_obj)
        if '..' in path_str or path_str.startswith('/'):
            # Allow absolute paths if no base_path restriction
            if base_path is None and path_str.startswith('/'):
                return True
            return False
        
        # If base_path is specified, ensure path is within it
        if base_path:
            base_obj = normalize_path(base_path)
            if base_obj:
                try:
                    path_obj.resolve().relative_to(base_obj.resolve())
                    return True
                except ValueError:
                    return False
        
        return True
        
    except Exception:
        return False


def get_file_size_display(path: Union[str, Path]) -> str:
    """
    Get human-readable file size.
    
    Args:
        path: File path
        
    Returns:
        Formatted file size string
    """
    try:
        path_obj = normalize_path(path)
        if not path_obj or not path_obj.exists() or not path_obj.is_file():
            return "Unknown"
        
        size_bytes = path_obj.stat().st_size
        
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
            
    except Exception:
        return "Unknown"