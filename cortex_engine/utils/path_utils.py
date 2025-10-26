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
    Also handles Docker container mount paths.
    
    Args:
        path_str: Windows path string or Path object
        
    Returns:
        WSL-compatible or Docker mount path string
        
    Examples:
        'C:/Users/docs' -> '/mnt/c/Users/docs'
        'F:\\ai_databases' -> '/mnt/f/ai_databases'
        'E:\\KB_Test' -> '/mnt/e/KB_Test'
        '/mnt/c/existing' -> '/mnt/c/existing' (unchanged)
    """
    if not path_str:
        return ""
    
    path_str = str(path_str).strip()
    
    # Already WSL/Docker mount format, return as-is
    if path_str.startswith('/mnt/'):
        return path_str
    
    # Already Unix path, return as-is
    if path_str.startswith('/') and not path_str.startswith('/mnt/'):
        return path_str
    
    # Normalize backslashes to forward slashes
    normalized_path = path_str.replace('\\', '/')
    
    # Match Windows drive pattern (C:/, D:/, etc.)
    match = re.match(r'^([a-zA-Z]):/(.*)', normalized_path)
    if match:
        drive_letter, rest = match.groups()
        drive_mount = f"/mnt/{drive_letter.lower()}"
        wsl_path = f"{drive_mount}/{rest}" if rest else drive_mount
        # Prefer WSL mounts when available (WSL / Docker Desktop)
        if os.path.exists(drive_mount):
            return wsl_path
        # Fallback to normalized Windows path for native Windows environments
        return f"{drive_letter.upper()}:/{rest}" if rest else f"{drive_letter.upper()}:/"
    
    # Handle bare drive patterns like C:, D:, E:
    match = re.match(r'^([a-zA-Z]):$', normalized_path)
    if match:
        drive_letter = match.group(1)
        drive_mount = f"/mnt/{drive_letter.lower()}"
        if os.path.exists(drive_mount):
            return drive_mount
        return f"{drive_letter.upper()}:/"
    
    return normalized_path


def _in_docker() -> bool:
    return bool(os.path.exists('/.dockerenv') or os.environ.get('container') or os.environ.get('DOCKER_CONTAINER'))


def _resolve_docker_host_path(drive: str, rest: str) -> str:
    """Try common Docker Desktop host mount roots and return the first that exists."""
    drive = drive.lower()
    # Allow override via env
    preferred_root = os.environ.get('DOCKER_HOST_MOUNT_ROOT')
    candidate_roots = [r for r in [preferred_root, '/host_mnt', '/run/desktop/mnt/host', '/mnt'] if r]
    for root in candidate_roots:
        candidate = f"{root}/{drive}/{rest}"
        if os.path.exists(candidate):
            return candidate
    # Fall back to the first root even if it doesn't exist yet
    return f"{candidate_roots[0]}/{drive}/{rest}" if candidate_roots else f"/host_mnt/{drive}/{rest}"


def convert_source_path_to_docker_mount(path_str: Union[str, Path, None]) -> str:
    """
    Convert a source directory path to Docker mount path - specifically for ingestion source directories.
    Unlike convert_to_docker_mount_path, this never routes to AI_DATABASE_PATH.
    """
    if not path_str:
        return ""

    raw = str(path_str).strip()
    in_docker = _in_docker()

    # Special-case: POSIX-looking inputs
    if raw.startswith('/'):
        if in_docker:
            mnt_match = re.match(r'^/mnt/([a-zA-Z])/(.*)$', raw)
            if mnt_match:
                drive, rest = mnt_match.groups()
                # For source directories, always map to Docker Desktop's host mount root
                return _resolve_docker_host_path(drive, rest)
        return raw

    # Normalize backslashes first for Windows-style inputs
    normalized_path = raw.replace('\\', '/')

    # Windows drive patterns (C:/..., D:/...)
    m = re.match(r'^([a-zA-Z]):/(.*)', normalized_path)
    if m:
        drive, rest = m.groups()
        if in_docker:
            # For source directories, resolve to an existing host mount path inside the container
            return _resolve_docker_host_path(drive, rest)
        return convert_windows_to_wsl_path(normalized_path)

    # Bare drive letter (e.g., 'D:')
    m = re.match(r'^([a-zA-Z]):$', normalized_path)
    if m:
        drive = m.group(1)
        if in_docker:
            return _resolve_docker_host_path(drive, '')
        return convert_windows_to_wsl_path(normalized_path)

    # Fallback to general conversion
    return convert_windows_to_wsl_path(raw)


def convert_to_docker_mount_path(path_str: Union[str, Path, None]) -> str:
    """
    Convert host paths to container-visible paths when running in Docker.
    - In Docker, prefer `AI_DATABASE_PATH` or known host mount roots over `/mnt/*`.
    - Outside Docker, return WSL-style conversions.
    """
    if not path_str:
        return ""

    raw = str(path_str).strip()
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('container') or os.environ.get('DOCKER_CONTAINER')

    # If already POSIX-like, handle WSL-style mounts specially when in Docker
    if raw.startswith('/'):
        if in_docker:
            mnt = re.match(r'^/mnt/([a-zA-Z])/(.*)$', raw)
            if mnt:
                drive, rest = mnt.groups()
                ai_db_env = os.environ.get('AI_DATABASE_PATH')
                if ai_db_env and ai_db_env.strip():
                    return ai_db_env.rstrip('/')
                return f"/host_mnt/{drive.lower()}/{rest}"
        return raw

    # Normalize backslashes first
    normalized_path = raw.replace('\\', '/')

    # Windows drive pattern
    m = re.match(r'^([a-zA-Z]):/(.*)', normalized_path)
    if m:
        drive, rest = m.groups()
        if in_docker:
            ai_db_env = os.environ.get('AI_DATABASE_PATH')
            if ai_db_env and ai_db_env.strip():
                return ai_db_env.rstrip('/')
            return f"/host_mnt/{drive.lower()}/{rest}"
        return convert_windows_to_wsl_path(normalized_path)

    # Bare drive letter (e.g., 'D:')
    m = re.match(r'^([a-zA-Z]):$', normalized_path)
    if m:
        drive = m.group(1)
        if in_docker:
            ai_db_env = os.environ.get('AI_DATABASE_PATH')
            if ai_db_env and ai_db_env.strip():
                return ai_db_env.rstrip('/')
            return f"/host_mnt/{drive.lower()}"
        return convert_windows_to_wsl_path(normalized_path)

    # No special handling needed
    return convert_windows_to_wsl_path(raw)


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
    Handles Docker container environments where Windows paths need special handling.
    
    Args:
        path: Path to validate
        must_be_dir: If True, path must be a directory
        
    Returns:
        True if path exists and meets requirements
    """
    if not path:
        return False
    
    # First try the normalized path
    path_obj = normalize_path(path)
    if path_obj and path_obj.exists():
        if must_be_dir:
            return path_obj.is_dir()
        return True
    
    # Docker container environment: try Docker volume mount paths
    path_str = str(path).strip()
    
    # Check if it's a Windows path that might be mounted in Docker
    docker_mount_paths = []
    
    if path_str.startswith(('E:\\', 'E:/', 'e:\\', 'e:/')):
        docker_mount_paths.append(Path('/mnt/e') / path_str[3:].replace('\\', '/'))
    elif path_str.startswith(('D:\\', 'D:/', 'd:\\', 'd:/')):
        docker_mount_paths.append(Path('/mnt/d') / path_str[3:].replace('\\', '/'))
    elif path_str.startswith(('C:\\', 'C:/', 'c:\\', 'c:/')):
        docker_mount_paths.append(Path('/mnt/c') / path_str[3:].replace('\\', '/'))
    elif path_str.startswith(('F:\\', 'F:/', 'f:\\', 'f:/')):
        docker_mount_paths.append(Path('/mnt/f') / path_str[3:].replace('\\', '/'))
    
    # Check Docker mount paths
    for docker_path in docker_mount_paths:
        try:
            if docker_path.exists():
                if must_be_dir:
                    return docker_path.is_dir()
                return True
        except (OSError, PermissionError):
            continue
    
    return False


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
