import re
import os
from pathlib import Path
from typing import Union, Optional, List
import urllib.parse


def convert_windows_to_wsl_path(path_str: Union[str, Path, None]) -> str:
    if not path_str:
        return ""

    path_str = str(path_str).strip()

    if path_str.startswith('/mnt/'):
        return path_str

    if path_str.startswith('/') and not path_str.startswith('/mnt/'):
        return path_str

    normalized_path = path_str.replace('\\', '/')

    m = re.match(r'^([a-zA-Z]):/(.*)', normalized_path)
    if m:
        drive, rest = m.groups()
        drive_mount = f"/mnt/{drive.lower()}"
        wsl_path = f"{drive_mount}/{rest}" if rest else drive_mount
        if os.path.exists(drive_mount):
            return wsl_path
        return f"{drive.upper()}:/{rest}" if rest else f"{drive.upper()}:/"

    m = re.match(r'^([a-zA-Z]):$', normalized_path)
    if m:
        drive = m.group(1)
        drive_mount = f"/mnt/{drive.lower()}"
        if os.path.exists(drive_mount):
            return drive_mount
        return f"{drive.upper()}:/"

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
    if not path_str:
        return ""

    raw = str(path_str).strip()
    in_docker = _in_docker()

    # Special-case: POSIX-looking inputs
    if raw.startswith('/'):
        # If running in Docker and the path is a WSL-style mount (e.g., /mnt/c/..),
        # translate it to a container-visible host mount path to avoid read-only /mnt/*.
        if in_docker:
            mnt_match = re.match(r'^/mnt/([a-zA-Z])/(.*)$', raw)
            if mnt_match:
                drive, rest = mnt_match.groups()
                # Prefer the configured AI database mount inside the container if provided
                ai_db_env = os.environ.get('AI_DATABASE_PATH')
                if ai_db_env and ai_db_env.strip():
                    # If user passed a directory under the Windows drive root, but we're in Docker,
                    # route to the configured writable mount root (e.g., /data/ai_databases)
                    return ai_db_env.rstrip('/')
                # Otherwise, map to Docker Desktop's host mount root if accessible
                return _resolve_docker_host_path(drive, rest)
        # Non-WSL POSIX path or not in Docker -> return as-is
        return raw

    # Normalize backslashes first for Windows-style inputs
    normalized_path = raw.replace('\\', '/')

    # Windows drive patterns (C:/..., D:/...)
    m = re.match(r'^([a-zA-Z]):/(.*)', normalized_path)
    if m:
        drive, rest = m.groups()
        if in_docker:
            # Resolve to an existing host mount path inside the container
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


def normalize_path(path_str: Union[str, Path, None]) -> Optional[Path]:
    if not path_str:
        return None
    normalized_str = convert_to_docker_mount_path(path_str)
    try:
        return Path(normalized_str).resolve()
    except (OSError, ValueError):
        return None


def ensure_directory(path: Union[str, Path]) -> Path:
    path_obj = normalize_path(path)
    if not path_obj:
        raise ValueError(f"Invalid path: {path}")
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def validate_path_exists(path: Union[str, Path], must_be_dir: bool = False) -> bool:
    if not path:
        return False
    path_obj = normalize_path(path)
    if path_obj and path_obj.exists():
        return path_obj.is_dir() if must_be_dir else True
    return False


def process_drag_drop_path(raw_path: str) -> Optional[Path]:
    """Process a drag-and-drop path from any platform into a normalized Path."""
    if not raw_path or not raw_path.strip():
        return None
    path_str = raw_path.strip()
    # Strip file:// and file:\ prefixes
    if path_str.startswith('file://'):
        path_str = path_str[7:]
    if path_str.startswith('file:\\\\'):
        path_str = path_str[7:]
    # Trim quotes
    if path_str.startswith('"') and path_str.endswith('"'):
        path_str = path_str[1:-1]
    if path_str.startswith("'") and path_str.endswith("'"):
        path_str = path_str[1:-1]
    # URL decode and unescape common sequences
    try:
        path_str = urllib.parse.unquote(path_str)
    except Exception:
        pass
    path_str = path_str.replace('\\ ', ' ')
    path_str = path_str.replace('\\&', '&').replace('\\(', '(').replace('\\)', ')')
    # Normalize and return
    norm = normalize_path(path_str)
    return norm if norm and norm.exists() else norm


def process_multiple_drag_drop_paths(raw_paths: str) -> List[Path]:
    """Process newline-separated drag-and-drop paths into a list of Paths."""
    if not raw_paths or not raw_paths.strip():
        return []
    out: List[Path] = []
    seen = set()
    for line in raw_paths.strip().split('\n'):
        p = process_drag_drop_path(line.strip())
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out
