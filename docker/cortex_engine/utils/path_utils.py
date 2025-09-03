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
        return f"/mnt/{drive.lower()}/{rest}"
    m = re.match(r'^([a-zA-Z]):$', normalized_path)
    if m:
        drive = m.group(1)
        return f"/mnt/{drive.lower()}"
    return normalized_path


def convert_to_docker_mount_path(path_str: Union[str, Path, None]) -> str:
    if not path_str:
        return ""
    raw = str(path_str).strip()
    if raw.startswith('/'):
        return raw
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('container') or os.environ.get('DOCKER_CONTAINER')
    normalized_path = raw.replace('\\', '/')
    m = re.match(r'^([a-zA-Z]):/(.*)', normalized_path)
    if m:
        drive, rest = m.groups()
        if in_docker:
            # Prefer explicit container-mapped DB root
            ai_db_env = os.environ.get('AI_DATABASE_PATH')
            if ai_db_env:
                return ai_db_env if rest.lower().endswith('ai_databases') else ai_db_env
            return f"/host_mnt/{drive.lower()}/{rest}"
        return f"/mnt/{drive.lower()}/{rest}"
    m = re.match(r'^([a-zA-Z]):$', normalized_path)
    if m:
        drive = m.group(1)
        if in_docker:
            ai_db_env = os.environ.get('AI_DATABASE_PATH')
            if ai_db_env:
                return ai_db_env
            return f"/host_mnt/{drive.lower()}"
        return f"/mnt/{drive.lower()}"
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
