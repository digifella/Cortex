import os
from pathlib import Path
from .path_utils import convert_to_docker_mount_path


def _safe_path_exists(path: Path) -> bool:
    """Safely check mount points without surfacing device errors."""
    try:
        return path.exists()
    except (OSError, PermissionError):
        return False

def get_default_ai_database_path() -> str:
    # Prefer container env path
    env_path = os.getenv("AI_DATABASE_PATH")
    if env_path:
        return convert_to_docker_mount_path(env_path)
    
    # In Docker: prefer standard mount
    if _safe_path_exists(Path("/data")):
        return "/data/ai_databases"
    
    # Non-Docker environment: find accessible path
    candidate_paths = [
        Path("/mnt/c/ai_databases") if _safe_path_exists(Path("/mnt/c")) else None,
        Path("/mnt/d/ai_databases") if _safe_path_exists(Path("/mnt/d")) else None,
        Path.home() / "ai_databases",
        Path(__file__).parent.parent.parent / "data" / "ai_databases"
    ]
    
    # Find first accessible parent
    for path in candidate_paths:
        if path is not None:
            try:
                parent = path.parent
                if parent.exists() or parent == path:
                    return str(path)
            except (OSError, PermissionError):
                continue
    
    # Ultimate fallback
    return str(Path.cwd() / "ai_databases")

def get_default_knowledge_source_path() -> str:
    env_path = os.getenv("KNOWLEDGE_SOURCE_PATH")
    if env_path:
        return convert_to_docker_mount_path(env_path)
    
    if _safe_path_exists(Path("/data")):
        return "/data/knowledge_base"
    
    # Non-Docker: prefer user Documents
    home = Path.home()
    return str(home / "Documents")
