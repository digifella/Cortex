import os
from pathlib import Path
from .path_utils import convert_to_docker_mount_path

def get_default_ai_database_path() -> str:
    # Prefer container env path
    env_path = os.getenv("AI_DATABASE_PATH")
    if env_path:
        return convert_to_docker_mount_path(env_path)
    # Fallback to standard mount used in docker-compose
    return "/data/ai_databases" if Path("/data").exists() else "/home/cortex/data/ai_databases"

def get_default_knowledge_source_path() -> str:
    env_path = os.getenv("KNOWLEDGE_SOURCE_PATH")
    if env_path:
        return convert_to_docker_mount_path(env_path)
    return "/data/knowledge_base" if Path("/data").exists() else "/home/cortex/data/knowledge_base"

