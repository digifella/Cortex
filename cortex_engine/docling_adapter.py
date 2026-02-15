"""
Docling backend selection helpers for ingestion.
"""

from __future__ import annotations

import os
from typing import Optional


VALID_INGEST_BACKENDS = {"default", "docling", "auto"}
VALID_MIGRATION_MODES = {"legacy", "docling", "hybrid", "gradual"}


def is_docker_environment() -> bool:
    """Detect Docker runtime."""
    return os.path.exists("/.dockerenv")


def resolve_ingest_backend(
    ingest_backend: Optional[str] = None,
    env_var: str = "CORTEX_INGEST_BACKEND",
) -> str:
    """
    Resolve ingest backend from explicit value or environment.

    Backends:
    - default: preserve Cortex defaults (legacy in Docker, gradual elsewhere)
    - docling: force docling migration mode
    - auto: force gradual migration mode with fallback
    """
    backend = (ingest_backend or os.getenv(env_var, "default")).strip().lower()
    if backend not in VALID_INGEST_BACKENDS:
        raise ValueError(
            f"Invalid ingest backend '{backend}'. Expected one of: {sorted(VALID_INGEST_BACKENDS)}"
        )
    return backend


def resolve_migration_mode(
    ingest_backend: Optional[str] = None,
    explicit_migration_mode: Optional[str] = None,
) -> str:
    """
    Resolve migration mode used by ingestion manager.

    Explicit migration mode still takes precedence for backward compatibility.
    """
    if explicit_migration_mode:
        mode = explicit_migration_mode.strip().lower()
        if mode not in VALID_MIGRATION_MODES:
            raise ValueError(
                f"Invalid migration mode '{mode}'. Expected one of: {sorted(VALID_MIGRATION_MODES)}"
            )
        return mode

    backend = resolve_ingest_backend(ingest_backend)
    if backend == "docling":
        return "docling"
    if backend == "auto":
        return "gradual"

    return "legacy" if is_docker_environment() else "gradual"
