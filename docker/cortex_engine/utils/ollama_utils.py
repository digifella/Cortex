import os
from typing import Optional, Tuple



def get_ollama_base_url() -> str:
    # Prefer explicit env, then config default, then common host endpoints
    env_url = os.getenv("OLLAMA_BASE_URL")
    return env_url or "http://localhost:11434"


def check_ollama_service(base_url: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
    """Probe Ollama API and return (ok, error_message, resolved_base_url). Tries common Docker host addresses."""
    candidates = []
    if base_url:
        candidates.append(base_url.rstrip('/'))
    # Try env/default
    candidates.append(get_ollama_base_url().rstrip('/'))
    # Docker host fallback commonly available in Desktop
    candidates.extend([
        "http://host.docker.internal:11434",
        "http://localhost:11434",
    ])

    tried = []
    for url in candidates:
        if not url or url in tried:
            continue
        tried.append(url)
        try:
            import requests  # Local import to avoid hard dependency at module import time
            resp = requests.get(f"{url}/api/tags", timeout=2)
            if resp.status_code == 200:
                # Minimal JSON presence check
                _ = resp.json()
                return True, None, url
        except Exception as e:
            last_err = str(e)
            continue

    return False, last_err if 'last_err' in locals() else "Service not reachable", None


def format_ollama_error_for_user(context: str, error_msg: str) -> str:
    return f"Ollama is not available for {context}. Error: {error_msg}. Ensure the service is running or reachable from the container."

def get_ollama_status_message():
    ok, err, url = check_ollama_service()
    return f"Running at {url}" if ok else f"Not running: {err}"

def get_ollama_instructions():
    return "Start Ollama inside the container or ensure the service is mapped."
