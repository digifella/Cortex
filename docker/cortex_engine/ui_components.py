from typing import Tuple

def llm_provider_selector(task: str, key: str, help_text: str) -> Tuple[str, dict]:
    # Minimal selector stub for Docker build
    display = "Ollama (Local)"
    status = {"ok": True, "provider": "ollama"}
    return display, status

