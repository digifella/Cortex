from .utils.logging_utils import get_logger
import os
import subprocess

logger = get_logger(__name__)

class _SystemStatus:
    def get_setup_progress(self):
        # Minimal Docker-aware progress: assume UI is up; check Ollama and API
        ollama_running = False
        api_running = False
        errors = []

        try:
            res = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], capture_output=True)
            ollama_running = res.returncode == 0
        except Exception as e:
            errors.append(f"Ollama check failed: {e}")

        try:
            res = subprocess.run(["curl", "-s", "http://localhost:8000/health"], capture_output=True)
            api_running = res.returncode == 0
        except Exception as e:
            errors.append(f"API check failed: {e}")

        setup_complete = ollama_running and api_running
        progress = 100 if setup_complete else (50 if api_running or ollama_running else 10)

        return {
            "setup_complete": setup_complete,
            "progress_percent": progress,
            "status_message": "Services running" if setup_complete else "Starting services...",
            "ollama_running": ollama_running,
            "api_running": api_running,
            "models": [],
            "errors": errors,
        }

system_status = _SystemStatus()

