def check_ollama_service():
    # Simple optimistic check
    try:
        import subprocess
        res = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], capture_output=True)
        return (res.returncode == 0, None if res.returncode == 0 else "Service not reachable")
    except Exception as e:
        return (False, str(e))


def format_ollama_error_for_user(context: str, error_msg: str) -> str:
    return f"Ollama is not available for {context}. Error: {error_msg}. Ensure the service is running."

def get_ollama_status_message():
    ok, err = check_ollama_service()
    return "Running" if ok else f"Not running: {err}"

def get_ollama_instructions():
    return "Start Ollama inside the container or ensure the service is mapped."

