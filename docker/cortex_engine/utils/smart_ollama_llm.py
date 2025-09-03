def create_smart_ollama_llm(model: str = "mistral:7b-instruct-v0.3-q4_K_M", request_timeout: float = 120.0):
    # Lightweight shim; real LLM client setup is optional in Docker UI
    class _DummyLLM:
        model = model
        request_timeout = request_timeout
    return _DummyLLM()

