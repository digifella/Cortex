import os
from typing import Optional


from .ollama_utils import get_ollama_base_url


class _OllamaLLM:
    def __init__(self, model: str, base_url: Optional[str] = None, request_timeout: float = 120.0):
        self.model = model
        self.base_url = (base_url or get_ollama_base_url()).rstrip('/')
        self.request_timeout = request_timeout

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            import requests  # Local import to avoid hard dependency at module import time
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            # Newer API returns {'response': '...'} for non-streaming
            if isinstance(data, dict) and 'response' in data:
                return data['response']
            # Fallback: return raw
            return str(data)
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")


def create_smart_ollama_llm(model: str = "mistral:7b-instruct-v0.3-q4_K_M", request_timeout: float = 120.0, base_url: Optional[str] = None):
    return _OllamaLLM(model=model, base_url=base_url, request_timeout=request_timeout)
