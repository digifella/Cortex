"""
LLM Interface
Version: 1.1.0
Date: 2026-02-15

Purpose: Simple interface for LLM operations with pluggable providers.
"""

import os
from typing import Optional, Dict, Any

import ollama
from openai import OpenAI
from .utils import get_logger

logger = get_logger(__name__)


class LLMInterface:
    """Simple LLM interface using Ollama or LM Studio (OpenAI-compatible)."""

    def __init__(
        self,
        model: str = "mistral-small3.2",
        temperature: float = 0.7,
        request_timeout: float = 90.0,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize LLM interface.

        Args:
            model: Model name
            temperature: Generation temperature
            request_timeout: HTTP timeout in seconds
            provider: Optional provider override (ollama|lmstudio)
            base_url: Optional provider base URL override
            api_key: Optional provider API key override
        """
        self.model = model
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.provider = (provider or os.getenv("CORTEX_LLM_PROVIDER", "ollama")).strip().lower()

        if self.provider == "lmstudio":
            lmstudio_base_url = (base_url or os.getenv("CORTEX_LMSTUDIO_BASE_URL", "http://localhost:1234/v1")).strip()
            lmstudio_api_key = api_key or os.getenv("CORTEX_LMSTUDIO_API_KEY", "lm-studio")
            self.client = OpenAI(base_url=lmstudio_base_url, api_key=lmstudio_api_key, timeout=request_timeout)
        elif self.provider == "ollama":
            self.client = ollama.Client(timeout=request_timeout)
        else:
            raise ValueError("Unsupported provider. Use 'ollama' or 'lmstudio'.")

        logger.info(
            f"LLMInterface initialized with provider={self.provider}, model={model} (timeout={request_timeout}s)"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Optional max tokens

        Returns:
            Generated text
        """
        try:
            messages = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            messages.append({
                "role": "user",
                "content": prompt
            })

            if self.provider == "lmstudio":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or 2048,
                )
                content = (response.choices[0].message.content or "").strip()
                return content

            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": max_tokens or 2048,
                },
            )
            return response["message"]["content"]

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_with_context(
        self,
        prompt: str,
        context: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text with additional context.

        Args:
            prompt: User prompt
            context: Context dictionary
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        # Format context into prompt
        context_str = "\n\n".join([
            f"**{key}:** {value}"
            for key, value in context.items()
        ])

        full_prompt = f"{context_str}\n\n{prompt}"

        return self.generate(full_prompt, system_prompt)
