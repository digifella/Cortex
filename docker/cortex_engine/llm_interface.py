"""
LLM Interface
Version: 1.0.0
Date: 2026-01-05

Purpose: Simple interface for LLM operations.
"""

import ollama
from typing import Optional, Dict, Any
from .utils import get_logger

logger = get_logger(__name__)


class LLMInterface:
    """Simple LLM interface using Ollama."""

    def __init__(
        self,
        model: str = "mistral-small3.2",
        temperature: float = 0.7,
        request_timeout: float = 90.0,
    ):
        """
        Initialize LLM interface.

        Args:
            model: Model name
            temperature: Generation temperature
            request_timeout: Ollama HTTP timeout in seconds
        """
        self.model = model
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.client = ollama.Client(timeout=request_timeout)

        logger.info(
            f"LLMInterface initialized with model: {model} (timeout={request_timeout}s)"
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

            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": max_tokens or 2048,
                },
            )

            return response['message']['content']

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
