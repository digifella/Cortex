"""
Modern Ollama LLM wrapper that uses updated API endpoints
Replacement for the deprecated llama-index Ollama class that still uses /api/generate
"""

import json
import requests
from typing import Dict, Any, Optional, Sequence
from llama_index.core.llms import LLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from .logging_utils import get_logger

logger = get_logger(__name__)


class ModernOllama(LLM):
    """Modern Ollama LLM implementation using /api/chat endpoint."""
    
    model: str = Field(description="The Ollama model to use.")
    base_url: str = Field(default="http://localhost:11434", description="Base URL for Ollama API.")
    request_timeout: float = Field(default=120.0, description="Request timeout in seconds.")
    
    _client: Optional[requests.Session] = PrivateAttr()
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        request_timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            request_timeout=request_timeout,
            **kwargs,
        )
        self._client = requests.Session()
    
    @classmethod
    def class_name(cls) -> str:
        return "ModernOllama"
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=4096,  # Default, should be model-specific
            num_output=1024,      # Default, should be model-specific
            model_name=self.model,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete a prompt using Ollama chat API."""
        from llama_index.core.base.llms.types import ChatMessage
        messages = [ChatMessage(role="user", content=prompt)]
        chat_response = self.chat(messages, **kwargs)
        return CompletionResponse(
            text=chat_response.message.content,
            raw=chat_response.raw
        )
    
    @llm_completion_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat with the model using modern Ollama API."""
        try:
            # Convert LlamaIndex ChatMessage to Ollama format
            ollama_messages = []
            for msg in messages:
                # Handle both ChatMessage objects and dict objects
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # ChatMessage object
                    role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                    content = msg.content
                elif isinstance(msg, dict):
                    # Dict object
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                else:
                    # Fallback - treat as string content
                    role = 'user'
                    content = str(msg)
                
                ollama_messages.append({
                    "role": role,
                    "content": content
                })
            
            # Extract model parameters from kwargs and put them in options
            options = kwargs.get("options", {})
            
            # Handle common LlamaIndex parameters by mapping them to Ollama options
            if "temperature" in kwargs:
                options["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                options["top_p"] = kwargs["top_p"]
            if "num_predict" in kwargs:
                options["num_predict"] = kwargs["num_predict"]
            
            # Set some good defaults for proposals if not specified
            if not options:
                options = {
                    "temperature": 0.3,    # Lower temperature for consistent outputs
                    "top_p": 0.8,         # Reduce randomness 
                    "num_predict": 2048   # Reasonable output length
                }
            
            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": options
            }
            
            # Make request to modern /api/chat endpoint
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.request_timeout
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")
            
            data = response.json()
            
            # Extract response from modern API format
            if "message" in data and "content" in data["message"]:
                content = data["message"]["content"]
            else:
                raise RuntimeError(f"Unexpected Ollama response format: {data}")
            
            # Return in LlamaIndex format
            from llama_index.core.base.llms.types import ChatMessage
            return ChatResponse(
                message=ChatMessage(role="assistant", content=content),
                raw=data
            )
            
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise RuntimeError(f"Failed to get response from Ollama: {e}")
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """Stream completion - not implemented for simplicity."""
        raise NotImplementedError("Streaming not implemented in ModernOllama")
    
    @llm_completion_callback() 
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        """Stream chat - not implemented for simplicity."""
        raise NotImplementedError("Streaming not implemented in ModernOllama")
    
    # Async methods required by LlamaIndex LLM interface
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async complete - delegates to sync method."""
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async chat - delegates to sync method.""" 
        return self.chat(messages, **kwargs)
    
    async def astream_complete(self, prompt: str, **kwargs: Any):
        """Async stream complete - not implemented."""
        raise NotImplementedError("Async streaming not implemented in ModernOllama")
    
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        """Async stream chat - not implemented."""
        raise NotImplementedError("Async streaming not implemented in ModernOllama")


def create_modern_ollama_llm(model: str = "mistral", base_url: str = "http://localhost:11434", request_timeout: float = 120.0) -> ModernOllama:
    """Factory function to create a modern Ollama LLM instance."""
    return ModernOllama(
        model=model,
        base_url=base_url,
        request_timeout=request_timeout
    )