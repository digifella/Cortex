"""
Modern Ollama LLM wrapper that uses updated API endpoints
Version: 2.0.0
Date: 2026-01-02

Replacement for the deprecated llama-index Ollama class that still uses /api/generate
Enhanced with proper async streaming, extended timeout support, and debug logging.
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


class ModernOllamaLLM(LLM):
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
        return "ModernOllamaLLM"
    
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
        return CompletionResponse(text=chat_response.message.content)
    
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
                    "num_predict": 512    # Sufficient for concise JSON metadata; lighter on resources
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
        """Stream completion using Ollama streaming API."""
        from llama_index.core.base.llms.types import ChatMessage
        messages = [ChatMessage(role="user", content=prompt)]
        yield from self.stream_chat(messages, **kwargs)

    @llm_completion_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        """Stream chat using Ollama streaming API."""
        try:
            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                    content = msg.content
                elif isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                else:
                    role = 'user'
                    content = str(msg)

                ollama_messages.append({
                    "role": role,
                    "content": content
                })

            # Build request payload with streaming enabled
            options = kwargs.get("options", {})
            if "temperature" in kwargs:
                options["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                options["top_p"] = kwargs["top_p"]
            if "num_predict" in kwargs:
                options["num_predict"] = kwargs["num_predict"]

            if not options:
                options = {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 2048
                }

            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": True,
                "options": options
            }

            # Make streaming request
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.request_timeout,
                stream=True
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")

            # Stream the response chunks
            from llama_index.core.llms import CompletionResponse
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            yield CompletionResponse(
                                text=content,
                                delta=content
                            )
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise RuntimeError(f"Failed to stream from Ollama: {e}")
    
    # Async methods required by LlamaIndex LLM interface
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async complete - direct implementation to avoid callback issues."""
        from llama_index.core.base.llms.types import ChatMessage
        messages = [ChatMessage(role="user", content=prompt)]

        # Call achat directly instead of going through sync methods
        chat_response = await self.achat(messages, **kwargs)
        return CompletionResponse(text=chat_response.message.content)
    
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async chat - direct async implementation."""
        import asyncio

        # For simplicity, run the sync version in a thread pool
        # This avoids callback validation issues in async context
        loop = asyncio.get_event_loop()

        # Call the internal logic directly without the callback decorator
        try:
            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                    content = msg.content
                elif isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                else:
                    role = 'user'
                    content = str(msg)

                ollama_messages.append({
                    "role": role,
                    "content": content
                })

            # Build payload
            options = kwargs.get("options", {})
            if "temperature" in kwargs:
                options["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                options["top_p"] = kwargs["top_p"]
            if "num_predict" in kwargs:
                options["num_predict"] = kwargs["num_predict"]

            if not options:
                options = {
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "num_predict": 512
                }

            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": options
            }

            # Make request in thread pool
            def _make_request():
                logger.info(f"Making Ollama request with timeout: {self.request_timeout}s")
                response = self._client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.request_timeout
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")

                data = response.json()

                if "message" in data and "content" in data["message"]:
                    content = data["message"]["content"]
                else:
                    raise RuntimeError(f"Unexpected Ollama response format: {data}")

                from llama_index.core.base.llms.types import ChatMessage
                return ChatResponse(
                    message=ChatMessage(role="assistant", content=content),
                    raw=data
                )

            return await loop.run_in_executor(None, _make_request)

        except Exception as e:
            logger.error(f"Ollama async chat error: {e}")
            raise RuntimeError(f"Failed to get response from Ollama: {e}")
    
    async def astream_complete(self, prompt: str, **kwargs: Any):
        """Async stream complete - yields completion chunks."""
        from llama_index.core.base.llms.types import ChatMessage
        messages = [ChatMessage(role="user", content=prompt)]
        async for chunk in self.astream_chat(messages, **kwargs):
            yield chunk

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any):
        """Async stream chat - yields chat response chunks."""
        import aiohttp

        try:
            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                    content = msg.content
                elif isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                else:
                    role = 'user'
                    content = str(msg)

                ollama_messages.append({
                    "role": role,
                    "content": content
                })

            # Build request payload with streaming enabled
            options = kwargs.get("options", {})
            if "temperature" in kwargs:
                options["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                options["top_p"] = kwargs["top_p"]
            if "num_predict" in kwargs:
                options["num_predict"] = kwargs["num_predict"]

            if not options:
                options = {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 2048
                }

            payload = {
                "model": self.model,
                "messages": ollama_messages,
                "stream": True,
                "options": options
            }

            # Make async streaming request
            # Use sock_read timeout instead of total timeout for streaming
            # This allows long-running streams but times out if no data received for 300 seconds
            timeout = aiohttp.ClientTimeout(
                total=None,  # No total timeout for streaming
                sock_read=300  # 5 minutes between chunks (for slow 70B models)
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"Ollama API error {response.status}: {error_text}")

                    # Stream the response chunks line by line
                    from llama_index.core.llms import CompletionResponse

                    # Read response line by line (Ollama sends newline-delimited JSON)
                    async for line_bytes in response.content:
                        if not line_bytes:
                            continue

                        try:
                            # Decode and parse the JSON line
                            line = line_bytes.decode('utf-8').strip()
                            if not line:
                                continue

                            data = json.loads(line)

                            # Check if this is a content chunk
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                if content:  # Only yield non-empty content
                                    yield CompletionResponse(
                                        text=content,
                                        delta=content
                                    )

                            # Check if stream is done
                            if data.get("done", False):
                                break

                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            # Skip malformed lines
                            logger.debug(f"Skipping malformed streaming line: {e}")
                            continue

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Async Ollama streaming error: {e}")
            logger.error(f"Full traceback: {error_details}")
            raise RuntimeError(f"Failed to async stream from Ollama: {str(e)}")


def create_modern_ollama_llm(model: str = "mistral", base_url: str = "http://localhost:11434", request_timeout: float = 120.0) -> ModernOllamaLLM:
    """Factory function to create a modern Ollama LLM instance."""
    return ModernOllamaLLM(
        model=model,
        base_url=base_url,
        request_timeout=request_timeout
    )
