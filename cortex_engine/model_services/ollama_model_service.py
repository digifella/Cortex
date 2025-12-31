"""
Ollama Model Service Implementation
Provides compatibility with existing Ollama-based model management.
"""

import asyncio
import json
import aiohttp
import re
from typing import List, Optional, Dict, Any, AsyncIterator

from .interfaces import ModelServiceInterface, ModelInfo, ModelStatus, ModelDownloadProgress
from ..utils.logging_utils import get_logger
from ..utils.ollama_utils import check_ollama_service

logger = get_logger(__name__)


class OllamaModelService(ModelServiceInterface):
    """Ollama backend implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._session = None
    
    @property
    def backend_name(self) -> str:
        return "ollama"
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def is_available(self) -> bool:
        """Check if Ollama service is available."""
        is_running, _, _ = check_ollama_service(self.host, self.port)
        return is_running
    
    async def list_available_models(self) -> List[ModelInfo]:
        """List locally available Ollama models."""
        if not await self.is_available():
            return []
        
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    logger.error(f"Failed to list Ollama models: {response.status}")
                    return []
                
                data = await response.json()
                models = []
                
                for model_data in data.get("models", []):
                    name = model_data.get("name", "")
                    parts = name.split(":")
                    model_name = parts[0]
                    tag = parts[1] if len(parts) > 1 else "latest"
                    
                    # Convert size from bytes to GB
                    size_bytes = model_data.get("size", 0)
                    size_gb = size_bytes / (1024**3)
                    
                    models.append(ModelInfo(
                        name=model_name,
                        tag=tag,
                        size_gb=size_gb,
                        status=ModelStatus.AVAILABLE,
                        backend=self.backend_name,
                        description=f"Ollama model: {name}",
                        performance_tier="standard"
                    ))
                
                return models
                
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                logger.warning("Event loop closed, falling back to synchronous request")
                return self._list_models_sync()
            else:
                logger.error(f"Failed to list Ollama models: {e}")
                return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    def _list_models_sync(self) -> List[ModelInfo]:
        """Synchronous fallback for listing models when event loop is closed."""
        try:
            import requests
            
            # Check if service is available
            is_running, _, _ = check_ollama_service(self.host, self.port)
            if not is_running:
                return []
            
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to list Ollama models: {response.status_code}")
                return []
            
            data = response.json()
            models = []
            
            for model_data in data.get("models", []):
                name = model_data.get("name", "")
                parts = name.split(":")
                model_name = parts[0]
                tag = parts[1] if len(parts) > 1 else "latest"
                
                # Convert size from bytes to GB
                size_bytes = model_data.get("size", 0)
                size_gb = size_bytes / (1024**3)
                
                models.append(ModelInfo(
                    name=model_name,
                    tag=tag,
                    size_gb=size_gb,
                    status=ModelStatus.AVAILABLE,
                    backend=self.backend_name,
                    description=f"Ollama model: {name}",
                    performance_tier="standard"
                ))
            
            return models
            
        except Exception as e:
            logger.error(f"Synchronous model list failed: {e}")
            return []
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        models = await self.list_available_models()
        
        # Handle both "model:tag" and "model" formats
        for model in models:
            if model.full_name == model_name or model.name == model_name:
                return model
        
        return None
    
    async def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is locally available."""
        model_info = await self.get_model_info(model_name)
        return model_info is not None
    
    async def pull_model(self, model_name: str) -> AsyncIterator[ModelDownloadProgress]:
        """Download/pull a model with progress tracking."""
        if not await self.is_available():
            yield ModelDownloadProgress(
                model_name=model_name,
                total_size=0,
                downloaded_size=0,
                status="Ollama service not available"
            )
            return
        
        try:
            session = await self._get_session()
            
            payload = {"name": model_name}
            
            async with session.post(
                f"{self.base_url}/api/pull",
                json=payload
            ) as response:
                
                if response.status != 200:
                    yield ModelDownloadProgress(
                        model_name=model_name,
                        total_size=0,
                        downloaded_size=0,
                        status=f"Pull failed with status {response.status}"
                    )
                    return
                
                total_size = 0
                downloaded_size = 0
                
                async for line in response.content:
                    try:
                        line_str = line.decode().strip()
                        if not line_str:
                            continue
                        
                        data = json.loads(line_str)
                        status = data.get("status", "")
                        
                        # Extract progress information
                        if "total" in data:
                            total_size = data["total"]
                        if "completed" in data:
                            downloaded_size = data["completed"]
                        
                        yield ModelDownloadProgress(
                            model_name=model_name,
                            total_size=total_size,
                            downloaded_size=downloaded_size,
                            status=status
                        )
                        
                        # Check if download is complete
                        if status == "success" or "successfully" in status.lower():
                            break
                            
                    except json.JSONDecodeError:
                        # Skip non-JSON lines
                        continue
                    except Exception as e:
                        logger.error(f"Error processing pull progress: {e}")
                        continue
                
        except Exception as e:
            logger.error(f"Model pull failed: {e}")
            yield ModelDownloadProgress(
                model_name=model_name,
                total_size=0,
                downloaded_size=0,
                status=f"Error: {str(e)}"
            )
    
    async def remove_model(self, model_name: str) -> bool:
        """Remove a model from local storage."""
        if not await self.is_available():
            return False
        
        try:
            session = await self._get_session()
            payload = {"name": model_name}
            
            async with session.delete(
                f"{self.base_url}/api/delete",
                json=payload
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Failed to remove model {model_name}: {e}")
            return False
    
    async def get_model_endpoint(self, model_name: str) -> Optional[str]:
        """Get the API endpoint URL for a model."""
        if await self.is_model_available(model_name):
            return f"{self.base_url}/api/chat"
        return None
    
    async def test_model_inference(self, model_name: str, test_prompt: str = "Hello") -> bool:
        """Test if a model can perform inference."""
        if not await self.is_model_available(model_name):
            return False
        
        try:
            session = await self._get_session()
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": test_prompt}],
                "stream": False
            }
            
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status != 200:
                    return False
                
                data = await response.json()
                return "message" in data and "content" in data.get("message", {})
                
        except Exception as e:
            logger.error(f"Model inference test failed: {e}")
            return False
    
    async def get_performance_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a model."""
        if not await self.is_model_available(model_name):
            return {}
        
        return {
            "backend": self.backend_name,
            "execution_mode": "containerized",
            "performance_tier": "standard",
            "gpu_acceleration": False,  # Depends on Ollama configuration
            "memory_efficiency": "standard",
            "maturity": "high",
            "community_support": "excellent"
        }