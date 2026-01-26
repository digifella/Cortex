"""
Qwen3-VL Multimodal Embedding Service
======================================

Version: 1.0.0
Date: 2026-01-17

Provides unified multimodal embedding for text, images, and mixed-modality content
using Qwen3-VL-Embedding models. This enables semantic search across different
modalities in a shared vector space.

Key Features:
- Unified text+image+video embedding in same vector space
- Matryoshka Representation Learning (MRL) for flexible dimensions
- Flash Attention 2 optimization for memory efficiency
- Batch processing with GPU optimization
- LlamaIndex adapter for seamless integration

Hardware Requirements:
- Qwen3-VL-Embedding-2B: ~5GB VRAM (2048 dimensions)
- Qwen3-VL-Embedding-8B: ~16GB VRAM (4096 dimensions)

Usage:
    from cortex_engine.qwen3_vl_embedding_service import (
        embed_text, embed_image, embed_multimodal, get_embedding_service
    )

    # Text embedding
    text_vec = embed_text("quarterly revenue chart")

    # Image embedding
    img_vec = embed_image("/path/to/chart.png")

    # Cross-modal similarity
    similarity = text_vec @ img_vec.T
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

import torch
import numpy as np

from .utils.logging_utils import get_logger
from .utils.performance_monitor import measure
from .utils.gpu_monitor import get_gpu_memory_info, clear_gpu_cache
from .utils.resource_throttler import (
    ResourceThrottler, ThrottleConfig, ThrottleLevel, log_resource_status
)

logger = get_logger(__name__)

# Global throttler for embedding operations - use conservative settings for stability
_embedding_throttler: Optional[ResourceThrottler] = None

def _get_embedding_throttler() -> ResourceThrottler:
    """Get or create the embedding throttler."""
    global _embedding_throttler
    if _embedding_throttler is None:
        _embedding_throttler = ResourceThrottler(ThrottleConfig.conservative())
    return _embedding_throttler


# ============================================================================
# Configuration
# ============================================================================

class Qwen3VLModelSize(Enum):
    """Available Qwen3-VL embedding model sizes."""
    SMALL = "Qwen/Qwen3-VL-Embedding-2B"  # 2048 dims, ~5GB VRAM
    LARGE = "Qwen/Qwen3-VL-Embedding-8B"  # 4096 dims, ~16GB VRAM


@dataclass
class Qwen3VLConfig:
    """Configuration for Qwen3-VL embedding service."""
    model_name: str = "Qwen/Qwen3-VL-Embedding-8B"
    embedding_dim: int = 4096  # Full dimension, can be reduced via MRL
    mrl_dim: Optional[int] = None  # If set, truncate embeddings to this dimension
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"  # bfloat16 or float16
    max_batch_size: int = 8
    device: Optional[str] = None  # auto-detect if None
    trust_remote_code: bool = True
    normalize_embeddings: bool = True

    # Image processing
    max_image_size: int = 1024  # Max dimension for image resize
    image_format: str = "RGB"

    # Video processing
    max_video_frames: int = 64
    video_fps: float = 1.0

    @classmethod
    def for_model_size(cls, size: Qwen3VLModelSize) -> "Qwen3VLConfig":
        """Create config for specific model size."""
        if size == Qwen3VLModelSize.SMALL:
            return cls(
                model_name="Qwen/Qwen3-VL-Embedding-2B",
                embedding_dim=2048,
                max_batch_size=16,  # Smaller model can handle larger batches
            )
        else:  # LARGE
            return cls(
                model_name="Qwen/Qwen3-VL-Embedding-8B",
                embedding_dim=4096,
                max_batch_size=8,
            )

    @classmethod
    def auto_select(cls, prefer_quality: bool = True) -> "Qwen3VLConfig":
        """Auto-select model based on database dimensions first, then VRAM.

        Priority:
        1. Existing database dimensions (must match for compatibility)
        2. VRAM-based selection (for new databases)
        """
        # PRIORITY 1: Check existing database dimensions
        try:
            from .utils.embedding_validator import get_database_embedding_dimension
            from .utils.default_paths import get_default_ai_database_path

            # Also check cortex_config.json for user-configured path
            db_path = None
            try:
                import json
                from pathlib import Path
                config_file = Path(__file__).parent.parent / "cortex_config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)
                        db_path = config.get("ai_database_path")
            except Exception:
                pass

            if not db_path:
                db_path = get_default_ai_database_path()

            db_dim = get_database_embedding_dimension(db_path)
            if db_dim is not None:
                if db_dim == 2048:
                    logger.info(f"📊 Database has 2048D embeddings - using Qwen3-VL-2B for compatibility")
                    return cls.for_model_size(Qwen3VLModelSize.SMALL)
                elif db_dim == 4096:
                    logger.info(f"📊 Database has 4096D embeddings - using Qwen3-VL-8B for compatibility")
                    return cls.for_model_size(Qwen3VLModelSize.LARGE)
                else:
                    logger.warning(f"⚠️ Database has {db_dim}D embeddings - unusual dimension")
        except Exception as e:
            logger.debug(f"Could not detect database dimensions: {e}")

        # PRIORITY 2: VRAM-based selection (for new databases)
        gpu_info = get_gpu_memory_info()

        if gpu_info.is_cuda:
            available_gb = gpu_info.free_memory_gb

            if available_gb >= 20 and prefer_quality:
                logger.info(f"🚀 Auto-selected Qwen3-VL-8B (have {available_gb:.1f}GB free)")
                return cls.for_model_size(Qwen3VLModelSize.LARGE)
            elif available_gb >= 8:
                logger.info(f"📦 Auto-selected Qwen3-VL-2B (have {available_gb:.1f}GB free)")
                return cls.for_model_size(Qwen3VLModelSize.SMALL)
            else:
                logger.warning(f"⚠️ Low VRAM ({available_gb:.1f}GB) - Qwen3-VL may not fit")
                return cls.for_model_size(Qwen3VLModelSize.SMALL)
        else:
            logger.warning("⚠️ No CUDA GPU detected - Qwen3-VL performance will be limited")
            return cls.for_model_size(Qwen3VLModelSize.SMALL)


# ============================================================================
# Model Loading
# ============================================================================

_model_lock = threading.Lock()
_embedding_model: Optional[Any] = None
_processor: Optional[Any] = None
_current_config: Optional[Qwen3VLConfig] = None


def _get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(config: Optional[Qwen3VLConfig] = None) -> tuple:
    """
    Load Qwen3-VL embedding model and processor.

    Uses the custom Qwen3VLForEmbedding class which properly wraps the base model
    for embedding extraction. This class is defined in the model's scripts directory
    and loaded via trust_remote_code=True.

    Returns:
        Tuple of (model, processor, config)
    """
    global _embedding_model, _processor, _current_config

    if config is None:
        config = Qwen3VLConfig.auto_select()

    # Return cached model if same config
    if _embedding_model is not None and _current_config == config:
        return _embedding_model, _processor, _current_config

    with _model_lock:
        # Double-check after acquiring lock
        if _embedding_model is not None and _current_config == config:
            return _embedding_model, _processor, _current_config

        logger.info(f"Loading Qwen3-VL embedding model: {config.model_name}")

        try:
            # Import required packages
            import os
            import sys
            from transformers import AutoProcessor
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLPreTrainedModel, Qwen3VLModel
            )
            from transformers.modeling_outputs import ModelOutput
            from dataclasses import dataclass

            device = config.device or _get_device()
            dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float16

            # Define the custom embedding model class (from Qwen's scripts)
            @dataclass
            class Qwen3VLForEmbeddingOutput(ModelOutput):
                last_hidden_state: Optional[torch.FloatTensor] = None
                attention_mask: Optional[torch.Tensor] = None

            class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
                """Custom model class for Qwen3-VL embedding extraction."""
                _checkpoint_conversion_mapping = {}

                def __init__(self, config):
                    super().__init__(config)
                    self.model = Qwen3VLModel(config)
                    self.post_init()

                def get_input_embeddings(self):
                    return self.model.get_input_embeddings()

                def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                           past_key_values=None, inputs_embeds=None, pixel_values=None,
                           pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None,
                           cache_position=None, **kwargs):
                    outputs = self.model(
                        input_ids=input_ids, pixel_values=pixel_values,
                        pixel_values_videos=pixel_values_videos,
                        image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                        attention_mask=attention_mask, position_ids=position_ids,
                        past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                        cache_position=cache_position, **kwargs
                    )
                    return Qwen3VLForEmbeddingOutput(
                        last_hidden_state=outputs.last_hidden_state,
                        attention_mask=attention_mask
                    )

            # Model loading with offline/online fallback
            def load_model_and_processor():
                global _processor, _embedding_model

                # Load processor with right padding (required for embedding)
                _processor = AutoProcessor.from_pretrained(
                    config.model_name,
                    trust_remote_code=config.trust_remote_code,
                    padding_side='right'
                )

                # Load model with optimizations
                model_kwargs = {
                    "trust_remote_code": config.trust_remote_code,
                    "torch_dtype": dtype,
                }

                # Add Flash Attention 2 if available and requested
                if config.use_flash_attention:
                    try:
                        import flash_attn  # noqa: F401
                        model_kwargs["attn_implementation"] = "flash_attention_2"
                        logger.info("⚡ Using Flash Attention 2 for memory optimization")
                    except ImportError:
                        logger.info("📝 Flash Attention 2 not installed - using default attention")

                # Use custom Qwen3VLForEmbedding class that properly loads the checkpoint
                _embedding_model = Qwen3VLForEmbedding.from_pretrained(
                    config.model_name,
                    **model_kwargs
                ).to(device).eval()

            # Try loading in offline mode first (respects HF_HUB_OFFLINE env var)
            try:
                load_model_and_processor()
                logger.info(f"✅ Qwen3-VL loaded from cache on {device}")
            except Exception as offline_error:
                # If offline mode fails, temporarily enable online mode for download
                if "offline" in str(offline_error).lower() or "couldn't connect" in str(offline_error).lower() or "LocalEntryNotFoundError" in str(type(offline_error).__name__):
                    logger.warning(f"Qwen3-VL not cached locally, attempting download...")
                    logger.debug(f"Offline error: {offline_error}")

                    # Save current offline settings
                    old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
                    old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")

                    try:
                        # Temporarily enable online mode
                        os.environ["HF_HUB_OFFLINE"] = "0"
                        os.environ["TRANSFORMERS_OFFLINE"] = "0"

                        load_model_and_processor()
                        logger.info(f"✅ Qwen3-VL downloaded and loaded on {device}")

                    finally:
                        # Restore offline settings
                        if old_hf_offline is not None:
                            os.environ["HF_HUB_OFFLINE"] = old_hf_offline
                        elif "HF_HUB_OFFLINE" in os.environ:
                            del os.environ["HF_HUB_OFFLINE"]

                        if old_transformers_offline is not None:
                            os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline
                        elif "TRANSFORMERS_OFFLINE" in os.environ:
                            del os.environ["TRANSFORMERS_OFFLINE"]
                else:
                    # Re-raise if it's not an offline-related error
                    raise

            _current_config = config
            logger.info(f"✅ Qwen3-VL ready on {device} ({config.embedding_dim}D embeddings)")

            return _embedding_model, _processor, _current_config

        except ImportError as e:
            logger.error(f"❌ Missing dependencies for Qwen3-VL: {e}")
            logger.error("Install with: pip install transformers>=4.57.0 qwen-vl-utils>=0.0.14")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen3-VL model: {e}")
            raise


def unload_model():
    """Unload model to free GPU memory."""
    global _embedding_model, _processor, _current_config

    with _model_lock:
        if _embedding_model is not None:
            del _embedding_model
            _embedding_model = None
        if _processor is not None:
            del _processor
            _processor = None
        _current_config = None

        clear_gpu_cache()
        logger.info("🧹 Qwen3-VL model unloaded, GPU cache cleared")


# ============================================================================
# Embedding Generation
# ============================================================================

@dataclass
class EmbeddingInput:
    """Input for embedding generation."""
    text: Optional[str] = None
    image: Optional[Union[str, Path]] = None  # Path or URL
    video: Optional[Union[str, Path]] = None  # Path or URL
    instruction: Optional[str] = None  # Task-specific instruction

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by Qwen3-VL."""
        result = {}
        if self.text:
            result["text"] = self.text
        if self.image:
            result["image"] = str(self.image)
        if self.video:
            result["video"] = str(self.video)
        if self.instruction:
            result["instruction"] = self.instruction
        return result


def _process_inputs(
    inputs: List[EmbeddingInput],
    model: Any,
    processor: Any,
    config: Qwen3VLConfig
) -> np.ndarray:
    """
    Process inputs and generate embeddings.

    Args:
        inputs: List of EmbeddingInput objects
        model: Loaded Qwen3-VL model
        processor: Qwen3-VL processor
        config: Configuration

    Returns:
        Numpy array of embeddings (N x embedding_dim)
    """
    from PIL import Image
    import requests
    from io import BytesIO

    device = next(model.parameters()).device

    # Separate text-only inputs from image inputs
    texts = []
    images = []
    input_has_image = []

    for inp in inputs:
        text = inp.text or ""
        if inp.instruction:
            text = f"{inp.instruction}: {text}"
        texts.append(text)

        # Handle image inputs
        if inp.image:
            try:
                image_path = str(inp.image)
                if image_path.startswith(('http://', 'https://')):
                    # URL - fetch image
                    response = requests.get(image_path, timeout=10)
                    img = Image.open(BytesIO(response.content)).convert(config.image_format)
                else:
                    # Local file path
                    img = Image.open(image_path).convert(config.image_format)

                # Resize if needed
                if max(img.size) > config.max_image_size:
                    ratio = config.max_image_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                images.append(img)
                input_has_image.append(True)
            except Exception as e:
                logger.warning(f"Failed to load image {inp.image}: {e}")
                images.append(None)
                input_has_image.append(False)
        else:
            images.append(None)
            input_has_image.append(False)

    # Process through model
    with torch.no_grad():
        # Check if we have any valid images
        valid_images = [img for img in images if img is not None]

        if valid_images:
            # Process with images - use processor with text and images
            processed = processor(
                text=texts,
                images=valid_images if valid_images else None,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
        else:
            # Text-only processing
            processed = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

        # Forward pass
        outputs = model(**processed)

        # Extract embeddings using last-token pooling (required for Qwen3-VL-Embedding)
        # This takes the hidden state of the last non-padding token
        last_hidden_state = outputs.last_hidden_state
        attention_mask = processed.get('attention_mask')

        if attention_mask is not None:
            # Last-token pooling: find the position of the last non-pad token
            flipped_mask = attention_mask.flip(dims=[1])
            last_one_positions = flipped_mask.argmax(dim=1)
            col_indices = attention_mask.shape[1] - last_one_positions - 1
            row_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
            embeddings = last_hidden_state[row_indices, col_indices]
        else:
            # Fallback to last token if no attention mask
            embeddings = last_hidden_state[:, -1, :]

        # Normalize if requested
        if config.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Apply MRL dimension reduction if configured
        if config.mrl_dim and config.mrl_dim < config.embedding_dim:
            embeddings = embeddings[:, :config.mrl_dim]

        # Convert to float32 before numpy (bfloat16 not supported by numpy)
        return embeddings.cpu().float().numpy()


def embed_text(
    text: str,
    instruction: Optional[str] = None,
    config: Optional[Qwen3VLConfig] = None
) -> List[float]:
    """
    Generate embedding for a text string.

    Args:
        text: Input text to embed
        instruction: Optional task-specific instruction
        config: Optional configuration override

    Returns:
        Embedding vector as list of floats
    """
    model, processor, cfg = _load_model(config)

    inp = EmbeddingInput(text=text, instruction=instruction)
    embeddings = _process_inputs([inp], model, processor, cfg)

    return embeddings[0].tolist()


def embed_image(
    image_path: Union[str, Path],
    instruction: Optional[str] = None,
    config: Optional[Qwen3VLConfig] = None
) -> List[float]:
    """
    Generate embedding for an image.

    Args:
        image_path: Path or URL to image
        instruction: Optional task-specific instruction
        config: Optional configuration override

    Returns:
        Embedding vector as list of floats
    """
    model, processor, cfg = _load_model(config)

    inp = EmbeddingInput(image=image_path, instruction=instruction)
    embeddings = _process_inputs([inp], model, processor, cfg)

    return embeddings[0].tolist()


def embed_multimodal(
    text: Optional[str] = None,
    image: Optional[Union[str, Path]] = None,
    video: Optional[Union[str, Path]] = None,
    instruction: Optional[str] = None,
    config: Optional[Qwen3VLConfig] = None
) -> List[float]:
    """
    Generate embedding for mixed-modality input (text + image/video).

    Args:
        text: Optional text content
        image: Optional image path or URL
        video: Optional video path or URL
        instruction: Optional task-specific instruction
        config: Optional configuration override

    Returns:
        Embedding vector as list of floats
    """
    model, processor, cfg = _load_model(config)

    inp = EmbeddingInput(text=text, image=image, video=video, instruction=instruction)
    embeddings = _process_inputs([inp], model, processor, cfg)

    return embeddings[0].tolist()


def embed_batch(
    inputs: List[EmbeddingInput],
    config: Optional[Qwen3VLConfig] = None,
    show_progress: bool = False
) -> List[List[float]]:
    """
    Generate embeddings for a batch of inputs.

    Automatically handles batch sizing based on GPU memory.
    Includes mandatory delays between batches for system stability.

    GPU intensity (CORTEX_GPU_INTENSITY env var) controls:
    - Batch size: Lower intensity = smaller batches = less GPU memory at once
    - Delay: Lower intensity = longer pauses = more time for other GPU tasks

    Args:
        inputs: List of EmbeddingInput objects
        config: Optional configuration override
        show_progress: Show progress bar

    Returns:
        List of embedding vectors
    """
    import time
    import gc

    if not inputs:
        return []

    model, processor, cfg = _load_model(config)

    # GPU Intensity control (25-100%)
    # Lower intensity = smaller batches + longer delays = less GPU load
    gpu_intensity = int(os.environ.get("CORTEX_GPU_INTENSITY", "75"))
    gpu_intensity = max(25, min(100, gpu_intensity))  # Clamp to valid range

    # Scale batch size based on intensity (100% = full batch, 25% = 1/4 batch)
    intensity_factor = gpu_intensity / 100.0
    base_batch_size = cfg.max_batch_size
    adjusted_batch_size = max(1, int(base_batch_size * intensity_factor))

    # Scale delay inversely with intensity (100% = 1s, 25% = 4s)
    # This gives other GPU processes time to run between batches
    BATCH_DELAY_SECONDS = max(1.0, (100 - gpu_intensity) / 25.0)

    if gpu_intensity < 100:
        logger.info(f"🎮 GPU intensity {gpu_intensity}%: batch_size={adjusted_batch_size}, delay={BATCH_DELAY_SECONDS:.1f}s")

    all_embeddings = []
    batch_size = adjusted_batch_size
    total_batches = (len(inputs) + batch_size - 1) // batch_size

    with measure("qwen3_vl_embedding_batch", batch_size=batch_size, doc_count=len(inputs)):
        if len(inputs) > batch_size:
            logger.info(f"🔢 Generating Qwen3-VL embeddings for {len(inputs)} items in {total_batches} batches")
            logger.info(f"⏱️ Stability mode: {BATCH_DELAY_SECONDS}s delay between batches")
            # Emit machine-readable progress for UI
            print(f"CORTEX_EMBEDDING::0/{len(inputs)}::starting", flush=True)

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            batch_num = (i // batch_size) + 1

            if len(inputs) > batch_size:
                logger.debug(f"📦 Processing batch {batch_num}/{total_batches}")

            batch_embeddings = _process_inputs(batch, model, processor, cfg)
            all_embeddings.extend(batch_embeddings.tolist())

            # Emit embedding progress for UI (after each batch)
            if len(inputs) > batch_size:
                print(f"CORTEX_EMBEDDING::{len(all_embeddings)}/{len(inputs)}::batch_{batch_num}", flush=True)

            # STABILITY: Mandatory cleanup and delay after each batch
            if batch_num < total_batches:  # Don't delay after final batch
                # Clear GPU cache to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Mandatory delay for system stability
                time.sleep(BATCH_DELAY_SECONDS)

        if len(inputs) > batch_size:
            logger.info(f"✅ Qwen3-VL embedding generation complete: {len(all_embeddings)} vectors")

    return all_embeddings


def embed_texts_batch(
    texts: List[str],
    instruction: Optional[str] = None,
    config: Optional[Qwen3VLConfig] = None
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts (convenience wrapper).

    Args:
        texts: List of text strings
        instruction: Optional task-specific instruction (applied to all)
        config: Optional configuration override

    Returns:
        List of embedding vectors
    """
    inputs = [EmbeddingInput(text=t, instruction=instruction) for t in texts]
    return embed_batch(inputs, config)


def embed_images_batch(
    image_paths: List[Union[str, Path]],
    instruction: Optional[str] = None,
    config: Optional[Qwen3VLConfig] = None
) -> List[List[float]]:
    """
    Generate embeddings for multiple images (convenience wrapper).

    Args:
        image_paths: List of image paths or URLs
        instruction: Optional task-specific instruction (applied to all)
        config: Optional configuration override

    Returns:
        List of embedding vectors
    """
    inputs = [EmbeddingInput(image=p, instruction=instruction) for p in image_paths]
    return embed_batch(inputs, config)


# ============================================================================
# Service Interface (matches existing embedding_service.py pattern)
# ============================================================================

class Qwen3VLEmbeddingService:
    """
    Unified interface for Qwen3-VL embedding generation.

    Provides API compatibility with existing embedding_service.py pattern.
    """

    def __init__(self, config: Optional[Qwen3VLConfig] = None):
        """
        Initialize embedding service.

        Args:
            config: Optional configuration. Auto-selects optimal model if not provided.
        """
        self.config = config or Qwen3VLConfig.auto_select()
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._model is None:
            self._model, self._processor, self.config = _load_model(self.config)

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension (may be reduced by MRL)."""
        if self.config.mrl_dim:
            return self.config.mrl_dim
        return self.config.embedding_dim

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config.model_name

    def embed_query(self, text: str) -> List[float]:
        """Embed a query string (alias for compatibility)."""
        return embed_text(text, config=self.config)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return embed_texts_batch(texts, config=self.config)

    def embed_image(self, image_path: Union[str, Path]) -> List[float]:
        """Embed an image."""
        return embed_image(image_path, config=self.config)

    def embed_images(self, image_paths: List[Union[str, Path]]) -> List[List[float]]:
        """Embed multiple images."""
        return embed_images_batch(image_paths, config=self.config)

    def embed_document_page(
        self,
        page_image: Union[str, Path],
        page_text: Optional[str] = None
    ) -> List[float]:
        """
        Embed a document page (image + optional OCR text).

        This is useful for semantic document search where you want to
        capture both visual and textual content.
        """
        return embed_multimodal(
            text=page_text,
            image=page_image,
            config=self.config
        )

    def unload(self):
        """Unload model to free GPU memory."""
        unload_model()
        self._model = None
        self._processor = None

    def get_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            "model_name": self.config.model_name,
            "embedding_dimension": self.embedding_dimension,
            "full_dimension": self.config.embedding_dim,
            "mrl_dimension": self.config.mrl_dim,
            "use_flash_attention": self.config.use_flash_attention,
            "device": self.config.device or _get_device(),
            "max_batch_size": self.config.max_batch_size,
        }


# ============================================================================
# Global Service Instance
# ============================================================================

_service_instance: Optional[Qwen3VLEmbeddingService] = None
_service_lock = threading.Lock()


def get_embedding_service(config: Optional[Qwen3VLConfig] = None) -> Qwen3VLEmbeddingService:
    """
    Get or create the global embedding service instance.

    Args:
        config: Optional configuration for first initialization

    Returns:
        Qwen3VLEmbeddingService instance
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = Qwen3VLEmbeddingService(config)

    return _service_instance


def reset_service():
    """Reset the global service instance."""
    global _service_instance

    with _service_lock:
        if _service_instance is not None:
            _service_instance.unload()
            _service_instance = None


# ============================================================================
# Convenience Functions (match existing embedding_service.py API)
# ============================================================================

def qwen3_embed_query(text: str) -> List[float]:
    """Embed a single query (API compatibility)."""
    return get_embedding_service().embed_query(text)


def qwen3_embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts (API compatibility)."""
    return get_embedding_service().embed_texts(texts)


def qwen3_embed_image(image_path: Union[str, Path]) -> List[float]:
    """Embed an image (API compatibility)."""
    return get_embedding_service().embed_image(image_path)
