# ## File: query_cortex.py
# Version: 5.4.0 (LlamaIndex Compatibility Fix)
# Date: 2025-08-26
# Purpose: Backend module providing models and prompts.
#          - CRITICAL FIX (v5.4.0): Added get_agg_embedding_from_queries method and full LlamaIndex
#            compatibility to resolve 'SentenceTransformerWrapper' object has no attribute error.
#          - CRITICAL FIX (v5.3.0): Fixed emergency embedding model to use 768 dimensions matching
#            BAAI/bge-base-en-v1.5 standard, resolving ChromaDB dimension mismatch errors.
#          - CRITICAL FIX (v5.2.0): Added emergency embedding model for transformers compatibility
#            issues. Prevents complete system failure when PyTorch/transformers conflicts occur.
#          - CRITICAL BUGFIX (v5.1.1): Fixed UnboundLocalError for 'os' import that was
#            preventing proper model initialization. Moved os import to proper scope.
#          - CRITICAL FIX (v5.1.0): Applied PyTorch 2.8+ meta tensor fix to embedding
#            model initialization. Multi-layered fallback system for embedding models.

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .embedding_adapters import EmbeddingServiceAdapter
import torch
import ollama
import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading
from typing import Optional

from .config import KB_LLM_MODEL, EMBED_MODEL, VLM_MODEL
from .utils import get_logger

# Set up logging
logger = get_logger(__name__)

# Module-level executor for async image processing (shared across calls)
_image_executor = None
_executor_lock = threading.Lock()

def _get_image_executor():
    """Get or create the shared image processing executor."""
    global _image_executor
    if _image_executor is None:
        with _executor_lock:
            if _image_executor is None:
                _image_executor = ThreadPoolExecutor(
                    max_workers=3,
                    thread_name_prefix="vlm_image"
                )
    return _image_executor

# --- Prompt Templates ---

FINAL_SYNTHESIS_PROMPT = """
You are a helpful AI assistant for Project Cortex. Your task is to synthesize a clear and concise answer to the user's question based *only* on the provided context, which may contain both text and image descriptions.
Do not use any prior knowledge. If the context does not contain the answer, state that clearly.
Reference the source documents by their filenames when possible.

QUESTION:
{question}

CONTEXT:
---
{context}
---

ANSWER:
"""

AUTHOR_SUMMARY_PROMPT = """
You are a helpful AI assistant for Project Cortex. You have been asked to provide a summary of works by a specific author based on the provided context.
Synthesize a summary of the documents and topics associated with this author.
List the document titles you found. Do not use any prior knowledge.

AUTHOR:
{author_name}

CONTEXT:
---
{context}
---

SUMMARY OF WORKS:
"""

VLM_QA_PROMPT = """
You are a helpful AI assistant for Project Cortex. Your task is to synthesize a clear and concise answer to the user's question based on the provided context, which includes text and descriptions of images.
When an image is relevant, explicitly mention it in your answer (e.g., "According to the image description of the Venn diagram...").
Do not use any prior knowledge. If the context does not contain the answer, state that clearly.

QUESTION:
{question}

CONTEXT:
---
{context}
---

ANSWER:
"""

# --- Model Setup ---

def setup_models():
    """Configures and initializes the LLM and embedding model (LOCAL ONLY)."""
    logger.info("--- Configuring KB query models (LOCAL) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Check if Ollama is available first
        from cortex_engine.utils.ollama_utils import check_ollama_service
        
        is_running, error_msg = check_ollama_service()
        if not is_running:
            logger.warning(f"⚠️ Ollama service not available: {error_msg}")
            logger.warning("   Knowledge Base will operate in limited mode (vector search only)")
            Settings.llm = None
        else:
            # Knowledge Base operations MUST be local when available
            Settings.llm = Ollama(
                model=KB_LLM_MODEL, 
                request_timeout=300.0,
                temperature=0.1,  # Very low temperature for factual retrieval
            )
            logger.info(f"✅ KB models configured (LOCAL): LLM={KB_LLM_MODEL}, Embed={EMBED_MODEL}, Device={device}")
        
        # Embedding model initialization - prefer centralized embedding via adapter (keeps ingest/search identical)
        import os  # Move os import to be available in function scope
        try:
            # Method 1: Use adapter around embedding_service
            try:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                Settings.embed_model = EmbeddingServiceAdapter(model_name=EMBED_MODEL)
                test_embedding = Settings.embed_model.get_text_embedding("test")
                logger.info(f"Query engine: EmbeddingServiceAdapter successful, dimension: {len(test_embedding)}")
                logger.info("✅ Query engine: Initialized using EmbeddingServiceAdapter")
                
            except Exception as hf_e:
                logger.warning(f"Query engine: Adapter approach failed: {hf_e}")
                
                # Method 2: Sentence-transformers with full wrapper (fallback)
                try:
                    from sentence_transformers import SentenceTransformer
                    
                    st_model = SentenceTransformer(EMBED_MODEL, device=device, trust_remote_code=True)
                    
                    # Test the model
                    test_output = st_model.encode("test", convert_to_tensor=False)
                    logger.info(f"Query engine: Sentence-transformers test successful, dimension: {len(test_output)}")
                    
                    # Create FULL wrapper for LlamaIndex compatibility with all required methods
                    class CompleteSentenceTransformerWrapper:
                        def __init__(self, model, model_name):
                            self.model = model
                            self.model_name = model_name
                            
                        def get_text_embedding(self, text):
                            return self.model.encode(text, convert_to_tensor=False).tolist()
                        
                        def get_text_embeddings(self, texts):
                            embeddings = self.model.encode(texts, convert_to_tensor=False)
                            return [emb.tolist() for emb in embeddings]
                        
                        def get_text_embedding_batch(self, texts):
                            return self.get_text_embeddings(texts)
                        
                        def get_query_embedding(self, query):
                            return self.get_text_embedding(query)
                        
                        def get_agg_embedding_from_queries(self, queries):
                            """Aggregate multiple query embeddings by averaging them"""
                            embeddings = self.get_text_embeddings(queries)
                            if not embeddings:
                                return []
                            # Average the embeddings
                            import numpy as np
                            avg_embedding = np.mean(embeddings, axis=0)
                            return avg_embedding.tolist()
                    
                    Settings.embed_model = CompleteSentenceTransformerWrapper(st_model, EMBED_MODEL)
                    logger.info("✅ Query engine: Successfully initialized using complete sentence-transformers wrapper")
                    
                except Exception as st_e:
                    logger.warning(f"Query engine: Sentence-transformers approach failed: {st_e}")
                    
                    # Method 3: Basic HuggingFace fallback with simpler config
                    try:
                        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=device)
                        logger.info("✅ Query engine: Successfully initialized using basic HuggingFaceEmbedding")
                    
                    except Exception as basic_e:
                        logger.error(f"Query engine: All embedding initialization methods failed: {basic_e}")
                        # Try alternative model as last resort before emergency
                        try:
                            alt_model = "sentence-transformers/all-MiniLM-L6-v2"
                            Settings.embed_model = HuggingFaceEmbedding(model_name=alt_model, device=device)
                            logger.warning(f"⚠️ Query engine: Using alternative embedding model: {alt_model}")
                            
                        except Exception as alt_e:
                            logger.error(f"Query engine: Alternative model failed: {alt_e}")
                            # Emergency fallback with full LlamaIndex compatibility
                            logger.warning("🚨 Creating emergency embedding model with full LlamaIndex compatibility")
                            
                            class EmergencyEmbeddingModel:
                                """Emergency embedding model with full LlamaIndex interface"""
                                def __init__(self):
                                    self.embed_dim = 768
                                    self.model_name = "emergency-fallback"
                                    logger.warning("⚠️ Using emergency embedding model - search quality will be limited")
                                
                                def get_text_embedding(self, text):
                                    import hashlib
                                    hash_obj = hashlib.sha256(text.encode('utf-8'))
                                    hash_bytes = hash_obj.digest()
                                    embedding = []
                                    for i in range(self.embed_dim):
                                        byte_idx = (i * 4) % len(hash_bytes)
                                        byte_val = hash_bytes[byte_idx] if byte_idx < len(hash_bytes) else 0
                                        float_val = (float(byte_val) - 127.5) / 127.5
                                        embedding.append(float_val)
                                    return embedding
                                
                                def get_text_embeddings(self, texts):
                                    return [self.get_text_embedding(text) for text in texts]
                                
                                def get_text_embedding_batch(self, texts):
                                    return self.get_text_embeddings(texts)
                                
                                def get_query_embedding(self, query):
                                    return self.get_text_embedding(query)
                                
                                def get_agg_embedding_from_queries(self, queries):
                                    """Aggregate multiple query embeddings by averaging them"""
                                    embeddings = self.get_text_embeddings(queries)
                                    if not embeddings:
                                        return []
                                    import numpy as np
                                    avg_embedding = np.mean(embeddings, axis=0)
                                    return avg_embedding.tolist()
                            
                            Settings.embed_model = EmergencyEmbeddingModel()
                            logger.warning("🆘 Emergency embedding model active with full compatibility")
                
        except Exception as e:
            logger.error(f"❌ CRITICAL: Complete embedding system failure: {e}")
            raise RuntimeError(f"Failed to configure KB models: {e}")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to configure KB models: {e}")
        if Settings.llm is None:
            logger.info("   Falling back to basic vector search without AI enhancements")
        else:
            raise RuntimeError(f"Failed to configure KB models: {e}")


# --- SPRINT 21: VLM Utility for Ingestion ---
def describe_image_with_vlm_for_ingestion(image_path: str) -> str:
    """
    Uses a local VLM via Ollama to generate a text description for an image file.
    This function is intended to be called ONLY during the ingestion process.

    Args:
        image_path: The file path to the image.

    Returns:
        A text description of the image, or an error message.
    """
    try:
        print(f"  -> VLM Ingestion: Describing '{image_path}' with model '{VLM_MODEL}'...")
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # This uses the official ollama client with reduced timeout (30s)
        client = ollama.Client(timeout=30)  # 30 second timeout for faster processing
        response = client.chat(
            model=VLM_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': '''Analyze this image comprehensively for a professional knowledge management system. Please provide:

1. **Visual Content**: Describe what you see (objects, people, scenes, layout)
2. **Text Content**: Extract and transcribe any visible text, labels, titles, or captions
3. **Technical Elements**: Identify charts, graphs, diagrams, tables, or technical drawings
4. **Context & Purpose**: What is the likely purpose or message of this image?
5. **Key Information**: What are the most important details for knowledge retrieval?

Be specific and detailed. Include any data, numbers, or technical specifications visible. This description will be used for document search and analysis.''',
                    'images': [encoded_image]
                }
            ],
            options={
                "temperature": 0.1,  # Very low temperature for factual accuracy
                "top_p": 0.9,       # Slightly reduce randomness
                "num_predict": 500   # Allow longer, more detailed responses
            }
        )
        description = response['message']['content']
        print(f"  -> VLM Ingestion: Success. Description length: {len(description)} chars.")
        return description
    except FileNotFoundError:
        error_msg = f"VLM Error: Image file not found at {image_path}"
        print(f"  -> {error_msg}")
        return error_msg
    except Exception as e:
        # Check for specific 404 error indicating missing model
        if "404" in str(e) or "not found" in str(e).lower():
            error_msg = f"VLM Error: Model '{VLM_MODEL}' not found. Please install with: ollama pull {VLM_MODEL}"
            logger.error(f"Missing VLM model detected for {image_path}: {error_msg}")
        else:
            error_msg = f"VLM Error: An unexpected error occurred while processing {image_path}. Is Ollama running and the '{VLM_MODEL}' model pulled? Error: {e}"
            logger.error(f"VLM processing failed for {image_path}: {error_msg}")

        print(f"  -> {error_msg}")
        return error_msg


def describe_image_with_vlm_async(
    image_path: str,
    timeout: int = 30
) -> Optional[str]:
    """
    Async wrapper for VLM image description with timeout.
    Processes image in background thread with configurable timeout.

    This function enables parallel image processing during ingestion,
    significantly improving performance when processing multiple images.

    Args:
        image_path: Path to image file
        timeout: Timeout in seconds (default 30)

    Returns:
        Image description string, or None if timeout/error occurs

    Example:
        >>> description = describe_image_with_vlm_async("photo.jpg", timeout=30)
        >>> if description:
        >>>     print(f"Success: {description}")
        >>> else:
        >>>     print("Timeout or error occurred")
    """
    executor = _get_image_executor()

    try:
        # Submit task to thread pool
        future = executor.submit(
            describe_image_with_vlm_for_ingestion,
            image_path
        )

        # Wait for result with timeout
        result = future.result(timeout=timeout)

        # Check if result is an error message
        if result and result.startswith("VLM Error:"):
            logger.warning(f"❌ VLM processing error for {image_path}")
            return None

        return result

    except FuturesTimeoutError:
        logger.warning(f"⏱️ VLM timeout after {timeout}s for {image_path}")
        return None
    except Exception as e:
        logger.error(f"❌ Async VLM error for {image_path}: {e}")
        return None
