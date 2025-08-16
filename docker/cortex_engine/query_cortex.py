# ## File: query_cortex.py
# Version: 5.0.0 (VLM Utility for Ingestion)
# Date: 2025-07-22
# Purpose: Backend module providing models and prompts.
#          - FEATURE (v5.0.0): Re-introduced a VLM utility function, specifically
#            scoped for use during the ingestion process. This allows images to
#            be described by a local model like Llava without complicating the
#            main query engine.

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
import ollama
import base64

from .config import KB_LLM_MODEL, EMBED_MODEL, VLM_MODEL
from .utils import get_logger

# Set up logging
logger = get_logger(__name__)

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
        
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=device)
        
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

        # This uses the official ollama client with timeout protection
        client = ollama.Client(timeout=120)  # 2 minute timeout
        response = client.chat(
            model=VLM_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe this image in detail for a searchable knowledge base. Be specific about any text, diagrams, or data presented. What is the key information conveyed by this image?',
                    'images': [encoded_image]
                }
            ],
            options={"temperature": 0.2}  # Lower temperature for more factual descriptions
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