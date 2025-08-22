# ## File: cortex_engine/task_engine.py
# Version: 13.2.0 (Smart Ollama LLM Selector)
# Date: 2025-08-22
# Purpose: Core AI task execution engine.
#          - FEATURE (v13.0.0): Updated to use Mistral Small 3.2 for improved proposal
#            generation with better instruction following and reduced repetition.

import docx
import io
import os
from typing import Dict, List, Any, Optional

from docx.shared import RGBColor
from llama_index.core import VectorStoreIndex, Settings
from .instruction_parser import CortexInstruction, INSTRUCTION_REGEX
from .collection_manager import WorkingCollectionManager
from .config import PROPOSAL_LLM_MODEL
from .utils import get_logger

# Set up logging
logger = get_logger(__name__)

# --- PROMPT PERSONAS ---
PROMPTS = {
    "green": "You are a factual, concise AI assistant. Your tone should be professional, direct, and formal.",
    "orange": "You are an expert proposal writer. Your tone should be confident, persuasive, and benefits-oriented.",
    "red": "You are a world-class strategy consultant. Your tone should be authoritative, visionary, and inspiring."
}

# --- PROMPT TEMPLATES (RESTORED) ---
REFINE_PROMPT = """
{persona}
Your task is to refine and enhance the following raw text for a proposal section, using the provided context to add relevant details.
Do not invent facts. If the context is irrelevant, simply improve the existing text's clarity and style based on the guidance.
PROPOSAL SECTION HEADING: {section_heading}
SPECIFIC GUIDANCE: {sub_instruction}
RAW TEXT FROM USER:
---
{raw_text}
---
BACKGROUND CONTEXT FROM KNOWLEDGE BASE:
---
{context}
---
REFINED AND ENHANCED TEXT:
"""

GENERATE_KB_PROMPT = """
{persona}
Your task is to draft a compelling paragraph for the proposal section "{section_heading}", based ONLY on the provided context.
Use the specific guidance to focus your response. Synthesize information from multiple sources if available.
Do not use any information not present in the context.
SPECIFIC GUIDANCE: {sub_instruction}
BACKGROUND CONTEXT FROM KNOWLEDGE BASE:
---
{context}
---
DRAFT FOR "{section_heading}":
"""

GENERATE_FROM_KB_AND_PROPOSAL_PROMPT = """
{persona}
Your task is to draft a compelling paragraph for the proposal section "{section_heading}", using both the existing proposal content and relevant knowledge base information.
Use the specific guidance to focus your response. Synthesize information from the proposal context and knowledge base sources.
PROPOSAL SECTION HEADING: {section_heading}
SPECIFIC GUIDANCE: {sub_instruction}
EXISTING PROPOSAL CONTENT:
---
{proposal_context}
---
RELEVANT KNOWLEDGE BASE CONTEXT:
---
{kb_context}
---
Based on all the information above, draft a compelling response for the "{section_heading}" section:
"""

GENERATE_RESOURCES_PROMPT = """
{persona}
Your task is to suggest a team of resources for a project. Use the full proposal draft so far to understand the project's scope and requirements. Use the additional background context from the knowledge base to inform your suggestions about specific technologies or methodologies mentioned.
PROPOSAL SECTION HEADING: {section_heading}
SPECIFIC GUIDANCE: {sub_instruction}
FULL PROPOSAL DRAFT SO FAR:
---
{proposal_context}
---
ADDITIONAL BACKGROUND CONTEXT FROM KNOWLEDGE BASE:
---
{kb_context}
---
Based on all the information above, please suggest the required team roles and their key responsibilities.
SUGGESTED RESOURCES:
"""

# --- Color mapping ---
COLOR_MAP = {
    "green": RGBColor(0x00, 0x80, 0x00),
    "orange": RGBColor(0xFF, 0x8C, 0x00),
    "red": RGBColor(0xB2, 0x22, 0x22),
}

class TaskExecutionEngine:
    def __init__(self, main_index: VectorStoreIndex, collection_manager: WorkingCollectionManager):
        self.main_index = main_index
        self.collection_manager = collection_manager
        self._setup_proposal_llm()
    
    def _setup_proposal_llm(self):
        """Configure LLM specifically optimized for proposal generation (LOCAL ONLY)."""
        try:
            # Check if Ollama is available first
            from cortex_engine.utils.ollama_utils import check_ollama_service
            
            is_running, error_msg = check_ollama_service()
            if not is_running:
                logger.error(f"❌ CRITICAL: Ollama service not available for proposal generation: {error_msg}")
                logger.error("❌ Proposal generation requires Ollama to be running locally for privacy and control.")
                raise Exception(f"Ollama service unavailable: {error_msg}")
            
            # ENFORCE LOCAL-ONLY: Proposals MUST run locally for privacy/control
            from .utils.smart_ollama_llm import create_smart_ollama_llm
            proposal_llm = create_smart_ollama_llm(
                model=PROPOSAL_LLM_MODEL,
                request_timeout=300.0
            )
            # Note: Model parameters (temperature, top_p, etc.) are now handled at request level in modern API
            Settings.llm = proposal_llm
            logger.info(f"✅ Proposal LLM configured (LOCAL): {PROPOSAL_LLM_MODEL}")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to configure local proposal LLM: {e}")
            logger.error("   Proposals require local models. Please ensure Ollama is running and model is installed.")
            # Don't fall back to cloud models for proposals
            raise RuntimeError(f"Local LLM required for proposals but failed to load: {PROPOSAL_LLM_MODEL}")

    def _get_retriever(self, similarity_top_k=5, doc_id_filter: Optional[List[str]] = None):
        if doc_id_filter:
            from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
            filters = MetadataFilters(filters=[ExactMatchFilter(key="doc_id", value=doc_id) for doc_id in doc_id_filter])
            return self.main_index.as_retriever(
                similarity_top_k=similarity_top_k,
                filters=filters
            )
        return self.main_index.as_retriever(similarity_top_k=similarity_top_k)

    def _resolve_knowledge_sources(self, knowledge_sources: List[str]) -> Optional[List[str]]:
        if not knowledge_sources or "Main Cortex Knowledge Base" in knowledge_sources:
            return None
        all_doc_ids = set()
        for source_name in knowledge_sources:
            all_doc_ids.update(self.collection_manager.get_doc_ids_by_name(source_name))
        return list(all_doc_ids) if all_doc_ids else None

    # --- refine_with_ai METHOD (RESTORED) ---
    def refine_with_ai(self, section_heading: str, raw_text: str, creativity: str, sub_instruction: Optional[str]) -> str:
        if not raw_text.strip(): return ""
        search_query = f"Context for proposal section '{section_heading}'. User input: '{raw_text}'"
        retriever = self._get_retriever()
        retrieved_nodes = retriever.retrieve(search_query)
        context_str = "\n\n---\n\n".join([node.get_content() for node in retrieved_nodes])
        final_prompt = REFINE_PROMPT.format(
            persona=PROMPTS.get(creativity, PROMPTS['green']),
            section_heading=section_heading,
            sub_instruction=sub_instruction or "N/A",
            raw_text=raw_text,
            context=context_str or "No context found."
        )
        response = Settings.llm.complete(final_prompt)
        return response.text.strip()

    def generate_from_kb(self, instruction: CortexInstruction, creativity: str, knowledge_sources: List[str], session_state: dict) -> str:
        logger.info(f"🔍 DEBUG: generate_from_kb called for task_type='{instruction.task_type}', section='{instruction.section_heading}'")
        
        doc_id_filter = self._resolve_knowledge_sources(knowledge_sources)
        logger.info(f"🔍 DEBUG: Knowledge sources: {knowledge_sources}, doc_id_filter: {len(doc_id_filter) if doc_id_filter else 'None'}")
        
        retriever = self._get_retriever(doc_id_filter=doc_id_filter)

        proposal_context = []
        parsed_instructions = session_state.get('parsed_instructions', [])
        section_content = session_state.get('section_content', {})
        logger.info(f"🔍 DEBUG: Found {len(parsed_instructions)} parsed instructions, {len(section_content)} section contents")
        
        for i, inst in enumerate(parsed_instructions):
            content_key = f"content_inst_{i}"
            content = section_content.get(content_key, {}).get('text', '')
            if content:
                proposal_context.append(f"Content from section '{inst.section_heading}':\n{content}\n")
                logger.info(f"🔍 DEBUG: Added proposal context from section '{inst.section_heading}' ({len(content)} chars)")

        full_proposal_context = "\n---\n".join(proposal_context)
        logger.info(f"🔍 DEBUG: Full proposal context length: {len(full_proposal_context)} chars")

        search_query = f"FULL PROPOSAL CONTEXT SO FAR:\n{full_proposal_context}\n\nGenerate suggestions for proposal section '{instruction.section_heading}'."
        logger.info(f"🔍 DEBUG: Search query: {search_query[:200]}...")
        
        retrieved_nodes = retriever.retrieve(search_query)
        logger.info(f"🔍 DEBUG: Retrieved {len(retrieved_nodes)} nodes from KB")
        
        kb_context_str = "\n\n---\n\n".join([node.get_content() for node in retrieved_nodes]) or "No relevant information found."
        logger.info(f"🔍 DEBUG: KB context length: {len(kb_context_str)} chars")

        # Select the appropriate prompt template
        if instruction.task_type == "GENERATE_RESOURCES":
            prompt_template = GENERATE_RESOURCES_PROMPT
        elif instruction.task_type == "GENERATE_FROM_KB_AND_PROPOSAL":
            prompt_template = GENERATE_FROM_KB_AND_PROPOSAL_PROMPT
        else:
            prompt_template = GENERATE_KB_PROMPT

        # Format the prompt based on the template type
        if instruction.task_type in ["GENERATE_RESOURCES", "GENERATE_FROM_KB_AND_PROPOSAL"]:
            final_prompt = prompt_template.format(
                persona=PROMPTS.get(creativity, PROMPTS['green']),
                section_heading=instruction.section_heading,
                sub_instruction=instruction.sub_instruction or "N/A",
                proposal_context=full_proposal_context or "No prior proposal content available.",
                kb_context=kb_context_str
            )
        else: # Standard GENERATE tasks that only use KB context
            final_prompt = prompt_template.format(
                persona=PROMPTS.get(creativity, PROMPTS['green']),
                section_heading=instruction.section_heading,
                sub_instruction=instruction.sub_instruction or "N/A",
                context=kb_context_str
            )

        logger.info(f"🔍 DEBUG: Final prompt length: {len(final_prompt)} chars")
        logger.info(f"🔍 DEBUG: Final prompt preview: {final_prompt[:300]}...")
        
        response = Settings.llm.complete(final_prompt)
        logger.info(f"🔍 DEBUG: LLM response length: {len(response.text)} chars")
        return response.text.strip()

    def retrieve_from_kb(self, instruction: CortexInstruction, knowledge_sources: List[str], session_state: dict) -> str:
        doc_id_filter = self._resolve_knowledge_sources(knowledge_sources)
        retriever = self._get_retriever(similarity_top_k=5, doc_id_filter=doc_id_filter)

        search_query = f"Find the most relevant case studies or project examples for the proposal section '{instruction.section_heading}'. Focus on project outcomes, challenges, and solutions."
        retrieved_nodes = retriever.retrieve(search_query)

        if not retrieved_nodes: return "[No relevant case studies found.]"

        content = [f"**Source:** {os.path.basename(node.metadata.get('doc_posix_path', 'N/A'))}\n**Summary:** {node.metadata.get('summary', 'N/A')}\n\n**Excerpt:**\n{node.get_content().strip()}" for node in retrieved_nodes if node.score > 0.7]

        return "\n\n---\n\n".join(content) if content else "[No relevant case studies found with sufficient confidence.]"

    def assemble_document(self, instructions: List[CortexInstruction], section_content: dict, doc_template_bytes: bytes) -> io.BytesIO:
        doc = docx.Document(io.BytesIO(doc_template_bytes))

        # Find all paragraphs that contain an instruction tag
        instruction_paragraphs = []
        for p in doc.paragraphs:
            if INSTRUCTION_REGEX.search(p.text.strip()):
                instruction_paragraphs.append(p)

        for i, p in enumerate(instruction_paragraphs):
            # Ensure we don't go out of bounds if the number of instructions and paragraphs mismatches
            if i >= len(instructions):
                break

            content_key = f"content_inst_{i}"
            content_data = section_content.get(content_key, {'text': '', 'creativity': 'green'})
            generated_text = content_data.get('text', '')
            creativity_level = content_data.get('creativity')

            # Clear the placeholder text (e.g., "[PROMPT_HUMAN]") from the paragraph
            p.text = ""

            if generated_text:
                run = p.add_run()
                font_color = COLOR_MAP.get(creativity_level)
                if font_color:
                    run.font.color.rgb = font_color

                # Add the generated text, preserving line breaks
                lines = generated_text.split('\n')
                for j, line in enumerate(lines):
                    run.add_text(line)
                    if j < len(lines) - 1:
                        run.add_break()

        final_bio = io.BytesIO()
        doc.save(final_bio)
        final_bio.seek(0)
        return final_bio