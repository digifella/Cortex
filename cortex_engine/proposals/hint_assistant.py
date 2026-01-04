"""
Hint-Based MoE Assistant - Help Anywhere in Tender Documents
Version: 2.0.0
Date: 2026-01-02

Purpose: Provide MoE assistance based on user hints, not rigid instructions.
User can select any section and say "help me with this" + provide context/hint.

Key Features:
- Works with FlexibleSection (no rigid format)
- User provides natural language hints
- MoE decides which experts to use based on hint + section
- Iterative refinement support (generate → review → refine)
"""

import asyncio
from typing import Dict, List, Any, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum

from cortex_engine.adaptive_model_manager import AdaptiveModelManager, TaskType
from cortex_engine.utils.modern_ollama_llm import ModernOllamaLLM
from cortex_engine.proposals.flexible_parser import FlexibleSection, SectionType, ContentStatus

import logging
logger = logging.getLogger(__name__)


class AssistanceMode(str, Enum):
    """How should AI help with this section?"""
    GENERATE_NEW = "generate_new"              # Create from scratch
    REFINE_EXISTING = "refine_existing"        # Improve what's there
    ANSWER_QUESTION = "answer_question"        # Direct question answering
    EXPAND_BRIEF = "expand_brief"              # Expand brief notes
    REWRITE_PROFESSIONAL = "rewrite_professional"  # Make more professional
    ADD_EVIDENCE = "add_evidence"              # Add KB citations/evidence
    BRAINSTORM = "brainstorm"                  # Creative ideation


@dataclass
class AssistanceRequest:
    """User's request for help with a section."""

    section: FlexibleSection
    user_hint: str                             # User's guidance in natural language
    mode: AssistanceMode                       # How to help
    use_moe: bool = False                      # Use multiple experts?
    kb_sources: Optional[List[str]] = None     # Which KB collections to use
    creativity: float = 0.7                    # Temperature (0.0-2.0)
    max_length: int = 500                      # Max words in response


@dataclass
class AssistanceResult:
    """Result of AI assistance."""

    success: bool
    content: str                               # Generated/refined content
    models_used: List[str]                     # Which models helped
    method: str                                # "single_expert" or "multi_expert"
    confidence: float = 0.0                    # AI's confidence in result
    sources: List[str] = None                  # KB sources cited
    reasoning: Optional[str] = None            # Why these models were chosen
    error: Optional[str] = None


class HintBasedAssistant:
    """
    MoE assistant that helps based on user hints, not rigid instructions.
    User can ask for help with ANY section in ANY way.
    """

    def __init__(
        self,
        model_manager: AdaptiveModelManager,
        vector_index: Any,
        collection_manager: Any
    ):
        self.model_manager = model_manager
        self.vector_index = vector_index
        self.collection_manager = collection_manager

    async def assist(
        self,
        request: AssistanceRequest,
        progress_callback=None
    ) -> AssistanceResult:
        """
        Provide assistance based on user's hint and section context.
        Automatically decides whether to use single or multiple experts.
        """

        if progress_callback:
            progress_callback("Analyzing your request...")

        # 1. Understand what user wants
        intent = self._understand_intent(request)

        # 2. Decide on model strategy
        if request.use_moe or request.section.complexity == "complex":
            # Use multiple experts
            return await self._assist_with_moe(request, intent, progress_callback)
        else:
            # Single expert is sufficient
            return await self._assist_with_single_model(request, intent, progress_callback)

    async def _assist_with_single_model(
        self,
        request: AssistanceRequest,
        intent: Dict[str, Any],
        progress_callback=None
    ) -> AssistanceResult:
        """Provide assistance using a single optimal model."""

        # Select best model for this task
        task_type = self._intent_to_task_type(intent)
        model_name = await self.model_manager.recommend_model(task_type)

        if not model_name:
            return AssistanceResult(
                success=False,
                content="",
                models_used=[],
                method="single_expert",
                error="No suitable model available"
            )

        if progress_callback:
            progress_callback(f"Using {model_name}...")

        # Build context from knowledge base
        kb_context = await self._build_kb_context(request)

        # Build prompt
        prompt = self._build_prompt(request, intent, kb_context)

        # Generate
        try:
            llm = ModernOllamaLLM(model=model_name, request_timeout=300.0)

            response = await llm.acomplete(
                prompt,
                options={
                    "temperature": request.creativity,
                    "top_p": 0.9,
                    "num_predict": request.max_length * 5  # words to tokens rough estimate
                }
            )

            return AssistanceResult(
                success=True,
                content=response.text,
                models_used=[model_name],
                method="single_expert",
                confidence=0.8,  # TODO: Implement confidence scoring
                sources=kb_context.get("sources", [])
            )

        except Exception as e:
            logger.error(f"Single model assistance failed: {e}")
            return AssistanceResult(
                success=False,
                content="",
                models_used=[model_name],
                method="single_expert",
                error=str(e)
            )

    async def _assist_with_moe(
        self,
        request: AssistanceRequest,
        intent: Dict[str, Any],
        progress_callback=None
    ) -> AssistanceResult:
        """Provide assistance using multiple expert models with synthesis."""

        # Get expert ensemble
        experts = await self._get_expert_ensemble(request, intent)

        if not experts or len(experts) < 1:
            # Fallback to single model
            return await self._assist_with_single_model(request, intent, progress_callback)

        if progress_callback:
            progress_callback(f"Using {len(experts)} expert models...")

        # Build context
        kb_context = await self._build_kb_context(request)
        prompt = self._build_prompt(request, intent, kb_context)

        # Run experts in parallel
        expert_results = []

        for i, expert_model in enumerate(experts, 1):
            if progress_callback:
                progress_callback(f"Expert {i}/{len(experts)}: {expert_model}")

            try:
                llm = ModernOllamaLLM(model=expert_model, request_timeout=600.0)

                response = await llm.acomplete(
                    prompt,
                    options={
                        "temperature": request.creativity,
                        "top_p": 0.9,
                        "num_predict": request.max_length * 5
                    }
                )

                expert_results.append({
                    "model": expert_model,
                    "content": response.text
                })

            except Exception as e:
                logger.warning(f"Expert {expert_model} failed: {e}")
                continue

        if not expert_results:
            return AssistanceResult(
                success=False,
                content="",
                models_used=experts,
                method="multi_expert",
                error="All expert models failed"
            )

        # Synthesize expert outputs
        if progress_callback:
            progress_callback("Synthesizing expert insights...")

        synthesized = await self._synthesize_expert_outputs(
            request,
            expert_results,
            kb_context
        )

        return AssistanceResult(
            success=True,
            content=synthesized,
            models_used=[r["model"] for r in expert_results],
            method="multi_expert",
            confidence=0.9,  # Higher confidence with MoE
            sources=kb_context.get("sources", []),
            reasoning=f"Used {len(expert_results)} experts for comprehensive analysis"
        )

    def _understand_intent(self, request: AssistanceRequest) -> Dict[str, Any]:
        """Analyze user's hint to understand what they want."""

        hint_lower = request.user_hint.lower()

        # Detect intent from hint keywords
        intent = {
            "primary_goal": "generate",
            "requires_creativity": False,
            "requires_evidence": False,
            "requires_technical_depth": False
        }

        # Creativity indicators
        if any(word in hint_lower for word in ['creative', 'innovative', 'unique', 'novel']):
            intent["requires_creativity"] = True

        # Evidence indicators
        if any(word in hint_lower for word in ['evidence', 'proof', 'example', 'case study', 'citation']):
            intent["requires_evidence"] = True

        # Technical depth indicators
        if any(word in hint_lower for word in ['technical', 'detailed', 'comprehensive', 'methodology']):
            intent["requires_technical_depth"] = True

        # Mode-specific intents
        if request.mode == AssistanceMode.ANSWER_QUESTION:
            intent["primary_goal"] = "answer"
        elif request.mode == AssistanceMode.BRAINSTORM:
            intent["primary_goal"] = "ideate"
            intent["requires_creativity"] = True
        elif request.mode == AssistanceMode.ADD_EVIDENCE:
            intent["primary_goal"] = "cite"
            intent["requires_evidence"] = True

        return intent

    def _intent_to_task_type(self, intent: Dict[str, Any]) -> TaskType:
        """Map intent to AdaptiveModelManager task type."""

        if intent["requires_technical_depth"]:
            return TaskType.RESEARCH

        if intent["primary_goal"] == "ideate":
            return TaskType.SYNTHESIS

        if intent["requires_evidence"]:
            return TaskType.RESEARCH

        return TaskType.ANALYSIS  # Default

    async def _get_expert_ensemble(
        self,
        request: AssistanceRequest,
        intent: Dict[str, Any]
    ) -> List[str]:
        """Get 2-3 expert models for MoE synthesis."""

        task_type = self._intent_to_task_type(intent)

        # Get best model
        best = await self.model_manager.recommend_model(task_type, preference="best")

        # Get balanced alternative
        balanced = await self.model_manager.recommend_model(task_type, preference="balanced")

        experts = []
        if best:
            experts.append(best)
        if balanced and balanced != best:
            experts.append(balanced)

        # For highly complex tasks, add third expert if available
        if request.section.complexity == "complex" and len(experts) < 3:
            fast = await self.model_manager.recommend_model(task_type, preference="fastest")
            if fast and fast not in experts:
                experts.append(fast)

        return experts

    async def _build_kb_context(self, request: AssistanceRequest) -> Dict[str, Any]:
        """Retrieve relevant context from knowledge base."""

        if not self.vector_index:
            return {"content": "", "sources": []}

        try:
            # Build search query from section + hint
            query = f"{request.section.heading}: {request.user_hint}"

            # Retrieve from KB
            query_engine = self.vector_index.as_query_engine(
                similarity_top_k=5,
                streaming=False
            )

            response = query_engine.query(query)

            # Extract sources
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes[:5]:
                    metadata = node.metadata or {}
                    source = metadata.get('file_name', 'Unknown')
                    if source not in sources:
                        sources.append(source)

            return {
                "content": str(response),
                "sources": sources
            }

        except Exception as e:
            logger.warning(f"KB context retrieval failed: {e}")
            return {"content": "", "sources": []}

    def _build_prompt(
        self,
        request: AssistanceRequest,
        intent: Dict[str, Any],
        kb_context: Dict[str, Any]
    ) -> str:
        """Build prompt for AI model based on request context."""

        # Base context
        prompt_parts = []

        # System role based on mode
        if request.mode == AssistanceMode.GENERATE_NEW:
            prompt_parts.append("You are an expert proposal writer creating compelling content for tender responses.")
        elif request.mode == AssistanceMode.ANSWER_QUESTION:
            prompt_parts.append("You are a knowledgeable expert answering tender questions accurately and comprehensively.")
        elif request.mode == AssistanceMode.BRAINSTORM:
            prompt_parts.append("You are a creative strategist brainstorming innovative approaches for tender responses.")
        elif request.mode == AssistanceMode.REWRITE_PROFESSIONAL:
            prompt_parts.append("You are a professional editor refining content to be clear, compelling, and professional.")
        else:
            prompt_parts.append("You are an expert proposal writer assisting with tender document completion.")

        # Section context
        prompt_parts.append(f"\n**Section:** {request.section.heading}")

        if request.section.numbering:
            prompt_parts.append(f"**Section Number:** {request.section.numbering}")

        if request.section.parent_heading:
            prompt_parts.append(f"**Parent Section:** {request.section.parent_heading}")

        # Existing content (if any)
        if request.section.content and request.mode != AssistanceMode.GENERATE_NEW:
            prompt_parts.append(f"\n**Current Content:**\n{request.section.content}")

        # User's hint/guidance
        prompt_parts.append(f"\n**Your Task:**\n{request.user_hint}")

        # Knowledge base context (if available)
        if kb_context.get("content"):
            prompt_parts.append(f"\n**Relevant Knowledge from Your Database:**\n{kb_context['content']}")

        # Output instructions based on mode
        if request.mode == AssistanceMode.GENERATE_NEW:
            prompt_parts.append("\nGenerate compelling, well-structured content that addresses the section requirements.")
        elif request.mode == AssistanceMode.ANSWER_QUESTION:
            prompt_parts.append("\nProvide a clear, comprehensive answer that demonstrates expertise and capability.")
        elif request.mode == AssistanceMode.BRAINSTORM:
            prompt_parts.append("\nGenerate 3-5 innovative ideas or approaches. Be creative and think outside the box.")
        elif request.mode == AssistanceMode.REFINE_EXISTING:
            prompt_parts.append("\nImprove the current content by making it clearer, more compelling, and more professional.")
        elif request.mode == AssistanceMode.ADD_EVIDENCE:
            prompt_parts.append("\nEnhance the content with specific examples, case studies, or evidence from the knowledge base.")

        # Length guidance
        prompt_parts.append(f"\n**Target Length:** Approximately {request.max_length} words")

        # Final instruction
        prompt_parts.append("\n**Response:**")

        return "\n".join(prompt_parts)

    async def _synthesize_expert_outputs(
        self,
        request: AssistanceRequest,
        expert_results: List[Dict[str, str]],
        kb_context: Dict[str, Any]
    ) -> str:
        """Synthesize multiple expert outputs into one superior response."""

        if len(expert_results) == 1:
            return expert_results[0]["content"]

        # Build synthesis prompt
        expert_texts = []
        for i, result in enumerate(expert_results, 1):
            expert_texts.append(f"### Expert {i} ({result['model']}):\n{result['content']}")

        synthesis_prompt = f"""You are a meta-analyst synthesizing multiple expert responses for a tender document section.

**Section:** {request.section.heading}
**User Guidance:** {request.user_hint}

Multiple AI experts have provided responses below. Your task is to:
1. Identify the best insights from each expert
2. Resolve any contradictions intelligently
3. Combine into ONE cohesive, high-quality response
4. Maintain professional tone and clarity
5. Target length: {request.max_length} words

**Expert Responses:**

{chr(10).join(expert_texts)}

**Synthesized Response:**"""

        # Use best available model for synthesis
        synthesis_model = await self.model_manager.recommend_model(
            TaskType.SYNTHESIS,
            preference="balanced"
        )

        try:
            llm = ModernOllamaLLM(model=synthesis_model or "mistral-small3.2", request_timeout=180.0)

            response = await llm.acomplete(
                synthesis_prompt,
                options={"temperature": 0.3, "top_p": 0.8}
            )

            return response.text

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback: Return best expert output
            return expert_results[0]["content"]
