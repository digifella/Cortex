"""
Response Generator for Intelligent Proposal Completion

Version: 1.0.0
Date: 2026-01-19

Purpose: Generates draft responses for substantive proposal questions
using LLM with evidence from the knowledge collection.

Key features:
- Question-type-specific prompts
- Evidence injection into generation
- Confidence scoring based on evidence quality
- Placeholder detection for incomplete responses
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import re

from .field_classifier import QuestionType, ClassifiedField
from .evidence_retriever import Evidence
from .llm_interface import LLMInterface
from .entity_profile_manager import EntityProfileManager
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class DraftResponse:
    """A generated draft response for a substantive question."""
    question: str
    question_type: QuestionType
    text: str                        # The generated response text
    evidence_used: List[Evidence]    # Evidence that informed the response
    confidence: float                # 0-1 confidence in response quality
    word_count: int
    needs_review: bool               # True if low confidence or placeholders
    placeholders: List[str]          # Any [NEEDS DETAIL] markers found
    generation_time: float           # Seconds taken to generate
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseGenerator:
    """
    Generates draft responses for substantive proposal questions.

    Uses LLM with evidence injection to create responses that are:
    - Specific (using evidence from knowledge collection)
    - Professional (appropriate tone for proposals)
    - Verifiable (cites sources that can be checked)
    """

    # Prompt templates by question type
    PROMPT_TEMPLATES = {
        QuestionType.CAPABILITY: """You are writing a capability response for a government tender proposal.

COMPANY: {company_name}
QUESTION: {question}

RELEVANT EVIDENCE FROM OUR KNOWLEDGE BASE:
{evidence_section}

INSTRUCTIONS:
1. Write a compelling response demonstrating our capability and track record
2. Use SPECIFIC details from the evidence - project names, outcomes, metrics
3. Write in first person plural ("we", "our team")
4. Focus on relevant experience that directly answers the question
5. If evidence is insufficient for a claim, mark it: [NEEDS DETAIL: what's needed]
{word_instruction}

RESPONSE:""",

        QuestionType.METHODOLOGY: """You are writing a methodology response for a government tender proposal.

COMPANY: {company_name}
QUESTION: {question}

RELEVANT EVIDENCE FROM OUR KNOWLEDGE BASE:
{evidence_section}

INSTRUCTIONS:
1. Outline a clear, structured approach to delivering the required services
2. Reference proven methodologies from the evidence where applicable
3. Include key phases, milestones, and deliverables
4. Write in first person plural ("we will", "our approach")
5. Be specific about tools, techniques, and processes we'll use
6. If a detail is needed, mark it: [NEEDS DETAIL: what's needed]
{word_instruction}

RESPONSE:""",

        QuestionType.VALUE_PROPOSITION: """You are writing a value proposition response for a government tender proposal.

COMPANY: {company_name}
QUESTION: {question}

RELEVANT EVIDENCE FROM OUR KNOWLEDGE BASE:
{evidence_section}

INSTRUCTIONS:
1. Clearly articulate the benefits we will deliver
2. Use SPECIFIC examples and metrics from the evidence
3. Focus on outcomes that matter to the client
4. If applicable, mention local/community benefits (jobs, local suppliers)
5. Quantify benefits where possible (%, $, time saved)
6. If a detail is needed, mark it: [NEEDS DETAIL: what's needed]
{word_instruction}

RESPONSE:""",

        QuestionType.COMPLIANCE: """You are writing a compliance response for a government tender proposal.

COMPANY: {company_name}
QUESTION: {question}

RELEVANT EVIDENCE FROM OUR KNOWLEDGE BASE:
{evidence_section}

INSTRUCTIONS:
1. Directly confirm compliance with the stated requirement
2. Reference specific certifications, policies, or accreditations from evidence
3. Be clear and unambiguous - avoid hedging language
4. If we cannot confirm compliance, state: [NEEDS VERIFICATION: what to check]
5. Include relevant dates, certificate numbers, or policy references
{word_instruction}

RESPONSE:""",

        QuestionType.INNOVATION: """You are writing an innovation response for a government tender proposal.

COMPANY: {company_name}
QUESTION: {question}

RELEVANT EVIDENCE FROM OUR KNOWLEDGE BASE:
{evidence_section}

INSTRUCTIONS:
1. Highlight innovative approaches, technologies, or solutions we offer
2. Explain how these innovations benefit the client
3. Use evidence to show proven results of our innovations
4. Balance innovation claims with practical delivery assurance
5. If a detail is needed, mark it: [NEEDS DETAIL: what's needed]
{word_instruction}

RESPONSE:""",

        QuestionType.RISK: """You are writing a risk management response for a government tender proposal.

COMPANY: {company_name}
QUESTION: {question}

RELEVANT EVIDENCE FROM OUR KNOWLEDGE BASE:
{evidence_section}

INSTRUCTIONS:
1. Identify key risks relevant to this engagement
2. Describe specific mitigation strategies for each risk
3. Reference our experience managing similar risks from evidence
4. Include contingency plans where appropriate
5. Be practical and realistic, not generic
6. If a detail is needed, mark it: [NEEDS DETAIL: what's needed]
{word_instruction}

RESPONSE:""",

        QuestionType.PERSONNEL: """You are writing a personnel/team response for a government tender proposal.

COMPANY: {company_name}
QUESTION: {question}

RELEVANT EVIDENCE FROM OUR KNOWLEDGE BASE:
{evidence_section}

INSTRUCTIONS:
1. Describe the team composition and key roles
2. Highlight relevant qualifications and experience from evidence
3. Focus on expertise directly relevant to this engagement
4. If specific CVs are needed, note: [ATTACH: CV for role name]
5. Emphasize team stability and availability
{word_instruction}

RESPONSE:""",

        QuestionType.PRICING: """You are writing a pricing/cost response for a government tender proposal.

COMPANY: {company_name}
QUESTION: {question}

RELEVANT EVIDENCE FROM OUR KNOWLEDGE BASE:
{evidence_section}

INSTRUCTIONS:
1. Provide clear pricing structure as requested
2. Break down costs into logical components
3. Explain the basis for pricing (rates, effort estimates)
4. Note any assumptions or exclusions
5. Mark any figures that need confirmation: [CONFIRM: pricing item]
{word_instruction}

RESPONSE:""",

        QuestionType.GENERAL: """You are writing a response for a government tender proposal.

COMPANY: {company_name}
QUESTION: {question}

RELEVANT EVIDENCE FROM OUR KNOWLEDGE BASE:
{evidence_section}

INSTRUCTIONS:
1. Directly answer the question asked
2. Use specific details from the evidence where relevant
3. Write in first person plural ("we", "our")
4. Be professional and concise
5. If evidence is insufficient, mark: [NEEDS DETAIL: what's needed]
{word_instruction}

RESPONSE:"""
    }

    def __init__(
        self,
        llm: LLMInterface,
        entity_manager: EntityProfileManager
    ):
        """
        Initialize the response generator.

        Args:
            llm: LLM interface for generation
            entity_manager: Entity profile manager
        """
        self.llm = llm
        self.entity_manager = entity_manager

        logger.info("ResponseGenerator initialized")

    def generate(
        self,
        classified_field: ClassifiedField,
        evidence: List[Evidence],
        entity_id: str,
        max_tokens: int = 1500
    ) -> DraftResponse:
        """
        Generate a draft response for a substantive question.

        Args:
            classified_field: The classified field/question
            evidence: Retrieved evidence from knowledge collection
            entity_id: Entity profile ID
            max_tokens: Maximum tokens for generation

        Returns:
            DraftResponse with generated text and metadata
        """
        start_time = datetime.now()

        # Get entity profile
        entity = self.entity_manager.get_entity_profile(entity_id)
        if not entity:
            raise ValueError(f"Entity not found: {entity_id}")

        company_name = getattr(entity, 'company_name', entity_id)

        # Build prompt
        prompt = self._build_prompt(
            question=classified_field.field_text,
            question_type=classified_field.question_type or QuestionType.GENERAL,
            evidence=evidence,
            company_name=company_name,
            word_limit=classified_field.word_limit
        )

        # Generate response
        try:
            response_text = self.llm.generate(prompt, max_tokens=max_tokens)
            response_text = self._clean_response(response_text)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response_text = f"[GENERATION FAILED: {str(e)}]\n\nPlease write this response manually."

        # Analyze response
        placeholders = self._find_placeholders(response_text)
        confidence = self._calculate_confidence(evidence, response_text, placeholders)
        word_count = len(response_text.split())

        generation_time = (datetime.now() - start_time).total_seconds()

        return DraftResponse(
            question=classified_field.field_text,
            question_type=classified_field.question_type or QuestionType.GENERAL,
            text=response_text,
            evidence_used=evidence,
            confidence=confidence,
            word_count=word_count,
            needs_review=confidence < 0.6 or len(placeholders) > 0,
            placeholders=placeholders,
            generation_time=generation_time,
            metadata={
                'entity_id': entity_id,
                'company_name': company_name,
                'evidence_count': len(evidence),
                'word_limit': classified_field.word_limit
            }
        )

    def _build_prompt(
        self,
        question: str,
        question_type: QuestionType,
        evidence: List[Evidence],
        company_name: str,
        word_limit: Optional[int]
    ) -> str:
        """Build the generation prompt with evidence injection."""

        # Build evidence section
        if evidence:
            evidence_parts = []
            for i, e in enumerate(evidence, 1):
                source_info = f"Source {i}: {e.source_doc}"
                if e.doc_type:
                    source_info += f" ({e.doc_type})"
                evidence_parts.append(f"**{source_info}**\n{e.text}")

            evidence_section = "\n\n---\n\n".join(evidence_parts)
        else:
            evidence_section = "[No specific evidence found in knowledge collection. Please add relevant source documents or write response manually.]"

        # Word limit instruction
        if word_limit:
            word_instruction = f"\nTarget length: approximately {word_limit} words. Be concise but complete."
        else:
            word_instruction = "\nBe thorough but concise. Aim for 150-300 words unless more detail is clearly needed."

        # Get template
        template = self.PROMPT_TEMPLATES.get(
            question_type,
            self.PROMPT_TEMPLATES[QuestionType.GENERAL]
        )

        # Fill template
        prompt = template.format(
            company_name=company_name,
            question=question,
            evidence_section=evidence_section,
            word_instruction=word_instruction
        )

        return prompt

    def _clean_response(self, text: str) -> str:
        """Clean up the generated response."""
        # Remove any "RESPONSE:" prefix the LLM might include
        text = re.sub(r'^RESPONSE:\s*', '', text, flags=re.IGNORECASE)

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    def _find_placeholders(self, text: str) -> List[str]:
        """Find placeholder markers in the response."""
        patterns = [
            r'\[NEEDS DETAIL:[^\]]+\]',
            r'\[NEEDS VERIFICATION:[^\]]+\]',
            r'\[CONFIRM:[^\]]+\]',
            r'\[ATTACH:[^\]]+\]',
            r'\[TODO:[^\]]+\]',
            r'\[INSERT:[^\]]+\]',
            r'\[GENERATION FAILED:[^\]]+\]',
        ]

        placeholders = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            placeholders.extend(matches)

        return placeholders

    def _calculate_confidence(
        self,
        evidence: List[Evidence],
        response_text: str,
        placeholders: List[str]
    ) -> float:
        """
        Calculate confidence score for the response.

        Based on:
        - Quality and quantity of evidence
        - Response length and substance
        - Presence of placeholders
        """
        score = 0.5  # Base score

        # Evidence quality (up to +0.3)
        if evidence:
            avg_relevance = sum(e.relevance_score for e in evidence) / len(evidence)
            evidence_bonus = min(len(evidence) / 5, 1.0) * 0.15  # More evidence = better
            relevance_bonus = avg_relevance * 0.15
            score += evidence_bonus + relevance_bonus
        else:
            score -= 0.2  # No evidence is concerning

        # Response substance (up to +0.15)
        word_count = len(response_text.split())
        if word_count > 100:
            score += 0.05
        if word_count > 200:
            score += 0.05
        if word_count > 300:
            score += 0.05

        # Placeholder penalty (-0.1 per placeholder, max -0.3)
        placeholder_penalty = min(len(placeholders) * 0.1, 0.3)
        score -= placeholder_penalty

        # Check for generic/vague language (-0.1)
        vague_phrases = [
            "we have experience", "we are able to", "we can provide",
            "our team is capable", "we will ensure", "as required"
        ]
        vague_count = sum(1 for phrase in vague_phrases if phrase in response_text.lower())
        if vague_count > 3:
            score -= 0.1

        # Clamp to 0-1 range
        return max(0.0, min(1.0, score))

    def regenerate(
        self,
        previous_response: DraftResponse,
        entity_id: str,
        additional_guidance: str = "",
        new_evidence: Optional[List[Evidence]] = None
    ) -> DraftResponse:
        """
        Regenerate a response with additional guidance.

        Args:
            previous_response: The previous draft to improve
            entity_id: Entity profile ID
            additional_guidance: User guidance for improvement
            new_evidence: Optional new/different evidence to use

        Returns:
            New DraftResponse
        """
        # Use new evidence if provided, otherwise use original
        evidence = new_evidence if new_evidence is not None else previous_response.evidence_used

        # Create enhanced prompt
        entity = self.entity_manager.get_entity_profile(entity_id)
        company_name = getattr(entity, 'company_name', entity_id)

        # Build base prompt
        base_prompt = self._build_prompt(
            question=previous_response.question,
            question_type=previous_response.question_type,
            evidence=evidence,
            company_name=company_name,
            word_limit=previous_response.metadata.get('word_limit')
        )

        # Add regeneration context
        regen_prompt = f"""{base_prompt}

PREVIOUS DRAFT (needs improvement):
{previous_response.text}

IMPROVEMENT GUIDANCE:
{additional_guidance if additional_guidance else "Please improve the response with more specific details from the evidence."}

IMPROVED RESPONSE:"""

        # Generate
        start_time = datetime.now()
        try:
            response_text = self.llm.generate(regen_prompt, max_tokens=1500)
            response_text = self._clean_response(response_text)

            # Remove any "IMPROVED RESPONSE:" prefix
            response_text = re.sub(r'^IMPROVED RESPONSE:\s*', '', response_text, flags=re.IGNORECASE)
        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            return previous_response  # Return original if regeneration fails

        # Analyze new response
        placeholders = self._find_placeholders(response_text)
        confidence = self._calculate_confidence(evidence, response_text, placeholders)
        word_count = len(response_text.split())
        generation_time = (datetime.now() - start_time).total_seconds()

        return DraftResponse(
            question=previous_response.question,
            question_type=previous_response.question_type,
            text=response_text,
            evidence_used=evidence,
            confidence=confidence,
            word_count=word_count,
            needs_review=confidence < 0.6 or len(placeholders) > 0,
            placeholders=placeholders,
            generation_time=generation_time,
            metadata={
                **previous_response.metadata,
                'regenerated': True,
                'regeneration_guidance': additional_guidance
            }
        )


class BatchResponseGenerator:
    """
    Generates responses for multiple questions efficiently.

    Handles batching and progress tracking for processing
    all substantive questions in a proposal.
    """

    def __init__(
        self,
        llm: LLMInterface,
        entity_manager: EntityProfileManager,
        evidence_retriever  # Type hint omitted to avoid circular import
    ):
        """
        Initialize batch generator.

        Args:
            llm: LLM interface
            entity_manager: Entity profile manager
            evidence_retriever: Evidence retriever instance
        """
        self.generator = ResponseGenerator(llm, entity_manager)
        self.evidence_retriever = evidence_retriever
        self.llm = llm
        self.entity_manager = entity_manager

    def generate_all(
        self,
        classified_fields: List[ClassifiedField],
        entity_id: str,
        collection_name: str,
        progress_callback=None
    ) -> List[DraftResponse]:
        """
        Generate responses for all intelligent-tier fields.

        Args:
            classified_fields: Fields to generate responses for
            entity_id: Entity profile ID
            collection_name: Knowledge collection to search
            progress_callback: Optional callback(current, total, field_text)

        Returns:
            List of DraftResponse objects
        """
        # Filter to intelligent tier only
        intelligent_fields = [
            f for f in classified_fields
            if f.tier.value == "intelligent"
        ]

        if not intelligent_fields:
            logger.info("No intelligent-tier fields to process")
            return []

        responses = []
        total = len(intelligent_fields)

        for i, field in enumerate(intelligent_fields):
            if progress_callback:
                progress_callback(i + 1, total, field.field_text[:50])

            try:
                # Retrieve evidence
                evidence_result = self.evidence_retriever.find_evidence(
                    question=field.field_text,
                    question_type=field.question_type or QuestionType.GENERAL,
                    collection_name=collection_name,
                    max_results=5,
                    use_reranker=True
                )

                # Generate response
                response = self.generator.generate(
                    classified_field=field,
                    evidence=evidence_result.evidence,
                    entity_id=entity_id
                )

                responses.append(response)

            except Exception as e:
                logger.error(f"Failed to generate response for field: {e}")
                # Create error response
                responses.append(DraftResponse(
                    question=field.field_text,
                    question_type=field.question_type or QuestionType.GENERAL,
                    text=f"[GENERATION FAILED: {str(e)}]\n\nPlease write this response manually.",
                    evidence_used=[],
                    confidence=0.0,
                    word_count=0,
                    needs_review=True,
                    placeholders=["[GENERATION FAILED]"],
                    generation_time=0,
                    metadata={'error': str(e)}
                ))

        logger.info(f"Generated {len(responses)} responses")
        return responses
