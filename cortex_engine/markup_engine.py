"""
Markup Engine
Version: 1.0.0
Date: 2026-01-05

Purpose: LLM-assisted markup of tender documents with @mention suggestions.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from .mention_parser import MentionParser
from .entity_profile_manager import EntityProfileManager
from .workspace_model import MentionBinding
from .utils import get_logger
from .llm_interface import LLMInterface

logger = get_logger(__name__)


class MarkupEngine:
    """LLM-assisted markup engine for suggesting @mentions."""

    def __init__(
        self,
        entity_manager: EntityProfileManager,
        llm: LLMInterface
    ):
        """
        Initialize markup engine.

        Args:
            entity_manager: Entity profile manager
            llm: LLM interface
        """
        self.entity_manager = entity_manager
        self.llm = llm
        self.parser = MentionParser()

        logger.info("MarkupEngine initialized")

    def analyze_document(
        self,
        document_text: str,
        entity_id: str
    ) -> List[MentionBinding]:
        """
        Analyze document and suggest @mention placements.

        Args:
            document_text: Full document text
            entity_id: Entity profile ID to use

        Returns:
            List of suggested mention bindings

        Example:
            >>> engine = MarkupEngine(entity_manager, llm)
            >>> mentions = engine.analyze_document(tender_text, "longboardfella_consulting")
            >>> print(f"Suggested {len(mentions)} mentions")
        """
        logger.info(f"Analyzing document for entity {entity_id}")

        # Get entity profile to know what data is available
        profile = self.entity_manager.get_entity_profile(entity_id)

        if not profile:
            raise ValueError(f"Entity not found: {entity_id}")

        suggestions = []

        # 1. Pattern-based detection (fast, deterministic)
        pattern_suggestions = self._detect_pattern_based_fields(document_text, entity_id)
        suggestions.extend(pattern_suggestions)

        # 2. LLM-assisted detection (slower, contextual)
        # Disabled for MVP - can enable later
        # llm_suggestions = self._detect_llm_based_fields(document_text, entity_id)
        # suggestions.extend(llm_suggestions)

        logger.info(f"Generated {len(suggestions)} mention suggestions")

        return suggestions

    def _detect_pattern_based_fields(
        self,
        text: str,
        entity_id: str
    ) -> List[MentionBinding]:
        """
        Detect fields using pattern matching.

        Args:
            text: Document text
            entity_id: Entity ID

        Returns:
            List of mention bindings
        """
        suggestions = []

        # Patterns for common fields
        patterns = {
            # Company details
            r'(?i)legal\s+(?:entity\s+)?name[:\s]*$': '@companyname',
            r'(?i)company\s+name[:\s]*$': '@companyname',
            r'(?i)business\s+name[:\s]*$': '@companyname',
            r'(?i)abn[:\s]*$': '@abn',
            r'(?i)australian\s+business\s+number[:\s]*$': '@abn',
            r'(?i)acn[:\s]*$': '@acn',
            r'(?i)australian\s+company\s+number[:\s]*$': '@acn',

            # Contact details
            r'(?i)registered\s+(?:office\s+)?address[:\s]*$': '@registered_office',
            r'(?i)(?:business\s+)?address[:\s]*$': '@registered_office',
            r'(?i)phone[:\s]*$': '@phone',
            r'(?i)telephone[:\s]*$': '@phone',
            r'(?i)contact\s+(?:phone|number)[:\s]*$': '@phone',
            r'(?i)email[:\s]*$': '@email',
            r'(?i)e-mail[:\s]*$': '@email',
            r'(?i)contact\s+email[:\s]*$': '@email',
            r'(?i)website[:\s]*$': '@website',
            r'(?i)web\s+(?:site|address)[:\s]*$': '@website',

            # Executive summary / company overview
            r'(?i)executive\s+summary[:\s]*$': '@narrative[company_overview]',
            r'(?i)company\s+(?:overview|profile)[:\s]*$': '@narrative[company_overview]',
            r'(?i)about\s+(?:the\s+)?company[:\s]*$': '@narrative[company_overview]',

            # Insurance
            r'(?i)insurance\s+coverage[:\s]*$': '@insurance.public_liability.coverage',
            r'(?i)public\s+liability[:\s]*$': '@insurance.public_liability.coverage',
            r'(?i)insurance\s+(?:policy\s+)?number[:\s]*$': '@insurance.public_liability.policy_number',
        }

        lines = text.split('\n')

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()

            for pattern, mention in patterns.items():
                if re.search(pattern, line_stripped):
                    # Found a potential field
                    parsed = self.parser.parse(mention)

                    location = f"Line {line_num + 1}"

                    # Check if this section already exists
                    if line_num > 0:
                        # Look at previous lines for section heading
                        for i in range(max(0, line_num - 5), line_num):
                            if self._is_section_heading(lines[i]):
                                location = lines[i].strip()
                                break

                    binding = MentionBinding(
                        mention_text=mention,
                        mention_type=parsed.mention_type.value,
                        field_path=parsed.field_path,
                        location=location,
                        suggested_by_llm=False
                    )

                    suggestions.append(binding)

        return suggestions

    def _detect_llm_based_fields(
        self,
        text: str,
        entity_id: str
    ) -> List[MentionBinding]:
        """
        Use LLM to detect fields that need @mentions.

        Args:
            text: Document text
            entity_id: Entity ID

        Returns:
            List of mention bindings
        """
        # TODO: Implement LLM-based detection
        # This would use the LLM to understand context and suggest mentions
        # For example, detecting when a project example is needed, or a team member CV

        return []

    def _is_section_heading(self, line: str) -> bool:
        """Check if line is a section heading."""
        line = line.strip()

        if not line or len(line) < 5:
            return False

        # Common patterns
        patterns = [
            line.startswith('SECTION'),
            line.startswith('PART'),
            line.startswith('CHAPTER'),
            line.isupper() and len(line.split()) <= 10,
            line.startswith('##'),
        ]

        # Numbered sections
        if line[0].isdigit() and ('.' in line[:10]):
            return True

        return any(patterns)

    def insert_mentions_in_document(
        self,
        document_text: str,
        mention_bindings: List[MentionBinding]
    ) -> str:
        """
        Insert @mentions into document text at suggested locations.

        Args:
            document_text: Original document text
            mention_bindings: List of approved mention bindings

        Returns:
            Document text with @mentions inserted

        Example:
            >>> marked_up = engine.insert_mentions_in_document(
            ...     document_text,
            ...     approved_mentions
            ... )
        """
        lines = document_text.split('\n')

        # Group mentions by location
        mentions_by_location: Dict[str, List[MentionBinding]] = {}
        for binding in mention_bindings:
            if binding.approved and not binding.rejected:
                loc = binding.location
                if loc not in mentions_by_location:
                    mentions_by_location[loc] = []
                mentions_by_location[loc].append(binding)

        # Insert mentions
        for line_num, line in enumerate(lines):
            # Check if this line matches a location
            location_key = f"Line {line_num + 1}"

            if location_key in mentions_by_location:
                # Insert mentions after this line
                for binding in mentions_by_location[location_key]:
                    lines[line_num] += f"\n{binding.mention_text}"

        return '\n'.join(lines)

    def validate_mentions(
        self,
        document_text: str,
        entity_id: str
    ) -> Tuple[List[str], List[str]]:
        """
        Validate all @mentions in document.

        Args:
            document_text: Document text with @mentions
            entity_id: Entity ID

        Returns:
            Tuple of (valid_mentions, invalid_mentions)

        Example:
            >>> valid, invalid = engine.validate_mentions(marked_up_text, "longboardfella_consulting")
            >>> print(f"Valid: {len(valid)}, Invalid: {len(invalid)}")
        """
        # Find all mentions
        mentions = self.parser.parse_all(document_text)

        valid_mentions = []
        invalid_mentions = []

        # Try to resolve each mention
        from .field_substitution_engine import FieldSubstitutionEngine

        engine = FieldSubstitutionEngine(self.entity_manager)

        for mention in mentions:
            result = engine.resolve(mention, entity_id)

            if result.success or result.requires_llm:
                valid_mentions.append(mention.raw_text)
            else:
                invalid_mentions.append(mention.raw_text)

        logger.info(f"Validation: {len(valid_mentions)} valid, {len(invalid_mentions)} invalid")

        return valid_mentions, invalid_mentions
