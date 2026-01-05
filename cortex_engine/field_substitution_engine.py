"""
Field Substitution Engine - Resolve @mentions to Values
Version: 1.0.0
Date: 2026-01-05

Purpose: Resolve parsed @mentions to actual values from entity profiles.
Handles:
- Simple field substitution (@companyname → "Acme Corp")
- Structured data access (@insurance.public_liability.coverage → "$20,000,000")
- Content generation (@cv[paul_smith] → Formatted CV)
- Creative generation (@generate[type=executive_summary] → LLM-generated content)
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

from .mention_parser import ParsedMention, MentionType
from .entity_profile_manager import EntityProfileManager
from .entity_profile_schema import (
    EntityProfile,
    TeamMember,
    Project,
    Reference,
    Insurance,
    Capability
)
from .utils import get_logger

logger = get_logger(__name__)


class SubstitutionResult:
    """Result of mention substitution."""

    def __init__(
        self,
        success: bool,
        value: Optional[str] = None,
        error_message: Optional[str] = None,
        requires_llm: bool = False,
        generation_context: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.value = value
        self.error_message = error_message
        self.requires_llm = requires_llm
        self.generation_context = generation_context or {}

    def __str__(self) -> str:
        if self.success:
            return f"<Success: {self.value[:50]}...>" if self.value else "<Success: Empty>"
        else:
            return f"<Error: {self.error_message}>"


class FieldSubstitutionEngine:
    """Engine for resolving @mentions to actual values."""

    def __init__(self, profile_manager: EntityProfileManager):
        """
        Initialize engine.

        Args:
            profile_manager: EntityProfileManager instance
        """
        self.profile_manager = profile_manager
        logger.info("FieldSubstitutionEngine initialized")

    def resolve(
        self,
        mention: ParsedMention,
        entity_id: str
    ) -> SubstitutionResult:
        """
        Resolve a mention to its value.

        Args:
            mention: Parsed mention
            entity_id: Entity ID to resolve against

        Returns:
            SubstitutionResult

        Examples:
            >>> engine = FieldSubstitutionEngine(profile_manager)
            >>> mention = parser.parse("@companyname")
            >>> result = engine.resolve(mention, "longboardfella_consulting")
            >>> result.value
            'Longboardfella Consulting Pty Ltd'
        """
        if not mention.is_valid:
            return SubstitutionResult(
                success=False,
                error_message=mention.error_message
            )

        # Route to appropriate resolver based on type
        if mention.mention_type == MentionType.SIMPLE:
            return self._resolve_simple(mention, entity_id)

        elif mention.mention_type == MentionType.STRUCTURED:
            return self._resolve_structured(mention, entity_id)

        elif mention.mention_type == MentionType.CONTENT_GEN:
            return self._resolve_content_gen(mention, entity_id)

        elif mention.mention_type == MentionType.NARRATIVE:
            return self._resolve_narrative(mention, entity_id)

        elif mention.mention_type == MentionType.CREATIVE_GEN:
            return self._resolve_creative_gen(mention, entity_id)

        return SubstitutionResult(
            success=False,
            error_message=f"Unknown mention type: {mention.mention_type}"
        )

    def resolve_all(
        self,
        mentions: List[ParsedMention],
        entity_id: str
    ) -> Dict[str, SubstitutionResult]:
        """
        Resolve multiple mentions.

        Args:
            mentions: List of ParsedMention objects
            entity_id: Entity ID

        Returns:
            Dict mapping mention.raw_text → SubstitutionResult
        """
        results = {}

        for mention in mentions:
            results[mention.raw_text] = self.resolve(mention, entity_id)

        return results

    # ========================================
    # SIMPLE FIELD RESOLUTION
    # ========================================

    def _resolve_simple(self, mention: ParsedMention, entity_id: str) -> SubstitutionResult:
        """Resolve simple field mention."""
        profile = self.profile_manager.get_entity_profile(entity_id)

        if not profile:
            return SubstitutionResult(
                success=False,
                error_message=f"Entity profile not found: {entity_id}"
            )

        # Navigate to field using dot notation
        value = self._get_nested_field(profile, mention.field_path)

        if value is None:
            return SubstitutionResult(
                success=False,
                error_message=f"Field not found: {mention.field_path}"
            )

        # Apply formatting
        formatted_value = self._format_value(value, mention, profile)

        return SubstitutionResult(
            success=True,
            value=formatted_value
        )

    # ========================================
    # STRUCTURED FIELD RESOLUTION
    # ========================================

    def _resolve_structured(self, mention: ParsedMention, entity_id: str) -> SubstitutionResult:
        """Resolve structured data access (e.g., @insurance.public_liability.coverage)."""
        profile = self.profile_manager.get_entity_profile(entity_id)

        if not profile:
            return SubstitutionResult(
                success=False,
                error_message=f"Entity profile not found: {entity_id}"
            )

        parts = mention.field_path.split('.')
        root = parts[0]

        # Handle different root types
        if root == 'insurance':
            return self._resolve_insurance_field(mention, entity_id, parts[1:])

        elif root == 'team':
            return self._resolve_team_field(mention, entity_id, parts[1:])

        elif root == 'projects':
            return self._resolve_project_field(mention, entity_id, parts[1:])

        elif root == 'references':
            return self._resolve_reference_field(mention, entity_id, parts[1:])

        elif root == 'capabilities':
            return self._resolve_capability_field(mention, entity_id, parts[1:])

        # Try general nested field access
        value = self._get_nested_field(profile, mention.field_path)

        if value is None:
            return SubstitutionResult(
                success=False,
                error_message=f"Field not found: {mention.field_path}"
            )

        return SubstitutionResult(
            success=True,
            value=str(value)
        )

    def _resolve_insurance_field(
        self,
        mention: ParsedMention,
        entity_id: str,
        path_parts: List[str]
    ) -> SubstitutionResult:
        """Resolve insurance-specific field (e.g., @insurance.public_liability.coverage)."""
        if not path_parts:
            return SubstitutionResult(success=False, error_message="Missing insurance policy ID")

        policy_id = path_parts[0]
        insurance = self.profile_manager.get_insurance(entity_id, policy_id)

        if not insurance:
            return SubstitutionResult(
                success=False,
                error_message=f"Insurance policy not found: {policy_id}"
            )

        # Get specific field if requested
        if len(path_parts) > 1:
            field_path = '.'.join(path_parts[1:])
            value = self._get_nested_field(insurance, field_path)
        else:
            # Return formatted summary
            value = f"{insurance.policy_type.value.replace('_', ' ').title()}: {insurance.coverage.formatted}"

        if value is None:
            return SubstitutionResult(
                success=False,
                error_message=f"Insurance field not found: {'.'.join(path_parts[1:])}"
            )

        return SubstitutionResult(success=True, value=str(value))

    def _resolve_team_field(
        self,
        mention: ParsedMention,
        entity_id: str,
        path_parts: List[str]
    ) -> SubstitutionResult:
        """Resolve team-specific field (e.g., @team.paul_smith.role)."""
        if not path_parts:
            return SubstitutionResult(success=False, error_message="Missing person ID")

        person_id = path_parts[0]
        team_member = self.profile_manager.get_team_member(entity_id, person_id)

        if not team_member:
            return SubstitutionResult(
                success=False,
                error_message=f"Team member not found: {person_id}"
            )

        # Get specific field if requested
        if len(path_parts) > 1:
            field_path = '.'.join(path_parts[1:])
            value = self._get_nested_field(team_member, field_path)
        else:
            value = team_member.full_name

        if value is None:
            return SubstitutionResult(
                success=False,
                error_message=f"Team field not found: {'.'.join(path_parts[1:])}"
            )

        return SubstitutionResult(success=True, value=str(value))

    def _resolve_project_field(
        self,
        mention: ParsedMention,
        entity_id: str,
        path_parts: List[str]
    ) -> SubstitutionResult:
        """Resolve project-specific field (e.g., @projects.health_transformation_2023.value)."""
        if not path_parts:
            return SubstitutionResult(success=False, error_message="Missing project ID")

        project_id = path_parts[0]
        project = self.profile_manager.get_project(entity_id, project_id)

        if not project:
            return SubstitutionResult(
                success=False,
                error_message=f"Project not found: {project_id}"
            )

        # Get specific field if requested
        if len(path_parts) > 1:
            field_path = '.'.join(path_parts[1:])
            value = self._get_nested_field(project, field_path)
        else:
            value = project.project_name

        if value is None:
            return SubstitutionResult(
                success=False,
                error_message=f"Project field not found: {'.'.join(path_parts[1:])}"
            )

        return SubstitutionResult(success=True, value=str(value))

    def _resolve_reference_field(
        self,
        mention: ParsedMention,
        entity_id: str,
        path_parts: List[str]
    ) -> SubstitutionResult:
        """Resolve reference-specific field (e.g., @references.sarah_johnson.organization)."""
        if not path_parts:
            return SubstitutionResult(success=False, error_message="Missing reference ID")

        reference_id = path_parts[0]
        reference = self.profile_manager.get_reference(entity_id, reference_id)

        if not reference:
            return SubstitutionResult(
                success=False,
                error_message=f"Reference not found: {reference_id}"
            )

        # Get specific field if requested
        if len(path_parts) > 1:
            field_path = '.'.join(path_parts[1:])
            value = self._get_nested_field(reference, field_path)
        else:
            value = f"{reference.contact_name}, {reference.organization}"

        if value is None:
            return SubstitutionResult(
                success=False,
                error_message=f"Reference field not found: {'.'.join(path_parts[1:])}"
            )

        return SubstitutionResult(success=True, value=str(value))

    def _resolve_capability_field(
        self,
        mention: ParsedMention,
        entity_id: str,
        path_parts: List[str]
    ) -> SubstitutionResult:
        """Resolve capability-specific field (e.g., @capabilities.iso_9001_2015.certification_number)."""
        if not path_parts:
            return SubstitutionResult(success=False, error_message="Missing capability ID")

        capability_id = path_parts[0]
        capability = self.profile_manager.get_capability(entity_id, capability_id)

        if not capability:
            return SubstitutionResult(
                success=False,
                error_message=f"Capability not found: {capability_id}"
            )

        # Get specific field if requested
        if len(path_parts) > 1:
            field_path = '.'.join(path_parts[1:])
            value = self._get_nested_field(capability, field_path)
        else:
            value = capability.capability_name

        if value is None:
            return SubstitutionResult(
                success=False,
                error_message=f"Capability field not found: {'.'.join(path_parts[1:])}"
            )

        return SubstitutionResult(success=True, value=str(value))

    # ========================================
    # CONTENT GENERATION RESOLUTION
    # ========================================

    def _resolve_content_gen(self, mention: ParsedMention, entity_id: str) -> SubstitutionResult:
        """
        Resolve content generation mention (e.g., @cv[paul_smith, format=brief]).

        Note: This returns a result that indicates LLM generation is required.
        The actual generation happens in a separate LLM generation step.
        """
        root = mention.root_field
        item_id = mention.parameters.get('_positional')

        if not item_id:
            return SubstitutionResult(
                success=False,
                error_message=f"Missing item ID for {root} mention"
            )

        # Validate item exists
        if root == 'team':
            item = self.profile_manager.get_team_member(entity_id, item_id)
            if not item:
                return SubstitutionResult(
                    success=False,
                    error_message=f"Team member not found: {item_id}"
                )

        elif root == 'projects':
            item = self.profile_manager.get_project(entity_id, item_id)
            if not item:
                return SubstitutionResult(
                    success=False,
                    error_message=f"Project not found: {item_id}"
                )

        elif root == 'references':
            item = self.profile_manager.get_reference(entity_id, item_id)
            if not item:
                return SubstitutionResult(
                    success=False,
                    error_message=f"Reference not found: {item_id}"
                )

        # Return success but indicate LLM generation needed
        return SubstitutionResult(
            success=True,
            value=None,  # Will be filled by LLM
            requires_llm=True,
            generation_context={
                'type': root,
                'item_id': item_id,
                'parameters': mention.parameters,
                'entity_id': entity_id
            }
        )

    # ========================================
    # NARRATIVE RESOLUTION
    # ========================================

    def _resolve_narrative(self, mention: ParsedMention, entity_id: str) -> SubstitutionResult:
        """Resolve narrative section mention (e.g., @narrative[company_overview])."""
        section_name = mention.parameters.get('_positional')

        if not section_name:
            return SubstitutionResult(
                success=False,
                error_message="Missing section name for narrative mention"
            )

        # Get narrative section
        narrative_content = self.profile_manager.get_narrative_section(entity_id, section_name)

        if not narrative_content:
            return SubstitutionResult(
                success=False,
                error_message=f"Narrative section not found: {section_name}"
            )

        return SubstitutionResult(
            success=True,
            value=narrative_content
        )

    # ========================================
    # CREATIVE GENERATION RESOLUTION
    # ========================================

    def _resolve_creative_gen(self, mention: ParsedMention, entity_id: str) -> SubstitutionResult:
        """
        Resolve creative generation mention (e.g., @generate[type=executive_summary, ...]).

        Note: This returns a result that indicates LLM generation is required.
        """
        gen_type = mention.parameters.get('type')

        if not gen_type:
            return SubstitutionResult(
                success=False,
                error_message="Missing 'type' parameter for creative generation"
            )

        return SubstitutionResult(
            success=True,
            value=None,  # Will be filled by LLM
            requires_llm=True,
            generation_context={
                'type': 'creative',
                'generation_type': gen_type,
                'parameters': mention.parameters,
                'entity_id': entity_id
            }
        )

    # ========================================
    # HELPER METHODS
    # ========================================

    def _get_nested_field(self, obj: Any, field_path: str) -> Optional[Any]:
        """
        Get nested field value using dot notation.

        Args:
            obj: Object to traverse
            field_path: Dot-notation path (e.g., "company.legal_name")

        Returns:
            Field value or None if not found

        Examples:
            >>> _get_nested_field(profile, "company.legal_name")
            'Acme Corp'
        """
        parts = field_path.split('.')
        current = obj

        for part in parts:
            if current is None:
                return None

            # Try attribute access first
            if hasattr(current, part):
                current = getattr(current, part)
            # Try dict access
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _format_value(self, value: Any, mention: ParsedMention, profile: EntityProfile) -> str:
        """
        Format value according to entity formatting preferences.

        Args:
            value: Raw value
            mention: ParsedMention (for context)
            profile: EntityProfile (for formatting preferences)

        Returns:
            Formatted string
        """
        # Handle special formatting cases
        root = mention.root_field

        # ABN formatting
        if root == 'abn' or 'abn' in mention.field_path:
            return profile.format_abn()

        # ACN formatting
        if root == 'acn' or 'acn' in mention.field_path:
            return profile.format_acn()

        # Address formatting
        if 'registered_office' in mention.field_path or 'postal_address' in mention.field_path:
            if hasattr(value, 'formatted'):
                return value.formatted(single_line=True)

        # List formatting
        if isinstance(value, list):
            return ', '.join(str(v) for v in value)

        # Date formatting
        from datetime import date, datetime
        if isinstance(value, (date, datetime)):
            date_format = profile.formatting.date_format.value
            if date_format == "DD/MM/YYYY":
                return value.strftime("%d/%m/%Y")
            elif date_format == "MM/DD/YYYY":
                return value.strftime("%m/%d/%Y")
            elif date_format == "YYYY-MM-DD":
                return value.strftime("%Y-%m-%d")
            else:  # ISO8601
                return value.isoformat()

        # Default: convert to string
        return str(value)
