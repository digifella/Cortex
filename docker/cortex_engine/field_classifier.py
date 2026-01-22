"""
Field Classifier for Intelligent Proposal Completion

Version: 1.0.0
Date: 2026-01-19

Purpose: Classifies proposal fields into two tiers:
- Tier 1 (AUTO_COMPLETE): Simple fields fillable from entity profile
- Tier 2 (INTELLIGENT): Substantive questions requiring knowledge search + LLM

The auto-complete field mappings are flexible - users can add custom mappings
on a case-by-case basis or use pre-templated ones.
"""

import re
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

from .utils import get_logger

logger = get_logger(__name__)


class FieldTier(Enum):
    """Processing tier for a field."""
    AUTO_COMPLETE = "auto_complete"  # Tier 1: Direct substitution from entity profile
    INTELLIGENT = "intelligent"       # Tier 2: Requires knowledge search + LLM generation


class QuestionType(Enum):
    """Type of substantive question (Tier 2 only)."""
    CAPABILITY = "capability"           # Experience, track record, qualifications
    METHODOLOGY = "methodology"         # Approach, process, how you'll deliver
    VALUE_PROPOSITION = "value"         # Benefits, impact, outcomes
    COMPLIANCE = "compliance"           # Certifications, policies, standards
    INNOVATION = "innovation"           # Novel approaches, technology, R&D
    RISK = "risk"                       # Risk identification, mitigation strategies
    PERSONNEL = "personnel"             # Team composition, key staff
    PRICING = "pricing"                 # Cost breakdown, rates
    GENERAL = "general"                 # Catch-all for unclassified


@dataclass
class ClassifiedField:
    """Result of field classification."""
    field_text: str
    tier: FieldTier
    question_type: Optional[QuestionType] = None
    auto_complete_mapping: Optional[str] = None  # e.g., "company_name", "abn"
    confidence: float = 0.8
    word_limit: Optional[int] = None
    context_hint: Optional[str] = None  # Additional context for generation


@dataclass
class AutoCompleteMapping:
    """Mapping from field pattern to entity profile field."""
    pattern: str           # Regex pattern to match field text
    profile_field: str     # Entity profile field path (e.g., "company_name", "contact.email")
    description: str       # Human-readable description
    is_regex: bool = True  # If False, pattern is simple substring match
    priority: int = 0      # Higher priority mappings checked first


class FieldClassifier:
    """
    Classifies proposal fields into auto-complete vs intelligent tiers.

    Auto-complete mappings are flexible:
    - Built-in mappings for common tender fields
    - Custom mappings can be added per-entity or globally
    - Mappings stored in JSON for easy editing
    """

    # Patterns that indicate a REAL substantive question (strict mode)
    SUBSTANTIVE_QUESTION_PATTERNS = [
        # Explicit request patterns
        r"^(?:please\s+)?(?:provide|describe|detail|outline|explain)\s+.{20,}",
        r"^(?:please\s+)?(?:demonstrate|evidence|show)\s+.{15,}",
        # Question patterns
        r"^how\s+(?:will|would|do|can)\s+you\s+.{15,}",
        r"^what\s+(?:is|are|will)\s+your\s+.{15,}",
        r"^describe\s+(?:your|the)\s+.{15,}",
        # Specific tender question patterns
        r"capability\s+and\s+capacity",
        r"proposed\s+methodology",
        r"previous\s+(?:experience|projects)",
        r"track\s+record",
        r"(?:economic|social)\s+benefit",
        r"positive\s+impact",
        r"value\s+(?:for|to)\s+(?:money|the)",
        r"risk\s+(?:management|mitigation)",
        r"key\s+personnel",
        r"team\s+(?:composition|structure)",
        r"quality\s+(?:assurance|management)",
        r"how\s+many\s+.{10,}",
        r"will\s+you\s+(?:source|use|engage)",
    ]

    # Patterns that should NOT be extracted as substantive questions
    EXCLUDE_PATTERNS = [
        r"^(?:name|abn|acn|address|email|phone|fax|website)\s*:",
        r"^(?:date|time|signature|witness)\s*:",
        r"^(?:section|part|attachment|schedule)\s+\d",
        r"^table\s+\d",
        r"^\d+\.\d+\s+[A-Z]",  # Section numbers like "1.1 TITLE"
        r"^[A-Z\s]{3,20}:?\s*$",  # ALL CAPS short headers
        r"^\*",  # Footnotes
        r"^note:",
        r"^important:",
        r"^warning:",
    ]

    # Built-in auto-complete patterns (can be extended)
    DEFAULT_AUTO_COMPLETE_MAPPINGS = [
        # Company identification
        AutoCompleteMapping(
            pattern=r"(?:company|business|supplier|tenderer|organisation|organization)[\'\s\-]*(?:s\s+)?(?:legal\s+)?name",
            profile_field="company_name",
            description="Company legal name",
            priority=10
        ),
        AutoCompleteMapping(
            pattern=r"^name\s*:",
            profile_field="company_name",
            description="Name field (company)",
            priority=9
        ),
        AutoCompleteMapping(
            pattern=r"(?:trading|trade)\s*(?:as|name)",
            profile_field="trading_name",
            description="Trading name",
            priority=10
        ),
        AutoCompleteMapping(
            pattern=r"\bABN\b|australian\s+business\s+number",
            profile_field="abn",
            description="Australian Business Number",
            priority=10
        ),
        AutoCompleteMapping(
            pattern=r"\bACN\b|australian\s+company\s+number",
            profile_field="acn",
            description="Australian Company Number",
            priority=10
        ),
        AutoCompleteMapping(
            pattern=r"\bARBN\b",
            profile_field="arbn",
            description="Australian Registered Body Number",
            priority=10
        ),

        # Addresses
        AutoCompleteMapping(
            pattern=r"registered\s+(?:office\s+)?address",
            profile_field="registered_address",
            description="Registered office address",
            priority=9
        ),
        AutoCompleteMapping(
            pattern=r"(?:postal|mailing)\s+address",
            profile_field="postal_address",
            description="Postal address",
            priority=9
        ),
        AutoCompleteMapping(
            pattern=r"(?:business|street|physical)\s+address",
            profile_field="business_address",
            description="Business address",
            priority=9
        ),

        # Contact details
        AutoCompleteMapping(
            pattern=r"(?:email|e-mail)\s*(?:address)?",
            profile_field="contact_email",
            description="Contact email",
            priority=8
        ),
        AutoCompleteMapping(
            pattern=r"(?:phone|telephone|tel)\s*(?:number)?",
            profile_field="contact_phone",
            description="Contact phone",
            priority=8
        ),
        AutoCompleteMapping(
            pattern=r"(?:mobile|cell)\s*(?:number|phone)?",
            profile_field="contact_mobile",
            description="Mobile number",
            priority=8
        ),
        AutoCompleteMapping(
            pattern=r"(?:fax)\s*(?:number)?",
            profile_field="contact_fax",
            description="Fax number",
            priority=7
        ),
        AutoCompleteMapping(
            pattern=r"(?:website|web\s*site|url|web\s+address)",
            profile_field="website",
            description="Website URL",
            priority=8
        ),

        # Contact person
        AutoCompleteMapping(
            pattern=r"(?:contact|authorised|authorized)\s+(?:person|officer|representative)[\s\-]*(?:name)?",
            profile_field="contact_person_name",
            description="Contact person name",
            priority=8
        ),
        AutoCompleteMapping(
            pattern=r"(?:contact|authorised|authorized)\s+(?:person|officer)[\s\']*s?\s*(?:position|title|role)",
            profile_field="contact_person_title",
            description="Contact person title",
            priority=8
        ),

        # Financial/Legal
        AutoCompleteMapping(
            pattern=r"(?:bank|account)\s+(?:name|details)",
            profile_field="bank_name",
            description="Bank name",
            priority=7
        ),
        AutoCompleteMapping(
            pattern=r"\bBSB\b",
            profile_field="bank_bsb",
            description="Bank BSB",
            priority=7
        ),
        AutoCompleteMapping(
            pattern=r"(?:account)\s*(?:number|no\.?)",
            profile_field="bank_account_number",
            description="Bank account number",
            priority=7
        ),
        AutoCompleteMapping(
            pattern=r"(?:insurance|policy)\s*(?:number|no\.?)",
            profile_field="insurance_policy_number",
            description="Insurance policy number",
            priority=7
        ),
        AutoCompleteMapping(
            pattern=r"(?:public\s+liability|professional\s+indemnity)\s+(?:insurance|cover)",
            profile_field="insurance_details",
            description="Insurance details",
            priority=7
        ),

        # Dates
        AutoCompleteMapping(
            pattern=r"date\s+of\s+(?:incorporation|registration)",
            profile_field="incorporation_date",
            description="Date of incorporation",
            priority=6
        ),
    ]

    # Patterns that indicate substantive questions (Tier 2)
    SUBSTANTIVE_INDICATORS = [
        # Question words with substance - assign to likely types
        (r"(?:please\s+)?provide\s+details\s+of\s+(?:your\s+)?(?:capability|capacity)", QuestionType.CAPABILITY),
        (r"(?:please\s+)?(?:describe|detail|explain)\s+(?:your\s+)?(?:experience|capability)", QuestionType.CAPABILITY),
        (r"(?:please\s+)?(?:outline|describe)\s+(?:your\s+)?(?:proposed\s+)?(?:methodology|approach)", QuestionType.METHODOLOGY),
        (r"(?:please\s+)?(?:demonstrate|evidence|show)", QuestionType.CAPABILITY),

        # Capability indicators
        (r"(?:experience|track\s+record|history)\s+(?:in|with|of|delivering)", QuestionType.CAPABILITY),
        (r"(?:previous|past|similar)\s+(?:projects?|work|engagements?)", QuestionType.CAPABILITY),
        (r"(?:qualifications?|expertise|skills?)\s+(?:of|in|relevant)", QuestionType.CAPABILITY),
        (r"capability\s+(?:and\s+)?capacity", QuestionType.CAPABILITY),
        (r"proven\s+(?:ability|track\s+record)", QuestionType.CAPABILITY),
        (r"deliver(?:ing|ed)?\s+(?:the\s+)?(?:required\s+)?services?", QuestionType.CAPABILITY),

        # Methodology indicators
        (r"(?:proposed\s+)?(?:methodology|approach|method)", QuestionType.METHODOLOGY),
        (r"how\s+(?:will|would|do)\s+you", QuestionType.METHODOLOGY),
        (r"(?:outline|describe)\s+(?:your\s+)?(?:approach|process|steps)", QuestionType.METHODOLOGY),
        (r"(?:key\s+)?(?:steps|phases|stages)", QuestionType.METHODOLOGY),
        (r"(?:indicative\s+)?(?:timeline|schedule|milestones)", QuestionType.METHODOLOGY),
        (r"conducting\s+the\s+review", QuestionType.METHODOLOGY),

        # Value proposition indicators
        (r"(?:benefit|value|impact)\s+(?:to|for|on)", QuestionType.VALUE_PROPOSITION),
        (r"positive\s+impact", QuestionType.VALUE_PROPOSITION),
        (r"(?:economic|social)\s+(?:benefit|impact)", QuestionType.VALUE_PROPOSITION),
        (r"(?:local|community)\s+(?:benefit|impact|employment)", QuestionType.VALUE_PROPOSITION),
        (r"how\s+will\s+(?:this|you)\s+(?:benefit|support|contribute)", QuestionType.VALUE_PROPOSITION),

        # Compliance indicators
        (r"(?:confirm|certify|warrant)\s+(?:that|you)", QuestionType.COMPLIANCE),
        (r"(?:comply|compliance)\s+with", QuestionType.COMPLIANCE),
        (r"(?:meet|satisfy)\s+(?:the\s+)?(?:requirements?|standards?)", QuestionType.COMPLIANCE),
        (r"(?:accreditation|certification|ISO)", QuestionType.COMPLIANCE),

        # Innovation indicators
        (r"(?:innovative|novel|unique)\s+(?:approach|solution|method)", QuestionType.INNOVATION),
        (r"(?:innovation|creativity|new\s+ideas?)", QuestionType.INNOVATION),
        (r"(?:adding\s+value|value[\s-]add)", QuestionType.INNOVATION),

        # Risk indicators
        (r"(?:identify|describe)\s+(?:risks?|challenges?)", QuestionType.RISK),
        (r"(?:risk|issue)\s+(?:management|mitigation)", QuestionType.RISK),
        (r"(?:contingency|fallback)\s+(?:plan|arrangements?)", QuestionType.RISK),

        # Personnel indicators
        (r"(?:key\s+)?(?:personnel|staff|team\s+members?)", QuestionType.PERSONNEL),
        (r"(?:qualifications?|experience)\s+of\s+(?:key\s+)?(?:personnel|staff)", QuestionType.PERSONNEL),
        (r"(?:CVs?|resumes?|curriculum\s+vitae)", QuestionType.PERSONNEL),

        # Pricing indicators
        (r"(?:price|cost|fee|rate)\s+(?:breakdown|schedule|details)", QuestionType.PRICING),
        (r"(?:lump\s+sum|schedule\s+of\s+rates)", QuestionType.PRICING),
        (r"(?:pricing|quote|quotation)\s+(?:for|details)", QuestionType.PRICING),
    ]

    def __init__(self, custom_mappings_path: Optional[Path] = None):
        """
        Initialize field classifier.

        Args:
            custom_mappings_path: Optional path to JSON file with custom mappings
        """
        self.auto_complete_mappings: List[AutoCompleteMapping] = list(self.DEFAULT_AUTO_COMPLETE_MAPPINGS)
        self.custom_mappings_path = custom_mappings_path

        # Load custom mappings if provided
        if custom_mappings_path and custom_mappings_path.exists():
            self._load_custom_mappings(custom_mappings_path)

        # Sort by priority (higher first)
        self.auto_complete_mappings.sort(key=lambda m: -m.priority)

        logger.info(f"FieldClassifier initialized with {len(self.auto_complete_mappings)} auto-complete mappings")

    def _load_custom_mappings(self, path: Path):
        """Load custom mappings from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            for item in data.get('mappings', []):
                mapping = AutoCompleteMapping(
                    pattern=item['pattern'],
                    profile_field=item['profile_field'],
                    description=item.get('description', ''),
                    is_regex=item.get('is_regex', True),
                    priority=item.get('priority', 5)
                )
                self.auto_complete_mappings.append(mapping)

            logger.info(f"Loaded {len(data.get('mappings', []))} custom auto-complete mappings")

        except Exception as e:
            logger.error(f"Failed to load custom mappings from {path}: {e}")

    def add_mapping(self, mapping: AutoCompleteMapping):
        """Add a custom mapping at runtime."""
        self.auto_complete_mappings.append(mapping)
        self.auto_complete_mappings.sort(key=lambda m: -m.priority)
        logger.info(f"Added custom mapping: {mapping.description}")

    def save_mappings(self, path: Path):
        """Save current mappings to JSON file."""
        data = {
            'mappings': [
                {
                    'pattern': m.pattern,
                    'profile_field': m.profile_field,
                    'description': m.description,
                    'is_regex': m.is_regex,
                    'priority': m.priority
                }
                for m in self.auto_complete_mappings
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.auto_complete_mappings)} mappings to {path}")

    def is_substantive_question(self, text: str, strict: bool = True) -> bool:
        """
        Determine if text represents a substantive question requiring intelligent response.

        Args:
            text: The text to check
            strict: If True, requires matching substantive patterns; if False, more permissive

        Returns:
            True if this looks like a substantive question
        """
        text_lower = text.lower().strip()

        # Check exclusion patterns first - these are never substantive
        for pattern in self.EXCLUDE_PATTERNS:
            try:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return False
            except re.error:
                pass

        # Too short to be substantive
        if len(text_lower) < 25:
            return False

        # Pure numbers, dates, or codes
        if re.match(r'^[\d\s\.\-\/]+$', text_lower):
            return False

        # Just a label ending with colon or just punctuation
        if re.match(r'^[^:]+:\s*$', text) or len(text.replace(' ', '')) < 10:
            return False

        if strict:
            # Must match at least one substantive pattern
            for pattern in self.SUBSTANTIVE_QUESTION_PATTERNS:
                try:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        return True
                except re.error:
                    pass

            # Also check substantive indicators
            for pattern, _ in self.SUBSTANTIVE_INDICATORS:
                try:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        return True
                except re.error:
                    pass

            return False
        else:
            # Permissive mode - consider it substantive if it has question-like structure
            question_starters = [
                r'^(?:please\s+)?(?:provide|describe|detail|outline|explain)',
                r'^(?:how|what|when|where|why|which)',
                r'^(?:demonstrate|evidence|show)',
            ]
            for pattern in question_starters:
                if re.search(pattern, text_lower):
                    return True
            return False

    def classify(self, field_text: str, context: str = "", strict_filter: bool = True) -> ClassifiedField:
        """
        Classify a single field.

        Args:
            field_text: The field label/question text
            context: Surrounding context (e.g., section heading)
            strict_filter: If True, only substantive patterns pass; reduces false positives

        Returns:
            ClassifiedField with tier, type, and mapping info
        """
        field_lower = field_text.lower().strip()
        context_lower = context.lower() if context else ""

        # First pass: Check for auto-complete match
        for mapping in self.auto_complete_mappings:
            try:
                if mapping.is_regex:
                    if re.search(mapping.pattern, field_lower, re.IGNORECASE):
                        return ClassifiedField(
                            field_text=field_text,
                            tier=FieldTier.AUTO_COMPLETE,
                            question_type=None,
                            auto_complete_mapping=mapping.profile_field,
                            confidence=0.95,
                            word_limit=None,
                            context_hint=mapping.description
                        )
                else:
                    if mapping.pattern.lower() in field_lower:
                        return ClassifiedField(
                            field_text=field_text,
                            tier=FieldTier.AUTO_COMPLETE,
                            question_type=None,
                            auto_complete_mapping=mapping.profile_field,
                            confidence=0.90,
                            word_limit=None,
                            context_hint=mapping.description
                        )
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{mapping.pattern}': {e}")

        # Second pass: Check if this is actually a substantive question
        if strict_filter and not self.is_substantive_question(field_text, strict=True):
            # Not substantive enough - mark as auto-complete with unknown mapping
            return ClassifiedField(
                field_text=field_text,
                tier=FieldTier.AUTO_COMPLETE,
                question_type=None,
                auto_complete_mapping=None,  # Unknown - needs manual entry
                confidence=0.5,
                word_limit=None,
                context_hint="Not recognized as substantive question"
            )

        # Third pass: Classify as substantive question
        question_type = self._classify_question_type(field_lower, context_lower)
        word_limit = self._extract_word_limit(field_text, context)

        return ClassifiedField(
            field_text=field_text,
            tier=FieldTier.INTELLIGENT,
            question_type=question_type,
            auto_complete_mapping=None,
            confidence=0.8,
            word_limit=word_limit,
            context_hint=None
        )

    def _classify_question_type(self, field_lower: str, context_lower: str) -> QuestionType:
        """Determine the type of substantive question."""
        combined = f"{field_lower} {context_lower}"

        # Check each substantive indicator
        type_scores: Dict[QuestionType, int] = {}

        for pattern, qtype in self.SUBSTANTIVE_INDICATORS:
            try:
                if re.search(pattern, combined, re.IGNORECASE):
                    type_scores[qtype] = type_scores.get(qtype, 0) + 1
            except re.error:
                pass

        # Return highest scoring type, or GENERAL if no matches
        if type_scores:
            return max(type_scores, key=type_scores.get)

        return QuestionType.GENERAL

    def _extract_word_limit(self, field_text: str, context: str) -> Optional[int]:
        """Extract word limit if specified in field or context."""
        combined = f"{field_text} {context}"

        # Patterns for word limits
        patterns = [
            r"(?:maximum|max\.?|limit)\s*(?:of\s+)?(\d+)\s*words?",
            r"(\d+)\s*words?\s*(?:maximum|max\.?|limit)",
            r"(?:approximately|approx\.?|about|around)\s*(\d+)\s*words?",
            r"(\d+)[-–]\s*(\d+)\s*words?",  # Range like "200-300 words"
        ]

        for pattern in patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                # For ranges, take the upper limit
                groups = match.groups()
                if len(groups) == 2 and groups[1]:
                    return int(groups[1])
                return int(groups[0])

        return None

    def classify_batch(
        self,
        fields: List[Dict[str, str]],
        strict_filter: bool = True
    ) -> List[ClassifiedField]:
        """
        Classify multiple fields.

        Args:
            fields: List of dicts with 'text' and optional 'context' keys
            strict_filter: If True, only substantive patterns pass; reduces false positives

        Returns:
            List of ClassifiedField objects
        """
        results = []
        for field in fields:
            text = field.get('text', '')
            context = field.get('context', '')
            results.append(self.classify(text, context, strict_filter=strict_filter))
        return results

    def get_auto_complete_summary(self) -> List[Dict[str, str]]:
        """Get summary of all auto-complete mappings for UI display."""
        return [
            {
                'pattern': m.pattern,
                'field': m.profile_field,
                'description': m.description
            }
            for m in self.auto_complete_mappings
        ]


# Singleton instance for easy importing
_classifier_instance: Optional[FieldClassifier] = None

def get_field_classifier(custom_mappings_path: Optional[Path] = None) -> FieldClassifier:
    """Get or create the field classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FieldClassifier(custom_mappings_path)
    return _classifier_instance
