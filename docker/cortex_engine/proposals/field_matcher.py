"""
AI-Powered Field Matcher

Intelligently matches detected tender fields to workspace entity data using:
1. Structured field matching (direct lookups for ABN, address, insurance, etc.)
2. Semantic field matching (LLM-powered extraction from workspace documents)

Author: Cortex Suite
Created: 2026-01-04
Version: 1.0.0
"""

import json
import asyncio
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import logging
import ollama

# Handle imports for both module and direct execution
if TYPE_CHECKING or __name__ != "__main__":
    from ..workspace_schema import FieldMapping
    from .tender_field_parser import DetectedField
else:
    # Will be imported in __main__ block for direct execution
    FieldMapping = None
    DetectedField = None

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching a single field."""
    field_id: str
    matched: bool
    matched_data: Optional[str]
    data_source: str  # "organization.abn" | "semantic_search" | "insurance[0]"
    confidence: float  # 0.0 - 1.0
    reasoning: str  # Explanation for user
    alternatives: List[Dict[str, Any]]  # Alternative matches


class FieldMatcher:
    """
    Hybrid field matching engine using structured lookups + semantic search + LLM extraction.

    Matching Strategy:
    - Structured fields (ABN, ACN, address, etc.) → Direct lookup (confidence: 1.0)
    - Insurance fields → Filter list by type (confidence: 0.9-1.0)
    - Unstructured fields (experience, projects) → Semantic search + LLM (confidence: 0.3-0.9)
    """

    # Mapping of field types to entity data paths
    STRUCTURED_FIELD_PATHS = {
        "abn": "organization.abn",
        "acn": "organization.acn",
        "legal_name": "organization.legal_name",
        "address": "organization.address",
        "email": "organization.email",
        "phone": "organization.phone",
        "website": "organization.website",
        "gst_registered": "organization.gst_registered",
    }

    # Insurance type mappings
    INSURANCE_TYPE_MAP = {
        "insurance_public_liability": "Public Liability",
        "insurance_professional_indemnity": "Professional Indemnity",
        "insurance_workers_compensation": "Workers Compensation",
    }

    def __init__(
        self,
        entity_data: Dict[str, Any],
        workspace_id: str,
        model_name: str = "qwen2.5:14b-instruct-q4_K_M"
    ):
        """
        Initialize field matcher.

        Args:
            entity_data: Entity snapshot data (organization, insurances, projects, etc.)
            workspace_id: Workspace identifier
            model_name: LLM model for semantic extraction
        """
        self.entity_data = entity_data
        self.workspace_id = workspace_id
        self.model_name = model_name

        # Initialize Ollama client
        try:
            self.ollama_client = ollama.Client()
            logger.info(f"FieldMatcher initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            self.ollama_client = None

    def match_all_fields(
        self,
        detected_fields: List[DetectedField]
    ) -> List[FieldMapping]:
        """
        Match all detected fields using hybrid approach.

        Args:
            detected_fields: List of fields detected in tender

        Returns:
            List of FieldMapping objects with match results
        """
        logger.info(f"Matching {len(detected_fields)} detected fields...")

        field_mappings = []

        # Batch by match type for efficiency
        structured_fields = []
        insurance_fields = []
        semantic_fields = []

        for field in detected_fields:
            # Determine match strategy based on classification hints
            if self._is_structured_field(field):
                structured_fields.append(field)
            elif self._is_insurance_field(field):
                insurance_fields.append(field)
            else:
                semantic_fields.append(field)

        # Process structured fields (fast, synchronous)
        logger.info(f"Processing {len(structured_fields)} structured fields...")
        for field in structured_fields:
            mapping = self._match_structured_field(field)
            field_mappings.append(mapping)

        # Process insurance fields
        logger.info(f"Processing {len(insurance_fields)} insurance fields...")
        for field in insurance_fields:
            mapping = self._match_insurance_field(field)
            field_mappings.append(mapping)

        # Process semantic fields (slower, could be async in future)
        logger.info(f"Processing {len(semantic_fields)} semantic fields...")
        for field in semantic_fields:
            mapping = self._match_semantic_field(field)
            field_mappings.append(mapping)

        logger.info(f"Matching complete: {len(field_mappings)} field mappings created")

        return field_mappings

    # ==================== Structured Field Matching ====================

    def _is_structured_field(self, field: DetectedField) -> bool:
        """Check if field should use structured matching."""
        if not field.classification_hints:
            return False

        # Check if any classification hint maps to a structured path
        for hint in field.classification_hints:
            if hint in self.STRUCTURED_FIELD_PATHS:
                return True

        return False

    def _match_structured_field(self, field: DetectedField) -> FieldMapping:
        """
        Match field using direct lookup in entity data.

        Returns:
            FieldMapping with match result
        """
        # Get the highest confidence classification hint
        field_type = field.classification_hints[0] if field.classification_hints else None

        if not field_type or field_type not in self.STRUCTURED_FIELD_PATHS:
            return self._create_no_match_mapping(field, "No structured path found")

        # Get data path
        data_path = self.STRUCTURED_FIELD_PATHS[field_type]

        # Navigate to data
        matched_data = self._get_nested_value(self.entity_data, data_path)

        if matched_data is None:
            return self._create_no_match_mapping(
                field,
                f"No data found at path: {data_path}"
            )

        # Format the data for display
        formatted_data = self._format_value(matched_data)

        return FieldMapping(
            field_id=field.field_id,
            field_location=field.location,
            field_description=field.field_description,
            field_type=field_type,
            matched_data=formatted_data,
            data_source=data_path,
            confidence=1.0,  # Exact match
            user_approved=False,
            user_override=None
        )

    # ==================== Insurance Field Matching ====================

    def _is_insurance_field(self, field: DetectedField) -> bool:
        """Check if field is insurance-related."""
        if not field.classification_hints:
            return False

        return any(
            hint in self.INSURANCE_TYPE_MAP or hint.startswith("insurance_")
            for hint in field.classification_hints
        )

    def _match_insurance_field(self, field: DetectedField) -> FieldMapping:
        """
        Match insurance field by filtering insurance list.

        Returns:
            FieldMapping with matched insurance data
        """
        # Get insurance list from entity data
        insurances = self.entity_data.get("insurances", [])

        if not insurances:
            return self._create_no_match_mapping(field, "No insurance data available")

        # Determine which insurance type we're looking for
        insurance_type = None
        for hint in field.classification_hints:
            if hint in self.INSURANCE_TYPE_MAP:
                insurance_type = self.INSURANCE_TYPE_MAP[hint]
                break

        # If we have a specific type, filter for it
        if insurance_type:
            matching_insurance = next(
                (ins for ins in insurances if ins.get("insurance_type") == insurance_type),
                None
            )

            if matching_insurance:
                # Check what specific field is needed (policy number, expiry, etc.)
                if "policy_number" in field.classification_hints:
                    matched_data = matching_insurance.get("policy_number")
                    data_source = f"insurance[{insurance_type}].policy_number"
                elif "expiry" in field.classification_hints:
                    matched_data = matching_insurance.get("expiry_date")
                    data_source = f"insurance[{insurance_type}].expiry_date"
                else:
                    # Return full insurance description
                    matched_data = f"{matching_insurance.get('insurer')} - Policy: {matching_insurance.get('policy_number')}"
                    data_source = f"insurance[{insurance_type}]"

                return FieldMapping(
                    field_id=field.field_id,
                    field_location=field.location,
                    field_description=field.field_description,
                    field_type=field.classification_hints[0] if field.classification_hints else None,
                    matched_data=str(matched_data) if matched_data else None,
                    data_source=data_source,
                    confidence=0.95,
                    user_approved=False,
                    user_override=None
                )

        # No specific type - return first insurance as suggestion
        first_insurance = insurances[0]
        matched_data = f"{first_insurance.get('insurance_type')} - {first_insurance.get('insurer')}"

        return FieldMapping(
            field_id=field.field_id,
            field_location=field.location,
            field_description=field.field_description,
            field_type=field.classification_hints[0] if field.classification_hints else None,
            matched_data=matched_data,
            data_source="insurance[0]",
            confidence=0.7,  # Lower confidence - might not be the right type
            user_approved=False,
            user_override=None
        )

    # ==================== Semantic Field Matching ====================

    def _match_semantic_field(self, field: DetectedField) -> FieldMapping:
        """
        Match field using semantic search + LLM extraction.

        For complex fields like experience, projects, capabilities that require
        natural language understanding.

        Returns:
            FieldMapping with LLM-extracted data
        """
        if not self.ollama_client:
            return self._create_no_match_mapping(
                field,
                "LLM not available for semantic matching"
            )

        # Build context from entity data
        context = self._build_semantic_context(field)

        # Use LLM to extract relevant information
        prompt = self._create_extraction_prompt(field, context)

        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,  # Lower temperature for factual extraction
                    "num_predict": 500,
                }
            )

            response_text = response['response'].strip()

            # Parse the JSON response
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(json_text)

            matched_data = result.get("answer", "")
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "Extracted from entity data")

            # If confidence is too low or no answer, return no match
            if confidence < 0.3 or not matched_data:
                return self._create_no_match_mapping(field, reasoning)

            return FieldMapping(
                field_id=field.field_id,
                field_location=field.location,
                field_description=field.field_description,
                field_type=field.classification_hints[0] if field.classification_hints else "text_response",
                matched_data=matched_data,
                data_source="semantic_extraction",
                confidence=confidence,
                user_approved=False,
                user_override=None
            )

        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            return self._create_no_match_mapping(
                field,
                f"LLM extraction failed: {str(e)}"
            )

    def _build_semantic_context(self, field: DetectedField) -> str:
        """Build context string from entity data for semantic matching."""
        context_parts = []

        # Add organization info
        org = self.entity_data.get("organization", {})
        if org:
            context_parts.append(f"Organization: {org.get('legal_name', 'Unknown')}")

        # Add projects if available
        projects = self.entity_data.get("projects", [])
        if projects:
            project_summaries = []
            for proj in projects[:5]:  # Limit to 5 most relevant
                summary = f"- {proj.get('project_name', 'Unnamed')}: {proj.get('description', '')[:200]}"
                project_summaries.append(summary)
            context_parts.append("Past Projects:\n" + "\n".join(project_summaries))

        # Add capabilities if available
        capabilities = self.entity_data.get("capabilities", [])
        if capabilities:
            cap_list = [f"- {cap.get('capability_name', '')}: {cap.get('description', '')[:100]}" for cap in capabilities[:5]]
            context_parts.append("Capabilities:\n" + "\n".join(cap_list))

        return "\n\n".join(context_parts)

    def _create_extraction_prompt(self, field: DetectedField, context: str) -> str:
        """Create prompt for LLM extraction."""
        return f"""You are an expert at extracting relevant information to answer tender questions.

TENDER QUESTION:
"{field.field_description}"

AVAILABLE INFORMATION:
{context}

TASK:
Extract or generate a concise, accurate answer to the tender question using ONLY the information provided above.
If the information is not available, indicate that clearly.

Respond ONLY with valid JSON in this exact format:
{{
    "answer": "The extracted or generated answer (empty string if no relevant information)",
    "confidence": 0.75,
    "reasoning": "Brief explanation of why this answer was chosen or why no answer could be found"
}}

Confidence scoring:
- 0.9-1.0: Direct, exact information available
- 0.7-0.9: Good information available, minor interpretation needed
- 0.5-0.7: Partial information available, significant interpretation needed
- 0.3-0.5: Weak or tangential information
- 0.0-0.3: No relevant information found"""

    # ==================== Helper Methods ====================

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """
        Get value from nested dictionary using dot notation.

        Example: "organization.abn" → data["organization"]["abn"]
        """
        parts = path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _format_value(self, value: Any) -> str:
        """Format a value for display in tender document."""
        if value is None:
            return ""

        if isinstance(value, dict):
            # Format address dictionary
            if "street" in value:
                return self._format_address(value)
            return json.dumps(value)

        if isinstance(value, list):
            return ", ".join(str(v) for v in value)

        return str(value)

    def _format_address(self, address: Dict[str, str]) -> str:
        """Format address dictionary to single line."""
        parts = [
            address.get("street", ""),
            address.get("city", ""),
            address.get("state", ""),
            address.get("postcode", ""),
            address.get("country", "")
        ]
        return ", ".join(p for p in parts if p)

    def _create_no_match_mapping(self, field: DetectedField, reason: str) -> FieldMapping:
        """Create a FieldMapping for an unmatched field."""
        return FieldMapping(
            field_id=field.field_id,
            field_location=field.location,
            field_description=field.field_description,
            field_type=field.classification_hints[0] if field.classification_hints else None,
            matched_data=None,
            data_source=None,
            confidence=0.0,
            user_approved=False,
            user_override=None
        )


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from cortex_engine.proposals.field_classifier import get_classifier
    from cortex_engine.proposals.tender_field_parser import TenderFieldParser, DetectedField, LocationType
    from cortex_engine.workspace_schema import FieldMapping

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("FIELD MATCHER TEST")
    print("=" * 80)

    # Create sample entity data
    entity_data = {
        "organization": {
            "legal_name": "Acme Corporation Pty Ltd",
            "abn": "12345678901",
            "acn": "123456789",
            "address": {
                "street": "123 Business St",
                "city": "Sydney",
                "state": "NSW",
                "postcode": "2000",
                "country": "Australia"
            },
            "email": "contact@acme.com",
            "phone": "+61 2 1234 5678",
            "website": "https://acme.com"
        },
        "insurances": [
            {
                "insurance_type": "Public Liability",
                "insurer": "ABC Insurance",
                "policy_number": "PL-2024-001",
                "coverage_amount": 20000000,
                "expiry_date": "2025-12-31"
            },
            {
                "insurance_type": "Professional Indemnity",
                "insurer": "XYZ Insurance",
                "policy_number": "PI-2024-002",
                "coverage_amount": 10000000,
                "expiry_date": "2025-11-30"
            }
        ],
        "projects": [
            {
                "project_name": "Government Portal Modernization",
                "client": "Department of Digital Services",
                "description": "Modernized legacy government portal with React and cloud infrastructure, serving 100k+ daily users",
                "value": 500000,
                "deliverables": ["New UI/UX", "API Gateway", "Cloud Migration"]
            }
        ]
    }

    # Create sample detected fields
    detected_fields = [
        DetectedField(
            field_id="test_1",
            location="Row 1",
            location_type=LocationType.TABLE_CELL,
            location_coordinates={},
            field_description="ABN",
            field_context="",
            classification_hints=["abn"],
            classification_confidence=1.0,
            classification_method="regex"
        ),
        DetectedField(
            field_id="test_2",
            location="Row 2",
            location_type=LocationType.TABLE_CELL,
            location_coordinates={},
            field_description="Public Liability Insurance Policy Number",
            field_context="",
            classification_hints=["insurance_public_liability", "insurance_policy_number"],
            classification_confidence=0.9,
            classification_method="regex"
        ),
        DetectedField(
            field_id="test_3",
            location="Row 3",
            location_type=LocationType.PARAGRAPH,
            location_coordinates={},
            field_description="Describe your experience with government digital transformation projects",
            field_context="",
            classification_hints=["text_response", "relevant_experience"],
            classification_confidence=0.7,
            classification_method="llm"
        ),
    ]

    # Create matcher
    matcher = FieldMatcher(
        entity_data=entity_data,
        workspace_id="test_workspace",
        model_name="qwen2.5:14b-instruct-q4_K_M"
    )

    # Match fields
    mappings = matcher.match_all_fields(detected_fields)

    print(f"\n✅ Matched {len(mappings)} fields:\n")

    for mapping in mappings:
        print(f"Field: {mapping.field_description}")
        print(f"Matched: {mapping.matched_data or '[No match]'}")
        print(f"Source: {mapping.data_source or 'N/A'}")
        print(f"Confidence: {mapping.confidence:.2f}")
        print("-" * 80)
