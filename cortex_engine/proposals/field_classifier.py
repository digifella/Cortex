"""
Field Classification Engine for Tender Documents

This module provides hybrid regex + LLM classification of tender document fields,
mapping field descriptions to structured field types using:
1. Fast regex patterns for obvious cases (ABN, email, insurance, etc.)
2. LLM-based classification for ambiguous or context-dependent fields

Author: Cortex Suite
Created: 2026-01-04
Version: 2.0.0 - Hybrid Regex + LLM Classification
"""

import re
import json
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import logging
import ollama

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of field classification."""
    field_types: List[str]
    confidence: float  # 0.0-1.0
    matched_patterns: Dict[str, str]  # field_type -> matched pattern
    classification_method: str = "regex"  # "regex" | "llm" | "hybrid"
    llm_reasoning: Optional[str] = None  # LLM's reasoning (if used)


class FieldClassifier:
    """
    Pattern-based classification engine for tender document fields.

    Uses prioritized regex patterns to identify field types from
    field descriptions and labels in tender documents.
    """

    # Classification patterns organized by priority (most specific first)
    # Each entry: field_type -> (priority, [regex_patterns])
    CLASSIFICATION_RULES = {
        # Organization Identifiers (Priority: 1 - highest)
        "abn": (1, [
            r"\bABN\b",
            r"Australian\s+Business\s+Number",
            r"ABN\s*[:\-\s]*\d",
        ]),
        "acn": (1, [
            r"\bACN\b",
            r"Australian\s+Company\s+Number",
            r"ACN\s*[:\-\s]*\d",
        ]),
        "legal_name": (1, [
            r"legal\s+(?:entity\s+)?name",
            r"registered\s+(?:business\s+)?name",
            r"official\s+(?:company\s+)?name",
            r"trading\s+as",
        ]),

        # Contact Information (Priority: 2)
        "address": (2, [
            r"\baddress\b(?!\s*(?:email|web))",
            r"(?:street|postal|physical)\s+address",
            r"location",
            r"premises",
            r"\bstate\b",
            r"\bpostcode\b",
        ]),
        "email": (2, [
            r"e[\-\s]?mail(?:\s+address)?",
            r"email\s+contact",
            r"contact\s+email",
        ]),
        "phone": (2, [
            r"(?:phone|telephone)\s*(?:number)?",
            r"contact\s+(?:phone|number)",
            r"mobile(?:\s+number)?",
        ]),
        "website": (2, [
            r"web[\s\-]?site",
            r"web\s+address",
            r"url",
            r"online\s+presence",
        ]),

        # Insurance (Priority: 3)
        "insurance_public_liability": (3, [
            r"public\s+liability(?:\s+insurance)?",
            r"\bPL\s+insurance\b",
            r"liability\s+coverage",
        ]),
        "insurance_professional_indemnity": (3, [
            r"professional\s+indemnity(?:\s+insurance)?",
            r"\bPI\s+insurance\b",
            r"indemnity\s+coverage",
        ]),
        "insurance_workers_compensation": (3, [
            r"workers?\s+comp(?:ensation)?(?:\s+insurance)?",
            r"employee\s+insurance",
        ]),
        "insurance_policy_number": (3, [
            r"(?:insurance\s+)?policy\s+number",
            r"certificate\s+number",
            r"policy\s+(?:id|reference)",
        ]),
        "insurance_expiry": (3, [
            r"(?:insurance\s+)?expiry\s+date",
            r"policy\s+expiry",
            r"valid\s+(?:until|to)",
        ]),

        # Financial Information (Priority: 4)
        "gst_registered": (4, [
            r"\bGST\b(?:\s+registered)?",
            r"goods\s+and\s+services\s+tax",
            r"tax\s+registration",
        ]),
        "financial_turnover": (4, [
            r"annual\s+turnover",
            r"revenue",
            r"financial\s+capacity",
        ]),

        # Experience & Qualifications (Priority: 5)
        "years_experience": (5, [
            r"years?\s+(?:of\s+)?experience",
            r"years?\s+(?:in\s+)?(?:business|operation)",
            r"experience\s+(?:in\s+)?years?",
        ]),
        "relevant_experience": (5, [
            r"relevant\s+experience",
            r"similar\s+(?:projects?|work)",
            r"(?:project\s+)?experience",
        ]),
        "qualifications": (5, [
            r"qualifications?",
            r"certifications?",
            r"accreditations?",
            r"licenses?",
        ]),

        # Project References (Priority: 6)
        "project_name": (6, [
            r"project\s+(?:name|title)",
            r"contract\s+name",
        ]),
        "project_client": (6, [
            r"client(?:\s+name)?",
            r"customer",
            r"(?:project\s+)?organization",
        ]),
        "project_value": (6, [
            r"(?:project|contract)\s+value",
            r"contract\s+(?:sum|amount)",
            r"value\s+of\s+(?:works?|services?)",
        ]),
        "project_date": (6, [
            r"(?:project|contract)\s+(?:date|period)",
            r"completion\s+date",
            r"(?:start|end)\s+date",
        ]),
        "project_description": (6, [
            r"(?:project|work)\s+description",
            r"scope\s+of\s+(?:works?|services?)",
            r"brief\s+description",
        ]),

        # Personnel (Priority: 7)
        "contact_person": (7, [
            r"contact\s+(?:person|name)",
            r"(?:key|primary)\s+contact",
            r"representative",
        ]),
        "position_title": (7, [
            r"(?:position|job)\s+title",
            r"role",
            r"designation",
        ]),

        # Compliance & Safety (Priority: 8)
        "safety_policy": (8, [
            r"(?:WHS|OHS|safety)\s+policy",
            r"health\s+and\s+safety",
            r"workplace\s+safety",
        ]),
        "quality_system": (8, [
            r"quality\s+(?:system|management)",
            r"\bISO\b",
            r"quality\s+assurance",
        ]),

        # General (Priority: 9 - lowest, catch-all)
        "text_response": (9, [
            r"(?:please\s+)?(?:describe|provide|explain)",
            r"details?",
            r"information",
        ]),
        "number": (9, [
            r"\bnumber\b",
            r"\bqty\b",
            r"quantity",
        ]),
        "date": (9, [
            r"\bdate\b",
            r"when",
        ]),
    }

    def __init__(
        self,
        use_llm_fallback: bool = True,
        llm_model: str = "qwen2.5:3b-instruct-q8_0",
        llm_confidence_threshold: float = 0.4
    ):
        """
        Initialize the field classifier with compiled patterns.

        Args:
            use_llm_fallback: Enable LLM classification for ambiguous fields
            llm_model: Model to use for LLM classification (should be a fast router model)
            llm_confidence_threshold: If regex confidence < this, try LLM classification
        """
        self._compiled_patterns = self._compile_patterns()
        self.use_llm_fallback = use_llm_fallback
        self.llm_model = llm_model
        self.llm_confidence_threshold = llm_confidence_threshold
        self._ollama_client = None

        # Initialize Ollama client if LLM fallback is enabled
        if self.use_llm_fallback:
            try:
                self._ollama_client = ollama.Client()
                logger.info(f"FieldClassifier initialized with LLM fallback ({llm_model})")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama client: {e}. LLM fallback disabled.")
                self.use_llm_fallback = False

        logger.info(f"FieldClassifier initialized with {len(self.CLASSIFICATION_RULES)} field types")

    def _compile_patterns(self) -> Dict[str, tuple]:
        """
        Pre-compile all regex patterns for performance.

        Returns:
            Dictionary of field_type -> (priority, [compiled_patterns])
        """
        compiled = {}
        for field_type, (priority, patterns) in self.CLASSIFICATION_RULES.items():
            compiled_regex = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in patterns
            ]
            compiled[field_type] = (priority, compiled_regex)
        return compiled

    def classify(
        self,
        text: str,
        max_types: int = 3,
        min_confidence: float = 0.3
    ) -> ClassificationResult:
        """
        Classify a field description into one or more field types.

        Args:
            text: The field description or label text to classify
            max_types: Maximum number of field types to return
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            ClassificationResult with matched field types and confidence

        Examples:
            >>> classifier = FieldClassifier()
            >>> result = classifier.classify("Please provide your ABN")
            >>> result.field_types
            ['abn']

            >>> result = classifier.classify("Public Liability Insurance Policy Number")
            >>> result.field_types
            ['insurance_public_liability', 'insurance_policy_number']
        """
        if not text or not text.strip():
            return ClassificationResult(
                field_types=[],
                confidence=0.0,
                matched_patterns={}
            )

        matches = []
        matched_patterns = {}

        # Find all matching field types
        for field_type, (priority, compiled_patterns) in self._compiled_patterns.items():
            for pattern in compiled_patterns:
                match = pattern.search(text)
                if match:
                    matches.append((field_type, priority, pattern.pattern))
                    matched_patterns[field_type] = pattern.pattern
                    break  # Only need one pattern match per field type

        if not matches:
            logger.debug(f"No regex classification found for: '{text}'")

            # Try LLM classification if enabled
            if self.use_llm_fallback and self._ollama_client:
                logger.debug("Attempting LLM classification for unmatched field...")
                llm_result = self._classify_with_llm(text, max_types)
                if llm_result:
                    return llm_result

            # No matches from either regex or LLM
            return ClassificationResult(
                field_types=[],
                confidence=0.0,
                matched_patterns={},
                classification_method="none"
            )

        # Sort by priority (lower number = higher priority)
        matches.sort(key=lambda x: x[1])

        # Take top N matches
        top_matches = matches[:max_types]
        field_types = [match[0] for match in top_matches]

        # Calculate confidence based on:
        # 1. Number of matches (more matches = higher confidence)
        # 2. Priority of matches (higher priority = higher confidence)
        # 3. Pattern specificity (longer patterns = higher confidence)

        num_matches = len(matches)
        avg_priority = sum(m[1] for m in top_matches) / len(top_matches)

        # Base confidence from number of matches
        base_confidence = min(num_matches / 3.0, 1.0)  # 1-3 matches

        # Adjust for priority (priority 1-9, lower is better)
        priority_factor = 1.0 - (avg_priority - 1) / 8.0  # Normalize to 0-1

        # Final confidence calculation
        confidence = base_confidence * 0.7 + priority_factor * 0.3
        confidence = max(min_confidence, min(confidence, 1.0))

        logger.debug(f"Classified '{text}' as {field_types} (confidence: {confidence:.2f})")

        # Check if we should use LLM fallback
        if (self.use_llm_fallback and
            self._ollama_client and
            confidence < self.llm_confidence_threshold):

            logger.debug(f"Regex confidence ({confidence:.2f}) below threshold, trying LLM classification...")
            llm_result = self._classify_with_llm(text, max_types)

            # If LLM provides better confidence, use it
            if llm_result and llm_result.confidence > confidence:
                logger.info(f"LLM classification improved confidence: {confidence:.2f} → {llm_result.confidence:.2f}")
                return llm_result

        return ClassificationResult(
            field_types=field_types,
            confidence=confidence,
            matched_patterns={ft: matched_patterns[ft] for ft in field_types},
            classification_method="regex"
        )

    def _classify_with_llm(
        self,
        text: str,
        max_types: int = 3
    ) -> Optional[ClassificationResult]:
        """
        Use LLM to classify ambiguous field descriptions.

        Args:
            text: Field description to classify
            max_types: Maximum number of field types to return

        Returns:
            ClassificationResult from LLM, or None if LLM fails
        """
        if not self._ollama_client:
            return None

        # Get list of all available field types for the prompt
        all_field_types = self.get_all_field_types()

        # Create a structured prompt for the LLM
        prompt = f"""You are a field classification expert for tender documents.

Given the following field description from a tender document, classify it into the most appropriate field type(s) from the list below.

FIELD DESCRIPTION: "{text}"

AVAILABLE FIELD TYPES:
{chr(10).join(f"- {ft}" for ft in all_field_types)}

INSTRUCTIONS:
1. Analyze the field description carefully
2. Select up to {max_types} most appropriate field types
3. Assign a confidence score (0.0-1.0) based on how well the description matches
4. Provide brief reasoning for your classification

Respond ONLY with valid JSON in this exact format:
{{
    "field_types": ["type1", "type2"],
    "confidence": 0.85,
    "reasoning": "Brief explanation of classification"
}}"""

        try:
            response = self._ollama_client.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "num_predict": 200,   # Limit response length
                }
            )

            response_text = response['response'].strip()

            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()

            # Parse the JSON response
            result = json.loads(json_text)

            field_types = result.get("field_types", [])
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")

            # Validate field types
            valid_types = [ft for ft in field_types if ft in all_field_types]

            if not valid_types:
                logger.warning(f"LLM returned invalid field types: {field_types}")
                return None

            logger.debug(f"LLM classified '{text}' as {valid_types} (confidence: {confidence:.2f})")
            logger.debug(f"LLM reasoning: {reasoning}")

            return ClassificationResult(
                field_types=valid_types[:max_types],
                confidence=min(confidence, 1.0),
                matched_patterns={},
                classification_method="llm",
                llm_reasoning=reasoning
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw response: {response_text}")
            return None
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return None

    def get_field_type_info(self, field_type: str) -> Dict[str, any]:
        """
        Get information about a specific field type.

        Args:
            field_type: The field type to query

        Returns:
            Dictionary with priority and patterns
        """
        if field_type not in self.CLASSIFICATION_RULES:
            return None

        priority, patterns = self.CLASSIFICATION_RULES[field_type]
        return {
            "field_type": field_type,
            "priority": priority,
            "patterns": patterns,
            "pattern_count": len(patterns)
        }

    def get_all_field_types(self) -> List[str]:
        """Get list of all supported field types."""
        return sorted(self.CLASSIFICATION_RULES.keys())

    def classify_batch(
        self,
        texts: List[str],
        max_types: int = 3,
        min_confidence: float = 0.3
    ) -> List[ClassificationResult]:
        """
        Classify multiple field descriptions in batch.

        Args:
            texts: List of field descriptions
            max_types: Maximum number of field types per text
            min_confidence: Minimum confidence threshold

        Returns:
            List of ClassificationResult objects
        """
        return [
            self.classify(text, max_types, min_confidence)
            for text in texts
        ]


# Singleton instance for reuse
_classifier_instance = None

def get_classifier(
    use_llm_fallback: bool = True,
    llm_model: str = "qwen2.5:3b-instruct-q8_0",
    llm_confidence_threshold: float = 0.4
) -> FieldClassifier:
    """
    Get or create singleton FieldClassifier instance.

    Args:
        use_llm_fallback: Enable LLM classification for ambiguous fields
        llm_model: Model to use for LLM classification
        llm_confidence_threshold: Confidence threshold for LLM fallback

    Returns:
        FieldClassifier instance
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FieldClassifier(
            use_llm_fallback=use_llm_fallback,
            llm_model=llm_model,
            llm_confidence_threshold=llm_confidence_threshold
        )
    return _classifier_instance


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("HYBRID FIELD CLASSIFIER TEST (Regex + LLM)")
    print("=" * 80)

    # Test with LLM fallback disabled (regex only)
    print("\n" + "=" * 80)
    print("TEST 1: REGEX-ONLY CLASSIFICATION")
    print("=" * 80)

    classifier_regex = FieldClassifier(use_llm_fallback=False)

    test_cases = [
        "Please provide your ABN",
        "Public Liability Insurance Policy Number",
        "Describe your experience with similar projects",
        "Tell us about your company's sustainability initiatives",  # Ambiguous - should be low confidence
        "What makes your team uniquely qualified?",  # Very ambiguous
    ]

    for test_text in test_cases:
        result = classifier_regex.classify(test_text)
        print(f"\nText: '{test_text}'")
        print(f"Types: {result.field_types}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Method: {result.classification_method}")

    # Test with LLM fallback enabled
    print("\n" + "=" * 80)
    print("TEST 2: HYBRID (REGEX + LLM) CLASSIFICATION")
    print("=" * 80)

    classifier_hybrid = FieldClassifier(
        use_llm_fallback=True,
        llm_model="qwen2.5:3b-instruct-q8_0",
        llm_confidence_threshold=0.5
    )

    # Test cases that should trigger LLM
    llm_test_cases = [
        "Tell us about your company's sustainability initiatives",
        "What makes your team uniquely qualified?",
        "Explain your approach to risk management",
        "How do you ensure quality control?",
    ]

    for test_text in llm_test_cases:
        result = classifier_hybrid.classify(test_text)
        print(f"\nText: '{test_text}'")
        print(f"Types: {result.field_types}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Method: {result.classification_method}")
        if result.llm_reasoning:
            print(f"Reasoning: {result.llm_reasoning}")
