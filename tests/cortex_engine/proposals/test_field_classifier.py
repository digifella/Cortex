"""
Unit tests for Field Classifier

Tests pattern matching accuracy, confidence scoring, and edge cases.

Author: Cortex Suite
Created: 2026-01-04
Version: 1.0.0
"""

import pytest
from cortex_engine.proposals.field_classifier import (
    FieldClassifier,
    ClassificationResult,
    get_classifier
)


class TestFieldClassifier:
    """Test suite for FieldClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create a fresh classifier instance for each test (LLM disabled for predictable tests)."""
        return FieldClassifier(use_llm_fallback=False)

    # ==================== Basic Classification Tests ====================

    def test_classify_abn(self, classifier):
        """Test ABN field classification."""
        result = classifier.classify("Please provide your ABN")
        assert "abn" in result.field_types
        assert result.confidence > 0.3

    def test_classify_acn(self, classifier):
        """Test ACN field classification."""
        result = classifier.classify("Australian Company Number (ACN)")
        assert "acn" in result.field_types
        assert result.confidence > 0.3

    def test_classify_legal_name(self, classifier):
        """Test legal name classification."""
        result = classifier.classify("Legal entity name")
        assert "legal_name" in result.field_types

    def test_classify_address(self, classifier):
        """Test address field classification."""
        result = classifier.classify("Business address")
        assert "address" in result.field_types

        result2 = classifier.classify("Postal address including state and postcode")
        assert "address" in result2.field_types

    def test_classify_email(self, classifier):
        """Test email classification."""
        result = classifier.classify("Email address")
        assert "email" in result.field_types

    def test_classify_phone(self, classifier):
        """Test phone number classification."""
        result = classifier.classify("Contact phone number")
        assert "phone" in result.field_types

        result2 = classifier.classify("Mobile number")
        assert "phone" in result2.field_types

    # ==================== Insurance Field Tests ====================

    def test_classify_public_liability(self, classifier):
        """Test public liability insurance classification."""
        result = classifier.classify("Public Liability Insurance")
        assert "insurance_public_liability" in result.field_types

        result2 = classifier.classify("PL insurance policy")
        assert "insurance_public_liability" in result2.field_types

    def test_classify_professional_indemnity(self, classifier):
        """Test professional indemnity insurance classification."""
        result = classifier.classify("Professional Indemnity Insurance")
        assert "insurance_professional_indemnity" in result.field_types

        result2 = classifier.classify("PI insurance certificate")
        assert "insurance_professional_indemnity" in result2.field_types

    def test_classify_workers_comp(self, classifier):
        """Test workers compensation classification."""
        result = classifier.classify("Workers Compensation Insurance")
        assert "insurance_workers_compensation" in result.field_types

    def test_classify_insurance_policy_number(self, classifier):
        """Test insurance policy number classification."""
        result = classifier.classify("Policy number")
        assert "insurance_policy_number" in result.field_types

    def test_classify_insurance_expiry(self, classifier):
        """Test insurance expiry date classification."""
        result = classifier.classify("Insurance expiry date")
        assert "insurance_expiry" in result.field_types

    # ==================== Experience & Qualification Tests ====================

    def test_classify_years_experience(self, classifier):
        """Test years of experience classification."""
        result = classifier.classify("Years of experience in the industry")
        assert "years_experience" in result.field_types

    def test_classify_relevant_experience(self, classifier):
        """Test relevant experience classification."""
        result = classifier.classify("Describe your experience with similar projects")
        assert "relevant_experience" in result.field_types

    def test_classify_qualifications(self, classifier):
        """Test qualifications classification."""
        result = classifier.classify("Relevant qualifications and certifications")
        assert "qualifications" in result.field_types

    # ==================== Project Reference Tests ====================

    def test_classify_project_name(self, classifier):
        """Test project name classification."""
        result = classifier.classify("Project name")
        assert "project_name" in result.field_types

    def test_classify_project_client(self, classifier):
        """Test project client classification."""
        result = classifier.classify("Client name")
        assert "project_client" in result.field_types

    def test_classify_project_value(self, classifier):
        """Test project value classification."""
        result = classifier.classify("Contract value")
        assert "project_value" in result.field_types

    def test_classify_project_date(self, classifier):
        """Test project date classification."""
        result = classifier.classify("Project completion date")
        assert "project_date" in result.field_types

    # ==================== Multiple Type Tests ====================

    def test_classify_multiple_types(self, classifier):
        """Test classification with multiple field types."""
        result = classifier.classify(
            "Public Liability Insurance Policy Number"
        )
        assert "insurance_public_liability" in result.field_types
        assert "insurance_policy_number" in result.field_types
        assert len(result.field_types) <= 3  # Default max_types

    def test_classify_contact_with_multiple(self, classifier):
        """Test contact field with multiple identifiable components."""
        result = classifier.classify("Contact person name and phone number")
        assert "contact_person" in result.field_types or "phone" in result.field_types

    # ==================== Priority System Tests ====================

    def test_priority_abn_before_number(self, classifier):
        """Test that ABN is matched before generic number."""
        result = classifier.classify("ABN number")
        # ABN should be first due to higher priority
        assert result.field_types[0] == "abn"

    def test_priority_specific_insurance_before_generic(self, classifier):
        """Test specific insurance types have priority."""
        result = classifier.classify("Public Liability insurance")
        # Should match specific insurance type
        assert "insurance_public_liability" in result.field_types

    # ==================== Confidence Score Tests ====================

    def test_confidence_high_for_specific_match(self, classifier):
        """Test higher confidence for specific, single matches."""
        result = classifier.classify("ABN")
        assert result.confidence >= 0.3

    def test_confidence_increases_with_multiple_matches(self, classifier):
        """Test confidence increases with multiple pattern matches."""
        result1 = classifier.classify("Number")
        result2 = classifier.classify("Public Liability Insurance Policy Number")

        # Multiple specific matches should have higher confidence
        assert result2.confidence > result1.confidence

    def test_minimum_confidence_threshold(self, classifier):
        """Test minimum confidence threshold."""
        result = classifier.classify("Some text", min_confidence=0.5)
        if result.field_types:
            assert result.confidence >= 0.5

    # ==================== Edge Case Tests ====================

    def test_empty_text(self, classifier):
        """Test classification with empty text."""
        result = classifier.classify("")
        assert result.field_types == []
        assert result.confidence == 0.0

    def test_whitespace_only(self, classifier):
        """Test classification with whitespace only."""
        result = classifier.classify("   ")
        assert result.field_types == []
        assert result.confidence == 0.0

    def test_no_match(self, classifier):
        """Test classification with no matching patterns."""
        result = classifier.classify("xyzabc123randomtext")
        assert result.field_types == []
        assert result.confidence == 0.0

    def test_case_insensitive(self, classifier):
        """Test case-insensitive matching."""
        result1 = classifier.classify("ABN")
        result2 = classifier.classify("abn")
        result3 = classifier.classify("Abn")

        assert result1.field_types == result2.field_types == result3.field_types

    def test_special_characters(self, classifier):
        """Test classification with special characters."""
        result = classifier.classify("ABN: ___________")
        assert "abn" in result.field_types

    # ==================== Max Types Tests ====================

    def test_max_types_limit(self, classifier):
        """Test max_types parameter limits results."""
        result = classifier.classify(
            "Public Liability Insurance Policy Number",
            max_types=1
        )
        assert len(result.field_types) <= 1

        result2 = classifier.classify(
            "Public Liability Insurance Policy Number",
            max_types=2
        )
        assert len(result2.field_types) <= 2

    # ==================== Batch Classification Tests ====================

    def test_classify_batch(self, classifier):
        """Test batch classification."""
        texts = [
            "ABN",
            "Email address",
            "Public Liability Insurance"
        ]
        results = classifier.classify_batch(texts)

        assert len(results) == 3
        assert "abn" in results[0].field_types
        assert "email" in results[1].field_types
        assert "insurance_public_liability" in results[2].field_types

    def test_classify_batch_empty(self, classifier):
        """Test batch classification with empty list."""
        results = classifier.classify_batch([])
        assert results == []

    # ==================== Utility Method Tests ====================

    def test_get_field_type_info(self, classifier):
        """Test getting field type information."""
        info = classifier.get_field_type_info("abn")
        assert info is not None
        assert info["field_type"] == "abn"
        assert "priority" in info
        assert "patterns" in info
        assert info["pattern_count"] > 0

    def test_get_field_type_info_invalid(self, classifier):
        """Test getting info for invalid field type."""
        info = classifier.get_field_type_info("invalid_field_type")
        assert info is None

    def test_get_all_field_types(self, classifier):
        """Test getting all field types."""
        field_types = classifier.get_all_field_types()
        assert isinstance(field_types, list)
        assert len(field_types) > 0
        assert "abn" in field_types
        assert "insurance_public_liability" in field_types

    # ==================== Singleton Tests ====================

    def test_get_classifier_singleton(self):
        """Test singleton pattern for get_classifier()."""
        classifier1 = get_classifier()
        classifier2 = get_classifier()
        assert classifier1 is classifier2

    # ==================== Real-World Tender Examples ====================

    def test_real_world_tender_field_1(self, classifier):
        """Test real-world tender field: Organization details."""
        result = classifier.classify(
            "Tenderer's legal entity name and ABN"
        )
        assert "legal_name" in result.field_types or "abn" in result.field_types

    def test_real_world_tender_field_2(self, classifier):
        """Test real-world tender field: Insurance."""
        result = classifier.classify(
            "Current Public Liability Insurance policy number and expiry date"
        )
        assert "insurance_public_liability" in result.field_types
        assert "insurance_policy_number" in result.field_types or \
               "insurance_expiry" in result.field_types

    def test_real_world_tender_field_3(self, classifier):
        """Test real-world tender field: Experience."""
        result = classifier.classify(
            "Provide details of at least 3 similar projects completed in the last 5 years"
        )
        assert "relevant_experience" in result.field_types or \
               "project_description" in result.field_types

    def test_real_world_tender_field_4(self, classifier):
        """Test real-world tender field: Contact."""
        result = classifier.classify(
            "Primary contact person, position, email and phone"
        )
        field_types = result.field_types
        assert any(ft in field_types for ft in [
            "contact_person", "position_title", "email", "phone"
        ])


# ==================== Integration Tests ====================

class TestFieldClassifierIntegration:
    """Integration tests for complete workflows."""

    def test_complete_tender_field_set(self):
        """Test classifying a complete set of tender fields."""
        classifier = FieldClassifier()

        tender_fields = {
            "ABN": "abn",
            "Legal entity name": "legal_name",
            "Registered address": "address",
            "Contact email": "email",
            "Public Liability Insurance": "insurance_public_liability",
            "Professional Indemnity Insurance": "insurance_professional_indemnity",
            "Years in business": "years_experience",
            "Relevant project experience": "relevant_experience",
        }

        for field_text, expected_type in tender_fields.items():
            result = classifier.classify(field_text)
            assert expected_type in result.field_types, \
                f"Failed to classify '{field_text}' as '{expected_type}'"

    def test_performance_batch_classification(self):
        """Test performance with large batch."""
        classifier = FieldClassifier()

        # Create 100 test fields
        test_fields = [
            f"Field {i}: ABN or email or address"
            for i in range(100)
        ]

        import time
        start = time.time()
        results = classifier.classify_batch(test_fields)
        elapsed = time.time() - start

        assert len(results) == 100
        assert elapsed < 5.0  # Should complete in under 5 seconds


# ==================== LLM Hybrid Classification Tests ====================

class TestLLMHybridClassification:
    """Test suite for LLM-enhanced classification."""

    @pytest.fixture
    def hybrid_classifier(self):
        """Create classifier with LLM fallback enabled."""
        return FieldClassifier(
            use_llm_fallback=True,
            llm_model="qwen2.5:3b-instruct-q8_0",
            llm_confidence_threshold=0.5
        )

    def test_llm_classifies_ambiguous_fields(self, hybrid_classifier):
        """Test LLM can classify ambiguous field descriptions."""
        result = hybrid_classifier.classify("Tell us about your company's sustainability initiatives")

        # Should get classification from LLM
        assert len(result.field_types) > 0
        assert result.classification_method in ["llm", "regex"]

        # LLM should provide reasoning if used
        if result.classification_method == "llm":
            assert result.llm_reasoning is not None

    def test_llm_improves_low_confidence_matches(self, hybrid_classifier):
        """Test LLM improves classification when regex confidence is low."""
        # This field has low regex confidence
        result = hybrid_classifier.classify("What makes your team uniquely qualified?")

        # Should have some classification
        assert len(result.field_types) > 0

        # Check if LLM was used
        if result.classification_method == "llm":
            assert result.confidence >= 0.5  # LLM should be confident

    def test_llm_handles_unmatched_fields(self, hybrid_classifier):
        """Test LLM provides classification for fields with no regex match."""
        result = hybrid_classifier.classify("Explain your approach to risk management")

        # LLM should find a classification
        assert len(result.field_types) > 0
        assert result.confidence > 0.0

    def test_regex_still_preferred_for_clear_fields(self, hybrid_classifier):
        """Test regex is still used for clearly defined fields."""
        result = hybrid_classifier.classify("Please provide your ABN")

        # Should use regex for obvious fields
        assert "abn" in result.field_types
        # Method might be regex (if confidence is high enough)
        assert result.confidence > 0.3

    def test_llm_disabled_mode(self):
        """Test classifier works without LLM fallback."""
        classifier = FieldClassifier(use_llm_fallback=False)

        result = classifier.classify("Tell us about your sustainability initiatives")

        # Should have low/no confidence without LLM
        # classification_method should be regex or none
        assert result.classification_method in ["regex", "none"]
        assert result.llm_reasoning is None

    def test_llm_classification_result_structure(self, hybrid_classifier):
        """Test LLM classification returns proper structure."""
        result = hybrid_classifier.classify("How do you ensure quality control?")

        # Check structure
        assert isinstance(result.field_types, list)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert result.classification_method in ["regex", "llm", "hybrid", "none"]

        # If LLM was used, should have reasoning
        if result.classification_method == "llm":
            assert isinstance(result.llm_reasoning, str)
            assert len(result.llm_reasoning) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
