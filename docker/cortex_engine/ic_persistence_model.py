"""
Intelligent Completion Persistence Models

Version: 1.0.0
Date: 2026-01-23

Purpose: Pydantic models for serializing/deserializing Intelligent Completion
workflow state, enabling "Save As You Go" persistence.

The IC workflow state includes:
- Extracted and classified fields
- Per-question status, responses, and evidence
- User settings per question (collection, creativity)
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class PersistedFieldTier(str, Enum):
    """Serializable tier enum."""
    AUTO_COMPLETE = "auto_complete"
    INTELLIGENT = "intelligent"


class PersistedQuestionType(str, Enum):
    """Serializable question type enum."""
    CAPABILITY = "capability"
    METHODOLOGY = "methodology"
    VALUE_PROPOSITION = "value"
    COMPLIANCE = "compliance"
    INNOVATION = "innovation"
    RISK = "risk"
    PERSONNEL = "personnel"
    PRICING = "pricing"
    GENERAL = "general"


class PersistedEvidence(BaseModel):
    """Persisted evidence passage."""
    text: str
    source_doc: str
    source_chunk_id: str
    relevance_score: float
    doc_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PersistedClassifiedField(BaseModel):
    """Persisted field classification result."""
    field_text: str
    tier: PersistedFieldTier
    question_type: Optional[PersistedQuestionType] = None
    auto_complete_mapping: Optional[str] = None
    confidence: float = 0.8
    word_limit: Optional[int] = None
    context_hint: Optional[str] = None


class PersistedQuestionState(BaseModel):
    """Per-question status, response, and evidence."""
    status: str  # 'pending', 'skipped', 'completed', 'editing'
    response: str = ""
    evidence: List[PersistedEvidence] = Field(default_factory=list)
    confidence: Optional[float] = None
    collection_name: Optional[str] = None  # Per-question collection choice
    creativity_level: Optional[int] = None  # 0=Factual, 1=Balanced, 2=Creative


class ICCompletionState(BaseModel):
    """Root model containing all IC workflow state."""
    workspace_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # All classified fields
    classified_fields: List[PersistedClassifiedField] = Field(default_factory=list)

    # Separated field lists (indices into classified_fields)
    auto_complete_field_indices: List[int] = Field(default_factory=list)
    intelligent_field_indices: List[int] = Field(default_factory=list)

    # Questions grouped by type (type_value -> list of field indices)
    questions_by_type: Dict[str, List[int]] = Field(default_factory=dict)

    # Per-question status keyed by field_text
    question_status: Dict[str, PersistedQuestionState] = Field(default_factory=dict)

    # Evidence cache keyed by field_text
    evidence_cache: Dict[str, List[PersistedEvidence]] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Conversion functions between dataclasses and Pydantic models

def evidence_to_persisted(evidence) -> PersistedEvidence:
    """Convert Evidence dataclass to PersistedEvidence."""
    return PersistedEvidence(
        text=evidence.text,
        source_doc=evidence.source_doc,
        source_chunk_id=evidence.source_chunk_id,
        relevance_score=evidence.relevance_score,
        doc_type=evidence.doc_type,
        metadata=evidence.metadata if evidence.metadata else {}
    )


def persisted_to_evidence(persisted: PersistedEvidence):
    """Convert PersistedEvidence to Evidence dataclass."""
    from .evidence_retriever import Evidence
    return Evidence(
        text=persisted.text,
        source_doc=persisted.source_doc,
        source_chunk_id=persisted.source_chunk_id,
        relevance_score=persisted.relevance_score,
        doc_type=persisted.doc_type,
        metadata=persisted.metadata
    )


def classified_field_to_persisted(field) -> PersistedClassifiedField:
    """Convert ClassifiedField dataclass to PersistedClassifiedField."""
    from .field_classifier import FieldTier, QuestionType

    tier = PersistedFieldTier(field.tier.value) if field.tier else PersistedFieldTier.AUTO_COMPLETE

    question_type = None
    if field.question_type:
        question_type = PersistedQuestionType(field.question_type.value)

    return PersistedClassifiedField(
        field_text=field.field_text,
        tier=tier,
        question_type=question_type,
        auto_complete_mapping=field.auto_complete_mapping,
        confidence=field.confidence,
        word_limit=field.word_limit,
        context_hint=field.context_hint
    )


def persisted_to_classified_field(persisted: PersistedClassifiedField):
    """Convert PersistedClassifiedField to ClassifiedField dataclass."""
    from .field_classifier import ClassifiedField, FieldTier, QuestionType

    tier = FieldTier(persisted.tier.value) if persisted.tier else FieldTier.AUTO_COMPLETE

    question_type = None
    if persisted.question_type:
        question_type = QuestionType(persisted.question_type.value)

    return ClassifiedField(
        field_text=persisted.field_text,
        tier=tier,
        question_type=question_type,
        auto_complete_mapping=persisted.auto_complete_mapping,
        confidence=persisted.confidence,
        word_limit=persisted.word_limit,
        context_hint=persisted.context_hint
    )
