# Intelligent Proposal Completion System - Architectural Plan

**Status:** Planning
**Date:** 2026-01-19
**Priority:** High - Core Proposal Workflow Enhancement

---

## Executive Summary

The current proposal system treats all fields equally, spending LLM cycles on simple fields like "Registered Office Address" that could be template-filled in milliseconds. Meanwhile, the substantive fields that actually win contracts (capability statements, methodology, value propositions) don't leverage the knowledge collection at all.

**The Solution:** A two-tier field processing architecture that:
1. **Auto-completes** simple fields from entity profile (no LLM needed)
2. **Intelligently generates** substantive responses using knowledge collection + LLM

---

## Problem Analysis

### Current State Issues

From analyzing the example RFQ (DPFEM-FSST-2026.docx):

**Fields the system currently focuses on (low value):**
- Supplier's name, ACN, ABN
- Postal address, Email address, Telephone number
- Registered office address
- Contact person details
- Signature blocks

**Fields the system should focus on (high value):**
- "Please provide details of your capability and capacity to deliver the required services"
- "Please outline your proposed methodology and approach"
- "Detail how you will have a positive impact on the Tasmanian community or economy"
- "Are you a Tasmanian SME? How many Tasmanian jobs will be supported?"
- "Will you source components from other Tasmanian SMEs or sub-contractors?"

### Root Cause

The `markup_engine.py` current prompt only offers simple @mentions:
```python
available_fields = {
    '@companyname': 'Company legal name',
    '@abn': 'Australian Business Number',
    '@acn': 'Australian Company Number',
    '@registered_office': 'Registered office address',
    '@email': 'Contact email',
    '@phone': 'Contact phone number',
    '@website': 'Company website',
    '@narrative[company_overview]': 'Company overview/executive summary'
}
```

There's no mechanism to:
1. Recognize substantive questions requiring drafted responses
2. Search the knowledge collection for evidence
3. Generate intelligent responses based on that evidence

---

## Proposed Architecture

### Two-Tier Processing Model

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION                        │
│  • Parse document structure                                  │
│  • Extract all fillable fields/questions                     │
│  • Classify each field → Tier 1 or Tier 2                   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│   TIER 1: AUTO-COMPLETE │     │   TIER 2: INTELLIGENT GEN   │
│   ─────────────────────│     │   ─────────────────────────  │
│   • Pattern matching    │     │   • Question classification  │
│   • Entity profile lookup│    │   • Knowledge search         │
│   • Template substitution│    │   • Evidence retrieval       │
│   • Zero LLM calls      │     │   • Draft generation         │
│   • Instant completion  │     │   • Confidence scoring       │
└─────────────────────────┘     └─────────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    USER REVIEW INTERFACE                     │
│  • Auto-complete summary (quick verify/edit)                │
│  • Substantive responses queue (review drafts + evidence)   │
│  • Evidence panel (source documents from knowledge base)    │
│  • Export completed proposal                                │
└─────────────────────────────────────────────────────────────┘
```

### Tier 1: Auto-Complete Fields

**Definition:** Fields that can be filled directly from entity profile data.

**Identification Patterns:**
```python
SIMPLE_FIELD_PATTERNS = [
    r'(?:company|business|supplier|tenderer).*(?:name|title)',
    r'(?:ABN|ACN|ARBN|TFN)',
    r'(?:postal|registered|business|street).*address',
    r'(?:email|e-mail).*address',
    r'(?:phone|telephone|mobile|fax).*(?:number)?',
    r'(?:contact|authorised|authorized).*(?:person|officer|representative)',
    r'(?:bank|account|BSB).*(?:details|number)',
    r'(?:insurance|policy).*(?:number|details)',
    r'(?:website|URL|web address)',
    r'(?:date of (?:incorporation|registration))',
]
```

**Processing:**
1. Match field against patterns
2. Look up corresponding entity profile field
3. Substitute value directly
4. No LLM call needed

**User Interface:**
- Show summary: "12 fields auto-completed from entity profile"
- Expandable list for quick verification
- Edit button if changes needed

### Tier 2: Intelligence-Required Fields

**Definition:** Fields requiring substantive responses that should leverage the knowledge collection.

**Question Types:**

| Type | Examples | Evidence Needed |
|------|----------|-----------------|
| CAPABILITY | "Describe your experience...", "Proven history in..." | Past project summaries, case studies |
| METHODOLOGY | "Outline your approach...", "Proposed method..." | Process documents, frameworks |
| VALUE_PROPOSITION | "How will you benefit...", "What value..." | Impact statements, testimonials |
| COMPLIANCE | "Confirm you can meet...", "Certify that..." | Policies, certifications |
| INNOVATION | "Describe any innovative...", "Novel approach..." | R&D docs, unique capabilities |
| RISK | "Identify risks...", "Mitigation strategies..." | Risk registers, contingency plans |

**Processing Pipeline:**

```
Question → Classify Type → Formulate Search Query → Search Knowledge Collection
                                                            │
                                        ┌───────────────────┴───────────────────┐
                                        │     EVIDENCE RETRIEVAL                │
                                        │  • Vector search for semantic match   │
                                        │  • Rerank with Qwen3-VL for precision │
                                        │  • Extract relevant passages          │
                                        │  • Score evidence quality             │
                                        └───────────────────┬───────────────────┘
                                                            │
                                        ┌───────────────────┴───────────────────┐
                                        │     RESPONSE GENERATION               │
                                        │  • Build prompt with:                 │
                                        │    - Question requirements            │
                                        │    - Entity profile context           │
                                        │    - Retrieved evidence               │
                                        │    - Word/format constraints          │
                                        │  • Generate draft response            │
                                        │  • Calculate confidence score         │
                                        └───────────────────────────────────────┘
```

---

## Implementation Components

### 1. Field Classifier (`field_classifier.py`)

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class FieldTier(Enum):
    AUTO_COMPLETE = "auto_complete"  # Tier 1
    INTELLIGENT = "intelligent"       # Tier 2

class QuestionType(Enum):
    CAPABILITY = "capability"
    METHODOLOGY = "methodology"
    VALUE_PROPOSITION = "value_proposition"
    COMPLIANCE = "compliance"
    INNOVATION = "innovation"
    RISK = "risk"
    GENERAL = "general"

@dataclass
class ClassifiedField:
    field_text: str
    tier: FieldTier
    question_type: Optional[QuestionType]
    auto_complete_mapping: Optional[str]  # e.g., "entity.company_name"
    confidence: float
    word_limit: Optional[int]  # Extracted from context if specified

class FieldClassifier:
    """Classifies proposal fields into auto-complete vs intelligent tiers."""

    def classify(self, field_text: str, context: str = "") -> ClassifiedField:
        """Classify a single field."""
        # First pass: Rule-based for obvious auto-complete fields
        if mapping := self._check_auto_complete(field_text):
            return ClassifiedField(
                field_text=field_text,
                tier=FieldTier.AUTO_COMPLETE,
                question_type=None,
                auto_complete_mapping=mapping,
                confidence=0.95,
                word_limit=None
            )

        # Second pass: Classify question type
        question_type = self._classify_question_type(field_text, context)
        word_limit = self._extract_word_limit(field_text, context)

        return ClassifiedField(
            field_text=field_text,
            tier=FieldTier.INTELLIGENT,
            question_type=question_type,
            auto_complete_mapping=None,
            confidence=0.8,
            word_limit=word_limit
        )
```

### 2. Evidence Retriever (`evidence_retriever.py`)

```python
@dataclass
class Evidence:
    text: str
    source_doc: str
    relevance_score: float
    doc_type: str  # case_study, policy, capability_statement, etc.

class EvidenceRetriever:
    """Retrieves relevant evidence from knowledge collection for proposal responses."""

    def __init__(self, knowledge_search, collection_name: str):
        self.search = knowledge_search
        self.collection = collection_name

    def find_evidence(
        self,
        question: str,
        question_type: QuestionType,
        max_results: int = 5
    ) -> List[Evidence]:
        """Find relevant evidence from knowledge collection."""

        # Reformulate query based on question type
        search_query = self._build_search_query(question, question_type)

        # Search with reranking for precision
        results = self.search.hybrid_search(
            query=search_query,
            collection=self.collection,
            use_reranker=True,
            top_k=max_results * 2  # Over-fetch then filter
        )

        # Extract and score evidence passages
        evidence = self._extract_evidence(results, question)

        return evidence[:max_results]

    def _build_search_query(self, question: str, qtype: QuestionType) -> str:
        """Reformulate question into effective search query."""
        prefixes = {
            QuestionType.CAPABILITY: "experience delivering projects involving",
            QuestionType.METHODOLOGY: "approach process methodology for",
            QuestionType.VALUE_PROPOSITION: "benefits impact outcomes of",
            QuestionType.COMPLIANCE: "certification policy compliance with",
            QuestionType.INNOVATION: "innovative solution technology for",
            QuestionType.RISK: "risk mitigation strategy for",
        }
        prefix = prefixes.get(qtype, "")
        return f"{prefix} {question}"
```

### 3. Response Generator (`response_generator.py`)

```python
@dataclass
class DraftResponse:
    text: str
    evidence_used: List[Evidence]
    confidence: float  # 0-1 based on evidence quality
    word_count: int
    needs_review: bool  # True if low confidence or no evidence

class ResponseGenerator:
    """Generates draft responses using LLM + evidence."""

    def __init__(self, llm: LLMInterface, entity_manager: EntityProfileManager):
        self.llm = llm
        self.entity_manager = entity_manager

    def generate(
        self,
        question: str,
        question_type: QuestionType,
        evidence: List[Evidence],
        entity_id: str,
        word_limit: Optional[int] = None
    ) -> DraftResponse:
        """Generate a draft response for a substantive question."""

        entity = self.entity_manager.get_entity_profile(entity_id)

        prompt = self._build_prompt(
            question=question,
            question_type=question_type,
            evidence=evidence,
            entity=entity,
            word_limit=word_limit
        )

        response_text = self.llm.generate(prompt)

        return DraftResponse(
            text=response_text,
            evidence_used=evidence,
            confidence=self._calculate_confidence(evidence, response_text),
            word_count=len(response_text.split()),
            needs_review=len(evidence) < 2 or self._has_placeholders(response_text)
        )

    def _build_prompt(self, question, question_type, evidence, entity, word_limit):
        """Build generation prompt with evidence injection."""

        evidence_section = "\n\n".join([
            f"**Source: {e.source_doc}**\n{e.text}"
            for e in evidence
        ])

        word_instruction = f"\nTarget length: approximately {word_limit} words." if word_limit else ""

        return f"""You are writing a proposal response for {entity.company_name}.

QUESTION TO ANSWER:
{question}

RELEVANT EVIDENCE FROM OUR KNOWLEDGE BASE:
{evidence_section}

COMPANY CONTEXT:
- Company: {entity.company_name}
- Overview: {entity.narratives.get('company_overview', 'N/A')}

INSTRUCTIONS:
1. Write a professional, compelling response that directly answers the question
2. Use specific details from the evidence provided - reference actual projects, numbers, outcomes
3. Write in first person plural ("we", "our team")
4. Be concrete and specific, avoid generic statements
5. If evidence is insufficient, note [NEEDS DETAIL: specific info needed]
{word_instruction}

DRAFT RESPONSE:"""
```

### 4. Enhanced Review UI

**New workflow in Proposal_Chunk_Review_V2.py:**

```python
# Instead of linear chunk review, offer two panels:

# PANEL 1: Auto-Complete Summary
st.subheader("Auto-Completed Fields")
st.success(f"✅ {len(auto_completed)} fields filled from entity profile")

with st.expander("Review auto-completed fields", expanded=False):
    for field in auto_completed:
        col1, col2, col3 = st.columns([2, 3, 1])
        col1.write(field.label)
        col2.write(field.value)
        if col3.button("Edit", key=f"edit_{field.id}"):
            # Open edit modal
            pass

# PANEL 2: Substantive Responses Queue
st.subheader("Responses Requiring Review")
st.info(f"📝 {len(substantive)} responses drafted - review and approve")

for response in substantive:
    with st.container(border=True):
        # Question header
        st.markdown(f"**{response.question_type.value.title()}:** {response.question[:100]}...")

        # Confidence indicator
        confidence_color = "green" if response.confidence > 0.7 else "orange" if response.confidence > 0.4 else "red"
        st.markdown(f"Confidence: :{confidence_color}[{response.confidence:.0%}]")

        # Draft response (editable)
        edited = st.text_area(
            "Draft Response",
            value=response.text,
            height=200,
            key=f"response_{response.id}"
        )

        # Evidence panel
        with st.expander("📚 Evidence Sources"):
            for evidence in response.evidence_used:
                st.markdown(f"**{evidence.source_doc}** (relevance: {evidence.relevance_score:.0%})")
                st.markdown(f"> {evidence.text[:300]}...")

        # Actions
        col1, col2, col3 = st.columns(3)
        if col1.button("✅ Approve", key=f"approve_{response.id}"):
            pass
        if col2.button("🔄 Regenerate", key=f"regen_{response.id}"):
            pass
        if col3.button("❌ Skip", key=f"skip_{response.id}"):
            pass
```

---

## Implementation Phases

### Phase 1: Field Classifier (Foundation)
**Effort:** 2-3 days

1. Create `field_classifier.py` with FieldTier and QuestionType enums
2. Implement rule-based auto-complete detection
3. Implement question type classification
4. Add word limit extraction
5. Unit tests

**Deliverable:** Working classifier that can categorize fields from the example RFQ.

### Phase 2: Auto-Complete Engine
**Effort:** 2-3 days

1. Extend entity profile schema with common tender fields
2. Create field → profile mapping configuration
3. Implement template substitution engine
4. Build verification UI component
5. Integration tests

**Deliverable:** Simple fields auto-fill instantly from entity profile.

### Phase 3: Evidence Retrieval
**Effort:** 3-4 days

1. Create `evidence_retriever.py`
2. Implement query reformulation per question type
3. Integrate with existing knowledge search + Qwen3-VL reranker
4. Add evidence extraction and scoring
5. Test with real knowledge collection

**Deliverable:** Can retrieve relevant evidence for capability/methodology questions.

### Phase 4: Response Generation
**Effort:** 3-4 days

1. Create `response_generator.py`
2. Design prompts for each question type
3. Implement confidence scoring
4. Add placeholder detection ([NEEDS DETAIL])
5. Test quality of generated responses

**Deliverable:** Can generate draft responses with evidence injection.

### Phase 5: Enhanced Review UI
**Effort:** 4-5 days

1. Redesign Proposal_Chunk_Review_V2.py
2. Split into Auto-Complete Summary + Substantive Queue
3. Add evidence panel
4. Add regeneration capability
5. User testing and refinement

**Deliverable:** New review workflow that focuses user attention on high-value content.

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Time on simple fields | ~60% of review time | ~5% (quick verify) |
| Time on substantive fields | ~40% (no drafts) | ~95% (review drafts) |
| Evidence utilization | 0% (not integrated) | 80% of responses cite evidence |
| User edits required | High (blank fields) | Low (refine drafts) |
| Proposal completion time | Hours | ~30-45 minutes |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Poor field classification | Fields incorrectly routed | Hybrid rule+LLM approach, user override |
| Irrelevant evidence retrieved | Bad draft responses | Strict reranking, evidence scoring |
| Generated responses too generic | Don't win contracts | Require specific evidence citations |
| Knowledge collection gaps | Can't answer questions | Flag low-confidence, prompt user to add docs |
| LLM hallucination | False claims in proposal | Evidence-grounded generation, confidence scores |

---

## Questions for User

Before proceeding, please confirm:

1. **Knowledge Collection Scope:** Should the system search the entire knowledge base, or should we require users to specify a collection (e.g., "Past Projects", "Capability Statements")?

2. **Entity Profile Extension:** Do you want to add more auto-completable fields to entity profiles (e.g., bank details, insurance numbers, certifications)?

3. **Draft Generation Model:** Should we use the same LLM (qwen2.5:72b) for draft generation, or a different model optimized for longer outputs?

4. **Priority Questions:** Which question types are most important to get right first - Capability, Methodology, or Value Proposition?

---

## Next Steps

1. Review this plan and provide feedback
2. Answer the questions above
3. Begin Phase 1 (Field Classifier) implementation
4. Iterate based on testing with real proposals
