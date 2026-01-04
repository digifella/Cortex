"""
Flexible Proposal System - Version 2.0.0
No rigid instructions required - works with ANY tender document structure.
"""

from cortex_engine.proposals.flexible_parser import (
    FlexibleTemplateParser,
    FlexibleSection,
    SectionType,
    ContentStatus
)

from cortex_engine.proposals.hint_assistant import (
    HintBasedAssistant,
    AssistanceRequest,
    AssistanceResult,
    AssistanceMode
)

__all__ = [
    "FlexibleTemplateParser",
    "FlexibleSection",
    "SectionType",
    "ContentStatus",
    "HintBasedAssistant",
    "AssistanceRequest",
    "AssistanceResult",
    "AssistanceMode"
]
