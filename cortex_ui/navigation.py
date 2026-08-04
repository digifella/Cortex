"""Declarative navigation structure for the Cortex Suite Streamlit app."""

from typing import Final


PageDefinition = tuple[str, str, str]
NavigationSection = tuple[str, tuple[PageDefinition, ...]]


NAVIGATION_SECTIONS: Final[tuple[NavigationSection, ...]] = (
    (
        "Start",
        (
            ("cortex_ui/home.py", "Home", "🏠"),
        ),
    ),
    (
        "Core Workflow",
        (
            ("cortex_pages/1_AI_Assisted_Research.py", "Discovery Research", "🤖"),
            ("cortex_pages/2_Knowledge_Ingest.py", "Knowledge Ingest", "🧠"),
            ("cortex_pages/3_Knowledge_Search.py", "Knowledge Search", "🔍"),
            ("cortex_pages/4_Collection_Management.py", "Collections", "📚"),
            ("cortex_pages/5_Knowledge_Analytics.py", "Analytics", "📊"),
        ),
    ),
    (
        "Research & Documents",
        (
            ("cortex_pages/7_Document_Extract.py", "Document Processing", "📄"),
            ("cortex_pages/8_Document_Summarizer.py", "Document Summarizer", "📝"),
            ("cortex_pages/12_Document_Dialog.py", "Document Dialog", "💬"),
            ("cortex_pages/9_Knowledge_Synthesizer.py", "Knowledge Synthesizer", "🧩"),
            ("cortex_pages/15_Claim_Citation_Mapper.py", "Claim-Citation Mapper", "🧷"),
        ),
    ),
    (
        "Media & Metadata",
        (
            ("cortex_pages/10_Visual_Analysis.py", "Visual Analysis", "👁️"),
            ("cortex_pages/11_Metadata_Management.py", "Metadata Management", "🏷️"),
            ("cortex_pages/20_Photo_Metadata_Tools.py", "Photo & Metadata Tools", "📷"),
            ("cortex_pages/21_Audio_Cleanup.py", "Audio Cleanup", "🎙️"),
        ),
    ),
    (
        "Proposals",
        (
            ("cortex_pages/13_Proposal_Manager.py", "Proposal Manager", "📋"),
            ("cortex_pages/Entity_Profile_Manager.py", "Entity Profiles", "🏢"),
        ),
    ),
    (
        "Specialist Tools",
        (
            ("cortex_pages/17_Stakeholder_Signals.py", "Stakeholder Signals", "🎯"),
            ("cortex_pages/18_Private_Vault_GraphRAG.py", "Private Vault GraphRAG", "🔐"),
        ),
    ),
    (
        "System",
        (
            ("cortex_pages/16_Queue_Monitor.py", "Queue Monitor", "📡"),
            ("cortex_pages/6_Maintenance.py", "Maintenance", "🔧"),
        ),
    ),
)


# Retained in the repository for compatibility and historical reference, but
# intentionally omitted from the sidebar because their workflows are available
# through a canonical page above.
HIDDEN_LEGACY_PAGES: Final[tuple[str, ...]] = (
    "cortex_pages/14_URL_Ingestor.py",
    "cortex_pages/19_Researcher_Assistant.py",
    "cortex_pages/Proposal_Workspace.py",
    "cortex_pages/Proposal_Chunk_Review_V2.py",
    "cortex_pages/Proposal_Intelligent_Completion.py",
)
