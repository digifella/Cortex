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
            ("pages/1_AI_Assisted_Research.py", "Discovery Research", "🤖"),
            ("pages/2_Knowledge_Ingest.py", "Knowledge Ingest", "🧠"),
            ("pages/3_Knowledge_Search.py", "Knowledge Search", "🔍"),
            ("pages/4_Collection_Management.py", "Collections", "📚"),
            ("pages/5_Knowledge_Analytics.py", "Analytics", "📊"),
        ),
    ),
    (
        "Research & Documents",
        (
            ("pages/7_Document_Extract.py", "Document Processing", "📄"),
            ("pages/8_Document_Summarizer.py", "Document Summarizer", "📝"),
            ("pages/12_Document_Dialog.py", "Document Dialog", "💬"),
            ("pages/9_Knowledge_Synthesizer.py", "Knowledge Synthesizer", "🧩"),
            ("pages/15_Claim_Citation_Mapper.py", "Claim-Citation Mapper", "🧷"),
        ),
    ),
    (
        "Media & Metadata",
        (
            ("pages/10_Visual_Analysis.py", "Visual Analysis", "👁️"),
            ("pages/11_Metadata_Management.py", "Metadata Management", "🏷️"),
            ("pages/20_Photo_Metadata_Tools.py", "Photo & Metadata Tools", "📷"),
            ("pages/21_Audio_Cleanup.py", "Audio Cleanup", "🎙️"),
        ),
    ),
    (
        "Proposals",
        (
            ("pages/13_Proposal_Manager.py", "Proposal Manager", "📋"),
            ("pages/Entity_Profile_Manager.py", "Entity Profiles", "🏢"),
        ),
    ),
    (
        "Specialist Tools",
        (
            ("pages/17_Stakeholder_Signals.py", "Stakeholder Signals", "🎯"),
            ("pages/18_Private_Vault_GraphRAG.py", "Private Vault GraphRAG", "🔐"),
        ),
    ),
    (
        "System",
        (
            ("pages/16_Queue_Monitor.py", "Queue Monitor", "📡"),
            ("pages/6_Maintenance.py", "Maintenance", "🔧"),
        ),
    ),
)


# Retained in the repository for compatibility and historical reference, but
# intentionally omitted from the sidebar because their workflows are available
# through a canonical page above.
HIDDEN_LEGACY_PAGES: Final[tuple[str, ...]] = (
    "pages/14_URL_Ingestor.py",
    "pages/19_Researcher_Assistant.py",
    "pages/Proposal_Workspace.py",
    "pages/Proposal_Chunk_Review_V2.py",
    "pages/Proposal_Intelligent_Completion.py",
)
