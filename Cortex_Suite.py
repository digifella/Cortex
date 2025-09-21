# ## File: Cortex_Suite.py
# Version: v4.8.0
# Date: 2025-08-30
# Purpose: Main entry point for the Cortex Suite application

import streamlit as st
import sys
from pathlib import Path
import time

# Add the project root to the Python path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cortex_engine.system_status import system_status
from cortex_engine.version_config import get_version_display, VERSION_METADATA
from cortex_engine.utils.model_checker import model_checker
from cortex_engine.help_system import help_system

def load_recent_changelog_entries(max_versions=3):
    """Load recent changelog entries for What's New section"""
    try:
        # Try to find CHANGELOG.md in project directory or docker directory
        changelog_paths = [
            Path(__file__).parent / "CHANGELOG.md",  # Project root
            Path(__file__).parent / "docker" / "CHANGELOG.md"  # Docker directory
        ]
        
        changelog_content = None
        for changelog_path in changelog_paths:
            if changelog_path.exists():
                with open(changelog_path, 'r', encoding='utf-8') as f:
                    changelog_content = f.read()
                break
        
        if not changelog_content:
            return "📋 Changelog not found - using version config information instead."
        
        # Parse recent version entries
        lines = changelog_content.split('\n')
        versions = []
        current_version = None
        current_content = []
        
        for line in lines:
            # Look for version headers (## v4.4.0 - Release Name)
            if line.startswith('## v') and len(versions) < max_versions:
                # Save previous version if exists
                if current_version and current_content:
                    versions.append({
                        'header': current_version,
                        'content': '\n'.join(current_content).strip()
                    })
                    current_content = []
                
                current_version = line.replace('## ', '### ')  # Convert to smaller header for display
                
            elif current_version and line.strip():  # Only collect non-empty lines
                # Skip certain sections we don't want in What's New
                if not any(skip in line.lower() for skip in ['### 🔥 breaking changes', '### breaking changes']):
                    current_content.append(line)
        
        # Add the last version
        if current_version and current_content and len(versions) < max_versions:
            versions.append({
                'header': current_version,
                'content': '\n'.join(current_content).strip()
            })
        
        if not versions:
            return f"📋 **Latest Version:** {get_version_display()}\n\n{VERSION_METADATA.get('description', 'Latest updates and improvements.')}"
        
        # Format for display
        result = ""
        for version in versions:
            result += f"{version['header']}\n{version['content']}\n\n"
        
        return result.strip()
        
    except Exception as e:
        # Fallback to version config if changelog parsing fails
        return f"📋 **Latest Version:** {get_version_display()}\n\n{VERSION_METADATA.get('description', 'Recent updates and improvements.')}\n\n*For complete version history, see CHANGELOG.md*"

st.set_page_config(
    page_title="Cortex Suite",
    page_icon="🚀",
    layout="wide"
)

# Check system setup status
try:
    setup_info = system_status.get_setup_progress()
    setup_complete = setup_info.get("setup_complete", False)
except Exception:
    setup_complete = False
    setup_info = {"progress_percent": 0, "status_message": "⚠️ System status check failed"}

if not setup_complete:
    # Show setup progress page
    st.title("🔧 Cortex Suite Setup in Progress")
    st.caption(get_version_display())
    
    # Progress bar
    progress_percent = setup_info.get("progress_percent", 0)
    st.progress(progress_percent / 100.0)
    
    # Status message
    status_message = setup_info.get("status_message", "Setting up...")
    st.markdown(f"### {status_message}")
    
    # Detailed status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Services Status")
        ollama_status = "✅ Running" if setup_info.get("ollama_running") else "🔄 Starting..."
        api_status = "✅ Running" if setup_info.get("api_running") else "🔄 Starting..."
        st.write(f"🤖 Ollama Service: {ollama_status}")
        st.write(f"🔗 API Server: {api_status}")
    
    with col2:
        st.markdown("#### AI Models")
        models = setup_info.get("models", [])
        for model in models:
            status_icon = "✅" if model["available"] else "⬇️"
            model_name = model["name"].split(":")[-1] if ":" in model["name"] else model["name"]
            st.write(f"{status_icon} {model_name} ({model['size_gb']}GB)")
    
    # Error messages
    errors = setup_info.get("errors", [])
    if errors:
        st.markdown("#### Issues")
        for error in errors:
            st.error(error)
    
    # Instructions
    st.markdown("""
    ---
    ### What's happening?
    
    🚀 **Good news!** The Cortex Suite interface is running and accessible.
    
    ⬇️ **AI models are downloading** in the background (this can take 15-30 minutes for ~20GB total).
    
    ✨ **You can already explore** the interface and configure settings while models download.
    
    🎯 **Full AI features** will become available automatically once downloads complete.
    
    ### While you wait:
    - Browse the navigation menu to see available tools
    - Check out the Knowledge Search and Collection Management
    - Review the system documentation
    """)
    
    # Auto-refresh
    st.markdown("*This page will auto-refresh every 30 seconds...*")
    time.sleep(30)
    st.rerun()

else:
    # Normal main page
    st.title("🚀 Welcome to the Project Cortex Suite")
    st.caption(get_version_display())
    
    # What's New section - automatically loaded from CHANGELOG.md
    with st.expander("✨ What's New in Recent Updates", expanded=False):
        changelog_content = load_recent_changelog_entries(max_versions=3)
        st.markdown(changelog_content)
        
        # Add link to full changelog
        st.markdown("---")
        st.markdown("📋 **[View Complete Changelog](./CHANGELOG.md)** • All version history and detailed changes")

st.markdown("""
This is the central hub for the Cortex Suite, an integrated workbench for building a knowledge base and using it for AI-assisted proposal development.

Please select a tool from the sidebar on the left to begin your workflow:

-   **🤖 AI Assisted Research:** Use a multi-agent system to perform external research on a topic and synthesize the findings into a "Discovery Note".

-   **🧠 Knowledge Ingest:** Ingest new documents (your own files or generated Discovery Notes) into the central knowledge base.

-   **🔬 Knowledge Search:** Explore the entire knowledge base, build curated "Working Collections", and prune unwanted documents from the KB.

-   **📚 Collection Management:** View the contents of, rename, export, or delete your Working Collections.

-   **📝 Proposal Step 1 Prep:** (Formerly Template Editor) Create and modify `.docx` templates with interactive Cortex instructions.

-   **🗂️ Proposal Step 2 Make:** (Formerly Proposal Management) Create new proposals or load, manage, and delete existing ones.

-   **📊 Knowledge Analytics:** NEW! Comprehensive analytics dashboard providing insights into knowledge base usage, entity relationships, knowledge gaps, and optimization opportunities.

-   **📝 Document Anonymizer:** NEW! Replace identifying information in documents with generic placeholders (Person A, Company 1, etc.) for privacy protection.

-   **💡 Idea Generator:** NEW! Transform your knowledge into innovative concepts using the Double Diamond methodology (Discover, Define, Develop, Deliver) for structured ideation.

-   **📄 Document Summarizer:** NEW! Generate intelligent summaries from any document with multiple detail levels (Highlights, Summary, Detailed Analysis). Perfect for extracting key insights from lengthy reports, research papers, and proposals.

This unified interface provides a seamless workflow from initial research to final document generation.
""")

st.divider()

# Add model status check in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    # Display platform configuration and hybrid backend info
    try:
        setup_info = system_status.get_setup_progress()
        if "platform_config" in setup_info:
            st.info(setup_info["platform_config"])
            
        # Display backend information if available
        if "backends" in setup_info and setup_info["backends"]:
            st.markdown("**🚀 AI Backends:**")
            for backend in setup_info["backends"]:
                name = backend["name"].replace("_", " ").title()
                status_icon = "🟢" if backend["available"] else "🔴"
                tier_icon = "⭐" if backend["performance_tier"] == "premium" else "📦"
                model_info = f"({backend['model_count']} models)" if backend['model_count'] > 0 else "(no models)"
                
                st.caption(f"{status_icon} {tier_icon} {name} {model_info}")
                
            # Show active strategy
            if "hybrid_strategy" in setup_info and setup_info["hybrid_strategy"]:
                strategy_display = setup_info["hybrid_strategy"].replace("_", " ").title()
                st.caption(f"🎯 **Strategy:** {strategy_display}")
                
    except Exception:
        st.info("💻 Platform: Detecting...")
    
    # Quick model availability check
    ingestion_check = model_checker.check_ingestion_requirements(include_images=True)
    research_check = model_checker.check_research_requirements()
    
    if ingestion_check["can_proceed"]:
        st.success("✅ Ingestion: Ready")
    else:
        st.error("❌ Ingestion: Missing models")
        with st.expander("View Details"):
            st.markdown(model_checker.format_status_message(ingestion_check))
    
    if research_check["local_research_available"]:
        st.success("✅ Research: Ready")
    elif research_check["ollama_running"]:
        st.warning("⚠️ Research: Cloud only")
        st.caption("Local research model not available")
    else:
        st.error("❌ Research: Ollama down")
    
    st.divider()
    
    # Database Status Check
    st.subheader("💾 Database Status")
    
    # Import required modules for database checking
    try:
        from cortex_engine.session_state import initialize_app_session_state
        from cortex_engine.utils import convert_windows_to_wsl_path
        import os
        
        # Initialize session state to get database path
        initialize_app_session_state()
        
        current_db_path = st.session_state.get("db_path_input", "")
        
        if current_db_path:
            # Convert and check paths
            wsl_path = convert_windows_to_wsl_path(current_db_path)
            chroma_path = os.path.join(wsl_path, "knowledge_hub_db")
            
            # Display current path
            st.caption(f"📁 **Path:** `{current_db_path}`")
            
            # Check base directory
            if os.path.exists(wsl_path):
                st.success("✅ Base directory exists")
                
                # Check knowledge base
                if os.path.exists(chroma_path):
                    st.success("✅ Knowledge base found")
                    
                    # Try to count documents
                    try:
                        import chromadb
                        from chromadb.config import Settings as ChromaSettings
                        from cortex_engine.config import COLLECTION_NAME
                        
                        db_settings = ChromaSettings(anonymized_telemetry=False)
                        db = chromadb.PersistentClient(path=chroma_path, settings=db_settings)
                        collection = db.get_or_create_collection(COLLECTION_NAME)
                        doc_count = collection.count()
                        
                        if doc_count > 0:
                            st.info(f"📚 **{doc_count} documents** in knowledge base")
                        else:
                            st.warning("⚠️ Knowledge base is empty")
                            st.caption("Run Knowledge Ingest to add documents")
                            
                    except Exception as db_e:
                        st.warning("⚠️ Cannot inspect knowledge base")
                        st.caption(f"Error: {str(db_e)[:50]}...")
                else:
                    st.error("❌ Knowledge base missing")
                    st.caption("Run Knowledge Ingest to create it")
            else:
                st.error("❌ Database directory not found")
                st.caption("Check path in Knowledge Search")
        else:
            st.warning("⚠️ No database path configured")
            st.caption("Configure in Knowledge Search")
            
        # Quick fix button
        if st.button("🔧 Fix Database Path", use_container_width=True):
            st.info("💡 Go to **Knowledge Search** page to configure the database path")
            
    except Exception as status_e:
        st.error("❌ Database status check failed")
        st.caption(f"Error: {str(status_e)[:50]}...")

# Add help system
help_system.show_help_menu()

# Show help modal if requested
if st.session_state.get("show_help_modal", False):
    help_topic = st.session_state.get("help_topic", "overview")
    help_system.show_help_modal(help_topic)
else:
    st.info("💡 **Need help?** Use the Help menu in the sidebar to get detailed guidance on any module.")
    
    # Quick start guide on main page
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        ### New to Cortex Suite? Follow these steps:
        
        **1. 📚 Start with Knowledge Ingest**
        - Add your first documents to build your knowledge base
        - Try **Batch Mode** for large document collections
        - **Standard Mode** for careful curation of smaller sets
        
        **2. 🔍 Explore with Knowledge Search**
        - Search your documents using natural language
        - Try different search modes (Semantic, GraphRAG, Hybrid)
        - Create collections to organize useful findings
        
        **3. 🗂️ Organize with Collections**
        - Group related documents by project, client, or topic
        - Use collections as input for proposal generation
        
        **4. 📝 Generate Proposals**
        - Prepare templates in Step 1
        - Create proposals using your knowledge base in Step 2
        - Use Proposal Copilot for interactive assistance
        
        **5. 📊 Monitor with Analytics**
        - Understand your knowledge patterns
        - Identify gaps and opportunities
        - Track system usage and effectiveness
        
        **💡 Pro Tip**: Start small with a focused set of documents, then expand as you become comfortable with the system!
        """)

st.info("To get started, click on one of the pages in the navigation sidebar.")

st.divider()

# Attribution footer
# Latest code changes footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; font-size: 0.85em; margin: 1em 0;'>
        <strong>🕒 Latest Code Changes:</strong> {VERSION_METADATA['release_date']}<br>
        <em>{VERSION_METADATA['description']}</em>
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown("""
<div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 5px; margin-top: 20px;">
    <p style="margin: 0; color: #666; font-size: 14px;">
        <strong>System designed and built by Paul Cooper, Director of Longboardfella Consulting Pty Ltd.</strong><br>
        <a href="https://www.longboardfella.com.au" target="_blank" style="color: #1f77b4;">www.longboardfella.com.au</a><br><br>
        Design and concepts demonstrated in the Cortex Suite may be freely shared provided attribution to Paul Cooper is made.<br>
        System is a beta - no support available. Code may not be fully stable.<br>
        Feedback is welcome and can be made via email to <a href="mailto:paul@longboardfella.com.au" style="color: #1f77b4;">paul@longboardfella.com.au</a>
    </p>
</div>
""", unsafe_allow_html=True)