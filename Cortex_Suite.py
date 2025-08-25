# ## File: Cortex_Suite.py
# Version: v3.0.0 (Enhanced Visual Processing Integration)
# Date: 2025-08-24
# Purpose: A central Streamlit launchpad for the integrated Cortex Suite.

import streamlit as st
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cortex_engine.help_system import help_system
from cortex_engine.utils.model_checker import model_checker
from cortex_engine.system_status import system_status

# --- Page Setup ---
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
    st.caption("Version v3.0.0 - Enhanced Visual Processing Integration")
    
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
    st.caption("Version v3.0.0 (Enhanced Visual Processing Integration)")

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
    """
    <div style='text-align: center; color: #666; font-size: 0.85em; margin: 1em 0;'>
        <strong>🕒 Latest Code Changes:</strong> 2025-08-26<br>
        <em>Code Quality Cleanup: Removed duplicate utilities, optimized imports, cleaned up legacy files, and improved logging consistency across modules</em>
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