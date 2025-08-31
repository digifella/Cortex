# Knowledge Management Hub
# Version: v4.4.2
# Date: 2025-08-31
# Central hub for all knowledge management operations

import streamlit as st
from pathlib import Path

# Import version from centralized config
from cortex_engine.version_config import VERSION_STRING

st.set_page_config(page_title="Knowledge Management", layout="wide", page_icon="🧠")

# Page configuration
PAGE_VERSION = VERSION_STRING

def main():
    st.title("🧠 Knowledge Management Hub")
    st.caption(f"Version: {PAGE_VERSION} • Central hub for all knowledge operations")
    
    st.markdown("""
    **Knowledge Management provides comprehensive tools for:**
    - 📄 **Document Ingestion** - Import and process documents into your knowledge base
    - 🔍 **Knowledge Search** - Search through your indexed documents and data
    - 💡 **Idea Generation** - Transform knowledge into innovative concepts using Double Diamond methodology
    - 🔒 **Document Anonymization** - Remove sensitive information from documents
    - 📝 **Document Summarization** - Generate intelligent summaries from any document
    """)
    
    # Create a grid layout for the knowledge management functions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Core Operations")
        
        # Knowledge Ingest
        with st.container():
            st.markdown("""
            #### 📄 Knowledge Ingest
            Import documents, PDFs, text files, and other content into your knowledge base with AI-powered processing.
            """)
            if st.button("🚀 Start Knowledge Ingest", use_container_width=True, key="nav_ingest"):
                st.switch_page("pages/_Knowledge_Ingest.py")
        
        st.divider()
        
        # Knowledge Search  
        with st.container():
            st.markdown("""
            #### 🔍 Knowledge Search
            Search through your knowledge base using vector similarity, text search, and GraphRAG enhanced queries.
            """)
            if st.button("🔍 Start Knowledge Search", use_container_width=True, key="nav_search"):
                st.switch_page("pages/_Knowledge_Search.py")
        
        st.divider()
        
        # Idea Generator
        with st.container():
            st.markdown("""
            #### 💡 Idea Generator
            Transform your knowledge into innovative concepts using the Double Diamond methodology (Discover, Define, Develop, Deliver).
            """)
            if st.button("💡 Start Idea Generator", use_container_width=True, key="nav_ideas"):
                st.switch_page("pages/_Idea_Generator.py")
    
    with col2:
        st.markdown("### 🛠️ Specialized Tools")
        
        # Document Anonymizer
        with st.container():
            st.markdown("""
            #### 🔒 Document Anonymizer
            Remove or replace sensitive information (names, organizations, emails) from documents with configurable anonymization.
            """)
            if st.button("🔒 Start Document Anonymizer", use_container_width=True, key="nav_anon"):
                st.switch_page("pages/_Document_Anonymizer.py")
        
        st.divider()
        
        # Document Summarizer
        with st.container():
            st.markdown("""
            #### 📝 Document Summarizer  
            Generate intelligent summaries from any document with multiple detail levels and formats.
            """)
            if st.button("📝 Start Document Summarizer", use_container_width=True, key="nav_summ"):
                st.switch_page("pages/_Document_Summarizer.py")
        
        st.divider()
        
        # Knowledge Analytics (link to existing page)
        with st.container():
            st.markdown("""
            #### 📊 Knowledge Analytics
            Analyze your knowledge base with comprehensive dashboards, entity insights, and usage statistics.
            """)
            if st.button("📊 View Knowledge Analytics", use_container_width=True, key="nav_analytics"):
                st.switch_page("pages/7_Knowledge_Analytics.py")
    
    st.markdown("---")
    
    # Quick status and help information
    st.markdown("### ℹ️ Quick Status")
    
    # Load configuration to show current database path
    try:
        from cortex_engine.config_manager import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.get_config()
        db_path = config.get('ai_database_path', 'Not configured')
        
        st.info(f"**Current Knowledge Base:** `{db_path}`")
        
        # Check if knowledge base exists
        if db_path and db_path != 'Not configured':
            kb_path = Path(db_path) / "knowledge_hub_db"
            if kb_path.exists():
                st.success("✅ Knowledge base is accessible")
            else:
                st.warning("⚠️ Knowledge base directory not found - use Knowledge Ingest to create")
        else:
            st.warning("⚠️ Database path not configured - use Knowledge Ingest to set up")
            
    except Exception as e:
        st.error(f"Could not load configuration: {e}")
    
    # Help and workflow guidance
    with st.expander("💡 Workflow Guidance", expanded=False):
        st.markdown("""
        ### Recommended Knowledge Management Workflow:
        
        1. **📄 Start with Knowledge Ingest** - Import your documents and build your knowledge base
        2. **🔍 Use Knowledge Search** - Find relevant information and create working collections
        3. **📊 Check Knowledge Analytics** - Understand your data through visual insights
        4. **💡 Generate Ideas** - Transform knowledge into innovative concepts
        5. **🛠️ Use Specialized Tools** - Anonymize sensitive content or create summaries as needed
        
        ### Integration Points:
        - **Search → Ideas**: Use search results as input for idea generation
        - **Analytics → Search**: Identify knowledge gaps and search for specific content
        - **Ingest → All**: Fresh content immediately available across all tools
        """)

if __name__ == "__main__":
    main()