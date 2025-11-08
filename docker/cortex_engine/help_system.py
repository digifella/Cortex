# ## File: cortex_engine/help_system.py
# Version: 1.0.0 (Help System Implementation)
# Date: 2025-07-29
# Purpose: Centralized help system with tooltips and comprehensive documentation

import streamlit as st
from typing import Dict, List
from pathlib import Path

class HelpSystem:
    """Centralized help system for Cortex Suite"""
    
    def __init__(self):
        self.help_content = self._load_help_content()
    
    def _load_help_content(self) -> Dict:
        """Load all help content"""
        return {
            "overview": {
                "title": "Cortex Suite Overview",
                "description": "AI-powered knowledge management and proposal generation system",
                "content": """
                ### Welcome to Cortex Suite! 🚀
                
                Cortex Suite is your comprehensive AI-powered knowledge management system that helps you:
                
                **📚 Manage Knowledge**
                - Ingest documents (PDFs, Word, PowerPoint, images)
                - Extract entities and relationships automatically
                - Search using advanced AI and graph technology
                
                **🧠 Generate Intelligence**
                - Create proposals using your knowledge base
                - Research topics with AI assistance
                - Analyze document collections and relationships
                
                **🔧 Stay Organized**
                - Manage document collections
                - Track ingestion history and errors
                - Backup and restore your data
                
                ### Quick Start Guide
                1. **Start with Knowledge Ingest** to add your documents
                2. **Use Knowledge Search** to explore your content
                3. **Create Collections** to organize related documents
                4. **Generate Proposals** using your curated knowledge
                """
            },
            "ai_research": {
                "title": "AI Assisted Research",
                "description": "Multi-agent research system with AI-powered web search and analysis",
                "content": """
                ### AI Assisted Research 🔍
                
                **Purpose**: Conduct comprehensive research on any topic using AI agents.
                
                **Key Features**:
                - **Multi-Agent System**: Different AI agents handle different aspects of research
                - **Web Integration**: Searches the web for current information
                - **Knowledge Integration**: Combines web results with your knowledge base
                - **Visual Outputs**: Creates mind maps and structured reports
                
                **How to Use**:
                1. **Enter Research Topic**: Describe what you want to research
                2. **Select Scope**: Choose depth (quick overview vs deep dive)
                3. **Choose Sources**: Web search, knowledge base, or both
                4. **Review Results**: Get structured reports, mind maps, and source lists
                
                **Best Practices**:
                - Be specific in your research queries
                - Use this for topics requiring current information
                - Review sources for credibility
                - Save important findings to your knowledge base
                
                **Tips**:
                - Start broad, then narrow down with follow-up research
                - Use the mind maps to identify knowledge gaps
                - Export results to use in proposals or presentations
                """
            },
            "knowledge_ingest": {
                "title": "Knowledge Ingest",
                "description": "Import and process documents with AI-powered metadata extraction",
                "content": """
                ### Knowledge Ingest 📚
                
                **Purpose**: Add documents to your knowledge base with intelligent processing.
                
                **Supported Formats**:
                - **Documents**: PDF, Word (.docx), PowerPoint (.pptx)
                - **Images**: PNG, JPG, JPEG (with AI vision analysis)
                - **Text Files**: TXT, Markdown, CSV, JSON
                
                **Processing Modes**:
                
                **🔍 Standard Mode (Recommended for small batches)**:
                - Preview each file before processing
                - Review and edit AI-generated metadata
                - Perfect for careful curation
                
                **🚀 Batch Mode (NEW - For large collections)**:
                - Skip preview and process hundreds of files automatically
                - Errors logged separately for later review
                - Ideal for bulk imports
                
                **Workflow**:
                1. **Set Paths**: Configure source and database locations
                2. **Select Directories**: Choose folders to scan
                3. **Choose Mode**: Standard preview or batch processing
                4. **Apply Filters**: Exclude temp files, prefer certain formats
                5. **Review/Process**: Standard mode shows preview, batch mode runs automatically
                6. **Organize**: Create collections from newly ingested documents
                
                **Advanced Features**:
                - **Smart Filtering**: Automatically excludes temp files and duplicates
                - **Entity Extraction**: Identifies people, organizations, and projects
                - **Error Handling**: Failed documents logged with detailed error messages
                - **Progress Tracking**: Real-time progress updates during processing
                
                **Tips for Success**:
                - Use batch mode for initial bulk imports
                - Use standard mode for ongoing document additions
                - Check the failure log if documents don't appear
                - Organize documents into collections immediately after ingestion
                """
            },
            "knowledge_search": {
                "title": "Knowledge Search",
                "description": "Advanced AI-powered search with GraphRAG technology",
                "content": """
                ### Knowledge Search 🔍
                
                **Purpose**: Find information in your knowledge base using advanced AI search.
                
                **Search Technologies**:
                
                **🔤 Semantic Search**: 
                - Understands meaning, not just keywords
                - Finds conceptually related content
                - Great for broad topic exploration
                
                **🕸️ GraphRAG Search**:
                - Uses knowledge graph relationships
                - Finds connections between entities
                - Discovers insights through relationships
                
                **🔗 Hybrid Search**:
                - Combines semantic + graph approaches
                - Most comprehensive results
                - Recommended for complex queries
                
                **Search Interface**:
                - **Query Input**: Natural language questions work best
                - **Collection Filter**: Search within specific collections
                - **Result Limit**: Control number of results returned
                - **Search History**: Review and reuse previous searches
                
                **Understanding Results**:
                - **Relevance Score**: How well content matches your query
                - **Source Information**: Document type, date, collection
                - **Entity Connections**: Related people, organizations, projects
                - **Content Preview**: Key excerpts with highlighting
                
                **Advanced Features**:
                - **Entity Relationships**: Explore connections between concepts
                - **Collection Analytics**: Understanding your document landscape
                - **Export Options**: Save results for later reference
                
                **Search Tips**:
                - Ask questions like "What projects involved Company X?"
                - Use specific terms when you know them
                - Try different search modes for different types of queries
                - Review entity relationships for discovery insights
                """
            },
            "collections": {
                "title": "Collection Management",
                "description": "Organize and curate document collections for specific purposes",
                "content": """
                ### Collection Management 🗂️
                
                **Purpose**: Organize your documents into themed collections for easy access.
                
                **What are Collections?**
                Collections are curated groups of documents organized around:
                - **Projects**: All documents for a specific project
                - **Clients**: Documents related to particular clients
                - **Topics**: Documents covering specific subject areas
                - **Proposals**: Supporting materials for proposal development
                
                **Collection Operations**:
                
                **📝 Create Collections**:
                - From search results (save useful findings)
                - From ingestion results (organize new documents)
                - Manually by selecting documents
                
                **✏️ Manage Collections**:
                - Add/remove documents
                - Rename collections
                - Merge collections
                - Export collection contents
                
                **🔍 Use Collections**:
                - Filter searches to specific collections
                - Generate proposals from collection content
                - Analyze collection themes and relationships
                
                **Smart Features**:
                - **Duplicate Detection**: Warns about documents in multiple collections
                - **Collection Statistics**: Shows document counts, types, dates
                - **Related Suggestions**: AI suggests related documents to add
                - **Export Options**: Various formats for external use
                
                **Best Practices**:
                - Create collections with clear, descriptive names
                - Keep collections focused and purposeful
                - Regular cleanup to remove outdated documents
                - Use collections as input for proposal generation
                
                **Collection Strategies**:
                - **Project-Based**: One collection per client project
                - **Topic-Based**: Collections for expertise areas
                - **Temporal**: Collections for specific time periods
                - **Hybrid**: Combine approaches as needed
                """
            },
            "proposals": {
                "title": "Proposal Generation",
                "description": "AI-powered proposal creation using your knowledge base",
                "content": """
                ### Proposal Generation 📝
                
                **Purpose**: Create professional proposals using your knowledge base content.
                
                **Two-Step Process**:
                
                **📋 Step 1: Template Preparation**
                - **Upload Templates**: Use existing Word templates
                - **Define Placeholders**: Mark sections for AI completion
                - **Set Instructions**: Guide AI on content requirements
                - **Map Knowledge**: Connect template sections to collections
                
                **✍️ Step 2: Proposal Creation**
                - **Select Template**: Choose prepared template
                - **Configure Content**: Select knowledge collections to use
                - **AI Generation**: AI fills template using your knowledge
                - **Review & Edit**: Refine generated content
                - **Export**: Download completed proposal
                
                **🤖 Proposal Copilot**:
                - Interactive proposal assistant
                - Real-time content suggestions
                - Knowledge base integration
                - Collaborative editing experience
                
                **Template Features**:
                - **Smart Placeholders**: AI understands section purpose
                - **Content Mapping**: Links template sections to knowledge
                - **Style Preservation**: Maintains document formatting
                - **Reusable Templates**: Save templates for future use
                
                **AI Capabilities**:
                - **Content Generation**: Creates relevant content from knowledge
                - **Context Awareness**: Understands proposal requirements
                - **Style Matching**: Adapts to your writing style
                - **Fact Checking**: Ensures accuracy against knowledge base
                
                **Quality Assurance**:
                - **Source Attribution**: Tracks content sources
                - **Consistency Checks**: Ensures coherent messaging
                - **Completeness Review**: Identifies missing sections
                - **Version Control**: Tracks proposal iterations
                
                **Best Practices**:
                - Prepare templates with clear section definitions
                - Curate relevant collections before proposal generation
                - Review and refine AI-generated content
                - Use Proposal Copilot for iterative improvement
                """
            },
            "analytics": {
                "title": "Knowledge Analytics",
                "description": "Analyze your knowledge base patterns and relationships",
                "content": """
                ### Knowledge Analytics 📊
                
                **Purpose**: Understand patterns and relationships in your knowledge base.
                
                **Analysis Types**:
                
                **📈 Document Analytics**:
                - Document type distribution
                - Ingestion patterns over time
                - Collection size and growth
                - Processing success rates
                
                **🕸️ Entity Analysis**:
                - People, organizations, and project relationships
                - Entity co-occurrence patterns
                - Knowledge graph visualizations
                - Relationship strength analysis
                
                **🔍 Content Analysis**:
                - Topic modeling and clustering
                - Keyword frequency analysis
                - Content similarity patterns
                - Knowledge gap identification
                
                **📊 Usage Analytics**:
                - Search pattern analysis
                - Most accessed documents
                - Collection usage statistics
                - Proposal generation patterns
                
                **Visualization Features**:
                - **Interactive Charts**: Explore data dynamically
                - **Network Graphs**: Visualize entity relationships
                - **Timeline Views**: See knowledge evolution
                - **Heatmaps**: Identify patterns and clusters
                
                **Insights Provided**:
                - **Knowledge Gaps**: Areas needing more content
                - **Expert Networks**: Who knows what
                - **Content Relationships**: Unexpected connections
                - **Usage Patterns**: How knowledge is accessed
                
                **Business Intelligence**:
                - **Client Analysis**: Understanding client relationships
                - **Project Patterns**: Successful project characteristics
                - **Expertise Mapping**: Team knowledge distribution
                - **Growth Tracking**: Knowledge base evolution
                
                **Using Analytics**:
                - Identify areas for targeted content collection
                - Understand team expertise distribution
                - Discover hidden relationships in your knowledge
                - Optimize knowledge organization strategies
                """
            },
            "tips_best_practices": {
                "title": "Tips & Best Practices",
                "description": "Expert tips for getting the most out of Cortex Suite",
                "content": """
                ### Tips & Best Practices 💡
                
                **🚀 Getting Started**:
                - Start with a small, focused set of documents for your first ingestion
                - Use batch mode for large initial imports, standard mode for ongoing additions
                - Create meaningful collection names that your team will understand
                - Set up your paths once and save them in the configuration
                
                **📚 Knowledge Management**:
                - **Document Naming**: Use consistent, descriptive file names
                - **Folder Structure**: Organize source documents logically
                - **Regular Maintenance**: Periodically review and clean up collections
                - **Error Monitoring**: Check ingestion failure logs regularly
                
                **🔍 Search Optimization**:
                - **Natural Language**: Ask complete questions, not just keywords
                - **Be Specific**: Include context in your queries when possible
                - **Try Different Modes**: Semantic for concepts, Graph for relationships
                - **Use Collections**: Filter searches to relevant document sets
                
                **📝 Proposal Excellence**:
                - **Template Preparation**: Invest time in creating good templates
                - **Collection Curation**: Organize relevant knowledge before proposal creation
                - **Iterative Refinement**: Use Proposal Copilot for multiple rounds of improvement
                - **Quality Review**: Always review AI-generated content for accuracy
                
                **🔧 System Optimization**:
                - **Regular Backups**: Use the backup system for important knowledge bases
                - **Monitor Performance**: Check logs if ingestion or search seems slow
                - **Update Models**: Keep AI models updated for best performance
                - **Clean Environment**: Regularly clean up temporary files and logs
                
                **👥 Team Collaboration**:
                - **Naming Conventions**: Establish team standards for collections and documents
                - **Knowledge Sharing**: Use collections to share curated knowledge sets
                - **Documentation**: Keep notes on important searches and findings
                - **Training**: Ensure team members understand different search modes
                
                **🛠️ Troubleshooting**:
                - **Ingestion Issues**: Check file permissions and formats
                - **Search Problems**: Try different query formulations
                - **Performance**: Monitor system resources during large operations
                - **Errors**: Check logs directory for detailed error information
                
                **📈 Continuous Improvement**:
                - Use analytics to understand knowledge patterns
                - Regularly review and optimize your collection structure
                - Learn from search patterns to improve knowledge organization
                - Gather team feedback on system usage and needs
                """
            },
            "anonymizer": {
                "title": "Document Anonymizer",
                "description": "Replace identifying information with generic placeholders",
                "content": """
                ### Document Anonymizer 📝
                
                The Document Anonymizer helps protect privacy by replacing identifying information with generic placeholders.
                
                **What Gets Anonymized:**
                - **People names** → Person A, Person B, etc.
                - **Company names** → Company 1, Company 2, etc.
                - **Project names** → Project 1, Project 2, etc.
                - **Email addresses** → [EMAIL]
                - **Phone numbers** → [PHONE]
                - **URLs** → [URL]
                - **Headers/footers** → [HEADER/FOOTER REMOVED]
                
                **Supported Formats:**
                - TXT (plain text files)
                - PDF (Adobe PDF documents)
                - DOCX (Microsoft Word documents)
                
                **Key Features:**
                - **Smart Detection**: Uses AI to identify entities with high accuracy
                - **Consistent Mapping**: Same entities get same anonymous names across files
                - **Batch Processing**: Handle multiple files at once
                - **Cross-Platform**: Works on Mac, Windows, Linux, and Docker
                - **Mapping Report**: Creates reference file showing replacements
                
                **How to Use:**
                1. **Select Files**: Drag-drop, browse files, or select entire directories
                2. **Configure**: Choose shared mapping and confidence settings
                3. **Set Output**: Specify where anonymized files should be saved
                4. **Run**: Click "Start Anonymization" and review results
                
                **Best Practices:**
                - Test with sample documents first
                - Review anonymized output before sharing
                - Keep mapping files secure (they can reverse anonymization)
                - Use shared mapping for related document sets
                - Adjust confidence threshold based on your needs
                
                **Platform Support:**
                - **Mac**: Full drag-drop support from Finder
                - **Windows**: Drag-drop from File Explorer
                - **Linux**: Standard file path support
                - **Docker**: Complete functionality in containers
                
                **Troubleshooting:**
                - **"File not found"**: Ensure paths are accessible
                - **"Unsupported format"**: Convert to TXT, PDF, or DOCX
                - **Over-anonymization**: Increase confidence threshold
                - **Under-anonymization**: Lower confidence threshold
                """
            }
        }
    
    def show_help_menu(self):
        """Display the main help menu in sidebar"""
        with st.sidebar:
            st.markdown("---")
            st.subheader("📖 Help & Documentation")
            
            help_options = [
                ("🏠 Overview", "overview"),
                ("🔍 AI Research", "ai_research"), 
                ("📚 Knowledge Ingest", "knowledge_ingest"),
                ("🔎 Knowledge Search", "knowledge_search"),
                ("🗂️ Collections", "collections"),
                ("📝 Proposals", "proposals"),
                ("📊 Analytics", "analytics"),
                ("🔒 Document Anonymizer", "anonymizer"),
                ("💡 Tips & Best Practices", "tips_best_practices")
            ]
            
            selected_help = st.selectbox(
                "Select Help Topic:",
                options=[option[1] for option in help_options],
                format_func=lambda x: next(option[0] for option in help_options if option[1] == x),
                key="help_topic_selector"
            )
            
            if st.button("📖 Show Help", use_container_width=True):
                st.session_state.show_help_modal = True
                st.session_state.help_topic = selected_help
    
    def show_help_modal(self, topic: str):
        """Display help content in a modal-like container"""
        help_data = self.help_content.get(topic, {})
        
        if not help_data:
            st.error(f"Help topic '{topic}' not found.")
            return
        
        # Create a prominent help display
        st.markdown("---")
        st.markdown(f"# 📖 {help_data['title']}")
        st.markdown(f"*{help_data['description']}*")
        st.markdown("---")
        
        # Display the help content
        st.markdown(help_data['content'])
        
        # Close button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✅ Close Help", use_container_width=True):
                st.session_state.show_help_modal = False
                st.rerun()
    
    def add_tooltip(self, text: str, tooltip: str) -> str:
        """Add a tooltip to text using Streamlit's help parameter simulation"""
        return f"{text} ℹ️"
    
    def show_contextual_help(self, page: str):
        """Show contextual help for specific pages"""
        contextual_tips = {
            "ingest": {
                "batch_mode": "🚀 **Batch Mode**: Skip file preview and process hundreds of files automatically. Errors are logged separately for review.",
                "smart_filtering": "🔍 **Smart Filtering**: Automatically excludes temporary files, prefers newer versions, and removes duplicates.",
                "entity_extraction": "🧠 **Entity Extraction**: AI automatically identifies people, organizations, and projects in your documents.",
                "image_processing": "📸 **Image Processing**: AI can analyze images and extract text/descriptions using vision models."
            },
            "search": {
                "semantic_search": "🔤 **Semantic Search**: Understands meaning and context, not just exact keyword matches.",
                "graphrag": "🕸️ **GraphRAG**: Uses entity relationships to find connected information and insights.",
                "hybrid_mode": "🔗 **Hybrid Search**: Combines semantic understanding with graph relationships for comprehensive results.",
                "collection_filtering": "🗂️ **Collection Filtering**: Limit search to specific document collections for focused results."
            },
            "proposals": {
                "template_preparation": "📋 **Template Setup**: Define placeholders and instructions to guide AI content generation.",
                "knowledge_mapping": "🗺️ **Knowledge Mapping**: Connect template sections to relevant document collections.",
                "ai_generation": "🤖 **AI Generation**: AI creates content using your knowledge base while maintaining your template structure.",
                "iterative_refinement": "✏️ **Iterative Refinement**: Use Proposal Copilot to continuously improve generated content."
            }
        }
        
        tips = contextual_tips.get(page, {})
        if tips:
            with st.expander("💡 Quick Tips for This Page"):
                for tip_key, tip_text in tips.items():
                    st.markdown(f"- {tip_text}")

# Global help system instance
help_system = HelpSystem()