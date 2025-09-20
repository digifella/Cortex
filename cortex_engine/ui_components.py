"""
Reusable UI Components for Streamlit Pages
Provides consistent UI elements across all Cortex modules.

Version: 1.0.0  
Date: 2025-08-13
"""

import os
import streamlit as st
from typing import Dict, Optional, Any, Tuple
from .version_config import get_version_footer
from .llm_service import create_llm_service, TaskType, LLMProvider

def llm_provider_selector(task_type: str, key_prefix: str = "", help_text: str = None) -> Tuple[str, Dict[str, Any]]:
    """
    Standardized LLM provider selection component.
    
    Args:
        task_type: Type of task ("research", "proposals", "knowledge", "ideation")
        key_prefix: Unique prefix for session state keys
        help_text: Optional help text for the selector
        
    Returns:
        Tuple of (selected_provider_string, status_info_dict)
    """
    try:
        # Create service manager to get available options
        service_manager = create_llm_service(task_type)
        available_providers = service_manager.get_provider_display_names()
        
        # Task-specific defaults and help text
        task_configs = {
            "research": {
                "default_help": "Choose between local privacy or cloud power for research",
                "default_provider": "Cloud (Gemini)" if "Cloud (Gemini)" in available_providers else available_providers[0]
            },
            "proposals": {
                "default_help": "Proposals use local LLM only for maximum privacy", 
                "default_provider": "Local (Ollama)"
            },
            "knowledge": {
                "default_help": "Knowledge operations use local LLM for optimal performance",
                "default_provider": "Local (Ollama)"  
            },
            "ideation": {
                "default_help": "Choose your preferred LLM for creative ideation",
                "default_provider": "Local (Ollama)"
            }
        }
        
        config = task_configs.get(task_type, {})
        display_help = help_text or config.get("default_help", "Select LLM provider")
        
        # Provider selector (only show if multiple options available)
        if len(available_providers) == 1:
            selected_provider = available_providers[0]
            st.info(f"🤖 **LLM Provider**: {selected_provider}")
        else:
            selected_provider = st.selectbox(
                "🤖 Select LLM Provider",
                options=available_providers,
                index=available_providers.index(config.get("default_provider", available_providers[0])) 
                      if config.get("default_provider") in available_providers else 0,
                key=f"{key_prefix}_llm_provider" if key_prefix else "llm_provider",
                help=display_help
            )
        
        # Create service manager with user selection and get status
        user_service_manager = create_llm_service(task_type, selected_provider)
        status_info = user_service_manager.get_status_info()
        
        # Display status information
        if status_info["status"] == "ready":
            st.success(f"✅ **{status_info['provider']}** ready ({status_info['model']})")
        else:
            st.error(f"❌ **Error**: {status_info['message']}")
            
        return selected_provider, status_info
        
    except Exception as e:
        st.error(f"❌ LLM Provider configuration error: {e}")
        return "Local (Ollama)", {"status": "error", "message": str(e)}

def workflow_progress_display(workflow_orchestrator, compact: bool = False):
    """
    Display workflow progress with orchestrator integration.
    
    Args:
        workflow_orchestrator: WorkflowOrchestrator instance  
        compact: Whether to show compact view
    """
    import streamlit as st
    
    progress_info = workflow_orchestrator.get_step_progress()
    
    if compact:
        # Compact view - just progress bar and current step
        st.progress(progress_info["progress"], 
                   text=f"Step {progress_info['step_number']}/{progress_info['total_steps']}: {progress_info['current_step_title']}")
    else:
        # Full view with step indicators
        st.subheader(f"🔄 {workflow_orchestrator.workflow_name}")
        
        # Progress bar
        st.progress(progress_info["progress"], 
                   text=f"Progress: {progress_info['step_number']}/{progress_info['total_steps']} steps completed")
        
        # Step indicators
        all_steps = progress_info["completed_steps"] + [progress_info["current_step"]] + progress_info["remaining_steps"]
        cols = st.columns(len(all_steps))
        
        for i, step_id in enumerate(all_steps):
            step = workflow_orchestrator.steps[step_id] 
            with cols[i]:
                if step_id == progress_info["current_step"]:
                    st.markdown(f"**🟢 {step.title}**")
                elif step_id in progress_info["completed_steps"]:
                    st.markdown(f"✅ {step.title}")
                else:
                    st.markdown(f"⚪ {step.title}")

def workflow_phase_indicator(current_phase: str, phases: list, key_prefix: str = ""):
    """
    Display workflow progress indicator (legacy method - use workflow_progress_display for new code).
    
    Args:
        current_phase: Currently active phase
        phases: List of phase names
        key_prefix: Unique prefix for components
    """
    import streamlit as st
    
    # Create progress indicator
    if current_phase in phases:
        current_index = phases.index(current_phase)
        progress = (current_index + 1) / len(phases)
    else:
        progress = 0.0
    
    st.progress(progress, text=f"Progress: {current_phase.replace('_', ' ').title()}")
    
    # Create phase tabs or steps
    cols = st.columns(len(phases))
    for i, phase in enumerate(phases):
        with cols[i]:
            if phase == current_phase:
                st.markdown(f"**🟢 {phase.replace('_', ' ').title()}**")
            elif phases.index(phase) < phases.index(current_phase):
                st.markdown(f"✅ {phase.replace('_', ' ').title()}")
            else:
                st.markdown(f"⚪ {phase.replace('_', ' ').title()}")

def error_display(error_message: str, error_type: str = "Error", 
                 recovery_suggestion: str = None, show_details: bool = False):
    """
    Standardized error display with recovery suggestions.
    
    Args:
        error_message: Main error message
        error_type: Type of error (for categorization)
        recovery_suggestion: Optional suggestion for recovery
        show_details: Whether to show detailed error info
    """
    with st.container(border=True):
        st.error(f"❌ **{error_type}**: {error_message}")
        
        if recovery_suggestion:
            st.info(f"💡 **Suggestion**: {recovery_suggestion}")
            
        if show_details:
            with st.expander("🔍 Technical Details"):
                st.code(error_message)

def collection_selector(collection_manager, key_prefix: str = "", required: bool = True) -> Optional[str]:
    """
    Standardized collection selection component.
    
    Args:
        collection_manager: Instance of WorkingCollectionManager
        key_prefix: Unique prefix for session state keys  
        required: Whether collection selection is required
        
    Returns:
        Selected collection name or None
    """
    try:
        collection_names = collection_manager.get_collection_names()
        
        if not collection_names:
            if required:
                st.warning("⚠️ **No collections found**. Please create a collection first in Collection Management.")
                return None
            else:
                st.info("📁 No collections available. Some features will be limited.")
                return None
        
        # Add "default" option if not required
        options = collection_names if required else ["None"] + collection_names
        
        selected = st.selectbox(
            "📚 Select Knowledge Collection",
            options=options,
            index=0,
            key=f"{key_prefix}_collection" if key_prefix else "selected_collection",
            help="Choose the knowledge collection to analyze"
        )
        
        if selected == "None":
            return None
            
        # Display collection info
        doc_count = len(collection_manager.get_doc_ids_by_name(selected))
        st.info(f"📊 **{selected}**: {doc_count} documents")
        
        return selected
        
    except Exception as e:
        st.error(f"❌ Collection loading error: {e}")
        return None

def export_buttons(data: Dict[str, Any], filename_prefix: str, 
                  export_types: list = ["markdown", "json"]) -> None:
    """
    Standardized export functionality.
    
    Args:
        data: Data to export
        filename_prefix: Prefix for generated filenames
        export_types: Types of export to offer
    """
    from datetime import datetime
    import json
    
    cols = st.columns(len(export_types))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, export_type in enumerate(export_types):
        with cols[i]:
            if export_type == "markdown" and st.button(f"📄 Export Markdown", key=f"export_md_{filename_prefix}"):
                try:
                    # Generate markdown content
                    markdown_content = _generate_markdown_export(data)
                    
                    st.download_button(
                        label="📥 Download Markdown",
                        data=markdown_content,
                        file_name=f"{filename_prefix}_{timestamp}.md",
                        mime="text/markdown",
                        key=f"download_md_{filename_prefix}"
                    )
                except Exception as e:
                    st.error(f"Markdown export failed: {e}")
                    
            elif export_type == "json" and st.button(f"📋 Export JSON", key=f"export_json_{filename_prefix}"):
                try:
                    json_content = json.dumps(data, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="📥 Download JSON", 
                        data=json_content,
                        file_name=f"{filename_prefix}_{timestamp}.json",
                        mime="application/json",
                        key=f"download_json_{filename_prefix}"
                    )
                except Exception as e:
                    st.error(f"JSON export failed: {e}")

def render_version_footer(show_divider: bool = True):
    """Render a consistent version footer with environment indicator."""
    try:
        env = "🐳 Docker" if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') else "💻 Local"
        footer = get_version_footer()
        if show_divider:
            st.markdown("---")
        st.caption(f"{footer} • {env}")
    except Exception:
        pass

def _generate_markdown_export(data: Dict[str, Any]) -> str:
    """Generate markdown content from data structure."""
    from datetime import datetime
    
    markdown = f"""# Export Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
"""
    
    # Add data content based on structure
    if "status" in data and data["status"] == "success":
        if "idea_groups" in data:  # Idea Generator format
            markdown += f"- **Total Ideas**: {data.get('total_ideas', 0)}\n"
            markdown += f"- **Problems Addressed**: {data.get('total_problems', 0)}\n"
            markdown += f"- **LLM Provider**: {data.get('llm_provider', 'N/A')}\n\n"
            
            for i, group in enumerate(data.get("idea_groups", []), 1):
                problem = group.get("problem_statement", f"Problem {i}")
                ideas = group.get("ideas", [])
                
                markdown += f"## Problem {i}: {problem}\n\n"
                
                for j, idea in enumerate(ideas, 1):
                    if isinstance(idea, dict):
                        title = idea.get("title", f"Idea {j}")
                        description = idea.get("description", "No description")
                        
                        markdown += f"### {j}. {title}\n\n**Description:** {description}\n\n"
                        
                        if "implementation" in idea:
                            markdown += f"**Implementation:** {idea['implementation']}\n\n"
                        if "impact" in idea:
                            markdown += f"**Impact:** {idea['impact']}\n\n"
                    else:
                        markdown += f"### {j}. {idea}\n\n"
                
                markdown += "---\n\n"
    else:
        # Generic data export
        markdown += f"```json\n{json.dumps(data, indent=2)}\n```\n"
    
    return markdown
