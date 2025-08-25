"""
System Terminal
Version: v1.0.0
Date: 2025-08-25

Safe command execution interface for system management and troubleshooting.
Provides a secure, user-friendly terminal interface within the Streamlit GUI.
"""

import streamlit as st
import time
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="System Terminal", 
    page_icon="💻",
    layout="wide"
)

# Page configuration
PAGE_VERSION = "v1.0.0"

# Import Cortex modules
try:
    from cortex_engine.utils.command_executor import display_command_executor_widget, SafeCommandExecutor
except ImportError as e:
    st.error(f"Failed to import command executor: {e}")
    st.stop()

def display_header():
    """Display page header with information"""
    st.title("💻 System Terminal")
    st.caption(f"Version: {PAGE_VERSION} • Safe Command Execution Interface")
    
    st.markdown("""
    This secure terminal interface allows you to execute system commands safely within the Cortex Suite environment.
    Only whitelisted commands are permitted to ensure system security.
    """)

def display_quick_actions():
    """Display quick action buttons for common tasks"""
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📦 Check Models", use_container_width=True):
            st.session_state.quick_command = "ollama list"
            st.rerun()
    
    with col2:
        if st.button("🔍 System Info", use_container_width=True):
            st.session_state.quick_command = "python --version"
            st.rerun()
    
    with col3:
        if st.button("🐳 Docker Status", use_container_width=True):
            st.session_state.quick_command = "docker ps"
            st.rerun()
    
    with col4:
        if st.button("🤖 Ollama Version", use_container_width=True):
            st.session_state.quick_command = "ollama --version"
            st.rerun()

def display_model_management_section():
    """Display model management tools"""
    st.subheader("🤖 AI Model Management")
    
    # Model installation suggestions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📥 Install Popular Models:**")
        models_to_install = [
            ("llava:7b", "4.7GB", "Visual processing"),
            ("moondream", "1.6GB", "Lightweight vision"), 
            ("codellama", "3.8GB", "Code generation"),
            ("mistral", "4.4GB", "General purpose")
        ]
        
        for model, size, description in models_to_install:
            if st.button(f"📦 Install {model} ({size})", key=f"install_{model}", use_container_width=True):
                st.session_state.quick_command = f"ollama pull {model}"
                st.rerun()
            st.caption(f"   {description}")
    
    with col2:
        st.markdown("**ℹ️ Model Information:**")
        info_commands = [
            ("List all models", "ollama list"),
            ("Show model details", "ollama show llava:7b"),
            ("Check running models", "ollama ps")
        ]
        
        for description, command in info_commands:
            if st.button(f"🔍 {description}", key=f"info_{command.replace(' ', '_')}", use_container_width=True):
                st.session_state.quick_command = command
                st.rerun()

def display_troubleshooting_section():
    """Display troubleshooting tools and information"""
    with st.expander("🛠️ Troubleshooting & Help"):
        st.markdown("""
        ### Common Issues and Solutions
        
        **🔴 "Command not found" errors:**
        - Make sure Docker Desktop is running
        - The command might not be available in your environment
        - Try checking the Environment Information below
        
        **⚠️ Model download failures:**
        - Check your internet connection
        - Verify you have enough disk space (models can be several GB)
        - Try the command again - downloads can sometimes fail and need retry
        
        **🐳 Docker-specific issues:**
        - Commands run inside the Docker container, not on your host system
        - Models are stored in Docker volumes and persist between container restarts
        - If a command works in regular terminal but not here, it might be a path issue
        
        ### Security Features
        - Only whitelisted commands are allowed (ollama, docker info commands, python version, etc.)
        - No file system modification commands permitted
        - No network configuration commands allowed
        - Command history is logged for audit purposes
        
        ### Getting Help
        - Use the Environment Information section to understand your setup
        - Check the Command History to see what was run previously
        - All commands are logged with timestamps and success status
        """)

def main():
    """Main application logic"""
    display_header()
    
    # Handle quick command execution
    if 'quick_command' in st.session_state:
        st.info(f"📋 Executing: `{st.session_state.quick_command}`")
        # The command will be picked up by the command executor widget
    
    # Quick actions
    display_quick_actions()
    st.markdown("---")
    
    # Model management section
    display_model_management_section()
    st.markdown("---")
    
    # Main command executor widget
    suggested_commands = [
        "ollama list",
        "ollama pull llava:7b",
        "docker ps", 
        "python --version"
    ]
    
    # Pass the quick command if set
    initial_command = st.session_state.get('quick_command', '')
    if initial_command:
        del st.session_state.quick_command
    
    st.markdown("### 💻 Command Executor")
    display_command_executor_widget("Secure Terminal", suggested_commands)
    
    # Pre-fill command if from quick action
    if initial_command:
        st.session_state.command_to_execute = initial_command
        st.rerun()
    
    st.markdown("---")
    
    # Troubleshooting section
    display_troubleshooting_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666; font-size: 0.85em;'>"
        f"System Terminal v{PAGE_VERSION} • Safe Command Execution • "
        f"Security: Only whitelisted commands permitted"
        f"</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()