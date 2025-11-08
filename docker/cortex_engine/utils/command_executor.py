"""
Command Executor Utility
Safe command execution widget for Streamlit with Docker environment detection
"""

import subprocess
import os
import sys
import time
from typing import List, Dict, Optional, Tuple
import streamlit as st
from dataclasses import dataclass

@dataclass
class CommandResult:
    """Result of command execution"""
    success: bool
    output: str
    error: str
    return_code: int
    execution_time: float

class SafeCommandExecutor:
    """
    Safe command executor that can run commands in different environments.
    Automatically detects Docker vs host environment and adjusts commands accordingly.
    """
    
    # Whitelist of safe commands that can be executed
    SAFE_COMMANDS = {
        'ollama': {
            'subcommands': ['list', 'pull', 'show', 'ps', '--version', '--help'],
            'description': 'Ollama model management'
        },
        'docker': {
            'subcommands': ['ps', 'images', 'version', '--version', '--help'],
            'description': 'Docker container management (read-only)'
        },
        'python': {
            'subcommands': ['--version', '-V'],
            'description': 'Python version info'
        },
        'pip': {
            'subcommands': ['list', 'show', '--version'],
            'description': 'Python package info'
        }
    }
    
    def __init__(self):
        self.is_docker_env = self._detect_docker_environment()
        self.execution_history = []
    
    def _detect_docker_environment(self) -> bool:
        """Detect if we're running inside a Docker container"""
        # Check for Docker environment indicators
        docker_indicators = [
            os.path.exists('/.dockerenv'),
            os.environ.get('container') is not None,
            os.environ.get('DOCKER_CONTAINER') is not None,
            'docker' in os.environ.get('PATH', '').lower()
        ]
        return any(docker_indicators)
    
    def _validate_command(self, command_parts: List[str]) -> Tuple[bool, str]:
        """Validate that command is safe to execute"""
        if not command_parts:
            return False, "Empty command"
        
        main_command = command_parts[0].lower()
        
        # Check if main command is whitelisted
        if main_command not in self.SAFE_COMMANDS:
            return False, f"Command '{main_command}' not allowed. Allowed commands: {', '.join(self.SAFE_COMMANDS.keys())}"
        
        # Check subcommands if provided
        if len(command_parts) > 1:
            subcommand = command_parts[1].lower()
            allowed_subcommands = self.SAFE_COMMANDS[main_command]['subcommands']
            
            # Allow any subcommand that starts with allowed ones (for parameters)
            valid_subcommand = any(
                subcommand == allowed or subcommand.startswith(allowed + ' ')
                for allowed in allowed_subcommands
            )
            
            if not valid_subcommand:
                return False, f"Subcommand '{subcommand}' not allowed for {main_command}. Allowed: {', '.join(allowed_subcommands)}"
        
        return True, "Command validated"
    
    def execute_command(self, command: str, timeout: int = 60) -> CommandResult:
        """
        Execute a command safely with environment detection
        
        Args:
            command: Command string to execute
            timeout: Timeout in seconds
            
        Returns:
            CommandResult with execution details
        """
        start_time = time.time()
        command_parts = command.strip().split()
        
        # Validate command safety
        is_valid, validation_msg = self._validate_command(command_parts)
        if not is_valid:
            return CommandResult(
                success=False,
                output="",
                error=f"Security validation failed: {validation_msg}",
                return_code=-1,
                execution_time=time.time() - start_time
            )
        
        try:
            # Execute command with timeout
            result = subprocess.run(
                command_parts,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False  # Don't raise exception on non-zero exit
            )
            
            execution_time = time.time() - start_time
            success = result.returncode == 0
            
            # Store in history
            self.execution_history.append({
                'command': command,
                'success': success,
                'timestamp': time.strftime('%H:%M:%S'),
                'execution_time': execution_time
            })
            
            return CommandResult(
                success=success,
                output=result.stdout,
                error=result.stderr,
                return_code=result.returncode,
                execution_time=execution_time
            )
            
        except subprocess.TimeoutExpired:
            return CommandResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout} seconds",
                return_code=-1,
                execution_time=timeout
            )
        except FileNotFoundError:
            return CommandResult(
                success=False,
                output="",
                error=f"Command '{command_parts[0]}' not found",
                return_code=-1,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return CommandResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                return_code=-1,
                execution_time=time.time() - start_time
            )
    
    def get_environment_info(self) -> Dict[str, str]:
        """Get information about the current environment"""
        return {
            "Environment": "Docker Container" if self.is_docker_env else "Host System",
            "Platform": sys.platform,
            "Python Version": sys.version.split()[0],
            "Working Directory": os.getcwd()
        }

def display_command_executor_widget(title: str = "Command Executor", 
                                   suggested_commands: List[str] = None) -> None:
    """
    Display a Streamlit widget for safe command execution
    
    Args:
        title: Widget title
        suggested_commands: List of suggested commands to show
    """
    st.markdown(f"### 💻 {title}")
    
    executor = SafeCommandExecutor()
    
    # Show environment info
    with st.expander("🔍 Environment Information"):
        env_info = executor.get_environment_info()
        for key, value in env_info.items():
            st.text(f"{key}: {value}")
        
        # Show allowed commands
        st.markdown("**Allowed Commands:**")
        for cmd, info in executor.SAFE_COMMANDS.items():
            st.text(f"• {cmd}: {info['description']}")
    
    # Command input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        command = st.text_input(
            "Enter command:",
            placeholder="e.g., ollama pull llava:7b",
            help="Only whitelisted commands are allowed for security"
        )
    
    with col2:
        execute_button = st.button("▶️ Execute", type="primary", use_container_width=True)
    
    # Suggested commands
    if suggested_commands:
        st.markdown("**💡 Suggested Commands:**")
        cols = st.columns(len(suggested_commands))
        for i, suggested_cmd in enumerate(suggested_commands):
            with cols[i]:
                if st.button(f"📝 {suggested_cmd}", key=f"suggested_{i}", use_container_width=True):
                    st.session_state.command_to_execute = suggested_cmd
                    st.rerun()
    
    # Handle suggested command selection
    if 'command_to_execute' in st.session_state:
        command = st.session_state.command_to_execute
        execute_button = True
        del st.session_state.command_to_execute
    
    # Execute command
    if execute_button and command:
        st.markdown("---")
        st.markdown("### 📋 Execution Results")
        
        with st.spinner(f"Executing: `{command}`"):
            result = executor.execute_command(command)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if result.success else "🔴"
            st.metric("Status", f"{status_color} {'Success' if result.success else 'Failed'}")
        
        with col2:
            st.metric("Return Code", result.return_code)
        
        with col3:
            st.metric("Execution Time", f"{result.execution_time:.2f}s")
        
        # Output
        if result.output:
            st.markdown("**📤 Output:**")
            st.code(result.output, language="bash")
        
        # Errors
        if result.error:
            st.markdown("**⚠️ Error/Warning:**")
            st.code(result.error, language="bash")
        
        # Success feedback
        if result.success:
            st.success("✅ Command executed successfully!")
        else:
            st.error(f"❌ Command failed with return code {result.return_code}")
    
    # Execution history
    if executor.execution_history:
        with st.expander("📜 Command History"):
            for entry in executor.execution_history[-5:]:  # Show last 5 commands
                status_icon = "✅" if entry['success'] else "❌"
                st.text(f"{entry['timestamp']} {status_icon} {entry['command']} ({entry['execution_time']:.2f}s)")

# Convenience function for model installation
def display_model_installer_widget() -> None:
    """Display a specialized widget for model installation"""
    suggested_commands = [
        "ollama list",
        "ollama pull llava:7b", 
        "ollama pull moondream",
        "ollama pull codellama"
    ]
    
    st.info("🔒 **Security Note**: Only whitelisted commands are allowed. This widget cannot execute harmful system commands.")
    
    display_command_executor_widget(
        title="AI Model Installer", 
        suggested_commands=suggested_commands
    )