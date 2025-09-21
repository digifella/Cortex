"""
Ollama Utility Functions
Provides centralized Ollama connection checking and error handling
"""

import requests
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def check_ollama_service(host: str = "localhost", port: int = 11434) -> Tuple[bool, Optional[str]]:
    """
    Check if Ollama service is running and accessible.
    
    Returns:
        Tuple of (is_running, error_message)
        - is_running: True if Ollama is accessible
        - error_message: Description of the issue if not running
    """
    try:
        url = f"http://{host}:{port}/api/version"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True, None
        else:
            return False, f"Ollama service responded with status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Ollama service is not running or not accessible"
    except requests.exceptions.Timeout:
        return False, "Ollama service is not responding (timeout)"
    except Exception as e:
        return False, f"Error checking Ollama service: {str(e)}"

def get_ollama_status_message(is_running: bool, error_msg: Optional[str] = None) -> str:
    """
    Get user-friendly status message for Ollama service.
    
    Args:
        is_running: Whether Ollama is running
        error_msg: Optional error message from check_ollama_service
        
    Returns:
        User-friendly status message
    """
    if is_running:
        return "✅ Ollama service is running and ready"
    
    base_msg = "❌ Ollama service is not available"
    if error_msg:
        base_msg += f": {error_msg}"
    
    return base_msg

def get_ollama_instructions() -> str:
    """
    Get installation and setup instructions for Ollama.
    
    Returns:
        Formatted instructions for setting up Ollama
    """
    return """
**To install and start Ollama:**

1. **Install Ollama:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Start the service:**
   ```bash
   ollama serve
   ```

3. **Pull required model:**
   ```bash
   ollama pull mistral
   ```

4. **Verify it's working:**
   ```bash
   curl http://localhost:11434/api/version
   ```

**Alternative:** Use cloud-based LLM providers (Gemini, OpenAI) by configuring your `.env` file.
"""

def format_ollama_error_for_user(operation: str, error_details: str = "") -> str:
    """
    Format a user-friendly error message when Ollama operations fail.
    
    Args:
        operation: What operation was being attempted
        error_details: Technical error details (optional)
        
    Returns:
        User-friendly error message with actionable instructions
    """
    msg = f"🚫 **Unable to complete {operation}** - Ollama service is not available.\n\n"
    msg += "**This feature requires Ollama to be running.** " 
    msg += "Ollama provides local AI processing for enhanced document analysis, search, and content generation.\n\n"
    msg += get_ollama_instructions()
    
    if error_details:
        msg += f"\n\n**Technical details:** {error_details}"
    
    return msg