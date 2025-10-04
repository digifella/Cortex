# Setup Wizard Page
# Version: v4.4.1
# Date: 2025-08-21
# Purpose: Guided setup and onboarding for Cortex Suite with real-time logging

import streamlit as st
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import List, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex_engine.setup_manager import setup_manager, SetupStep, SetupStatus
from cortex_engine.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(page_title="Setup Wizard", layout="wide", page_icon="🧙‍♂️")

# Page version
PAGE_VERSION = "v4.4.1"

st.title("🧙‍♂️ Cortex Suite Setup Wizard")
st.caption(f"Version: {PAGE_VERSION}")

def display_progress_bar(progress):
    """Display setup progress bar."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.progress(progress.overall_progress_percent / 100.0)
    
    with col2:
        st.write(f"{progress.overall_progress_percent:.0f}% Complete")
    
    # Show completed steps
    if progress.completed_steps:
        st.success(f"✅ Completed: {', '.join([step.value.replace('_', ' ').title() for step in progress.completed_steps])}")

def display_step_welcome():
    """Display welcome step."""
    st.markdown("""
    ## 🚀 Welcome to Cortex Suite!
    
    This guided setup will help you configure your AI-powered knowledge management system in just a few minutes.
    
    ### What We'll Set Up:
    1. ✅ **System Environment Check** - Verify Docker, Python, and resources
    2. 📦 **Storage Location** - Choose Docker volume or host folder
    3. 🎯 **AI Model Strategy** - Choose between Docker Model Runner and Ollama
    4. 🔑 **API Configuration** - Set up cloud AI providers (optional)
    5. 📦 **Model Installation** - Download required AI models
    6. ✨ **System Validation** - Test everything is working
    
    ### Local vs Cloud AI:
    - **🏠 Local AI**: Complete privacy, no API costs, works offline
    - **☁️ Cloud AI**: Latest models, faster processing, requires API keys
    
    You can use Cortex Suite entirely locally or enhance it with cloud APIs.
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🚀 Start Setup", type="primary", use_container_width=True):
            return run_setup_step(SetupStep.WELCOME)
    
    return None

def display_step_environment_check(step_result):
    """Display environment check step."""
    st.markdown("## 🔍 System Environment Check")
    
    if step_result and step_result.details.get("checks"):
        checks = step_result.details["checks"]
        
        for check_name, result in checks.items():
            status_icon = result["status"]
            message = result["message"]
            
            if status_icon == "✅":
                st.success(f"{status_icon} {message}")
            elif status_icon == "⚠️":
                st.warning(f"{status_icon} {message}")
            elif status_icon == "❌":
                st.error(f"{status_icon} {message}")
            else:
                st.info(f"{status_icon} {message}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("Continue to Storage", type="primary", use_container_width=True):
            return run_setup_step(SetupStep.ENVIRONMENT_CHECK)
    
    return None

def display_step_model_strategy(step_result):
    """Display model strategy selection step."""
    st.markdown("## 🎯 Choose Your AI Model Strategy")
    
    # Strategy descriptions
    strategies = {
        "1": {
            "name": "Hybrid (Docker + Ollama)",
            "description": "✅ Best performance + compatibility\n✅ 15% faster inference\n✅ Future-proof",
            "recommended": True
        },
        "2": {
            "name": "Docker Model Runner Only", 
            "description": "✅ Enterprise-grade OCI distribution\n✅ CI/CD integration\n⚠️ Requires Docker Model Runner",
            "recommended": False
        },
        "3": {
            "name": "Ollama Only",
            "description": "✅ Simple and reliable\n✅ Large model library\n⚠️ Standard performance",
            "recommended": False
        },
        "4": {
            "name": "Auto-Optimal",
            "description": "✅ System chooses automatically\n✅ Environment-aware\n✅ Zero configuration",
            "recommended": False
        }
    }
    
    # Display strategy options
    choice = st.radio(
        "Select your preferred strategy:",
        options=list(strategies.keys()),
        format_func=lambda x: f"{strategies[x]['name']} {'(Recommended)' if strategies[x]['recommended'] else ''}",
        help="This affects performance, compatibility, and enterprise features"
    )
    
    # Show description of selected strategy
    if choice in strategies:
        st.info(strategies[choice]["description"])
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("Continue", type="primary", use_container_width=True):
            return run_setup_step(SetupStep.MODEL_STRATEGY, {"strategy": choice})
    
    return None

def display_step_storage_configuration(step_result):
    """Display storage configuration step."""
    st.markdown("## 📦 Storage Location")

    st.markdown("""
    Choose where to store your knowledge base (ChromaDB + metadata):
    - **Docker Volume (Default):** Persists across restarts. Easiest.
    - **Host Folder (Bind Mount):** Recommended if you want files on your OS for backups or sharing.
    """)

    choice = st.radio(
        "Select storage mode:",
        options=["volume", "host_bind"],
        format_func=lambda x: "Docker Volume (Default)" if x == "volume" else "Bind to Host Folder"
    )

    ai_db_path = ""
    source_path = ""
    if choice == "host_bind":
        st.info("Provide host paths. On Windows use paths like C:/ai_databases. On macOS/Linux use /Users/... or /home/...")
        ai_db_path = st.text_input("Host folder for AI database", placeholder="C:/ai_databases or /Users/you/ai_databases")
        source_path = st.text_input("(Optional) Host folder for Knowledge Source", placeholder="C:/Knowledge or /Users/you/Knowledge")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Continue", type="primary", use_container_width=True):
            return run_setup_step(SetupStep.STORAGE_CONFIGURATION, {
                "storage_mode": choice,
                "host_ai_database_path": ai_db_path,
                "host_source_path": source_path,
            })

    return None

def display_step_api_configuration(step_result):
    """Display API configuration step."""
    st.markdown("## 🔑 API Configuration")
    
    st.markdown("""
    ### Cloud API Benefits:
    - 🚀 Access to latest AI models (GPT-4, Gemini Pro, Claude)
    - ⚡ Faster processing for research tasks  
    - 🎯 Advanced reasoning capabilities
    - 🌐 YouTube transcript extraction
    """)
    
    # API configuration options
    config_choice = st.radio(
        "Choose configuration option:",
        ["1", "2", "3"],
        format_func=lambda x: {
            "1": "Local Only - Skip API setup (you can add keys later)",
            "2": "Configure APIs - Set up cloud providers now", 
            "3": "Guided Setup - Step-by-step API configuration"
        }[x]
    )
    
    api_keys = {}
    
    if config_choice in ["2", "3"]:
        st.markdown("### Configure API Keys")
        
        with st.expander("🔗 OpenAI (GPT-4, GPT-3.5)", expanded=False):
            st.markdown("""
            **Features:** GPT-4, GPT-3.5, Advanced reasoning, High-quality generation
            **Cost:** ~$0.03 per 1K tokens
            **Setup:** https://platform.openai.com/api-keys
            """)
            openai_key = st.text_input("OpenAI API Key", type="password", help="Starts with 'sk-'")
            if openai_key:
                api_keys["openai"] = openai_key
        
        with st.expander("🔗 Google Gemini", expanded=False):
            st.markdown("""
            **Features:** Gemini Pro, Long context, Multimodal, Free tier available
            **Cost:** Generous free tier: 60 requests/minute
            **Setup:** https://makersuite.google.com/app/apikey
            """)
            gemini_key = st.text_input("Gemini API Key", type="password", help="Starts with 'AI'")
            if gemini_key:
                api_keys["gemini"] = gemini_key
        
        with st.expander("🔗 YouTube Data API", expanded=False):
            st.markdown("""
            **Features:** Video search, Transcript extraction, Channel analysis
            **Cost:** Free: 10,000 units/day
            **Setup:** https://console.developers.google.com/apis/api/youtube.googleapis.com
            """)
            youtube_key = st.text_input("YouTube API Key", type="password")
            if youtube_key:
                api_keys["youtube"] = youtube_key
        
        with st.expander("🔗 Anthropic Claude", expanded=False):
            st.markdown("""
            **Features:** Claude 3, Long context, Advanced reasoning, Safety focused
            **Cost:** ~$0.25 per 1K tokens
            **Setup:** https://console.anthropic.com/
            """)
            anthropic_key = st.text_input("Anthropic API Key", type="password")
            if anthropic_key:
                api_keys["anthropic"] = anthropic_key
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("Continue", type="primary", use_container_width=True):
            return run_setup_step(SetupStep.API_CONFIGURATION, {
                "configure_apis": config_choice,
                "api_keys": api_keys
            })
    
    return None

def display_step_model_installation(step_result):
    """Display model installation step with real-time progress."""
    st.markdown("## 📦 AI Model Installation")
    
    # Show installation progress if in progress
    if (step_result and step_result.status == SetupStatus.IN_PROGRESS and 
        step_result.details and 'installation_progress' in step_result.details):
        
        progress_data = step_result.details['installation_progress']
        
        # Overall progress
        total = progress_data.get('total_models', 1)
        completed = len(progress_data.get('completed_models', []))
        failed = len(progress_data.get('failed_models', []))
        current = progress_data.get('current_model')
        
        # Progress bar
        progress_percent = (completed + failed) / total if total > 0 else 0
        st.progress(progress_percent, text=f"Installing models... {completed + failed}/{total} processed")
        
        # Current model status
        if current:
            st.info(f"🔄 Currently downloading: **{current}**")
        
        # Completed models
        if progress_data.get('completed_models'):
            st.success(f"✅ Completed: {', '.join(progress_data['completed_models'])}")
        
        # Failed models
        if progress_data.get('failed_models'):
            st.error(f"❌ Failed: {', '.join(progress_data['failed_models'])}")
        
        # Refresh every 5 seconds during installation
        time.sleep(5)
        st.rerun()
        
        return None
    
    st.markdown("""
    ### Required Models:
    - **Mistral 7B** (4.4GB) - General AI tasks and knowledge base operations
    - **Mistral Small 3.2** (7.2GB) - Enhanced proposal generation and complex analysis
    
    ### Optional Models:
    - **LLaVA** (4.5GB) - Image analysis and document processing
    - **Code Llama** (3.8GB) - Code generation and programming assistance
    """)
    
    # Installation options
    install_choice = st.radio(
        "Choose installation option:",
        ["1", "2", "3", "4"],
        format_func=lambda x: {
            "1": "Essential Only - Required models (~11.6GB)",
            "2": "Recommended - Include vision model (~16.1GB)",
            "3": "Complete - Install all models (~19.9GB)",
            "4": "Custom - Choose specific models"
        }[x]
    )
    
    custom_models = None
    if install_choice == "4":
        st.markdown("### Custom Model Selection")
        models = st.multiselect(
            "Select models to install:",
            ["1", "2", "3", "4"],
            default=["1", "2"],
            format_func=lambda x: {
                "1": "✅ Mistral 7B (4.4GB) - Required",
                "2": "✅ Mistral Small 3.2 (7.2GB) - Required",
                "3": "⭐ LLaVA (4.5GB) - Image analysis",
                "4": "⭐ Code Llama (3.8GB) - Code generation"
            }[x]
        )
        custom_models = ",".join(models)
    
    # Show estimated download times
    with st.expander("📊 Download Time Estimates", expanded=False):
        st.markdown("""
        **Internet Speed vs Download Time:**
        
        | Speed | Essential (~11.6GB) | Recommended (~16.1GB) | Complete (~19.9GB) |
        |-------|---------------------|----------------------|--------------------|
        | 10 Mbps | ~2.5 hours | ~3.5 hours | ~4.5 hours |
        | 50 Mbps | ~30 minutes | ~45 minutes | ~1 hour |
        | 100 Mbps | ~15 minutes | ~22 minutes | ~27 minutes |
        | 500 Mbps | ~3 minutes | ~4 minutes | ~5 minutes |
        
        **💡 Tip:** The setup wizard will show real-time progress and you can use Cortex Suite while models download in the background.
        """)
    
    st.warning("⏱️ Models download in background - you can use the app while they install!")
    
    # Initialize installation state
    if 'model_installation_active' not in st.session_state:
        st.session_state.model_installation_active = False
    if 'setup_logs' not in st.session_state:
        st.session_state.setup_logs = []
    if 'model_installation_started' not in st.session_state:
        st.session_state.model_installation_started = False
    
    # Check if we already have models installed (bypass installation if already done)
    try:
        from cortex_engine.utils.model_checker import model_checker
        available_models = model_checker.get_available_models()
        
        expected_models = []
        if install_choice == "Recommended":
            expected_models = ["mistral:latest", "mistral-small3.2", "llava:7b"]
        elif install_choice == "Complete":
            expected_models = ["mistral:latest", "mistral-small3.2", "llava:7b", "codellama"]
        elif install_choice == "Essential Only":
            expected_models = ["mistral:latest", "mistral-small3.2"]
        
        # If we have some models, show current status and option to continue
        if available_models and not st.session_state.model_installation_active:
            st.success(f"✅ Found {len(available_models)} existing models!")
            
            for model in available_models:
                st.success(f"✅ {model}")
            
            if len(available_models) >= 2:  # At least basic models
                st.info("🎯 You have enough models to proceed with setup. You can continue and download additional models later.")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("✅ Continue with Existing Models", type="primary", use_container_width=True):
                        return run_setup_step_with_logging(SetupStep.SYSTEM_VALIDATION)
            
            st.markdown("---")
            st.markdown("**Or install additional models:**")
    except Exception:
        pass  # Continue with normal flow
    
    # Show installation progress if active
    if st.session_state.model_installation_active:
        st.info("🔄 Model installation in progress...")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🔄 Reset Installation", help="Reset installation state if stuck"):
                st.session_state.model_installation_active = False
                st.session_state.setup_logs = []
                st.success("✅ Installation state reset")
                st.rerun()
        
        # Show expandable installation log
        with st.expander("📋 Installation Progress Log", expanded=True):
            log_container = st.container()
            with log_container:
                if st.session_state.setup_logs:
                    for log_entry in st.session_state.setup_logs[-20:]:  # Show last 20 entries
                        st.code(log_entry, language=None)
                else:
                    st.text("Initializing installation...")
                
                # Add helpful instructions
                st.markdown("---")
                st.markdown("""
                **💡 Installation Tips:**
                - Models download in the background via Ollama
                - You can use other Cortex Suite features while models install
                - Check the main dashboard for real-time model availability
                - Large models (LLaVA 7B ~4.7GB) may take several minutes
                """)
        
        # Check if we should show completion
        st.markdown("### 🔍 Current Model Status")
        try:
            from cortex_engine.utils.model_checker import model_checker
            from cortex_engine.utils.ollama_progress import ollama_progress_monitor
            
            available_models = model_checker.get_available_models()
            
            # Create detailed model status display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📦 Available Models:**")
                if available_models:
                    for model in available_models:
                        st.success(f"✅ {model}")
                else:
                    st.info("⏳ No models available yet...")
            
            with col2:
                st.markdown("**🔍 Expected Models:**")
                expected_models = []
                if install_choice == "Recommended":
                    expected_models = ["mistral:latest", "mistral-small3.2", "llava:7b"]
                elif install_choice == "Complete":
                    expected_models = ["mistral:latest", "mistral-small3.2", "llava:7b", "codellama"]
                elif install_choice == "Essential Only":
                    expected_models = ["mistral:latest", "mistral-small3.2"]
                
                for model in expected_models:
                    if any(model in available for available in available_models):
                        st.success(f"✅ {model}")
                    else:
                        st.warning(f"⏳ {model} (downloading...)")
            
            # Show progress summary
            if available_models:
                st.success(f"📊 **Status**: {len(available_models)}/{len(expected_models) if expected_models else '?'} models ready")
                
                # Check for visual models specifically
                visual_models = [m for m in available_models if 'llava' in m.lower() or 'moondream' in m.lower()]
                if visual_models:
                    st.success(f"👁️ **Visual Processing Ready**: {', '.join(visual_models)}")
                    # Add log entry only once
                    visual_log_entry = f"[{time.strftime('%H:%M:%S')}] ✅ Visual processing models detected: {', '.join(visual_models)}"
                    if visual_log_entry not in st.session_state.setup_logs:
                        st.session_state.setup_logs.append(visual_log_entry)
                
                # Check if installation is complete
                if len(available_models) >= len(expected_models):
                    st.success("🎉 **Installation Complete!** All expected models are ready.")
                    completion_log = f"[{time.strftime('%H:%M:%S')}] 🎉 Model installation completed successfully!"
                    if completion_log not in st.session_state.setup_logs:
                        st.session_state.setup_logs.append(completion_log)
                    
                    # Show continue button
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        if st.button("✅ Continue to Next Step", type="primary", use_container_width=True):
                            st.session_state.model_installation_active = False
                            return run_setup_step_with_logging(SetupStep.SYSTEM_VALIDATION)
            else:
                st.info("⏳ **Status**: Models are still downloading in the background...")
                
                # Add periodic progress updates
                current_time = time.strftime('%H:%M:%S')
                progress_update = f"[{current_time}] 📥 Download in progress - Ollama is fetching model files..."
                if len(st.session_state.setup_logs) < 10 or progress_update not in st.session_state.setup_logs[-5:]:
                    st.session_state.setup_logs.append(progress_update)
                
                # Add option to continue anyway after reasonable wait time
                st.markdown("---")
                st.warning("💡 **Having issues with downloads?** You can continue setup and models will download in the background.")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("⏭️ Continue Setup (Models will download in background)", help="Continue to next step - you can use basic features while models finish downloading"):
                        st.session_state.model_installation_active = False
                        st.session_state.setup_logs.append(f"[{time.strftime('%H:%M:%S')}] ⏭️ User chose to continue setup - models downloading in background")
                        return run_setup_step_with_logging(SetupStep.SYSTEM_VALIDATION)
                
                # Add manual model installer widget
                st.markdown("---")
                st.markdown("### 🛠️ Manual Model Installation")
                try:
                    from cortex_engine.utils.command_executor import display_model_installer_widget
                    display_model_installer_widget()
                except ImportError as e:
                    st.error(f"Command executor not available: {e}")
                    st.markdown("""
                    **Manual Installation (Advanced Users):**
                    
                    If you're comfortable with command line:
                    ```bash
                    # For Docker users:
                    docker exec -it cortex-suite ollama pull llava:7b
                    
                    # For direct installation:
                    ollama pull llava:7b
                    ```
                    """)
        
        except Exception as e:
            st.warning(f"⚠️ Cannot check model status: {e}")
            error_log = f"[{time.strftime('%H:%M:%S')}] ⚠️ Status check failed: {str(e)}"
            if error_log not in st.session_state.setup_logs:
                st.session_state.setup_logs.append(error_log)
        
        # Auto-refresh every 10 seconds instead of 2
        time.sleep(10)
        st.rerun()
    
    else:
        # Show installation button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 Start Installation", type="primary", use_container_width=True):
                user_input = {"install_models": install_choice}
                if custom_models:
                    user_input["custom_models"] = custom_models
                
                # Set installation active state
                st.session_state.model_installation_active = True
                st.session_state.setup_logs = []  # Clear previous logs
                st.session_state.setup_logs.append(f"[{time.strftime('%H:%M:%S')}] 🚀 Starting model installation: {install_choice}")
                
                if install_choice == "Recommended":
                    st.session_state.setup_logs.append(f"[{time.strftime('%H:%M:%S')}] 📦 Installing: Mistral 7B, Mistral Small 3.2, LLaVA 7B")
                elif install_choice == "Complete":
                    st.session_state.setup_logs.append(f"[{time.strftime('%H:%M:%S')}] 📦 Installing: All models including LLaVA 7B and Code Llama")
                elif install_choice == "Essential Only":
                    st.session_state.setup_logs.append(f"[{time.strftime('%H:%M:%S')}] 📦 Installing: Essential models only (Mistral 7B, Mistral Small 3.2)")
                else:
                    st.session_state.setup_logs.append(f"[{time.strftime('%H:%M:%S')}] 📦 Installing: Custom model selection")
                
                st.session_state.setup_logs.append(f"[{time.strftime('%H:%M:%S')}] ⏳ This may take several minutes depending on your internet speed...")
                
                # Trigger rerun to show progress
                st.rerun()
    
    return None

def run_setup_step_with_progress_tracking(step, user_input=None):
    """Run setup step with enhanced progress tracking for model installation."""
    if step == SetupStep.MODEL_INSTALLATION:
        # Create progress placeholders
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with status_placeholder.container():
            st.info("🔄 Initializing model installation...")
        
        # Start the installation in a way that allows progress updates
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Use a custom coroutine that can update progress
            async def installation_with_progress():
                result = await setup_manager.run_setup_step(step, user_input or {})
                return result
            
            result = loop.run_until_complete(installation_with_progress())
            loop.close()
            
            return result
            
        except Exception as e:
            with status_placeholder.container():
                st.error(f"Installation failed: {str(e)}")
            return None
    else:
        # Use regular setup step execution
        return run_setup_step_with_logging(step, user_input)

def display_step_system_validation(step_result):
    """Display system validation step with comprehensive model verification."""
    st.markdown("## ✨ System Validation")
    
    # Run validation if not already done
    if not step_result or step_result.status == SetupStatus.PENDING:
        with st.spinner("🔍 Running comprehensive system validation..."):
            return run_setup_step_with_logging(SetupStep.SYSTEM_VALIDATION)
    
    if step_result and step_result.details:
        validation = step_result.details
        
        # Create validation summary
        validation_items = []
        
        # Backend status
        if "backends" in validation:
            backends = validation["backends"]
            if backends["status"] == "✅":
                st.success(f"✅ **Model Backends**: {', '.join(backends['available'])} ({backends['count']} available)")
                validation_items.append(("✅", "Model Backends", "Available and responsive"))
            else:
                st.error("❌ **Model Backends**: No backends available")
                validation_items.append(("❌", "Model Backends", "No backends found"))
        
        # Model status with detailed verification
        if "models" in validation:
            models = validation["models"]
            if models["status"] == "✅":
                st.success(f"✅ **AI Models**: {models['available_count']} models installed and verified")
                validation_items.append(("✅", "AI Models", f"{models['available_count']} models ready"))
                
                # Show detailed model list
                with st.expander("📊 View Installed Models", expanded=False):
                    model_data = []
                    for model in models["models"]:
                        # Parse model info if available
                        parts = model.split(':')
                        model_name = parts[0] if parts else model
                        model_version = parts[1] if len(parts) > 1 else "latest"
                        
                        # Estimate model size (rough estimates)
                        size_estimates = {
                            "mistral": "~4.4GB",
                            "mistral-small3.2": "~7.2GB", 
                            "llava": "~4.5GB",
                            "codellama": "~3.8GB"
                        }
                        estimated_size = size_estimates.get(model_name.lower(), "~3-8GB")
                        
                        model_data.append({
                            "Model": model_name,
                            "Version": model_version,
                            "Size": estimated_size,
                            "Status": "✅ Ready"
                        })
                    
                    if model_data:
                        import pandas as pd
                        df = pd.DataFrame(model_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.write("No detailed model information available")
            else:
                st.error("❌ **AI Models**: No models available")
                validation_items.append(("❌", "AI Models", "No models found"))
        
        # Inference test with details
        if "inference" in validation:
            inference = validation["inference"]
            if inference["status"] == "✅":
                st.success(f"✅ **Model Inference**: Successfully tested with {inference['test_model']}")
                validation_items.append(("✅", "Model Inference", f"Test passed with {inference['test_model']}"))
                
                # Show test details
                with st.expander("🧪 Inference Test Details", expanded=False):
                    st.info(f"**Test Model**: {inference['test_model']}")
                    st.info(f"**Test Input**: 'Hello, test!'")
                    if 'test_output' in inference:
                        st.success(f"**Test Output**: {inference['test_output']}")
                    st.success("✅ Model can generate responses correctly")
            else:
                st.error(f"❌ **Model Inference**: Failed test with {inference.get('test_model', 'unknown model')}")
                validation_items.append(("❌", "Model Inference", "Test failed"))
        
        # System status overview
        if "system_status" in validation:
            system_info = validation["system_status"]
            with st.expander("📊 System Status Overview", expanded=False):
                if isinstance(system_info, dict):
                    st.json(system_info)
                else:
                    st.write(str(system_info))
        
        # Validation summary table
        st.markdown("### 📊 Validation Summary")
        
        if validation_items:
            import pandas as pd
            df = pd.DataFrame(validation_items, columns=["Status", "Component", "Details"])
            st.dataframe(df, use_container_width=True)
        
        # Overall status
        all_passed = all(item[0] == "✅" for item in validation_items)
        
        if all_passed:
            st.success("🎉 **All validation checks passed!** Your Cortex Suite is ready for use.")
        else:
            st.warning("⚠️ **Some validation checks failed.** You can still use Cortex Suite, but some features may be limited.")
            
            # Provide troubleshooting suggestions
            with st.expander("🔧 Troubleshooting Suggestions", expanded=True):
                st.markdown("""
                **If models are missing:**
                1. Check internet connectivity
                2. Ensure sufficient disk space (10GB+)
                3. Wait for background downloads to complete
                4. Try restarting the setup wizard
                
                **If inference fails:**
                1. Restart Docker Desktop
                2. Check Docker logs: `docker logs cortex-suite`
                3. Verify models downloaded completely
                4. Try switching model distribution strategy
                """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🏁 Complete Setup", type="primary", use_container_width=True):
            return run_setup_step_with_logging(SetupStep.COMPLETE)
    
    return None

def display_step_complete():
    """Display setup complete step."""
    st.markdown("## 🎉 Setup Complete!")
    
    st.balloons()
    
    st.markdown("""
    ### 🚀 Your Cortex Suite is Ready!
    
    **What You Can Do Now:**
    1. 📚 **Knowledge Ingest** - Upload and process documents
    2. 🔍 **Knowledge Search** - Find information across your documents  
    3. 🤖 **AI Research** - Automated research with multiple AI agents
    4. 📝 **Proposal Generation** - Create professional proposals
    5. 💡 **Idea Generator** - Structured innovation methodology
    6. 📊 **Analytics** - Visualize knowledge relationships
    
    ### 🎯 Quick Start Tips:
    - Upload a few PDF documents to build your knowledge base
    - Try the Knowledge Search to test retrieval capabilities
    - Generate a proposal to see AI-assisted writing in action
    - Use the System Status sidebar for monitoring
    """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📚 Start with Knowledge Ingest", use_container_width=True):
            st.switch_page("pages/2_Knowledge_Ingest.py")
    
    with col2:
        if st.button("🤖 Try AI Research", use_container_width=True):
            st.switch_page("pages/1_AI_Assisted_Research.py")
    
    with col3:
        if st.button("📝 Generate Proposal", use_container_width=True):
            st.switch_page("pages/6_Proposal_Step_2_Make.py")

def display_debug_log(logs: List[str]):
    """Display debug log window."""
    if logs:
        with st.expander("🐛 Debug Log", expanded=False):
            log_container = st.container()
            with log_container:
                for log in logs[-50:]:  # Show last 50 log entries
                    st.code(log, language=None)

def run_setup_step_with_logging(step, user_input=None):
    """Run a setup step asynchronously with real-time logging."""
    # Initialize session state for logging
    if 'setup_logs' not in st.session_state:
        st.session_state.setup_logs = []
    if 'current_operation' not in st.session_state:
        st.session_state.current_operation = None
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    log_placeholder = st.empty()
    
    def add_log(message: str):
        """Add a log message to the session state."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        st.session_state.setup_logs.append(log_entry)
        logger.info(message)
    
    try:
        add_log(f"Starting setup step: {step.value}")
        
        with status_placeholder.container():
            st.info(f"🔄 Running step: {step.value.replace('_', ' ').title()}...")
        
        # Show current logs
        with log_placeholder.container():
            display_debug_log(st.session_state.setup_logs)
        
        # Run the async function
        add_log("Initializing async event loop...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        add_log("Executing setup step...")
        result = loop.run_until_complete(setup_manager.run_setup_step(step, user_input or {}))
        
        loop.close()
        add_log(f"Setup step completed: {result.status.value if result else 'failed'}")
        
        # Update UI with result
        if result:
            if result.status == SetupStatus.COMPLETED:
                with status_placeholder.container():
                    st.success(f"✅ {step.value.replace('_', ' ').title()} completed successfully!")
            elif result.status == SetupStatus.FAILED:
                with status_placeholder.container():
                    st.error(f"❌ {step.value.replace('_', ' ').title()} failed: {result.message}")
                add_log(f"Error details: {result.details}")
            else:
                with status_placeholder.container():
                    st.warning(f"⚠️ {step.value.replace('_', ' ').title()}: {result.message}")
        
        # Update logs display
        with log_placeholder.container():
            display_debug_log(st.session_state.setup_logs)
        
        return result
        
    except Exception as e:
        error_msg = f"Setup step failed: {str(e)}"
        add_log(error_msg)
        
        with status_placeholder.container():
            st.error(error_msg)
        
        with log_placeholder.container():
            display_debug_log(st.session_state.setup_logs)
        
        logger.error(f"Setup step {step} failed: {e}")
        return None

def run_setup_step(step, user_input=None):
    """Legacy wrapper for compatibility."""
    return run_setup_step_with_logging(step, user_input)

def main():
    """Main setup wizard interface with improved state management."""
    
    # Initialize session state
    if 'setup_step_completed' not in st.session_state:
        st.session_state.setup_step_completed = False
    if 'last_step_result' not in st.session_state:
        st.session_state.last_step_result = None
    if 'show_step_result' not in st.session_state:
        st.session_state.show_step_result = True
    
    # Check if setup is already complete
    if setup_manager.is_setup_complete():
        st.success("✅ Setup has already been completed!")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 Run Setup Again", use_container_width=True):
                setup_manager.reset_setup()
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key.startswith('setup_'):
                        del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("📊 View System Status", use_container_width=True):
                # Show current system status
                try:
                    from cortex_engine.system_status import system_status
                    status = system_status.get_setup_progress()
                    st.json(status)
                except Exception as e:
                    st.error(f"Failed to get system status: {e}")
        
        with col3:
            if st.button("🚀 Go to Main App", use_container_width=True):
                st.switch_page("Cortex_Suite.py")
        
        return
    
    # Get current setup progress
    progress = setup_manager.get_setup_progress()
    
    # Display progress bar
    display_progress_bar(progress)
    
    # Display current step
    current_step = progress.current_step
    current_result = progress.step_results.get(current_step)
    
    # Handle step result display (prevent UI jumping)
    if current_result and current_result != st.session_state.last_step_result:
        st.session_state.last_step_result = current_result
        st.session_state.show_step_result = True
    
    # Show step result if available and should be shown
    if (st.session_state.show_step_result and st.session_state.last_step_result and 
        st.session_state.last_step_result.message):
        
        result = st.session_state.last_step_result
        
        with st.container():
            if result.status == SetupStatus.COMPLETED:
                st.success(result.message)
            elif result.status == SetupStatus.FAILED:
                st.error(result.message)
            elif result.status == SetupStatus.IN_PROGRESS:
                st.info(result.message)
            else:
                st.info(result.message)
    
    # Display current step interface
    st.markdown("---")
    
    if current_step == SetupStep.WELCOME:
        step_result = display_step_welcome()
    elif current_step == SetupStep.ENVIRONMENT_CHECK:
        step_result = display_step_environment_check(current_result)
    elif current_step == SetupStep.STORAGE_CONFIGURATION:
        step_result = display_step_storage_configuration(current_result)
    elif current_step == SetupStep.MODEL_STRATEGY:
        step_result = display_step_model_strategy(current_result)
    elif current_step == SetupStep.API_CONFIGURATION:
        step_result = display_step_api_configuration(current_result)
    elif current_step == SetupStep.MODEL_INSTALLATION:
        step_result = display_step_model_installation(current_result)
    elif current_step == SetupStep.SYSTEM_VALIDATION:
        step_result = display_step_system_validation(current_result)
    elif current_step == SetupStep.COMPLETE:
        step_result = display_step_complete()
    else:
        step_result = None
    
    # Handle step completion (trigger rerun only when necessary)
    if step_result and step_result != st.session_state.last_step_result:
        st.session_state.last_step_result = step_result
        st.session_state.show_step_result = True
        # Small delay to prevent rapid UI updates
        time.sleep(1)
        st.rerun()
    
    # Sidebar with help
    with st.sidebar:
        st.markdown("## 🆘 Setup Help")
        
        with st.expander("💡 Setup Tips"):
            st.markdown("""
            **Docker Issues:**
            - Ensure Docker Desktop is running
            - Check available disk space (10GB+)
            - Try: `docker system prune -f`
            
            **Model Download Issues:**
            - Check internet connectivity
            - Models are large (4-15GB each)
            - Download can take 10-30 minutes
            
            **API Key Issues:**
            - Ensure keys are correct format
            - Check API provider documentation
            - Keys can be added later in Settings
            """)
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Reset Setup:**
            Use this if setup gets stuck
            """)
            if st.button("🔄 Reset Setup"):
                setup_manager.reset_setup()
                st.success("Setup reset! Please refresh the page.")
                st.rerun()
        
        with st.expander("📋 System Info"):
            try:
                import platform
                import sys
                st.write(f"**OS:** {platform.system()} {platform.release()}")
                st.write(f"**Python:** {sys.version.split()[0]}")
                
                # Check Docker
                try:
                    import subprocess
                    result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.write(f"**Docker:** ✅ Available")
                    else:
                        st.write(f"**Docker:** ❌ Not available")
                except:
                    st.write(f"**Docker:** ❓ Unknown")
                    
            except Exception as e:
                st.write(f"Error getting system info: {e}")

if __name__ == "__main__":
    main()
