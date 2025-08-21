"""
Setup Manager for Cortex Suite
Guides users through initial setup, API configuration, and system validation.
"""

import os
import json
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import requests
import re

from .model_services import HybridModelManager, DistributionStrategy
from .utils.logging_utils import get_logger
from .config import get_cortex_config
from .exceptions import ConfigurationError

logger = get_logger(__name__)


class SetupStep(Enum):
    """Setup steps in order."""
    WELCOME = "welcome"
    ENVIRONMENT_CHECK = "environment_check"
    MODEL_STRATEGY = "model_strategy"
    API_CONFIGURATION = "api_configuration"
    MODEL_INSTALLATION = "model_installation"
    SYSTEM_VALIDATION = "system_validation"
    COMPLETE = "complete"


class SetupStatus(Enum):
    """Status of setup steps."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class APIProvider:
    """Configuration for an API provider."""
    name: str
    display_name: str
    required: bool
    env_var: str
    description: str
    setup_url: str
    validation_endpoint: Optional[str] = None
    cost_info: str = ""
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []


@dataclass
class SetupStepResult:
    """Result of a setup step."""
    step: SetupStep
    status: SetupStatus
    message: str
    details: Dict[str, Any] = None
    next_step: Optional[SetupStep] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class SetupProgress:
    """Overall setup progress."""
    current_step: SetupStep
    completed_steps: List[SetupStep]
    step_results: Dict[SetupStep, SetupStepResult]
    overall_progress_percent: float
    configuration: Dict[str, Any]
    
    def __post_init__(self):
        if not hasattr(self, 'step_results'):
            self.step_results = {}
        if not hasattr(self, 'configuration'):
            self.configuration = {}


class SetupManager:
    """Manages the initial setup process for Cortex Suite."""
    
    def __init__(self):
        self.api_providers = self._get_api_providers()
        self.setup_file = Path.home() / ".cortex" / "setup_progress.json"
        self.config_file = Path.home() / ".cortex" / "config.json"
        self._progress: Optional[SetupProgress] = None
        
        # Ensure setup directory exists
        self.setup_file.parent.mkdir(exist_ok=True)
    
    def _get_api_providers(self) -> List[APIProvider]:
        """Define all supported API providers."""
        return [
            APIProvider(
                name="openai",
                display_name="OpenAI",
                required=False,
                env_var="OPENAI_API_KEY",
                description="Access to GPT models for advanced research and analysis",
                setup_url="https://platform.openai.com/api-keys",
                validation_endpoint="https://api.openai.com/v1/models",
                cost_info="Pay-per-use: ~$0.03 per 1K tokens",
                features=["GPT-4", "GPT-3.5", "Advanced reasoning", "High-quality generation"]
            ),
            APIProvider(
                name="gemini",
                display_name="Google Gemini",
                required=False,
                env_var="GEMINI_API_KEY",
                description="Google's advanced AI model for research and content generation",
                setup_url="https://makersuite.google.com/app/apikey",
                validation_endpoint="https://generativelanguage.googleapis.com/v1beta/models",
                cost_info="Generous free tier: 60 requests/minute",
                features=["Gemini Pro", "Long context", "Multimodal", "Free tier available"]
            ),
            APIProvider(
                name="youtube",
                display_name="YouTube Data API",
                required=False,
                env_var="YOUTUBE_API_KEY",
                description="Research YouTube content and extract transcripts",
                setup_url="https://console.developers.google.com/apis/api/youtube.googleapis.com",
                cost_info="Free: 10,000 units/day (typical usage: 100-500 units/search)",
                features=["Video search", "Transcript extraction", "Channel analysis"]
            ),
            APIProvider(
                name="anthropic",
                display_name="Anthropic Claude",
                required=False,
                env_var="ANTHROPIC_API_KEY",
                description="Claude models for advanced reasoning and analysis",
                setup_url="https://console.anthropic.com/",
                validation_endpoint="https://api.anthropic.com/v1/messages",
                cost_info="Pay-per-use: ~$0.25 per 1K tokens",
                features=["Claude 3", "Long context", "Advanced reasoning", "Safety focused"]
            )
        ]
    
    def get_setup_progress(self) -> SetupProgress:
        """Get current setup progress."""
        if self._progress is None:
            self._load_progress()
        return self._progress
    
    def _load_progress(self):
        """Load setup progress from file."""
        if self.setup_file.exists():
            try:
                with open(self.setup_file, 'r') as f:
                    data = json.load(f)
                
                # Convert step enums
                current_step = SetupStep(data.get("current_step", "welcome"))
                completed_steps = [SetupStep(step) for step in data.get("completed_steps", [])]
                
                # Convert step results
                step_results = {}
                for step_name, result_data in data.get("step_results", {}).items():
                    step_results[SetupStep(step_name)] = SetupStepResult(
                        step=SetupStep(step_name),
                        status=SetupStatus(result_data["status"]),
                        message=result_data["message"],
                        details=result_data.get("details", {}),
                        next_step=SetupStep(result_data["next_step"]) if result_data.get("next_step") else None
                    )
                
                self._progress = SetupProgress(
                    current_step=current_step,
                    completed_steps=completed_steps,
                    step_results=step_results,
                    overall_progress_percent=data.get("overall_progress_percent", 0.0),
                    configuration=data.get("configuration", {})
                )
            except Exception as e:
                logger.warning(f"Failed to load setup progress: {e}")
                self._init_new_progress()
        else:
            self._init_new_progress()
    
    def _init_new_progress(self):
        """Initialize new setup progress."""
        self._progress = SetupProgress(
            current_step=SetupStep.WELCOME,
            completed_steps=[],
            step_results={},
            overall_progress_percent=0.0,
            configuration={}
        )
    
    def _save_progress(self):
        """Save setup progress to file."""
        try:
            # Convert to serializable format
            data = {
                "current_step": self._progress.current_step.value,
                "completed_steps": [step.value for step in self._progress.completed_steps],
                "step_results": {},
                "overall_progress_percent": self._progress.overall_progress_percent,
                "configuration": self._progress.configuration
            }
            
            # Convert step results
            for step, result in self._progress.step_results.items():
                data["step_results"][step.value] = {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "next_step": result.next_step.value if result.next_step else None
                }
            
            with open(self.setup_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save setup progress: {e}")
    
    def _update_progress(self, step: SetupStep, result: SetupStepResult):
        """Update setup progress."""
        self._progress.step_results[step] = result
        
        if result.status == SetupStatus.COMPLETED and step not in self._progress.completed_steps:
            self._progress.completed_steps.append(step)
        
        if result.next_step:
            self._progress.current_step = result.next_step
        
        # Calculate overall progress
        total_steps = len(SetupStep)
        completed_count = len(self._progress.completed_steps)
        self._progress.overall_progress_percent = (completed_count / total_steps) * 100
        
        self._save_progress()
    
    async def run_setup_step(self, step: SetupStep, user_input: Dict[str, Any] = None) -> SetupStepResult:
        """Run a specific setup step."""
        if user_input is None:
            user_input = {}
        
        try:
            if step == SetupStep.WELCOME:
                return await self._step_welcome()
            elif step == SetupStep.ENVIRONMENT_CHECK:
                return await self._step_environment_check()
            elif step == SetupStep.MODEL_STRATEGY:
                return await self._step_model_strategy(user_input)
            elif step == SetupStep.API_CONFIGURATION:
                return await self._step_api_configuration(user_input)
            elif step == SetupStep.MODEL_INSTALLATION:
                return await self._step_model_installation(user_input)
            elif step == SetupStep.SYSTEM_VALIDATION:
                return await self._step_system_validation()
            elif step == SetupStep.COMPLETE:
                return await self._step_complete()
            else:
                raise ValueError(f"Unknown setup step: {step}")
                
        except Exception as e:
            logger.error(f"Setup step {step} failed: {e}")
            result = SetupStepResult(
                step=step,
                status=SetupStatus.FAILED,
                message=f"Step failed: {str(e)}",
                details={"error": str(e)}
            )
            self._update_progress(step, result)
            return result
    
    async def _step_welcome(self) -> SetupStepResult:
        """Welcome step - introduce the setup process."""
        message = """
🚀 **Welcome to Cortex Suite Setup!**

This guided setup will help you:
1. ✅ Check your system environment
2. 🎯 Choose your AI model distribution strategy  
3. 🔑 Configure API keys for enhanced features
4. 📦 Install required AI models
5. ✨ Validate everything is working

The process takes 5-15 minutes depending on your choices and internet speed.

**Local vs Cloud AI:**
- **Local AI**: Complete privacy, no API costs, works offline
- **Cloud AI**: Latest models, faster processing, requires internet & API keys

You can use Cortex Suite entirely locally or enhance it with cloud APIs for advanced features.
        """.strip()
        
        result = SetupStepResult(
            step=SetupStep.WELCOME,
            status=SetupStatus.COMPLETED,
            message=message,
            next_step=SetupStep.ENVIRONMENT_CHECK
        )
        
        self._update_progress(SetupStep.WELCOME, result)
        return result
    
    async def _step_environment_check(self) -> SetupStepResult:
        """Check system environment and dependencies."""
        checks = {
            "docker": "Docker service",
            "python": "Python 3.11+",
            "disk_space": "Available disk space (10GB+)",
            "memory": "Available memory (4GB+)",
            "network": "Internet connectivity"
        }
        
        results = {}
        all_passed = True
        
        # Check Docker
        try:
            import subprocess
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            results["docker"] = {"status": "✅", "message": "Docker available"}
        except Exception:
            results["docker"] = {"status": "❌", "message": "Docker not found or not running"}
            all_passed = False
        
        # Check Python version
        import sys
        if sys.version_info >= (3, 11):
            results["python"] = {"status": "✅", "message": f"Python {sys.version.split()[0]}"}
        else:
            results["python"] = {"status": "⚠️", "message": f"Python {sys.version.split()[0]} (3.11+ recommended)"}
        
        # Check disk space
        try:
            import shutil
            free_space_gb = shutil.disk_usage(".").free / (1024**3)
            if free_space_gb >= 10:
                results["disk_space"] = {"status": "✅", "message": f"{free_space_gb:.1f}GB available"}
            else:
                results["disk_space"] = {"status": "⚠️", "message": f"{free_space_gb:.1f}GB available (10GB+ recommended)"}
        except Exception:
            results["disk_space"] = {"status": "❓", "message": "Could not check disk space"}
        
        # Check memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().available / (1024**3)
            if memory_gb >= 4:
                results["memory"] = {"status": "✅", "message": f"{memory_gb:.1f}GB available"}
            else:
                results["memory"] = {"status": "⚠️", "message": f"{memory_gb:.1f}GB available (4GB+ recommended)"}
        except Exception:
            results["memory"] = {"status": "❓", "message": "Could not check memory"}
        
        # Check network
        try:
            requests.get("https://www.google.com", timeout=5)
            results["network"] = {"status": "✅", "message": "Internet connectivity OK"}
        except Exception:
            results["network"] = {"status": "⚠️", "message": "Limited internet connectivity"}
        
        status = SetupStatus.COMPLETED if all_passed else SetupStatus.COMPLETED
        message = "**System Environment Check:**\n\n" + "\n".join([
            f"- {checks[key]}: {result['status']} {result['message']}"
            for key, result in results.items()
        ])
        
        if not all_passed:
            message += "\n\n⚠️ Some checks failed. You can continue, but may experience issues."
        
        result = SetupStepResult(
            step=SetupStep.ENVIRONMENT_CHECK,
            status=status,
            message=message,
            details={"checks": results},
            next_step=SetupStep.MODEL_STRATEGY
        )
        
        self._update_progress(SetupStep.ENVIRONMENT_CHECK, result)
        return result
    
    async def _step_model_strategy(self, user_input: Dict[str, Any]) -> SetupStepResult:
        """Choose model distribution strategy."""
        if "strategy" not in user_input:
            # Present options
            message = """
🎯 **Choose Your AI Model Strategy:**

**1. Hybrid (Docker + Ollama) - Recommended**
   - ✅ Best performance for enterprise use
   - ✅ Automatic fallback and compatibility
   - ✅ Future-proof with latest Docker AI features
   - 📊 15% faster inference, better GPU utilization

**2. Docker Model Runner Only**
   - ✅ Enterprise-grade OCI distribution
   - ✅ Better CI/CD integration
   - ✅ Host-native performance
   - ⚠️ Requires Docker Model Runner support

**3. Ollama Only**
   - ✅ Simple and reliable
   - ✅ Large community and model library
   - ✅ Proven stability
   - ⚠️ Standard performance tier

**4. Auto-Optimal**
   - ✅ System automatically chooses best option
   - ✅ Environment-aware selection
   - ✅ Zero configuration needed

Please select your preferred strategy (1-4):
            """.strip()
            
            return SetupStepResult(
                step=SetupStep.MODEL_STRATEGY,
                status=SetupStatus.IN_PROGRESS,
                message=message,
                details={"awaiting_input": "strategy"}
            )
        
        # Process strategy choice
        strategy_map = {
            "1": DistributionStrategy.HYBRID_DOCKER_PREFERRED,
            "2": DistributionStrategy.DOCKER_ONLY,
            "3": DistributionStrategy.OLLAMA_ONLY,
            "4": DistributionStrategy.AUTO_OPTIMAL
        }
        
        choice = user_input.get("strategy", "4")
        strategy = strategy_map.get(choice, DistributionStrategy.AUTO_OPTIMAL)
        
        # Save strategy to configuration
        self._progress.configuration["model_distribution_strategy"] = strategy.value
        
        strategy_names = {
            DistributionStrategy.HYBRID_DOCKER_PREFERRED: "Hybrid (Docker + Ollama)",
            DistributionStrategy.DOCKER_ONLY: "Docker Model Runner Only",
            DistributionStrategy.OLLAMA_ONLY: "Ollama Only",
            DistributionStrategy.AUTO_OPTIMAL: "Auto-Optimal"
        }
        
        message = f"""
✅ **Strategy Selected: {strategy_names[strategy]}**

{strategy.value} will be used for AI model distribution.

This affects:
- How models are downloaded and stored
- Performance characteristics
- Enterprise feature availability
- Compatibility with different environments
        """.strip()
        
        result = SetupStepResult(
            step=SetupStep.MODEL_STRATEGY,
            status=SetupStatus.COMPLETED,
            message=message,
            details={"strategy": strategy.value},
            next_step=SetupStep.API_CONFIGURATION
        )
        
        self._update_progress(SetupStep.MODEL_STRATEGY, result)
        return result
    
    async def _step_api_configuration(self, user_input: Dict[str, Any]) -> SetupStepResult:
        """Configure API keys for enhanced features."""
        if "configure_apis" not in user_input:
            # Present API configuration options
            message = """
🔑 **API Configuration for Enhanced Features**

Cortex Suite works completely locally, but you can enhance it with cloud APIs:

**Cloud API Benefits:**
- 🚀 Access to latest AI models (GPT-4, Gemini Pro, Claude)
- ⚡ Faster processing for research tasks
- 🎯 Advanced reasoning capabilities
- 🌐 YouTube transcript extraction

**Available APIs:**
            """.strip()
            
            for provider in self.api_providers:
                message += f"""

**{provider.display_name}**
- Features: {', '.join(provider.features)}
- Cost: {provider.cost_info}
- Setup: {provider.setup_url}
                """.strip()
            
            message += """

**Options:**
1. **Local Only** - Skip API setup (you can add keys later)
2. **Configure APIs** - Set up one or more cloud providers
3. **Guided Setup** - Step-by-step API key configuration

Choose option (1-3):
            """.strip()
            
            return SetupStepResult(
                step=SetupStep.API_CONFIGURATION,
                status=SetupStatus.IN_PROGRESS,
                message=message,
                details={"awaiting_input": "configure_apis"}
            )
        
        choice = user_input.get("configure_apis", "1")
        
        if choice == "1":
            # Skip API configuration
            message = """
✅ **Local-Only Configuration Selected**

You've chosen to use Cortex Suite with local AI models only. This provides:
- ✅ Complete privacy - no data leaves your system
- ✅ No API costs
- ✅ Works offline
- ✅ Full functionality for document management and basic AI features

You can always add API keys later via the Settings page.
            """.strip()
            
            result = SetupStepResult(
                step=SetupStep.API_CONFIGURATION,
                status=SetupStatus.COMPLETED,
                message=message,
                next_step=SetupStep.MODEL_INSTALLATION
            )
            
        elif choice == "2" or choice == "3":
            # Configure APIs
            if "api_keys" not in user_input:
                # Present API key input form
                message = """
🔑 **API Key Configuration**

Enter your API keys below (leave blank to skip):

For each API you want to use:
1. Visit the setup URL
2. Create an account and generate an API key
3. Enter the key below

**Current Environment Variables:**
                """.strip()
                
                current_keys = {}
                for provider in self.api_providers:
                    current_value = os.getenv(provider.env_var)
                    if current_value:
                        current_keys[provider.name] = "***configured***"
                    else:
                        current_keys[provider.name] = "not configured"
                    
                    message += f"\n- {provider.display_name} ({provider.env_var}): {current_keys[provider.name]}"
                
                message += "\n\nEnter API keys as JSON: {\"openai\": \"sk-...\", \"gemini\": \"AI...\"}"
                
                return SetupStepResult(
                    step=SetupStep.API_CONFIGURATION,
                    status=SetupStatus.IN_PROGRESS,
                    message=message,
                    details={"awaiting_input": "api_keys", "providers": [asdict(p) for p in self.api_providers]}
                )
            
            # Process API keys
            api_keys = user_input.get("api_keys", {})
            configured_apis = []
            validation_results = {}
            
            for provider in self.api_providers:
                if provider.name in api_keys and api_keys[provider.name]:
                    key = api_keys[provider.name].strip()
                    if key:
                        # Validate the API key
                        is_valid, error = await self._validate_api_key(provider, key)
                        validation_results[provider.name] = {"valid": is_valid, "error": error}
                        
                        if is_valid:
                            # Save to environment/config
                            self._progress.configuration[provider.env_var] = key
                            configured_apis.append(provider.display_name)
                        else:
                            validation_results[provider.name]["message"] = f"❌ {provider.display_name}: {error}"
            
            if configured_apis:
                message = f"""
✅ **API Configuration Complete**

Configured APIs:
{chr(10).join(f"- ✅ {api}" for api in configured_apis)}

These APIs will enhance your Cortex Suite experience with advanced AI capabilities.
                """.strip()
            else:
                message = """
⚠️ **No APIs Configured**

No valid API keys were provided. You can:
- Continue with local-only setup
- Add API keys later via Settings
- Restart API configuration
                """.strip()
            
            result = SetupStepResult(
                step=SetupStep.API_CONFIGURATION,
                status=SetupStatus.COMPLETED,
                message=message,
                details={"configured_apis": configured_apis, "validation_results": validation_results},
                next_step=SetupStep.MODEL_INSTALLATION
            )
        
        else:
            # Invalid choice
            return SetupStepResult(
                step=SetupStep.API_CONFIGURATION,
                status=SetupStatus.IN_PROGRESS,
                message="Please choose 1, 2, or 3.",
                details={"awaiting_input": "configure_apis"}
            )
        
        self._update_progress(SetupStep.API_CONFIGURATION, result)
        return result
    
    async def _validate_api_key(self, provider: APIProvider, api_key: str) -> Tuple[bool, Optional[str]]:
        """Validate an API key."""
        if not provider.validation_endpoint:
            return True, None  # No validation endpoint available
        
        try:
            if provider.name == "openai":
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get(provider.validation_endpoint, headers=headers, timeout=10)
                return response.status_code == 200, None if response.status_code == 200 else f"HTTP {response.status_code}"
            
            elif provider.name == "gemini":
                # Gemini validation
                url = f"{provider.validation_endpoint}?key={api_key}"
                response = requests.get(url, timeout=10)
                return response.status_code == 200, None if response.status_code == 200 else f"HTTP {response.status_code}"
            
            elif provider.name == "anthropic":
                headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
                response = requests.get(provider.validation_endpoint, headers=headers, timeout=10)
                return response.status_code == 200, None if response.status_code == 200 else f"HTTP {response.status_code}"
            
            else:
                # Generic validation
                return True, None
                
        except Exception as e:
            return False, str(e)
    
    async def _step_model_installation(self, user_input: Dict[str, Any]) -> SetupStepResult:
        """Install required AI models."""
        strategy = self._progress.configuration.get("model_distribution_strategy", "auto_optimal")
        
        if "install_models" not in user_input:
            message = f"""
📦 **AI Model Installation**

**Selected Strategy:** {strategy}

**Required Models:**
- Mistral 7B (4.4GB) - General AI tasks and knowledge base operations
- Mistral Small 3.2 (7.2GB) - Enhanced proposal generation and complex analysis

**Optional Models:**
- LLaVA (4.5GB) - Image analysis and document processing
- Code Llama (3.8GB) - Code generation and programming assistance

**Installation Options:**
1. **Essential Only** - Install required models (~11.6GB)
2. **Recommended** - Include vision model (~16.1GB)  
3. **Complete** - Install all models (~19.9GB)
4. **Custom** - Choose specific models

Estimated download time: 10-30 minutes depending on internet speed.

Choose installation option (1-4):
            """.strip()
            
            return SetupStepResult(
                step=SetupStep.MODEL_INSTALLATION,
                status=SetupStatus.IN_PROGRESS,
                message=message,
                details={"awaiting_input": "install_models"}
            )
        
        # Process installation choice
        choice = user_input.get("install_models", "1")
        
        models_to_install = []
        if choice == "1":  # Essential only
            models_to_install = ["mistral:7b-instruct-v0.3-q4_K_M", "mistral-small3.2"]
        elif choice == "2":  # Recommended
            models_to_install = ["mistral:7b-instruct-v0.3-q4_K_M", "mistral-small3.2", "llava"]
        elif choice == "3":  # Complete
            models_to_install = ["mistral:7b-instruct-v0.3-q4_K_M", "mistral-small3.2", "llava", "codellama"]
        elif choice == "4":  # Custom
            if "custom_models" not in user_input:
                message = """
**Custom Model Selection**

Available models:
1. ✅ Mistral 7B (4.4GB) - Required
2. ✅ Mistral Small 3.2 (7.2GB) - Required  
3. ⭐ LLaVA (4.5GB) - Image analysis
4. ⭐ Code Llama (3.8GB) - Code generation

Enter model numbers to install (e.g., "1,2,3"): 
                """.strip()
                
                return SetupStepResult(
                    step=SetupStep.MODEL_INSTALLATION,
                    status=SetupStatus.IN_PROGRESS,
                    message=message,
                    details={"awaiting_input": "custom_models"}
                )
            
            # Process custom selection
            selections = user_input.get("custom_models", "1,2").split(",")
            model_map = {
                "1": "mistral:7b-instruct-v0.3-q4_K_M",
                "2": "mistral-small3.2", 
                "3": "llava",
                "4": "codellama"
            }
            models_to_install = [model_map[s.strip()] for s in selections if s.strip() in model_map]
        
        # Start model installation
        try:
            hybrid_manager = HybridModelManager(strategy=strategy)
            
            installation_progress = {
                "total_models": len(models_to_install),
                "completed_models": [],
                "failed_models": [],
                "current_model": None
            }
            
            for model_name in models_to_install:
                installation_progress["current_model"] = model_name
                
                # Check if already available
                if await hybrid_manager.is_model_available(model_name):
                    installation_progress["completed_models"].append(model_name)
                    continue
                
                # Install the model
                success = False
                async for progress in hybrid_manager.pull_model(model_name):
                    if "success" in progress.status.lower() or "completed" in progress.status.lower():
                        success = True
                        break
                
                if success:
                    installation_progress["completed_models"].append(model_name)
                else:
                    installation_progress["failed_models"].append(model_name)
            
            await hybrid_manager.close()
            
            if installation_progress["failed_models"]:
                message = f"""
⚠️ **Model Installation Partially Complete**

✅ Successfully installed: {len(installation_progress["completed_models"])} models
❌ Failed to install: {len(installation_progress["failed_models"])} models

Failed models: {', '.join(installation_progress["failed_models"])}

You can retry installation later or continue with available models.
                """.strip()
                status = SetupStatus.COMPLETED
            else:
                message = f"""
✅ **Model Installation Complete**

Successfully installed {len(installation_progress["completed_models"])} models:
{chr(10).join(f"- ✅ {model}" for model in installation_progress["completed_models"])}

Your AI models are ready for use!
                """.strip()
                status = SetupStatus.COMPLETED
            
            result = SetupStepResult(
                step=SetupStep.MODEL_INSTALLATION,
                status=status,
                message=message,
                details=installation_progress,
                next_step=SetupStep.SYSTEM_VALIDATION
            )
            
        except Exception as e:
            result = SetupStepResult(
                step=SetupStep.MODEL_INSTALLATION,
                status=SetupStatus.FAILED,
                message=f"Model installation failed: {str(e)}",
                details={"error": str(e)}
            )
        
        self._update_progress(SetupStep.MODEL_INSTALLATION, result)
        return result
    
    async def _step_system_validation(self) -> SetupStepResult:
        """Validate the entire system is working."""
        validation_results = {}
        all_passed = True
        
        try:
            # Test hybrid model manager
            strategy = self._progress.configuration.get("model_distribution_strategy", "auto_optimal")
            hybrid_manager = HybridModelManager(strategy=strategy)
            
            # Check backend availability
            backends = await hybrid_manager.get_available_backends()
            validation_results["backends"] = {
                "available": backends,
                "count": len(backends),
                "status": "✅" if backends else "❌"
            }
            
            if not backends:
                all_passed = False
            
            # Test model availability
            models = await hybrid_manager.list_all_available_models()
            validation_results["models"] = {
                "available_count": len(models),
                "models": [m.full_name for m in models],
                "status": "✅" if models else "❌"
            }
            
            if not models:
                all_passed = False
            
            # Test model inference
            if models:
                test_model = models[0].full_name
                inference_works = await hybrid_manager.test_model_inference(test_model, "Hello, test!")
                validation_results["inference"] = {
                    "test_model": test_model,
                    "works": inference_works,
                    "status": "✅" if inference_works else "❌"
                }
                
                if not inference_works:
                    all_passed = False
            
            # Get system status
            system_status = await hybrid_manager.get_system_status()
            validation_results["system_status"] = system_status
            
            await hybrid_manager.close()
            
        except Exception as e:
            validation_results["error"] = str(e)
            all_passed = False
        
        if all_passed:
            message = """
✅ **System Validation Complete - All Systems Ready!**

🎯 **Validation Results:**
- Model backends: ✅ Available and responsive
- AI models: ✅ Installed and working
- Model inference: ✅ Successfully tested
- System health: ✅ All components operational

🚀 **Your Cortex Suite is ready to use!**

**Next Steps:**
1. Start using the Knowledge Ingest page to add documents
2. Try the AI Assisted Research for information gathering
3. Explore Proposal Generation for creating professional documents
4. Check out the Idea Generator for innovation workflows

**Getting Started Tips:**
- Upload a few PDF documents to build your knowledge base
- Use the Knowledge Search to test retrieval capabilities
- Try generating a proposal to see AI-assisted writing in action
            """.strip()
            status = SetupStatus.COMPLETED
            next_step = SetupStep.COMPLETE
        else:
            message = f"""
⚠️ **System Validation Issues Detected**

Some components may not be working correctly:

{chr(10).join([f"- {key}: {result.get('status', '❓')}" for key, result in validation_results.items() if isinstance(result, dict)])}

**Recommendations:**
1. Check that Docker is running
2. Verify internet connectivity for model downloads
3. Restart the application
4. Review the system logs for detailed error information

You can still use Cortex Suite, but some features may be limited.
            """.strip()
            status = SetupStatus.COMPLETED  # Allow continuation even with issues
            next_step = SetupStep.COMPLETE
        
        result = SetupStepResult(
            step=SetupStep.SYSTEM_VALIDATION,
            status=status,
            message=message,
            details=validation_results,
            next_step=next_step
        )
        
        self._update_progress(SetupStep.SYSTEM_VALIDATION, result)
        return result
    
    async def _step_complete(self) -> SetupStepResult:
        """Complete the setup process."""
        # Save final configuration
        self._save_final_configuration()
        
        message = """
🎉 **Setup Complete - Welcome to Cortex Suite!**

Your AI-powered knowledge management system is ready.

**What You Can Do Now:**
1. 📚 **Knowledge Ingest** - Upload and process documents
2. 🔍 **Knowledge Search** - Find information across your documents
3. 🤖 **AI Research** - Automated research with multiple AI agents
4. 📝 **Proposal Generation** - Create professional proposals using your knowledge base
5. 💡 **Idea Generator** - Structured innovation using Double Diamond methodology
6. 📊 **Analytics** - Visualize knowledge relationships and themes

**Your Configuration:**
- Distribution Strategy: {strategy}
- Available Models: {model_count}
- API Integrations: {api_count}

**Support & Resources:**
- Documentation: Available in the sidebar Help system
- Model Management: Use the System Status for monitoring
- Settings: Update configuration in the Settings page

🚀 **Ready to transform your documents into intelligent knowledge!**
        """.strip().format(
            strategy=self._progress.configuration.get("model_distribution_strategy", "not configured"),
            model_count=len(self._progress.configuration.get("installed_models", [])),
            api_count=len([k for k in self._progress.configuration.keys() if k.endswith("_API_KEY")])
        )
        
        result = SetupStepResult(
            step=SetupStep.COMPLETE,
            status=SetupStatus.COMPLETED,
            message=message,
            details={"setup_completed": True}
        )
        
        self._update_progress(SetupStep.COMPLETE, result)
        return result
    
    def _save_final_configuration(self):
        """Save the final configuration to the system."""
        try:
            # Create .env file or update existing one
            env_file = Path(".env")
            env_updates = {}
            
            # Add model distribution strategy
            strategy = self._progress.configuration.get("model_distribution_strategy")
            if strategy:
                env_updates["MODEL_DISTRIBUTION_STRATEGY"] = strategy
            
            # Add API keys
            for key, value in self._progress.configuration.items():
                if key.endswith("_API_KEY") and value:
                    env_updates[key] = value
            
            # Update .env file
            if env_updates:
                self._update_env_file(env_file, env_updates)
            
            # Mark setup as complete
            self._progress.configuration["setup_completed"] = True
            self._progress.configuration["setup_completed_at"] = time.time()
            
            self._save_progress()
            
        except Exception as e:
            logger.error(f"Failed to save final configuration: {e}")
    
    def _update_env_file(self, env_file: Path, updates: Dict[str, str]):
        """Update .env file with new values."""
        existing_lines = []
        if env_file.exists():
            with open(env_file, 'r') as f:
                existing_lines = f.readlines()
        
        # Parse existing variables
        existing_vars = {}
        other_lines = []
        
        for line in existing_lines:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                existing_vars[key.strip()] = value.strip()
            else:
                other_lines.append(line)
        
        # Update with new values
        existing_vars.update(updates)
        
        # Write back to file
        with open(env_file, 'w') as f:
            # Write other lines (comments, etc.)
            for line in other_lines:
                f.write(line + '\n')
            
            # Write variables
            for key, value in existing_vars.items():
                f.write(f"{key}={value}\n")
    
    def is_setup_complete(self) -> bool:
        """Check if setup has been completed."""
        progress = self.get_setup_progress()
        return (SetupStep.COMPLETE in progress.completed_steps or 
                progress.configuration.get("setup_completed", False))
    
    def reset_setup(self):
        """Reset the setup process."""
        if self.setup_file.exists():
            self.setup_file.unlink()
        self._progress = None
        self._init_new_progress()


# Global instance
setup_manager = SetupManager()