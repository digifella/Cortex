"""
Test script for Adaptive Model Manager
Displays available models, categorization, and recommendations
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cortex_engine.adaptive_model_manager import (
    AdaptiveModelManager,
    TaskType,
    ModelTier
)


async def test_model_manager():
    """Test the adaptive model manager with current Ollama models."""
    print("=" * 80)
    print("🧠 Adaptive Model Manager Test")
    print("=" * 80)
    print()

    # Initialize manager
    manager = AdaptiveModelManager()

    # Check system capabilities
    print("📊 System Summary:")
    print("-" * 80)
    summary = manager.get_system_summary()
    print(f"System Memory: {summary['system_memory_gb']:.1f}GB")
    print(f"Available Memory: {summary['available_memory_gb']:.1f}GB")
    print(f"NVIDIA GPU: {'Yes ✓' if summary['has_nvidia_gpu'] else 'No'}")
    if summary['has_nvidia_gpu'] and summary['gpu_info']:
        gpu = summary['gpu_info']
        print(f"GPU: {gpu.get('device_name', 'Unknown')}")
        if 'memory_total_gb' in gpu:
            print(f"GPU Memory: {gpu['memory_total_gb']:.1f}GB")
    print()

    # Discover models
    print("🔍 Discovering Available Models:")
    print("-" * 80)
    models = await manager.discover_models()

    if not models:
        print("❌ No models found - is Ollama running?")
        print("\nStart Ollama with: ollama serve")
        return

    print(f"Found {len(models)} models\n")

    # Categorize by tier
    print("📁 Models by Tier:")
    print("-" * 80)
    categorized = await manager.categorize_models()

    for tier in [ModelTier.FAST, ModelTier.MID, ModelTier.POWER]:
        tier_models = categorized[tier]
        print(f"\n{tier.value.upper()} ({len(tier_models)} models):")
        for model in tier_models:
            icon = "🚀" if tier == ModelTier.FAST else "⚡" if tier == ModelTier.MID else "💪"
            nvidia_badge = " [NVIDIA-optimized]" if model.nvidia_optimized else ""
            vision_badge = " [Vision]" if model.is_vision else ""
            print(f"  {icon} {model.full_name:45s} {model.size_gb:6.1f}GB  "
                  f"{model.family:10s}{nvidia_badge}{vision_badge}")
            if model.recommended_tasks:
                tasks = ", ".join([t.value for t in model.recommended_tasks[:3]])
                print(f"     └─ Best for: {tasks}")

    print()
    print("=" * 80)

    # Get recommendations for each task type
    print("\n💡 Model Recommendations by Task:")
    print("-" * 80)

    for task_type in TaskType:
        print(f"\n{task_type.value.upper()}:")

        for pref in ["fastest", "balanced", "best"]:
            model_name = await manager.recommend_model(task_type, pref)
            if model_name:
                model_info = await manager.get_model_info(model_name)
                pref_icon = "⚡" if pref == "fastest" else "⚖️" if pref == "balanced" else "🏆"
                print(f"  {pref_icon} {pref:10s}: {model_name:45s} "
                      f"({model_info.size_gb:.1f}GB)")
            else:
                print(f"  {pref:10s}: No suitable model found")

    print()
    print("=" * 80)

    # Show what models are needed for optimal operation
    print("\n🎯 Recommended Models to Pull (for optimal performance):")
    print("-" * 80)

    current_model_names = {m.name for m in models}

    recommendations = {
        "Fast Router/Classifier": [
            ("llama3.2:3b-instruct-q8_0", "Already have ✓" if "llama3.2" in current_model_names else "Recommended"),
            ("qwen2.5:3b-instruct-q8_0", "Already have ✓" if "qwen2.5" in current_model_names and any(m.params_estimate == 3 for m in models if m.family == "qwen") else "New - excellent reasoning"),
        ],
        "Mid-Range Analysis": [
            ("qwen2.5:14b-instruct-q4_K_M", "Already have ✓" if any(m.family == "qwen" and m.params_estimate == 14 for m in models) else "New - great balance"),
            ("mistral-small:latest", "Already have ✓" if "mistral-small" in current_model_names else "Consider"),
        ],
        "Power Models (Research/Ideation)": [
            ("llama3.3:70b-instruct-q4_K_M", "Already have ✓" if any(m.name == "llama3.3" and m.params_estimate == 70 for m in models) else "New - latest Llama"),
            ("qwen2.5:72b-instruct-q4_K_M", "Already have ✓" if any(m.family == "qwen" and m.params_estimate == 72 for m in models) else "New - SOTA reasoning"),
            ("nemotron:70b-instruct-q4_K_M", "Already have ✓" if "nemotron" in current_model_names else "New - NVIDIA optimized"),
        ]
    }

    for category, model_list in recommendations.items():
        print(f"\n{category}:")
        for model_name, status in model_list:
            icon = "✓" if "Already have" in status else "📥"
            print(f"  {icon} {model_name:45s} {status}")

    print()
    print("=" * 80)
    print("\n🚀 To pull a new model, run:")
    print("   ollama pull <model_name>")
    print("\nExample:")
    print("   ollama pull llama3.3:70b-instruct-q4_K_M")
    print()


if __name__ == "__main__":
    asyncio.run(test_model_manager())
