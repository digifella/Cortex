#!/usr/bin/env python3
"""
Ollama Progress Checker
Quick utility to check model download progress and status
"""

import subprocess
import sys
import json
import time
from pathlib import Path

def check_ollama_status():
    """Check if Ollama is running"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

def get_available_models():
    """Get list of available models"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        if len(parts) >= 3:
                            size = parts[2]
                            models.append({"name": model_name, "size": size})
                        else:
                            models.append({"name": model_name, "size": "Unknown"})
            return models
    except Exception as e:
        print(f"Error getting models: {e}")
    return []

def check_model_downloading():
    """Check if any models are currently downloading"""
    try:
        # Check if any ollama pull processes are running
        result = subprocess.run(["pgrep", "-f", "ollama pull"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def main():
    print("🔍 Cortex Suite - Ollama Progress Checker")
    print("=" * 50)
    
    # Check Ollama status
    print("\n📊 Ollama Service Status:")
    if check_ollama_status():
        print("✅ Ollama is running")
    else:
        print("❌ Ollama is not running or not accessible")
        return 1
    
    # Check available models
    print("\n📦 Available Models:")
    models = get_available_models()
    if models:
        for model in models:
            print(f"✅ {model['name']} ({model['size']})")
    else:
        print("⏳ No models available yet")
    
    # Check for expected visual processing models
    print("\n👁️ Visual Processing Models:")
    visual_models = [m for m in models if 'llava' in m['name'].lower() or 'moondream' in m['name'].lower()]
    if visual_models:
        for model in visual_models:
            print(f"✅ {model['name']} ({model['size']}) - Visual processing ready!")
    else:
        print("⏳ No visual models detected yet")
        
        # Check if downloading
        if check_model_downloading():
            print("📥 Models are currently downloading...")
        else:
            print("ℹ️ No download activity detected")
    
    # Check for required models
    print("\n🎯 Required Models Check:")
    required_models = ["mistral:7b-instruct-v0.3-q4_K_M", "mistral-small3.2", "llava:7b"]
    
    for required in required_models:
        found = False
        for available in models:
            if required in available['name'] or available['name'].startswith(required.split(':')[0]):
                print(f"✅ {required} - Available as {available['name']}")
                found = True
                break
        
        if not found:
            print(f"⏳ {required} - Not available yet")
    
    # Summary
    print(f"\n📈 Summary:")
    print(f"Total models available: {len(models)}")
    print(f"Visual models ready: {len(visual_models)}")
    print(f"Expected models: {len(required_models)}")
    
    if len(models) >= len(required_models):
        print("🎉 Installation appears complete!")
    else:
        print("⏳ Installation still in progress...")
        if check_model_downloading():
            print("📥 Downloads are active")
        else:
            print("ℹ️ No active downloads detected")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())