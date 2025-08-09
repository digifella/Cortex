#!/usr/bin/env python3
"""
Docker Setup Validation Script for Cortex Suite

This script validates that the Docker setup is ready for cross-platform deployment
and checks all required components including Windows path compatibility.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple

def run_command(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr"""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except FileNotFoundError:
        return 1, "", f"Command not found: {cmd[0]}"

def check_docker_installation() -> bool:
    """Check if Docker is installed and running"""
    print("🐳 Checking Docker installation...")
    
    # Check if docker command exists
    exit_code, stdout, stderr = run_command(["docker", "--version"])
    if exit_code != 0:
        print("❌ Docker is not installed or not in PATH")
        print("   Install Docker Desktop from: https://www.docker.com/products/docker-desktop/")
        return False
    
    print(f"✅ Docker version: {stdout.strip()}")
    
    # Check if Docker daemon is running
    exit_code, stdout, stderr = run_command(["docker", "info"])
    if exit_code != 0:
        print("❌ Docker daemon is not running")
        print("   Please start Docker Desktop and try again")
        return False
    
    print("✅ Docker daemon is running")
    return True

def check_required_files() -> bool:
    """Check if all required files are present"""
    print("\n📁 Checking required files...")
    
    required_files = [
        "Dockerfile",
        "docker-compose.yml", 
        "run-cortex.bat",
        "run-cortex.sh",
        ".env.example",
        "../requirements.txt",
        "../Cortex_Suite.py",
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_present = False
    
    return all_present

def validate_windows_compatibility() -> bool:
    """Check Windows-specific compatibility"""
    print("\n🪟 Checking Windows compatibility...")
    
    # Check batch file
    batch_file = Path(__file__).parent / "run-cortex.bat"
    if not batch_file.exists():
        print("❌ run-cortex.bat missing")
        return False
    
    # Check file encoding and line endings
    try:
        import subprocess
        file_result = subprocess.run(['file', str(batch_file)], capture_output=True, text=True)
        if file_result.returncode == 0:
            file_output = file_result.stdout.lower()
            if 'ascii text' in file_output and 'crlf' in file_output:
                print("✅ Batch file has correct encoding (ASCII) and line endings (CRLF)")
            elif 'utf-8' in file_output or 'unicode' in file_output:
                print("❌ Batch file has wrong encoding (UTF-8/Unicode) - must be ASCII")
                return False
            elif 'crlf' not in file_output:
                print("❌ Batch file missing CRLF line endings - will fail on Windows")
                return False
        else:
            print("⚠️  Could not check file encoding (file command failed)")
    except Exception as e:
        print(f"⚠️  Could not check file encoding: {e}")
    
    # Check for non-ASCII characters
    try:
        with open(batch_file, 'rb') as f:
            batch_bytes = f.read()
        
        # Check for Unicode characters that would break ASCII
        has_unicode = any(b > 127 for b in batch_bytes if b != 0x0d and b != 0x0a)  # Allow CR/LF
        if has_unicode:
            print("❌ Batch file contains non-ASCII characters - will fail on some Windows systems")
            return False
        else:
            print("✅ Batch file contains only ASCII characters")
    
    except Exception as e:
        print(f"❌ Could not check batch file characters: {e}")
        return False
    
    # Check batch file content
    batch_content = batch_file.read_text()
    required_bat_features = [
        "docker info",  # Docker check
        "findstr",      # Windows string search
        "timeout",      # Windows wait command
        "pause",        # Windows pause command
        "%errorlevel%", # Windows error level
    ]
    
    # Check for problematic Unicode that was previously used
    problematic_chars = ["🚀", "❌", "✅", "⏳", "📦", "🔨", "•"]
    found_unicode = []
    for char in problematic_chars:
        if char in batch_content:
            found_unicode.append(char)
    
    if found_unicode:
        print(f"❌ Batch file contains Unicode characters that will break Windows: {found_unicode}")
        return False
    
    missing_features = []
    for feature in required_bat_features:
        if feature not in batch_content:
            missing_features.append(feature)
    
    if missing_features:
        print(f"❌ Batch file missing features: {missing_features}")
        return False
    
    print("✅ Windows batch file has all required features and proper encoding")
    
    # Check environment file
    env_example = Path(__file__).parent / ".env.example"
    if env_example.exists():
        env_content = env_example.read_text()
        if "AI_DATABASE_PATH" in env_content and "GRAPHVIZ_DOT_EXECUTABLE" in env_content:
            print("✅ Environment configuration supports Windows paths")
        else:
            print("❌ Environment configuration missing path settings")
            return False
    
    return True

def check_path_utilities() -> bool:
    """Check if path utilities support Windows paths"""
    print("\n🛣️  Checking path utilities...")
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from cortex_engine.utils.path_utils import convert_windows_to_wsl_path, process_drag_drop_path
        
        # Test Windows path conversion
        test_cases = [
            ("C:\\Users\\Test", "/mnt/c/Users/Test"),
            ("F:/data/ai_databases", "/mnt/f/data/ai_databases"),
            ("D:\\My Documents\\file.txt", "/mnt/d/My Documents/file.txt"),
        ]
        
        all_passed = True
        for input_path, expected in test_cases:
            result = convert_windows_to_wsl_path(input_path)
            if result == expected:
                print(f"✅ Path conversion: {input_path} -> {result}")
            else:
                print(f"❌ Path conversion failed: {input_path} -> {result} (expected: {expected})")
                all_passed = False
        
        # Test drag-drop functionality
        test_drag_drops = [
            'file:///C:/Users/Test/document.pdf',
            '"C:\\Users\\Test\\My File.txt"',
            '/Users/test/document.pdf',  # Mac style
        ]
        
        for drag_path in test_drag_drops:
            result = process_drag_drop_path(drag_path)
            if result is not None:
                print(f"✅ Drag-drop processing: {drag_path} -> {result}")
            else:
                print(f"❌ Drag-drop processing failed: {drag_path}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Path utilities error: {e}")
        return False

def validate_dockerfile() -> bool:
    """Validate Dockerfile configuration"""
    print("\n🐳 Validating Dockerfile...")
    
    dockerfile_path = Path(__file__).parent / "Dockerfile"
    if not dockerfile_path.exists():
        print("❌ Dockerfile not found")
        return False
    
    dockerfile_content = dockerfile_path.read_text()
    
    required_components = [
        "FROM python:3.11",          # Python 3.11 requirement
        "graphviz",                  # Graphviz for mind maps
        "ollama",                    # Ollama installation
        "mistral-small3.2",          # New model requirement
        "en_core_web_sm",           # spaCy model
        "EXPOSE 8501",              # Streamlit port
        "8000",                     # API port (can be on same EXPOSE line)
    ]
    
    missing_components = []
    for component in required_components:
        if component not in dockerfile_content:
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Dockerfile missing components: {missing_components}")
        return False
    
    print("✅ Dockerfile has all required components")
    return True

def validate_new_features() -> bool:
    """Validate that new features are included"""
    print("\n🆕 Checking new features...")
    
    # Check if the latest Proposal Copilot with undo/hint features is present
    copilot_file = Path(__file__).parent.parent / "pages" / "Proposal_Copilot.py"
    if not copilot_file.exists():
        print("❌ Proposal_Copilot.py not found")
        return False
    
    copilot_content = copilot_file.read_text()
    
    new_features = [
        "handle_undo_action",                    # Undo functionality
        "hint_input_",                          # Hint input boxes
        "Additional guidance for AI",           # Hint UI text
        "enhanced_sub_instruction",             # Enhanced instruction handling
    ]
    
    missing_features = []
    for feature in new_features:
        if feature not in copilot_content:
            missing_features.append(feature)
    
    if missing_features:
        print(f"❌ Missing new features: {missing_features}")
        return False
    
    print("✅ All new features are included")
    
    # Check Knowledge Search bulk functionality
    search_file = Path(__file__).parent.parent / "pages" / "3_Knowledge_Search.py"
    if search_file.exists():
        search_content = search_file.read_text()
        if "➕ Add All to Collection" in search_content:
            print("✅ Knowledge Search bulk functionality included")
        else:
            print("❌ Knowledge Search bulk functionality missing")
            return False
    
    # Check task engine for new prompt template
    task_engine_file = Path(__file__).parent.parent / "cortex_engine" / "task_engine.py"
    if task_engine_file.exists():
        task_engine_content = task_engine_file.read_text()
        if "GENERATE_FROM_KB_AND_PROPOSAL_PROMPT" in task_engine_content:
            print("✅ New prompt template included in task engine")
        else:
            print("❌ New prompt template missing from task engine")
            return False
    
    return True

def validate_requirements() -> bool:
    """Validate requirements.txt is up to date"""
    print("\n📦 Checking requirements.txt...")
    
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    requirements_content = requirements_file.read_text()
    
    critical_packages = [
        "streamlit==1.36.0",
        "llama-index==0.10.50", 
        "numpy<2.0.0",              # Critical version constraint
        "spacy>=3.5.0,<3.8.0",      # spaCy compatibility
        "mistral-small3.2",          # Would be in Dockerfile/setup
        "chromadb==0.5.3",
        "fastapi==0.104.1",
    ]
    
    missing_packages = []
    for package in critical_packages:
        # For mistral-small3.2, it's a model not a package
        if package == "mistral-small3.2":
            continue
        if package not in requirements_content:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages in requirements.txt: {missing_packages}")
        return False
    
    print("✅ requirements.txt has all critical packages")
    return True

def main():
    """Main validation function"""
    print("🔍 Cortex Suite Docker Setup Validation")
    print("=" * 50)
    
    checks = [
        ("Docker Installation", check_docker_installation),
        ("Required Files", check_required_files), 
        ("Windows Compatibility", validate_windows_compatibility),
        ("Path Utilities", check_path_utilities),
        ("Dockerfile Validation", validate_dockerfile),
        ("New Features", validate_new_features),
        ("Requirements", validate_requirements),
    ]
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
            if not results[check_name]:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} validation failed with error: {e}")
            results[check_name] = False
            all_passed = False
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    print("=" * 50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Docker setup is ready for distribution")
        print("✅ Windows compatibility confirmed")
        print("✅ All new features included")
        print("\n📋 Ready to ship:")
        print("   1. ZIP file can be distributed")  
        print("   2. Users just need Docker Desktop")
        print("   3. Works on Windows, Mac, and Linux")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("🔧 Please fix the issues above before distribution")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())