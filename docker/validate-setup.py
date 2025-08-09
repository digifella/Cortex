#!/usr/bin/env python3
"""
Cortex Suite Setup Validator

This script validates that all required files are present for Docker distribution.
Run this before sharing the Cortex Suite with others.
"""

import os
import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists and report status"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_directory(dirpath, description):
    """Check if a directory exists and report status"""
    if Path(dirpath).is_dir():
        file_count = len(list(Path(dirpath).rglob('*')))
        print(f"✅ {description}: {dirpath} ({file_count} files)")
        return True
    else:
        print(f"❌ {description}: {dirpath} - MISSING")
        return False

def validate_docker_setup():
    """Validate Docker setup files"""
    print("🔍 Validating Docker Distribution Setup...")
    print("=" * 50)
    
    # Get the project root (parent of docker directory)
    docker_dir = Path(__file__).parent
    project_root = docker_dir.parent
    
    os.chdir(project_root)
    
    all_good = True
    
    print("\n📂 Essential Application Files:")
    all_good &= check_file("Cortex_Suite.py", "Main application")
    all_good &= check_file("requirements.txt", "Python dependencies")
    all_good &= check_directory("cortex_engine", "Core engine")
    all_good &= check_directory("pages", "UI pages")
    all_good &= check_directory("api", "REST API")
    
    print("\n🐳 Docker Files:")
    all_good &= check_file("docker/Dockerfile", "Main Docker container")
    all_good &= check_file("docker/.dockerignore", "Docker ignore rules")
    all_good &= check_file("docker/.env.example", "Environment template")
    
    print("\n🚀 Launcher Scripts:")
    all_good &= check_file("docker/run-cortex.sh", "Linux/Mac launcher")
    all_good &= check_file("docker/run-cortex.bat", "Windows launcher")
    
    print("\n📖 Documentation:")
    all_good &= check_file("docker/QUICK_START.md", "Quick start guide")
    all_good &= check_file("docker/EASY_SETUP.md", "Easy setup guide")
    all_good &= check_file("DISTRIBUTION_README.md", "Distribution instructions")
    
    print("\n🔧 Optional Files:")
    check_file("docker/deployment-scripts/deploy.sh", "Advanced deployment script")
    check_file("docker/docker-compose.yml", "Multi-container setup")
    check_file("docker/README.md", "Comprehensive Docker guide")
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🎉 SUCCESS: All essential files are present!")
        print("📦 Your Cortex Suite is ready for distribution!")
        print()
        print("📋 To share with others:")
        print("1. Create a ZIP file with the entire project folder")
        print("2. Include the DISTRIBUTION_README.md as instructions")
        print("3. Tell them to follow docker/QUICK_START.md")
        print()
        print("🌐 They'll access it at: http://localhost:8501")
        return True
    else:
        print("❌ FAILED: Some essential files are missing!")
        print("Please ensure all required files are present before distribution.")
        return False

def check_requirements():
    """Check if requirements.txt has all necessary dependencies"""
    print("\n🔍 Checking Python Dependencies...")
    
    required_packages = [
        'streamlit',
        'fastapi',
        'uvicorn',
        'ollama',
        'chromadb',
        'spacy',
        'networkx',
        'aiofiles',
        'pytest'
    ]
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().lower()
        
        missing = []
        for package in required_packages:
            if package not in requirements:
                missing.append(package)
        
        if missing:
            print(f"⚠️ WARNING: These packages might be missing: {', '.join(missing)}")
        else:
            print("✅ All essential packages found in requirements.txt")
            
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        return False
    
    return True

def estimate_distribution_size():
    """Estimate the size of the distribution package"""
    print("\n📊 Estimating Distribution Size...")
    
    total_size = 0
    file_count = 0
    
    # Essential directories to include
    dirs_to_check = ['cortex_engine', 'pages', 'api', 'docker']
    
    for dir_name in dirs_to_check:
        if Path(dir_name).exists():
            for file_path in Path(dir_name).rglob('*'):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    try:
                        total_size += file_path.stat().st_size
                        file_count += 1
                    except:
                        pass
    
    # Add main files
    main_files = ['Cortex_Suite.py', 'requirements.txt', 'DISTRIBUTION_README.md']
    for file_name in main_files:
        if Path(file_name).exists():
            try:
                total_size += Path(file_name).stat().st_size
                file_count += 1
            except:
                pass
    
    size_mb = total_size / (1024 * 1024)
    print(f"📦 Distribution package: ~{size_mb:.1f}MB ({file_count} files)")
    print(f"💾 User needs: ~10GB free space (including AI models)")
    
    return size_mb, file_count

def main():
    """Main validation function"""
    print("🚀 Cortex Suite Distribution Validator")
    print("====================================")
    
    # Validate setup
    setup_valid = validate_docker_setup()
    
    # Check requirements
    requirements_valid = check_requirements()
    
    # Estimate size
    size_mb, file_count = estimate_distribution_size()
    
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY:")
    print(f"   Setup files: {'✅ PASS' if setup_valid else '❌ FAIL'}")
    print(f"   Requirements: {'✅ PASS' if requirements_valid else '⚠️ WARNING'}")
    print(f"   Package size: {size_mb:.1f}MB ({file_count} files)")
    
    if setup_valid:
        print("\n🎉 READY FOR DISTRIBUTION!")
        print("   Users can run docker/run-cortex.sh or run-cortex.bat")
        print("   First launch takes 5-10 minutes (AI model download)")
        print("   Subsequent launches take ~30 seconds")
        return 0
    else:
        print("\n❌ NOT READY - Fix missing files first")
        return 1

if __name__ == "__main__":
    sys.exit(main())